"""Train only the additional correction while auditing the entire original model."""

import hashlib
import json
from pathlib import Path

import accelerate
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from safetensors import safe_open

from src.networks.time_series.causal_patch_transformer_residual_neighbor import freeze_original_model
from src.trainer.rain_trainer_ccc_frozen_ablation import frozen_digest
from src.trainer.rain_trainer_coarse_evolution import CoarseEvolutionTrainer
from src.trainer.rollout_dry_constraints import assess_candidate, bounded_auxiliary, dry_rollout_constraint


class ResidualNeighborTrainer(CoarseEvolutionTrainer):
    def __init__(self, cfg: DictConfig) -> None:
        if cfg.train.get("resume_path"):
            raise ValueError("Use a fresh paired correction experiment; exact-resume audit is not implemented")
        super().__init__(cfg)
        accelerate.utils.set_seed(int(self.train_cfg.seed))
        model = self.accelerator.unwrap_model(self.model)
        self.original_digest = frozen_digest(model)
        self.scope_audit["initial_frozen_sha256"] = self.original_digest
        self.batch_fingerprints = []
        self.gradient_audit = {}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            self.gradient_audit[name] = {"nonzero_calls": 0, "max_abs": 0.0}

            def record(gradient: torch.Tensor, key: str = name) -> None:
                if self.global_step >= 2:
                    return
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError(f"Nonfinite correction gradient: {key}")
                peak = gradient.detach().abs().max().item()
                self.gradient_audit[key]["nonzero_calls"] += int(peak > 0)
                self.gradient_audit[key]["max_abs"] = max(self.gradient_audit[key]["max_abs"], peak)

            parameter.register_hook(record)
        (self.proj_dir / "trainable_scope.json").write_text(json.dumps(self.scope_audit, indent=2))
        self.log_msg(f"[CorrectionScope] {self.scope_audit['parameter_counts']}")

    def _build_optim_sched(self) -> tuple:
        if self.accelerator.state.deepspeed_plugin is not None:
            raise ValueError("Correction experiment requires one GPU without DeepSpeed")
        model_state = self.model.state_dict()
        with safe_open(Path(self.train_cfg.init_model_path) / "model.safetensors", framework="pt", device="cpu") as source:
            keys = set(source.keys())
            if any(".correction." in key for key in keys) or keys - set(model_state):
                raise ValueError("Initialize from the original checkpoint without an added correction")
            if any(not key.startswith("cross_modal_adapter.correction.") for key in set(model_state) - keys):
                raise ValueError("Unexpected missing original-model tensor")
            for key in keys:
                if not torch.equal(model_state[key].cpu(), source.get_tensor(key)):
                    raise ValueError(f"Original checkpoint tensor not loaded exactly: {key}")
        self.model.cross_modal_adapter.initialize_correction()
        self.scope_audit = freeze_original_model(self.model)
        named = [(name, value) for name, value in self.model.named_parameters() if value.requires_grad]
        params = named if "muon" in self.train_cfg.optim._target_ else [value for _, value in named]
        optimizer = hydra.utils.instantiate(self.train_cfg.optim)(params)
        scheduler = hydra.utils.instantiate(self.train_cfg.scheduler)(optimizer=optimizer)
        return optimizer, scheduler

    def train_step(self, batch: dict[str, torch.Tensor]) -> tuple[dict, bool]:
        if self.global_step < 2:
            digest = hashlib.sha256()
            for name, value in sorted(batch.items()):
                if torch.is_tensor(value):
                    digest.update(name.encode())
                    digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
            self.batch_fingerprints.append(digest.hexdigest())
        return super().train_step(batch)

    def _next_prediction_loss(self, pred: dict, target_gt: dict, aux: dict | None = None,
                              loss_weight_override: dict | None = None) -> tuple[torch.Tensor, dict]:
        base, logs = super()._next_prediction_loss(pred, target_gt, aux, loss_weight_override)
        cfg = self.train_cfg.get("rollout_dry_constraint", {})
        if cfg.get("enabled", False) and aux and aux.get("target_mode") == "rollout_branch":
            if not aux.get("rain_residual_already_applied", False):
                raise ValueError("Expected final rainfall for dry supervision")
            settings = OmegaConf.to_container(cfg, resolve=True)
            rain = self._denormalize_rain_for_metrics(pred["rain"].float())
            target = self._denormalize_rain_for_metrics(target_gt["rain"].float()).clamp_min(0)
            raw, terms = dry_rollout_constraint(rain, target, settings)
            extra, ramp, scale = bounded_auxiliary(base, raw, self.global_step, settings)
            logs.update({f"loss/dry_constraint_{key}": value.detach() for key, value in terms.items()})
            logs.update({"loss/dry_constraint_added_inner": extra.detach(), "meta/dry_constraint_ramp": base.new_tensor(ramp),
                         "meta/dry_constraint_budget_scale": scale.detach()})
            base = base + extra
        return base, logs

    @torch.no_grad()
    def _run_val(self) -> None:
        model = self.accelerator.unwrap_model(self.model)
        if frozen_digest(model) != self.original_digest:
            raise RuntimeError("Original model weights changed")
        if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
            raise RuntimeError("Frozen original parameter received a gradient")
        if self.global_step >= 2:
            required = ("attention.out_proj.weight", "attention.in_proj_weight", "adapters.0.weight", "adapters.1.weight", "adapters.2.weight")
            for suffix in required:
                if self.gradient_audit[f"cross_modal_adapter.correction.{suffix}"]["nonzero_calls"] == 0:
                    raise RuntimeError(f"Correction did not learn in the first two updates: {suffix}")
        super()._run_val()
        report = json.loads((self.proj_dir / "fixed_validation" / f"step_{self.global_step:08d}" / "metrics.json").read_text())
        reference_path = self.proj_dir / "constraint_reference.json"
        if self.global_step == 0:
            reference_path.write_text(json.dumps(report, indent=2))
        else:
            self.pending_selection = {**assess_candidate(report, json.loads(reference_path.read_text()), self.train_cfg.constraint_selection), "step": self.global_step}
            with (self.proj_dir / "constraint_selection.jsonl").open("a") as handle:
                handle.write(json.dumps(self.pending_selection) + "\n")
            self.log_msg(f"[ConstraintSelection] {self.pending_selection}")
        with (self.proj_dir / "gradient_audit.jsonl").open("a") as handle:
            handle.write(json.dumps({"step": self.global_step, "frozen_unchanged": True, "gradients": self.gradient_audit,
                                     "first_two_updates_batches": self.batch_fingerprints}) + "\n")

    def _save_checkpoint(self) -> None:
        super()._save_checkpoint()
        result = getattr(self, "pending_selection", None)
        if result is None or result["step"] != self.global_step or not result["eligible"]:
            return
        best = self.proj_dir / "best_eligible.json"
        if best.exists() and json.loads(best.read_text())["long_lead_far"] <= result["long_lead_far"]:
            return
        saved = max((self.proj_dir / "checkpoints").glob("checkpoint_*"), key=lambda p: int(p.name.removeprefix("checkpoint_")))
        if not (saved / "model.safetensors").is_file():
            raise FileNotFoundError(saved)
        best.write_text(json.dumps({**result, "checkpoint": str(saved)}, indent=2))


@hydra.main(config_path="../config/ts_rain_train", config_name="rain_trainer_residual_neighbor", version_base=None)
def main(cfg: DictConfig) -> None:
    trainer = ResidualNeighborTrainer(cfg)
    try:
        trainer._run_val()
        trainer.run()
    finally:
        trainer._close_tensorboard_writer()


if __name__ == "__main__":
    main()
