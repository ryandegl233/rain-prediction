"""Train only a soft occurrence head while preserving the deterministic predictor."""

import hashlib
import json
from pathlib import Path

import accelerate
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from safetensors import safe_open

from src.networks.time_series.causal_patch_transformer_soft_occurrence import freeze_for_occurrence_training
from src.trainer.rain_trainer_ccc_frozen_ablation import frozen_digest
from src.trainer.rain_trainer_coarse_evolution import CoarseEvolutionTrainer
from src.trainer.rollout_dry_constraints import assess_candidate, bounded_auxiliary, dry_rollout_constraint
from src.trainer.soft_occurrence_objective import resolve_gate_strength, soft_occurrence_loss


class SoftOccurrenceTrainer(CoarseEvolutionTrainer):
    def __init__(self, cfg: DictConfig) -> None:
        if cfg.train.get("resume_path"):
            raise ValueError("Use a fresh paired occurrence experiment; exact-resume audit is not implemented")
        if bool(cfg.dataset.train.get("modality_zero_centering", False)):
            raise ValueError("Soft occurrence model currently requires rainfall in physical units")
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
                    raise FloatingPointError(f"Nonfinite occurrence gradient: {key}")
                peak = float(gradient.detach().abs().max())
                self.gradient_audit[key]["nonzero_calls"] += int(peak > 0)
                self.gradient_audit[key]["max_abs"] = max(self.gradient_audit[key]["max_abs"], peak)

            parameter.register_hook(record)
        (self.proj_dir / "trainable_scope.json").write_text(json.dumps(self.scope_audit, indent=2))
        self.log_msg(f"[OccurrenceScope] {self.scope_audit['parameter_counts']}")

    def _build_optim_sched(self) -> tuple:
        if self.accelerator.state.deepspeed_plugin is not None:
            raise ValueError("Occurrence experiment requires one GPU without DeepSpeed")
        model_state = self.model.state_dict()
        with safe_open(Path(self.train_cfg.init_model_path) / "model.safetensors", framework="pt", device="cpu") as source:
            source_keys = set(source.keys())
            if any(key.startswith("occurrence_head.") or key == "soft_gate_strength" for key in source_keys):
                raise ValueError("Initialize from a checkpoint without an occurrence head")
            if source_keys - set(model_state):
                raise ValueError("Initialization checkpoint has unexpected tensors")
            extra = set(model_state) - source_keys
            if any(not (key.startswith("occurrence_head.") or key == "soft_gate_strength") for key in extra):
                raise ValueError("Unexpected missing original-model tensor")
            for key in source_keys:
                if not torch.equal(model_state[key].cpu(), source.get_tensor(key)):
                    raise ValueError(f"Original checkpoint tensor not loaded exactly: {key}")
        self.scope_audit = freeze_for_occurrence_training(self.model)
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        optimizer = hydra.utils.instantiate(self.train_cfg.optim)(parameters)
        scheduler = hydra.utils.instantiate(self.train_cfg.scheduler)(optimizer=optimizer)
        return optimizer, scheduler

    def _set_gate_schedule(self) -> float:
        cfg = self.train_cfg.soft_occurrence
        strength = resolve_gate_strength(
            self.global_step,
            int(cfg.gate_start_step),
            int(cfg.gate_ramp_steps),
        )
        self.accelerator.unwrap_model(self.model).set_soft_gate_strength(strength)
        return strength

    def train_step(self, batch: dict[str, torch.Tensor]) -> tuple[dict, bool]:
        if self.global_step < 2:
            digest = hashlib.sha256()
            for name, value in sorted(batch.items()):
                if torch.is_tensor(value):
                    digest.update(name.encode())
                    digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
            self.batch_fingerprints.append(digest.hexdigest())
        self._set_gate_schedule()
        logs, did_step = super().train_step(batch)
        logs["meta/soft_gate_strength"] = torch.tensor(
            float(self.accelerator.unwrap_model(self.model).soft_gate_strength), device=self.device,
        )
        return logs, did_step

    def _next_prediction_loss(
        self,
        pred: dict,
        target_gt: dict,
        aux: dict | None = None,
        loss_weight_override: dict | None = None,
    ) -> tuple[torch.Tensor, dict]:
        base, logs = super()._next_prediction_loss(pred, target_gt, aux, loss_weight_override)
        occurrence_cfg = self.train_cfg.soft_occurrence
        logits = pred.get("occurrence_logits")
        if torch.is_tensor(logits):
            target_mm_min = self._denormalize_rain_for_metrics(target_gt["rain"].float()).clamp_min(0)
            raw, terms = soft_occurrence_loss(
                logits,
                target_mm_min,
                threshold_mm_min=float(occurrence_cfg.threshold_mm_min),
                false_positive_weight=float(occurrence_cfg.false_positive_weight),
                strong_threshold_mm_min=float(occurrence_cfg.strong_threshold_mm_min),
                strong_weight=float(occurrence_cfg.strong_weight),
                pod_floor=float(occurrence_cfg.pod_floor),
                pod_floor_weight=float(occurrence_cfg.pod_floor_weight),
            )
            weighted = float(occurrence_cfg.loss_weight) * raw
            base = base + weighted
            logs.update({f"loss/occurrence_{key}": value.detach() for key, value in terms.items()})
            logs["loss/occurrence_raw"] = raw.detach()
            logs["loss/occurrence_weighted"] = weighted.detach()

        dry_cfg = self.train_cfg.get("rollout_dry_constraint", {})
        if dry_cfg.get("enabled", False) and aux and aux.get("target_mode") == "rollout_branch":
            settings = OmegaConf.to_container(dry_cfg, resolve=True)
            rain = self._denormalize_rain_for_metrics(pred["rain"].float())
            target = self._denormalize_rain_for_metrics(target_gt["rain"].float()).clamp_min(0)
            raw, terms = dry_rollout_constraint(rain, target, settings)
            extra, ramp, scale = bounded_auxiliary(base, raw, self.global_step, settings)
            base = base + extra
            logs.update({f"loss/dry_constraint_{key}": value.detach() for key, value in terms.items()})
            logs["loss/dry_constraint_added_inner"] = extra.detach()
            logs["meta/dry_constraint_ramp"] = base.new_tensor(ramp)
            logs["meta/dry_constraint_budget_scale"] = scale.detach()
        return base, logs

    @torch.no_grad()
    def _run_val(self) -> None:
        strength = self._set_gate_schedule()
        model = self.accelerator.unwrap_model(self.model)
        if frozen_digest(model) != self.original_digest:
            raise RuntimeError("Frozen original predictor changed")
        super()._run_val()
        report_path = self.proj_dir / "fixed_validation" / f"step_{self.global_step:08d}" / "metrics.json"
        report = json.loads(report_path.read_text())
        report["soft_occurrence"] = {
            "gate_strength": strength,
            "gate_gamma": float(self.train_cfg.soft_occurrence.gate_gamma),
            "false_positive_weight": float(self.train_cfg.soft_occurrence.false_positive_weight),
        }
        report_path.write_text(json.dumps(report, indent=2))
        reference_path = self.proj_dir / "constraint_reference.json"
        if self.global_step == 0:
            reference_path.write_text(json.dumps(report, indent=2))
        else:
            result = assess_candidate(report, json.loads(reference_path.read_text()), self.train_cfg.constraint_selection)
            result["step"] = self.global_step
            with (self.proj_dir / "constraint_selection.jsonl").open("a") as handle:
                handle.write(json.dumps(result) + "\n")
            self.pending_selection = result
            self.log_msg(f"[ConstraintSelection] {result}")
        with (self.proj_dir / "gradient_audit.jsonl").open("a") as handle:
            handle.write(json.dumps({
                "step": self.global_step,
                "frozen_unchanged": True,
                "gradients": self.gradient_audit,
                "first_two_updates_batches": self.batch_fingerprints,
            }) + "\n")
        if self.global_step >= 2 and self.gradient_audit["occurrence_head.projection.weight"]["nonzero_calls"] == 0:
            raise RuntimeError("Occurrence projection received no gradient")

    def _save_checkpoint(self) -> None:
        super()._save_checkpoint()
        result = getattr(self, "pending_selection", None)
        if result is None or result["step"] != self.global_step or not result["eligible"]:
            return
        best = self.proj_dir / "best_eligible.json"
        if best.exists() and json.loads(best.read_text())["long_lead_far"] <= result["long_lead_far"]:
            return
        checkpoint = max(
            (self.proj_dir / "checkpoints").glob("checkpoint_*"),
            key=lambda path: int(path.name.removeprefix("checkpoint_")),
        )
        best.write_text(json.dumps({**result, "checkpoint": str(checkpoint)}, indent=2))


@hydra.main(config_path="../config/ts_rain_train", config_name="rain_trainer_soft_occurrence", version_base=None)
def main(cfg: DictConfig) -> None:
    trainer = SoftOccurrenceTrainer(cfg)
    try:
        trainer._run_val()
        trainer.run()
    finally:
        trainer._close_tensorboard_writer()


if __name__ == "__main__":
    main()
