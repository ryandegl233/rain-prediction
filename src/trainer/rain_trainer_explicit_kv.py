"""Train only explicit causal K/V encoding on top of the accepted occurrence model."""

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from safetensors import safe_open
import torch

from src.networks.time_series.causal_patch_transformer_explicit_kv import freeze_for_explicit_kv
from src.trainer.rain_trainer_ccc_frozen_ablation import frozen_digest
from src.trainer.rain_trainer_coarse_evolution import CoarseEvolutionTrainer
from src.trainer.rollout_dry_constraints import assess_candidate, bounded_auxiliary, dry_rollout_constraint
from src.trainer.soft_occurrence_objective import soft_occurrence_loss


class ExplicitKVTrainer(CoarseEvolutionTrainer):
    def __init__(self, cfg: DictConfig) -> None:
        if cfg.train.get("resume_path"):
            raise ValueError("Start this isolated K/V experiment from the configured occurrence checkpoint")
        super().__init__(cfg)
        model = self.accelerator.unwrap_model(self.model)
        self.initial_frozen_digest = frozen_digest(model)
        self.gradient_audit: dict[str, dict[str, float | int]] = {}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            self.gradient_audit[name] = {"calls": 0, "nonzero_calls": 0, "max_abs": 0.0}

            def record(gradient: torch.Tensor, key: str = name) -> None:
                if self.global_step >= 4:
                    return
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError(f"Nonfinite explicit K/V gradient: {key}")
                peak = float(gradient.detach().abs().max())
                row = self.gradient_audit[key]
                row["calls"] += 1
                row["nonzero_calls"] += int(peak > 0)
                row["max_abs"] = max(float(row["max_abs"]), peak)

            parameter.register_hook(record)
        self.scope_audit["initial_frozen_sha256"] = self.initial_frozen_digest
        (self.proj_dir / "trainable_scope.json").write_text(json.dumps(self.scope_audit, indent=2))
        self.log_msg(f"[ExplicitKVScope] {self.scope_audit['parameter_counts']}")

    def _build_optim_sched(self) -> tuple:
        if self.accelerator.state.deepspeed_plugin is not None:
            raise ValueError("Explicit K/V ablation requires one GPU without DeepSpeed")
        checkpoint_path = Path(self.train_cfg.init_model_path) / "model.safetensors"
        model_state = self.model.state_dict()
        with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
            source_keys = set(checkpoint.keys())
            if source_keys - set(model_state):
                raise ValueError("Occurrence checkpoint contains tensors absent from the explicit K/V model")
            missing = set(model_state) - source_keys
            if any(not key.startswith("cross_modal_adapter.correction.") for key in missing):
                raise ValueError("Only the new explicit K/V correction may be absent from the checkpoint")
            for key in source_keys:
                if not torch.equal(model_state[key].cpu(), checkpoint.get_tensor(key)):
                    raise ValueError(f"Checkpoint tensor was not loaded exactly: {key}")
        self.model.cross_modal_adapter.initialize_correction()
        self.model.set_soft_gate_strength(1.0)
        self.scope_audit = freeze_for_explicit_kv(self.model)
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        optimizer = hydra.utils.instantiate(self.train_cfg.optim)(parameters)
        scheduler = hydra.utils.instantiate(self.train_cfg.scheduler)(optimizer=optimizer)
        return optimizer, scheduler

    def _rollout_predict_with_settings(
        self,
        context: torch.Tensor,
        total_future_frames: int,
        mode: str,
        rollout_block_size: int,
        detach_history: bool,
        context_time: torch.Tensor | None = None,
        future_time: torch.Tensor | None = None,
        future_modalities: dict[str, torch.Tensor] | None = None,
        use_gt_future_modalities: bool = False,
        future_rain: torch.Tensor | None = None,
        use_gt_future_rain: bool = False,
        return_rain_delta: bool = False,
    ) -> dict[str, torch.Tensor]:
        if mode != "frame" or rollout_block_size != 1:
            raise ValueError("Explicit K/V experiment requires one-frame causal rollout")
        if use_gt_future_modalities or use_gt_future_rain:
            raise ValueError("Explicit K/V rollout must not read observed future modalities")
        if total_future_frames <= 0:
            raise ValueError("total_future_frames must be positive")
        if context_time is not None and (
            future_time is None
            or context_time.shape != (context.shape[0], context.shape[2])
            or future_time.shape != (context.shape[0], total_future_frames)
        ):
            raise ValueError("Invalid rollout time tensors")

        current = context
        current_time = context_time
        generated_mask = torch.zeros(
            context.shape[0], context.shape[2], dtype=torch.bool, device=context.device,
        )
        outputs: dict[str, list[torch.Tensor]] = {
            "radar": [], "satellite": [], "rain": [], "rain_delta": [], "occurrence_logits": [],
        }
        for lead in range(total_future_frames):
            seed = current[:, :, -1:]
            kwargs = {
                "context_x": current,
                "target_x": seed,
                "predict_frames": 1,
                "strict_target_isolation": bool(self.train_cfg.strict_target_isolation),
                "return_modality_dict": True,
                "context_generated_mask": generated_mask,
            }
            if current_time is not None:
                kwargs["context_time"] = current_time
                kwargs["target_time"] = current_time[:, -1:]
            with self.accelerator.autocast():
                prediction = self.model.forward_ar(**kwargs)
            outputs["radar"].append(prediction["radar"])
            outputs["satellite"].append(prediction["satellite"])
            outputs["rain"].append(prediction["rain"])
            outputs["rain_delta"].append(prediction["rain_delta"])
            outputs["occurrence_logits"].append(prediction["occurrence_logits"])
            history = self._merge_modalities(
                prediction["radar"], prediction["satellite"], prediction["rain_rollout"],
            )
            if detach_history:
                history = history.detach()
            current = torch.cat([current, history], dim=2)
            generated_mask = torch.cat([
                generated_mask,
                torch.ones(context.shape[0], 1, dtype=torch.bool, device=context.device),
            ], dim=1)
            if current_time is not None:
                current_time = torch.cat([current_time, future_time[:, lead : lead + 1]], dim=1)
        result = {key: torch.cat(value, dim=2) for key, value in outputs.items() if value}
        if not return_rain_delta:
            result.pop("rain_delta")
        return result

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
        model = self.accelerator.unwrap_model(self.model)
        if frozen_digest(model) != self.initial_frozen_digest:
            raise RuntimeError("Frozen occurrence baseline changed")
        super()._run_val()
        report_path = self.proj_dir / "fixed_validation" / f"step_{self.global_step:08d}" / "metrics.json"
        report = json.loads(report_path.read_text())
        reference_path = self.proj_dir / "constraint_reference.json"
        if self.global_step == 0:
            reference_path.write_text(json.dumps(report, indent=2))
        else:
            self.pending_selection = {
                **assess_candidate(report, json.loads(reference_path.read_text()), self.train_cfg.constraint_selection),
                "step": self.global_step,
            }
            with (self.proj_dir / "constraint_selection.jsonl").open("a") as handle:
                handle.write(json.dumps(self.pending_selection) + "\n")
            self.log_msg(f"[ConstraintSelection] {self.pending_selection}")
        with (self.proj_dir / "gradient_audit.jsonl").open("a") as handle:
            handle.write(json.dumps({
                "step": self.global_step,
                "frozen_unchanged": True,
                "gradients": self.gradient_audit,
            }) + "\n")

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


@hydra.main(config_path="../config/ts_rain_train", config_name="rain_trainer_explicit_kv", version_base=None)
def main(cfg: DictConfig) -> None:
    trainer = ExplicitKVTrainer(cfg)
    try:
        trainer._run_val()
        trainer.run()
    finally:
        trainer._close_tensorboard_writer()


if __name__ == "__main__":
    main()
