"""Frozen-backbone rollout fine-tuning with guarded dry-region supervision."""

import json
import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src.trainer.rain_trainer_ccc_frozen_ablation import CCCFrozenAblationTrainer
from src.trainer.rollout_dry_constraints import assess_candidate, bounded_auxiliary, dry_rollout_constraint


class RolloutDryTrainer(CCCFrozenAblationTrainer):
    def _next_prediction_loss(self, pred: dict[str, torch.Tensor], target_gt: dict[str, torch.Tensor],
                              aux: dict | None = None, loss_weight_override: dict[str, float] | None = None) -> tuple[torch.Tensor, dict]:
        base, logs = super()._next_prediction_loss(pred, target_gt, aux, loss_weight_override)
        cfg = self.train_cfg.get("rollout_dry_constraint", {})
        if not cfg.get("enabled", False) or not aux or aux.get("target_mode") != "rollout_branch":
            return base, logs
        if not aux.get("rain_residual_already_applied", False):
            raise ValueError("Dry supervision requires final rainfall, not an uncombined delta")
        settings = OmegaConf.to_container(cfg, resolve=True)
        rain = self._denormalize_rain_for_metrics(pred["rain"].float())
        target = self._denormalize_rain_for_metrics(target_gt["rain"].float()).clamp_min(0)
        raw, terms = dry_rollout_constraint(rain, target, settings)
        extra, ramp, scale = bounded_auxiliary(base, raw, self.global_step, settings)
        logs.update({f"loss/dry_constraint_{key}": value.detach() for key, value in terms.items()})
        logs["loss/dry_constraint_added_inner"] = extra.detach()
        logs["meta/dry_constraint_ramp"] = base.new_tensor(ramp)
        logs["meta/dry_constraint_budget_scale"] = scale.detach()
        return base + extra, logs

    @torch.no_grad()
    def _run_val(self) -> None:
        super()._run_val()
        report = json.loads((self.proj_dir / "fixed_validation" / f"step_{self.global_step:08d}" / "metrics.json").read_text())
        reference_path = self.proj_dir / "constraint_reference.json"
        if self.global_step == 0:
            reference_path.write_text(json.dumps(report, indent=2))
            return
        reference = json.loads(reference_path.read_text())
        result = assess_candidate(report, reference, self.train_cfg.constraint_selection)
        result["step"] = self.global_step
        with (self.proj_dir / "constraint_selection.jsonl").open("a") as handle:
            handle.write(json.dumps(result) + "\n")
        self.pending_selection = result
        self.log_msg(f"[ConstraintSelection] {result}")

    def _save_checkpoint(self) -> None:
        super()._save_checkpoint()
        result = getattr(self, "pending_selection", None)
        if result is None or result["step"] != self.global_step or not result["eligible"]:
            return
        best_path = self.proj_dir / "best_eligible.json"
        if best_path.exists() and json.loads(best_path.read_text())["long_lead_far"] <= result["long_lead_far"]:
            return
        checkpoints = list((self.proj_dir / "checkpoints").glob("checkpoint_*"))
        saved = max(checkpoints, key=lambda path: int(path.name.removeprefix("checkpoint_")))
        if not (saved / "model.safetensors").is_file():
            raise FileNotFoundError(f"Missing saved weights: {saved}")
        best_path.write_text(json.dumps({**result, "checkpoint": str(saved)}, indent=2))


@hydra.main(config_path="../config/ts_rain_train", config_name="rain_trainer_rollout_dry", version_base=None)
def main(cfg: DictConfig) -> None:
    trainer = RolloutDryTrainer(cfg)
    try:
        if not cfg.train.get("resume_path"):
            trainer._run_val()
        trainer.run()
    finally:
        trainer._close_tensorboard_writer()


if __name__ == "__main__":
    main()
