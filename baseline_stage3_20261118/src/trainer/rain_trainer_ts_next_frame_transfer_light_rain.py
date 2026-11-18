import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import hydra
import torch
from omegaconf import DictConfig

from src.trainer.rain_trainer_ts_next_frame_transfer import RainTSNextFrameTransferTrainer


class RainTSNextFrameTransferLightRainTrainer(RainTSNextFrameTransferTrainer):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        occurrence_cfg = self.train_cfg.get("loss", {}).get("light_rain_occurrence", {})
        self.light_rain_occurrence_enabled = bool(occurrence_cfg.get("enabled", False))
        self.light_rain_occurrence_weight = float(occurrence_cfg.get("weight", 0.0))
        self.light_rain_occurrence_warmup_steps = int(occurrence_cfg.get("warmup_steps", 0))
        self.light_rain_false_alarm_weight = float(occurrence_cfg.get("false_alarm_weight", 1.0))
        self.light_rain_miss_weight = float(occurrence_cfg.get("miss_weight", 1.0))
        self.light_rain_thresholds = [float(value) for value in occurrence_cfg.get("thresholds", [])]
        self.light_rain_temperatures = [float(value) for value in occurrence_cfg.get("temperatures", [])]
        self.light_rain_threshold_weights = [
            float(value) for value in occurrence_cfg.get("threshold_weights", [])
        ]

        if not self.light_rain_occurrence_enabled:
            return
        if self.light_rain_occurrence_weight < 0:
            raise ValueError("train.loss.light_rain_occurrence.weight must be >= 0")
        if self.light_rain_occurrence_warmup_steps < 0:
            raise ValueError("train.loss.light_rain_occurrence.warmup_steps must be >= 0")
        if self.light_rain_false_alarm_weight < 0 or self.light_rain_miss_weight < 0:
            raise ValueError("Light-rain false-alarm and miss weights must be >= 0")
        if not self.light_rain_thresholds:
            raise ValueError("train.loss.light_rain_occurrence.thresholds must not be empty")
        if len(self.light_rain_temperatures) != len(self.light_rain_thresholds):
            raise ValueError("Light-rain temperatures must match thresholds")
        if len(self.light_rain_threshold_weights) != len(self.light_rain_thresholds):
            raise ValueError("Light-rain threshold weights must match thresholds")
        if any(value <= 0 for value in self.light_rain_thresholds):
            raise ValueError("Light-rain thresholds must be > 0")
        if any(value <= 0 for value in self.light_rain_temperatures):
            raise ValueError("Light-rain temperatures must be > 0")
        if any(value < 0 for value in self.light_rain_threshold_weights):
            raise ValueError("Light-rain threshold weights must be >= 0")
        if sum(self.light_rain_threshold_weights) <= 0:
            raise ValueError("At least one light-rain threshold weight must be > 0")

    def _compute_light_rain_occurrence_loss(
        self,
        pred_rain: torch.Tensor,
        target_rain: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        total = pred_rain.new_zeros(())
        threshold_weight_sum = sum(self.light_rain_threshold_weights)
        logs: dict[str, torch.Tensor] = {}

        for threshold, temperature, threshold_weight in zip(
            self.light_rain_thresholds,
            self.light_rain_temperatures,
            self.light_rain_threshold_weights,
            strict=True,
        ):
            probability = torch.sigmoid((pred_rain - threshold) / temperature)
            rain_mask = target_rain >= threshold
            dry_mask = ~rain_mask
            rain_weight = rain_mask.to(dtype=pred_rain.dtype)
            dry_weight = dry_mask.to(dtype=pred_rain.dtype)

            false_alarm = (probability * dry_weight).sum() / dry_weight.sum().clamp_min(1.0)
            miss = ((1.0 - probability) * rain_weight).sum() / rain_weight.sum().clamp_min(1.0)
            threshold_loss = (
                self.light_rain_false_alarm_weight * false_alarm
                + self.light_rain_miss_weight * miss
            )
            total = total + threshold_weight * threshold_loss

            threshold_tag = f"{threshold:g}".replace(".", "p")
            logs[f"loss/light_rain_false_alarm_t{threshold_tag}"] = false_alarm.detach()
            logs[f"loss/light_rain_miss_t{threshold_tag}"] = miss.detach()

        return total / threshold_weight_sum, logs

    def _next_prediction_loss(
        self,
        pred: dict[str, torch.Tensor],
        target_gt: dict[str, torch.Tensor],
        aux: dict[str, float | int | str | torch.Tensor | dict[str, torch.Tensor]] | None = None,
        loss_weight_override: dict[str, float] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        base_loss, logs = super()._next_prediction_loss(pred, target_gt, aux, loss_weight_override)
        if not self.light_rain_occurrence_enabled or self.light_rain_occurrence_weight == 0:
            return base_loss, logs

        pred_rain = pred["rain"]
        is_rain_only_rollout = "radar" not in pred and "satellite" not in pred
        rain_residual_enabled = self._resolve_rain_residual_cfg()[0]
        rain_residual_already_applied = bool(aux.get("rain_residual_already_applied", 0)) if aux else False
        if rain_residual_enabled and not rain_residual_already_applied and not is_rain_only_rollout:
            rain_residual_ref = aux.get("rain_residual_ref") if aux else None
            if not torch.is_tensor(rain_residual_ref):
                raise ValueError("Light-rain occurrence loss requires aux['rain_residual_ref']")
            rain_ref = rain_residual_ref.to(device=pred_rain.device, dtype=pred_rain.dtype)
            pred_rain = self._apply_rain_residual_output(pred_rain, rain_ref)

        occurrence_raw, occurrence_logs = self._compute_light_rain_occurrence_loss(
            pred_rain,
            target_gt["rain"],
        )
        if self.light_rain_occurrence_warmup_steps == 0:
            warmup_scale = 1.0
        else:
            warmup_scale = min(
                float(self.global_step + 1) / float(self.light_rain_occurrence_warmup_steps),
                1.0,
            )
        occurrence_weight = self.light_rain_occurrence_weight * warmup_scale
        occurrence_loss = occurrence_raw * occurrence_weight
        total_loss = base_loss + occurrence_loss

        logs = dict(logs)
        logs.update(occurrence_logs)
        logs["loss/light_rain_occurrence_raw"] = occurrence_raw.detach()
        logs["loss/light_rain_occurrence"] = occurrence_loss.detach()
        logs["meta/light_rain_occurrence_weight"] = occurrence_loss.new_tensor(occurrence_weight)
        logs["loss"] = total_loss.detach()
        return total_loss, logs


@hydra.main(
    config_path="../config/ts_rain_train",
    config_name="rain_trainer_ts_next_frame_light_rain",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    trainer = RainTSNextFrameTransferLightRainTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
