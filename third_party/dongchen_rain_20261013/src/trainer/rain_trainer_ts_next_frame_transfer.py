import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import hydra
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig, OmegaConf

from src.networks.time_series.causal_patch_transformer_next_frame import RainCausalPatchTransformerNextFrame
from src.trainer.rain_trainer_ts_next_frame import RainTSNextFrameTrainer


class RainCausalPatchTransformerRainDecode(RainCausalPatchTransformerNextFrame):
    def _decode_target_tokens(
        self,
        target_tokens: torch.Tensor,
        hp: int,
        wp: int,
        h: int,
        w: int,
        batch_size: int,
        predict_frames: int,
        context_anchor: dict[str, torch.Tensor] | None = None,
        film_condition: dict[str, torch.Tensor] | None = None,
        return_modality_dict: bool = True,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        if return_modality_dict:
            return super()._decode_target_tokens(
                target_tokens, hp, wp, h, w, batch_size, predict_frames,
                context_anchor=context_anchor,
                film_condition=film_condition,
                return_modality_dict=True,
            )

        rain_cond = None
        if self.decoder_condition_mode == "film" and film_condition is not None:
            rain_cond = film_condition["rain"]
        if self.decoder_type == "2d":
            features = rearrange(target_tokens, "b p (hp wp) d -> (b p) d hp wp", hp=hp, wp=wp)
            if rain_cond is not None:
                rain_cond = rearrange(rain_cond, "b c p h w -> (b p) c h w")
            rain = self.rain_decoder(features, target_hw=(h, w), cond=rain_cond)
            rain = rearrange(rain, "(b p) c h w -> b c p h w", b=batch_size, p=predict_frames)
        else:
            features = rearrange(target_tokens, "b p (hp wp) d -> b d p hp wp", hp=hp, wp=wp)
            rain = self.rain_decoder(
                features, target_thw=(predict_frames * self.frame_patch_size, h, w), cond=rain_cond
            )
        if self.decoder_output_mode == "residual" and context_anchor is not None:
            rain = rain + context_anchor["rain"].expand(-1, -1, rain.shape[2], -1, -1)
        return rain


class RainTSNextFrameTransferTrainer(RainTSNextFrameTrainer):
    def __init__(self, cfg: DictConfig) -> None:
        if cfg.rain_prediction_model.get("_target_") == (
            "src.networks.time_series.causal_patch_transformer_next_frame.RainCausalPatchTransformerNextFrame"
        ):
            cfg = OmegaConf.merge(
                cfg,
                {
                    "rain_prediction_model": {
                        "_target_": "src.trainer.rain_trainer_ts_next_frame_transfer.RainCausalPatchTransformerRainDecode"
                    }
                },
            )
        super().__init__(cfg)
        self.train_dataloader = self.train_dataloadaer
        del self.train_dataloadaer

    def train_step(self, batch: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], bool]:
        model_fields = {
            f"{modality}_{period}"
            for modality in ("radar", "satellite", "rain", "time")
            for period in ("past", "future")
        }
        device_batch = {
            name: tensor.to(
                device=self.device,
                dtype=torch.float32 if name in model_fields else tensor.dtype,
                non_blocking=self.accelerator.non_blocking,
            )
            for name, tensor in batch.items()
        }
        return super().train_step(device_batch)

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
        model = self.accelerator.unwrap_model(self.model)
        rollout_cfg = self.train_cfg.next_pred.get("rollout_branch", {})
        loss_cfg = self.train_cfg.get("loss", {})
        rain_only = (
            self.cfg.rain_prediction_model.get("_target_")
            == "src.trainer.rain_trainer_ts_next_frame_transfer.RainCausalPatchTransformerRainDecode"
            and model.training
            and return_rain_delta
            and use_gt_future_modalities
            and not use_gt_future_rain
            and rollout_cfg.get("loss_on", "rain") == "rain"
            and loss_cfg.get("mode", "mse") == "mse"
            and not loss_cfg.get("temporal_diff", {}).get("enabled", False)
            and self.cfg.rain_prediction_model.get("dropout", 0.0) == 0.0
            and (mode == "frame" or (mode == "block" and rollout_block_size == model.frame_patch_size))
        )
        if not rain_only:
            return super()._rollout_predict_with_settings(
                context=context,
                total_future_frames=total_future_frames,
                mode=mode,
                rollout_block_size=rollout_block_size,
                detach_history=detach_history,
                context_time=context_time,
                future_time=future_time,
                future_modalities=future_modalities,
                use_gt_future_modalities=use_gt_future_modalities,
                future_rain=future_rain,
                use_gt_future_rain=use_gt_future_rain,
                return_rain_delta=return_rain_delta,
            )

        frame_patch_size = model.frame_patch_size
        if context.ndim != 5:
            raise ValueError("context must have shape [B,C,T,H,W]")
        if total_future_frames <= 0 or total_future_frames % frame_patch_size:
            raise ValueError("total_future_frames must be positive and divisible by frame_patch_size")
        if rollout_block_size <= 0 or context.shape[2] < frame_patch_size or context.shape[2] % frame_patch_size:
            raise ValueError("Invalid rollout block size or context length")
        if future_modalities is None:
            raise ValueError("Rain-only rollout requires GT radar and satellite")
        for name in ("radar", "satellite"):
            if future_modalities[name].shape != (
                context.shape[0],
                self.radar_c if name == "radar" else self.satellite_c,
                total_future_frames,
                *context.shape[-2:],
            ):
                raise ValueError(f"Invalid GT {name} shape for rain-only rollout")
        if context_time is not None or future_time is not None:
            if context_time is None or future_time is None:
                raise ValueError("context_time and future_time must both be provided")
            if context_time.shape != (context.shape[0], context.shape[2]) or future_time.shape != (
                context.shape[0], total_future_frames,
            ):
                raise ValueError("Time tensor shapes must match context and future frames")

        rain_predictions: list[torch.Tensor] = []
        rain_deltas: list[torch.Tensor] = []
        residual_enabled = self._resolve_rain_residual_cfg()[0]
        for start in range(0, total_future_frames, frame_patch_size):
            end = start + frame_patch_size
            seed = context[:, :, -frame_patch_size:]
            with self.accelerator.autocast():
                rain_delta = self.model.forward_ar(
                    context_x=context,
                    target_x=seed,
                    predict_frames=frame_patch_size,
                    strict_target_isolation=bool(self.train_cfg.strict_target_isolation),
                    return_modality_dict=False,
                    context_time=context_time,
                    target_time=None if context_time is None else context_time[:, -frame_patch_size:],
                )
            rain = rain_delta
            if residual_enabled:
                rain = self._apply_rain_residual_output(rain_delta, seed[:, -self.rain_c:])
            rain_predictions.append(rain)
            rain_deltas.append(rain_delta)
            if end < total_future_frames:
                history = self._merge_modalities(
                    future_modalities["radar"][:, :, start:end],
                    future_modalities["satellite"][:, :, start:end],
                    rain.detach() if detach_history else rain,
                )
                context = torch.cat([context, history], dim=2)
                if context_time is not None:
                    context_time = torch.cat([context_time, future_time[:, start:end]], dim=1)
        return {"rain": torch.cat(rain_predictions, dim=2), "rain_delta": torch.cat(rain_deltas, dim=2)}

    def _next_prediction_loss(
        self,
        pred: dict[str, torch.Tensor],
        target_gt: dict[str, torch.Tensor],
        aux: dict[str, float | int | str | torch.Tensor | dict[str, torch.Tensor]] | None = None,
        loss_weight_override: dict[str, float] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if "radar" in pred or "satellite" in pred:
            return super()._next_prediction_loss(pred, target_gt, aux, loss_weight_override)
        if aux is None or aux.get("target_mode") != "rollout_branch":
            raise ValueError("Rain-only loss is reserved for the training rollout branch")
        loss = F.mse_loss(pred["rain"], target_gt["rain"])
        logs = {"loss": loss.detach(), "loss/rain": loss.detach()}
        if self._resolve_rain_residual_cfg()[0]:
            logs["meta/rain_residual_enabled"] = loss.new_ones(())
        return loss, logs


@hydra.main(
    config_path="../config/ts_rain_train",
    config_name="rain_trainer_ts_next_frame",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    trainer = RainTSNextFrameTransferTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
