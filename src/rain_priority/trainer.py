import json
import math

import torch
from omegaconf import DictConfig

from src.gated_modality_rain.io_trainer import GatedModalityRainIOTrainer
from src.rain_priority.controller import HorizonController
from src.rain_priority.evaluation import RainLeadMetrics, save_raw_forecast
from src.rain_priority.loss import lead_mse, modality_losses
from src.rain_priority.model import RainPriorityModel
from src.rain_priority.optimization import backward_rain_priority


class RainPriorityTrainer(GatedModalityRainIOTrainer):
    def __init__(self, cfg: DictConfig) -> None:
        loss = cfg.train.loss
        for name in ("temporal_diff", "rain_region_weight", "rain_event_aux"):
            if loss.get(name, {}).get("enabled", False):
                raise ValueError(f"rain_priority does not enable legacy {name}; use the approved clean objective.")
        residual = loss.get("rain_residual", {})
        rollout = cfg.train.next_pred.rollout_branch
        sequence = cfg.train.next_pred.sequence_loss
        if loss.mode != "mse" or not residual.enabled or residual.delta_weight != 0 or residual.clamp_output:
            raise ValueError("Use MSE and unclamped residual output without auxiliary delta loss.")
        if not sequence.enabled or sequence.context_weight != 1 or sequence.future_weight != 1:
            raise ValueError("Keep the original equally weighted sequence segments.")
        if (
            not rollout.enabled
            or rollout.mode != "frame"
            or rollout.rollout_frames != 4
            or rollout.rollout_block_size != 1
            or not rollout.detach_history
            or rollout.use_gt_future_modalities
            or rollout.delta_loss.enabled
            or rollout.loss_on != "all"
        ):
            raise ValueError(
                "Use detached four-frame rollout with all outputs supervised and no GT forcing/delta loss."
            )
        if cfg.dataset.n_past != 4 or cfg.dataset.n_futures != 4 or cfg.rain_prediction_model.decoder_type != "3d":
            raise ValueError("Keep the original 3D decoder and four-in/four-out protocol.")
        if (
            cfg.train.strict_target_isolation
            or cfg.train.next_pred.target_mode != "next_frame"
            or cfg.train.next_pred.block_size != 1
            or cfg.val.rollout_mode != "frame"
            or cfg.val.rollout_block_size != 1
            or not cfg.val.rollout_history_detach
        ):
            raise ValueError("Unsupported override of the original sequence/fixed-origin protocol.")
        if cfg.train.gan.enabled or cfg.val.rollout_use_gt_future_modalities or cfg.val.after_roll_next.enabled:
            raise ValueError("GAN, future weather forcing, and extra forecast frames are outside this experiment.")
        if cfg.accelerator.mixed_precision not in {"no", "bf16"}:
            raise ValueError("Use FP32 or BF16 for rain-priority gradient routing.")
        for name, value in cfg.train.loss_weights.items():
            if not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"Invalid loss coefficient for {name}.")
        if cfg.train.loss_weights.rain != 1:
            raise ValueError("The primary rain coefficient is fixed at one.")
        if cfg.train.get("init_model_path"):
            raise ValueError("This experiment starts from scratch; use resume_path for its own checkpoints.")
        super().__init__(cfg)
        if self.accelerator.num_processes != 1 or self.accelerator.scaler is not None:
            raise ValueError("This release supports a single FP32/BF16 device only.")
        model = self.accelerator.unwrap_model(self.model)
        if not isinstance(model, RainPriorityModel):
            raise ValueError("RainPriorityTrainer requires RainPriorityModel.")
        self.controller = HorizonController(**dict(cfg.train.priority.horizon))
        self.accelerator.register_for_checkpointing(self.controller)
        gate = model.spatial_modality_gate
        self.gate_parameters = list(gate.parameters()) if gate is not None else []
        self.pending_error_sum = torch.zeros(4)
        self.pending_count = 0
        self.log_msg("Rain priority: original 3D decoder, 7 shifted targets, 4 fixed-origin forecast frames.")
        self.log_msg("Known limitation: original decoder temporal padding/normalization causality is unchanged.")

    def _device_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        past = batch.get("time_past_timestamp")
        future = batch.get("time_future_timestamp")
        reversed_time = batch.get("aug_time_reversed")
        if past is None or future is None or reversed_time is None:
            raise ValueError("Absolute timestamp and aug_time_reversed metadata are required.")
        if past.ndim != 2 or past.shape[1] != 4 or future.shape != past.shape or reversed_time.shape != past.shape[:1]:
            raise ValueError("Expected four past/future timestamps per sample and one reversal flag.")
        stamps = torch.cat([past.double(), future.double()], dim=1)
        if not torch.isfinite(stamps).all() or not ((reversed_time == 0) | (reversed_time == 1)).all():
            raise ValueError("Nonfinite timestamp or invalid time-reversal flag.")
        expected = (1 - 2 * reversed_time.long())[:, None] * 1800
        if not torch.isclose(stamps.diff(dim=1), expected.double(), rtol=0, atol=1e-6).all():
            raise ValueError("Expected consecutive 1800-second frames (or -1800 for reversed augmentation).")
        model_fields = {
            f"{modality}_{period}"
            for modality in ("rain", "radar", "satellite", "time")
            for period in ("past", "future")
        }
        return {
            key: value.to(
                self.device,
                dtype=torch.float32 if key in model_fields else value.dtype,
                non_blocking=self.accelerator.non_blocking,
            )
            for key, value in batch.items()
        }

    def _teacher_prediction(
        self,
        batch: dict[str, torch.Tensor],
        training: bool,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        context, seed, target, aux = self._build_next_pred_batch(batch, apply_missing_modality=training)
        if context.shape[2] != 1 or seed.shape[2] != 7:
            raise ValueError("Unexpected sequence preparation; expected one context and seven seed positions.")
        source = torch.ones(context.shape[0], 3, 8, dtype=torch.long, device=self.device)
        source[:, :, 0] = aux["context_modality_available"].long()
        steps = torch.tensor([-3, -3, -2, -1, 0, 1, 2, 3], device=self.device).expand(context.shape[0], -1)
        pred, stats = self.model.forward_ar(
            context,
            seed,
            predict_frames=7,
            strict_target_isolation=False,
            context_modality_available=aux["context_modality_available"],
            context_time=aux.get("context_time"),
            target_time=aux.get("target_seed_time"),
            gate_source=source,
            gate_steps=steps,
            return_gate_stats=True,
        )
        pred["rain"] = self._apply_rain_residual_output(pred["rain"], aux["rain_residual_ref"])
        return pred, target, stats

    def _forecast(
        self,
        context: torch.Tensor,
        context_time: torch.Tensor | None,
        future_time: torch.Tensor | None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if context.shape[2] != 4 or (context_time is None) != (future_time is None):
            raise ValueError("Forecast requires four history frames and matching optional time metadata.")
        if context_time is not None and (
            context_time.shape != (context.shape[0], 4) or future_time.shape != context_time.shape
        ):
            raise ValueError("Forecast timestamps must have shape [B,4].")
        current = context
        current_time = context_time
        predictions: dict[str, list[torch.Tensor]] = {name: [] for name in ("rain", "radar", "satellite")}
        stats: dict[str, torch.Tensor] = {}
        for horizon in range(4):
            b, _, t, _, _ = current.shape
            source = torch.ones(b, 3, t + 1, dtype=torch.long, device=current.device)
            if horizon:
                source[:, :, 4:] = 2
            steps = torch.cat(
                [torch.arange(-3, horizon + 1, device=current.device), torch.tensor([horizon], device=current.device)]
            ).expand(b, -1)
            with self.accelerator.autocast():
                pred, gate_stats = self.model.forward_ar(
                    current,
                    current[:, :, -1:],
                    predict_frames=1,
                    strict_target_isolation=False,
                    context_time=current_time,
                    target_time=None if current_time is None else current_time[:, -1:],
                    gate_source=source,
                    gate_steps=steps,
                    return_gate_stats=True,
                )
            pred["rain"] = self._apply_rain_residual_output(
                pred["rain"], self._split_modalities(current)["rain"][:, :, -1:]
            )
            for name in predictions:
                predictions[name].append(pred[name])
            stats.update({f"rollout/h{horizon + 1}/{name}": value for name, value in gate_stats.items()})
            generated = self._merge_modalities(pred["radar"], pred["satellite"], pred["rain"]).detach()
            current = torch.cat([current, generated], dim=2)
            if current_time is not None:
                current_time = torch.cat([current_time, future_time[:, horizon : horizon + 1]], dim=1)
        return {name: torch.cat(frames, dim=2) for name, frames in predictions.items()}, stats

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
        return_rain_delta: bool = False,
    ) -> dict[str, torch.Tensor]:
        if (
            total_future_frames != 4
            or mode != "frame"
            or rollout_block_size != 1
            or not detach_history
            or use_gt_future_modalities
            or return_rain_delta
        ):
            raise ValueError("Only the approved four-frame fixed-origin rollout is supported.")
        return self._forecast(context, context_time, future_time)[0]

    def train_step(self, batch: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], bool]:
        batch = self._device_batch(batch)
        with self.accelerator.accumulate(self.model):
            with self.accelerator.autocast():
                teacher, teacher_target, logs = self._teacher_prediction(batch, training=True)
                context, target, context_time, future_time = self._prepare_val_inference_batch(batch)
                rollout, rollout_stats = self._forecast(context, context_time, future_time)
                used_weights = self.controller.weights.to(self.device)
                losses, errors = modality_losses(
                    teacher,
                    teacher_target,
                    rollout,
                    target,
                    used_weights,
                    float(self.train_cfg.next_pred.rollout_branch.weight),
                )
                weighted = {name: value * float(self.train_cfg.loss_weights[name]) for name, value in losses.items()}
                total = sum(weighted.values())
            if not torch.isfinite(total):
                raise FloatingPointError("Nonfinite training loss; no optimizer update was performed.")
            diagnostics_every = int(self.train_cfg.priority.gradient_diagnostics_every)
            if diagnostics_every > 0 and (self.global_step + 1) % diagnostics_every == 0:
                shared = self.accelerator.unwrap_model(self.model).patch_embed.weight
                gradients = {
                    name: torch.autograd.grad(value, shared, retain_graph=True)[0].detach().float()
                    for name, value in weighted.items()
                }
                for name, grad in gradients.items():
                    logs[f"gradient/shared/{name}_norm"] = grad.norm()
                    if name != "rain":
                        rain_grad = gradients["rain"]
                        logs[f"gradient/shared/rain_{name}_cosine"] = (rain_grad * grad).sum() / (
                            rain_grad.norm() * grad.norm()
                        ).clamp_min(1e-12)
            if self.train_cfg.priority.rain_only_gate:
                backward_rain_priority(self.accelerator, total, losses["rain"], self.gate_parameters)
            else:
                self.accelerator.backward(total)
            count = target["rain"].numel() // 4
            self.pending_error_sum += errors.cpu() * count
            self.pending_count += count
            did_step = False
            if self.accelerator.sync_gradients:
                logs["gradient/gate_norm_before_clip"] = sum(
                    (
                        parameter.grad.detach().float().square().sum()
                        for parameter in self.gate_parameters
                        if parameter.grad is not None
                    ),
                    torch.zeros((), device=self.device),
                ).sqrt()
                max_norm = float(self.train_cfg.max_grad_norm)
                norm = self.accelerator.clip_grad_norm_(
                    self.model.parameters(), max_norm if max_norm > 0 else float("inf")
                )
                if not torch.isfinite(norm):
                    raise FloatingPointError("Nonfinite gradient norm; no optimizer update was performed.")
                logs["gradient/global_norm_before_clip"] = norm.detach()
                self.optim.step()
                did_step = not self.accelerator.optimizer_step_was_skipped
                if did_step:
                    self.sched.step()
                    self.global_step += 1
                    self.controller.observe(self.pending_error_sum, self.pending_count, self.global_step)
                    if (
                        self.global_step == self.controller.settings["warmup_steps"]
                        and self.controller.neutral_reference_count
                    ):
                        self.log_msg(
                            f"[Horizon warning] {self.controller.neutral_reference_count} near-zero reference horizons use neutral feedback ratios."
                        )
                    if self.ema_model is not None:
                        self.ema_model.update()
                self.optim.zero_grad(set_to_none=True)
                self.pending_error_sum.zero_()
                self.pending_count = 0
        logs.update(rollout_stats)
        logs.update({f"loss/{name}": value.detach() for name, value in losses.items()})
        logs["loss"] = total.detach()
        logs["horizon/neutral_reference_count"] = torch.tensor(float(self.controller.neutral_reference_count))
        for h in range(4):
            logs[f"horizon/h{h + 1}/weight_used"] = used_weights[h].detach().clone()
            logs[f"horizon/h{h + 1}/mse"] = errors[h]
        return logs, did_step

    @torch.no_grad()
    def val_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        with self.accelerator.autocast():
            pred, target, _ = self._teacher_prediction(self._device_batch(batch), training=False)
        losses = {}
        for name in ("rain", "radar", "satellite"):
            error = lead_mse(pred[name], target[name])
            losses[name] = 0.5 * error[:3].mean() + 0.5 * error[3:].mean()
        return {"loss": sum(float(self.train_cfg.loss_weights[m]) * value for m, value in losses.items())}

    @torch.no_grad()
    def _val_inference_step(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        context, target, context_time, future_time = self._prepare_val_inference_batch(self._device_batch(batch))
        prediction, stats = self._forecast(context, context_time, future_time)
        loss = sum(float(self.train_cfg.loss_weights[m]) * lead_mse(prediction[m], target[m]).mean() for m in target)
        return prediction, target, loss, stats

    @torch.no_grad()
    def _run_val(self) -> None:
        if int(self.val_cfg.max_val_iters) == 0:
            return
        was_training = self.model.training
        self.model.eval()
        metrics = RainLeadMetrics(list(self.val_cfg.csi_thresholds))
        persistence = RainLeadMetrics(list(self.val_cfg.csi_thresholds))
        directory = self.proj_dir / "validation"
        directory.mkdir(parents=True, exist_ok=True)
        batches = 0
        self.log_msg(
            f"[Val] start at step={self.global_step}; max_val_iters={self.val_cfg.max_val_iters} (-1=full set)"
        )
        try:
            for index, batch in enumerate(self.val_dataloader):
                if self.val_cfg.max_val_iters > 0 and index >= self.val_cfg.max_val_iters:
                    break
                batch = self._device_batch(batch)
                context, target, context_time, future_time = self._prepare_val_inference_batch(batch)
                prediction, _ = self._forecast(context, context_time, future_time)
                rain = self._denormalize_rain_for_metrics(prediction["rain"])
                truth = self._denormalize_rain_for_metrics(target["rain"])
                history = self._split_modalities(context)
                last = self._denormalize_rain_for_metrics(history["rain"][:, :, -1:]).expand_as(truth)
                metrics.update(rain, truth)
                persistence.update(last, truth)
                if index == 0 and self.val_cfg.save_visuals:
                    save_raw_forecast(directory / f"step-{self.global_step:08d}-raw.png", history, prediction, target)
                batches += 1
            if batches == 0:
                raise ValueError("Validation loader produced no samples.")
            result = {
                "step": self.global_step,
                "batches": batches,
                "rain": metrics.compute(),
                "persistence": persistence.compute(),
                "weights_used_for_training": self.controller.weights.tolist(),
                "evaluation_model": "current_model_not_ema",
                "units": "existing rain metric space",
            }
            (directory / f"step-{self.global_step:08d}.json").write_text(
                json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
            )
            for h in range(4):
                summary = {
                    k.split("/", 1)[1]: v
                    for k, v in result["rain"].items()
                    if k.startswith(f"h{h + 1}/") and ("csi@" in k or "far@" in k or "pod@" in k or k.endswith("/mse"))
                }
                self.log_msg(f"[Val][{self.global_step}][+{30 * (h + 1)}min] {json.dumps(summary)}")
            payload = {
                f"val/{group}/{key}": value
                for group in ("rain", "persistence")
                for key, value in result[group].items()
                if value is not None
            }
            if not self.train_cfg.debug:
                self._log_tensorboard_scalars(payload, step=self.global_step)
        finally:
            self.model.train(was_training)

    def _resume_if_needed(self) -> None:
        super()._resume_if_needed()
        if self.controller.step != self.global_step:
            raise ValueError("Controller step does not match model checkpoint step.")
        if self.global_step >= self.train_cfg.max_steps:
            raise ValueError("Checkpoint already reached max_steps; increase max_steps to continue.")
