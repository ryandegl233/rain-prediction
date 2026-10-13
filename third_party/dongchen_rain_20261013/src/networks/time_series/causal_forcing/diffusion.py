from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from .base import BaseCausalForcingModel
from .scheduler import DiffusionScheduler, FlowMatchScheduler, SchedulerInterface


class CausalDiffusion(BaseCausalForcingModel):
    """
    Causal diffusion wrapper with third_party-like interface.

    Key differences from third_party:
    - text encoder and vae are intentionally set to None
    - generator is your local rain model
    - scheduler defaults to local FlowMatchScheduler (or injected scheduler_cfg)
    """

    def __init__(self, args: DictConfig | Any, device: torch.device | None = None):
        super().__init__(args=args, device=device)

        self.num_train_timestep = int(getattr(args, "num_train_timestep", 1000))
        self.min_step = int(getattr(args, "min_step", int(0.02 * self.num_train_timestep)))
        self.max_step = int(getattr(args, "max_step", int(0.98 * self.num_train_timestep)))

        self.teacher_forcing = bool(getattr(args, "teacher_forcing", True))
        self.noise_augmentation_max_timestep = int(getattr(args, "noise_augmentation_max_timestep", 0))
        self.timestep_shift = float(getattr(args, "timestep_shift", 5.0))

        prediction_type = getattr(args, "prediction_type", getattr(args, "denoising_loss_type", "flow"))
        prediction_type = str(prediction_type).lower()
        if prediction_type in {"flow_prediction"}:
            prediction_type = "flow"
        if prediction_type in {"epsilon", "eps"}:
            prediction_type = "noise"
        if prediction_type not in {"flow", "noise", "x0", "velocity"}:
            raise ValueError(f"prediction_type must be one of [flow, noise, x0, velocity], got {prediction_type}")
        self.prediction_type = prediction_type

        self.strict_target_isolation = bool(getattr(args, "strict_target_isolation", True))
        self.channel_weights = {
            "radar": float(getattr(args, "loss_weight_radar", 1.0)),
            "satellite": float(getattr(args, "loss_weight_satellite", 1.0)),
            "rain": float(getattr(args, "loss_weight_rain", 1.0)),
        }

    def _initialize_models(self, args: DictConfig | Any, device: torch.device) -> None:
        generator = getattr(args, "generator", None)
        generator_cfg = getattr(args, "generator_cfg", None)

        if generator is not None and isinstance(generator, torch.nn.Module):
            self.generator = generator
        elif generator_cfg is not None:
            self.generator = hydra.utils.instantiate(generator_cfg)
        elif isinstance(generator, DictConfig) and "_target_" in generator:
            self.generator = hydra.utils.instantiate(generator)
        else:
            raise ValueError(
                "CausalDiffusion requires `generator` (nn.Module or config) or `generator_cfg` in args."
            )

        self.generator.to(device)
        self.generator.requires_grad_(True)

        # explicitly disabled in this project variant
        self.text_encoder = None
        self.vae = None

        scheduler_cfg = getattr(args, "scheduler_cfg", None)
        if scheduler_cfg is not None:
            self.scheduler: SchedulerInterface = hydra.utils.instantiate(scheduler_cfg)
        else:
            self.scheduler = FlowMatchScheduler(
                num_train_timesteps=int(getattr(args, "num_train_timestep", 1000)),
                shift=float(getattr(args, "timestep_shift", 5.0)),
                sigma_min=float(getattr(args, "sigma_min", 0.0)),
                extra_one_step=bool(getattr(args, "extra_one_step", True)),
            )
            self.scheduler.set_timesteps(int(getattr(args, "num_train_timestep", 1000)), training=True)

        if hasattr(self.scheduler, "timesteps"):
            self.scheduler.timesteps = self.scheduler.timesteps.to(device)
        if hasattr(self.scheduler, "sigmas"):
            self.scheduler.sigmas = self.scheduler.sigmas.to(device)
        if hasattr(self.scheduler, "alphas_cumprod"):
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)

    @property
    def in_channels(self) -> int:
        return int(getattr(self.generator, "in_channels", 12))

    def _to_bfchw(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, T, H, W] -> [B, T, C, H, W]
        return x.permute(0, 2, 1, 3, 4).contiguous()

    def _to_bcthw(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, C, H, W] -> [B, C, T, H, W]
        return x.permute(0, 2, 1, 3, 4).contiguous()

    def _split_modalities(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x: [B, C, T, H, W]
        radar_c = int(getattr(self.generator, "radar_out_channels", 1))
        sat_c = int(getattr(self.generator, "satellite_out_channels", 10))
        return {
            "radar": x[:, :radar_c],
            "satellite": x[:, radar_c : radar_c + sat_c],
            "rain": x[:, radar_c + sat_c :],
        }

    def _merge_modalities(self, modal: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([modal["radar"], modal["satellite"], modal["rain"]], dim=1)

    def _prepare_ar_teacher_forcing_batch(
        self, clean_latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Input clean_latent: [B, F, C, H, W]
        Returns:
            model_input_bcthw, model_timestep [B,L], target_xt [B,1,C,H,W], target_x0 [B,1,C,H,W], target_t [B], target_noise [B,1,C,H,W]
        """
        if clean_latent.ndim != 5:
            raise ValueError(f"clean_latent must be [B,F,C,H,W], got {tuple(clean_latent.shape)}")
        b, f, c, h, w = clean_latent.shape
        if c != self.in_channels:
            raise ValueError(f"channel mismatch: clean has C={c}, generator expects {self.in_channels}")
        if f < 1:
            raise ValueError("clean_latent must contain at least 1 frame.")

        if f == 1:
            target_idx = 0
        else:
            target_idx = int(torch.randint(1, f, (1,), device=self.device).item())

        context = clean_latent[:, :target_idx] if target_idx > 0 else clean_latent[:, :0]
        target_x0 = clean_latent[:, target_idx : target_idx + 1]

        target_index = self._get_timestep(
            min_timestep=self.min_step,
            max_timestep=min(self.max_step + 1, self.num_train_timestep),
            batch_size=b,
            num_frame=1,
            num_frame_per_block=1,
            uniform_timestep=True,
        ).squeeze(1)  # [B]
        target_t = self.scheduler.timesteps[target_index].to(device=self.device)

        target_noise = torch.randn_like(target_x0)
        target_xt = self.scheduler.add_noise(
            target_x0.reshape(-1, c, h, w),
            target_noise.reshape(-1, c, h, w),
            target_t,
        ).reshape(b, 1, c, h, w)

        if self.noise_augmentation_max_timestep > 0 and context.shape[1] > 0:
            context_index = self._get_timestep(
                min_timestep=0,
                max_timestep=min(self.noise_augmentation_max_timestep + 1, self.num_train_timestep),
                batch_size=b,
                num_frame=context.shape[1],
                num_frame_per_block=self.num_frame_per_block,
                uniform_timestep=False,
            )
            context_t = self.scheduler.timesteps[context_index].to(device=self.device)
            context_noise = torch.randn_like(context)
            context_noisy = self.scheduler.add_noise(
                context.reshape(-1, c, h, w),
                context_noise.reshape(-1, c, h, w),
                context_t.reshape(-1),
            ).reshape_as(context)
        else:
            context_t = torch.zeros((b, context.shape[1]), device=self.device, dtype=torch.float32)
            context_noisy = context

        model_input = torch.cat([context_noisy, target_xt], dim=1)  # [B,L,C,H,W]
        model_input_bcthw = self._to_bcthw(model_input)

        model_timestep = torch.zeros((b, model_input.shape[1]), device=self.device, dtype=torch.float32)
        if context.shape[1] > 0:
            model_timestep[:, :-1] = context_t.float()
        model_timestep[:, -1] = target_t.float()
        return model_input_bcthw, model_timestep, target_xt, target_x0, target_t, target_noise

    def _prediction_to_x0(
        self,
        pred: torch.Tensor,
        xt: torch.Tensor,
        timestep: torch.Tensor,
        prediction_type: str,
    ) -> torch.Tensor:
        # pred/xt: [B,1,C,H,W], timestep:[B]
        b, _, c, h, w = pred.shape
        pred_f = pred.reshape(b, c, h, w)
        xt_f = xt.reshape(b, c, h, w)

        if prediction_type == "x0":
            x0_f = pred_f
        elif prediction_type == "flow":
            if not hasattr(self.scheduler, "flow_pred_to_x0"):
                raise ValueError("Current scheduler does not support flow->x0 conversion.")
            x0_f = self.scheduler.flow_pred_to_x0(flow_pred=pred_f, xt=xt_f, timestep=timestep)
        elif prediction_type == "noise":
            if hasattr(self.scheduler, "convert_noise_to_x0"):
                x0_f = self.scheduler.convert_noise_to_x0(noise=pred_f, xt=xt_f, timestep=timestep)
            else:
                raise ValueError("Current scheduler does not support noise->x0 conversion.")
        elif prediction_type == "velocity":
            if hasattr(self.scheduler, "convert_velocity_to_x0"):
                x0_f = self.scheduler.convert_velocity_to_x0(velocity=pred_f, xt=xt_f, timestep=timestep)
            else:
                raise ValueError("Current scheduler does not support velocity->x0 conversion.")
        else:
            raise ValueError(f"Unsupported prediction_type={prediction_type}")
        return x0_f.reshape(b, 1, c, h, w)

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        third_party-compatible interface.

        Args:
            image_or_video_shape: kept for interface compatibility.
            conditional_dict/unconditional_dict: unused in this no-text variant.
            clean_latent: [B, F, C, H, W]
            initial_latent: unused.
        """
        _ = image_or_video_shape, conditional_dict, unconditional_dict, initial_latent

        clean_latent = clean_latent.to(device=self.device, dtype=self.dtype)
        model_input, model_timestep, target_xt, target_x0, target_t, target_noise = self._prepare_ar_teacher_forcing_batch(
            clean_latent
        )

        pred_modal = self.generator(
            x=model_input,
            diffusion_timestep=model_timestep,
            predict_frames=1,
            strict_target_isolation=self.strict_target_isolation,
            return_modality_dict=True,
        )
        pred = self._to_bfchw(self._merge_modalities(pred_modal))  # [B,1,C,H,W]

        if self.prediction_type == "flow":
            train_target = self.scheduler.training_target(sample=target_x0, noise=target_noise, timestep=target_t)
        elif self.prediction_type == "noise":
            train_target = target_noise
        elif self.prediction_type == "x0":
            train_target = target_x0
        else:
            if isinstance(self.scheduler, FlowMatchScheduler):
                raise ValueError("prediction_type='velocity' is not supported for FlowMatchScheduler.")
            v = self.scheduler.training_target(
                sample=target_x0.reshape(target_x0.shape[0], target_x0.shape[2], target_x0.shape[3], target_x0.shape[4]),
                noise=target_noise.reshape(target_noise.shape[0], target_noise.shape[2], target_noise.shape[3], target_noise.shape[4]),
                timestep=target_t,
                prediction_type="velocity",
            )
            train_target = v.reshape_as(target_x0)

        pred_c = self._to_bcthw(pred)
        tgt_c = self._to_bcthw(train_target)
        pred_split = self._split_modalities(pred_c)
        tgt_split = self._split_modalities(tgt_c)

        radar_loss_bt = F.mse_loss(pred_split["radar"], tgt_split["radar"], reduction="none").mean(dim=(1, 3, 4))
        sat_loss_bt = F.mse_loss(pred_split["satellite"], tgt_split["satellite"], reduction="none").mean(dim=(1, 3, 4))
        rain_loss_bt = F.mse_loss(pred_split["rain"], tgt_split["rain"], reduction="none").mean(dim=(1, 3, 4))

        loss_bt = (
            self.channel_weights["radar"] * radar_loss_bt
            + self.channel_weights["satellite"] * sat_loss_bt
            + self.channel_weights["rain"] * rain_loss_bt
        )
        if hasattr(self.scheduler, "training_weight"):
            timestep_weight = self.scheduler.training_weight(target_t)
            timestep_weight = timestep_weight.reshape_as(loss_bt)
            loss_bt = loss_bt * timestep_weight
        loss = loss_bt.mean()

        l_radar = radar_loss_bt.mean()
        l_sat = sat_loss_bt.mean()
        l_rain = rain_loss_bt.mean()

        pred_x0 = self._prediction_to_x0(
            pred=pred,
            xt=target_xt,
            timestep=target_t,
            prediction_type=self.prediction_type,
        )

        log_dict = {
            "loss": loss.detach(),
            "loss/radar": l_radar.detach(),
            "loss/satellite": l_sat.detach(),
            "loss/rain": l_rain.detach(),
            "x0": target_x0.detach(),
            "x0_pred": pred_x0.detach(),
            "timestep": target_t.detach(),
        }
        return loss, log_dict

    def save_checkpoint(
        self,
        ckpt_path: str | Path,
        optimizer: torch.optim.Optimizer | None = None,
        step: int | None = None,
        extra: dict | None = None,
    ) -> None:
        ckpt_path = Path(ckpt_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "generator": self.generator.state_dict(),
            "scheduler": {
                "class": self.scheduler.__class__.__name__,
                "num_train_timesteps": int(getattr(self.scheduler, "num_train_timesteps", self.num_train_timestep)),
            },
            "step": int(step or 0),
            "args": dict(self.args) if isinstance(self.args, dict) else None,
            "extra": extra or {},
        }
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, ckpt_path)

    def load_checkpoint(
        self,
        ckpt_path: str | Path,
        optimizer: torch.optim.Optimizer | None = None,
        strict: bool = True,
        map_location: str | torch.device = "cpu",
    ) -> dict:
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        payload = torch.load(ckpt_path, map_location=map_location)

        state = payload.get("generator", payload.get("model", payload))
        missing, unexpected = self.generator.load_state_dict(state, strict=False)
        if strict and (len(missing) > 0 or len(unexpected) > 0):
            raise RuntimeError(f"Strict load failed: missing={len(missing)}, unexpected={len(unexpected)}")

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])

        return {
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "step": int(payload.get("step", 0)),
            "extra": payload.get("extra", {}),
        }
