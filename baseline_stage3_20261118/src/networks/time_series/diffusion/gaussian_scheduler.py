from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Literal

import torch


class SchedulerInterface(ABC):
    """
    Base class for diffusion noise schedule.
    """

    alphas_cumprod: torch.Tensor  # [T], alphas for defining the noise schedule

    @abstractmethod
    def add_noise(
        self,
        clean_latent: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        pass

    def convert_x0_to_noise(
        self,
        x0: torch.Tensor,
        xt: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        noise = (xt - sqrt(alpha_t) * x0) / sqrt(beta_t)
        """
        x0_bt, xt_bt, timestep_bt, pack_shape = _flatten_bt_for_scheduler(x0, xt, timestep)
        original_dtype = x0_bt.dtype

        x0_bt = x0_bt.double()
        xt_bt = xt_bt.double()
        alphas_cumprod = self.alphas_cumprod.to(device=x0_bt.device, dtype=torch.float64)

        timestep_bt = timestep_bt.clamp(min=0, max=self.alphas_cumprod.shape[0] - 1)
        alpha_prod_t = alphas_cumprod[timestep_bt].reshape(-1, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t
        noise_pred = (xt_bt - alpha_prod_t.sqrt() * x0_bt) / beta_prod_t.sqrt()
        noise_pred = noise_pred.to(original_dtype)
        return _unflatten_bt_from_scheduler(noise_pred, pack_shape)

    def convert_noise_to_x0(
        self,
        noise: torch.Tensor,
        xt: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        x0 = (x_t - sqrt(beta_t) * noise) / sqrt(alpha_t)
        """
        noise_bt, xt_bt, timestep_bt, pack_shape = _flatten_bt_for_scheduler(noise, xt, timestep)
        original_dtype = noise_bt.dtype

        noise_bt = noise_bt.double()
        xt_bt = xt_bt.double()
        alphas_cumprod = self.alphas_cumprod.to(device=noise_bt.device, dtype=torch.float64)

        timestep_bt = timestep_bt.clamp(min=0, max=self.alphas_cumprod.shape[0] - 1)
        alpha_prod_t = alphas_cumprod[timestep_bt].reshape(-1, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t
        x0_pred = (xt_bt - beta_prod_t.sqrt() * noise_bt) / alpha_prod_t.sqrt()
        x0_pred = x0_pred.to(original_dtype)
        return _unflatten_bt_from_scheduler(x0_pred, pack_shape)


def _normalize_timestep(
    timestep: torch.Tensor,
    batch_size: int,
    frames: int,
    device: torch.device,
    num_train_timesteps: int,
) -> torch.Tensor:
    if timestep.ndim == 0:
        timestep = timestep[None]
    if timestep.ndim == 1:
        if timestep.shape[0] == batch_size:
            timestep = timestep[:, None].expand(batch_size, frames)
        elif timestep.shape[0] == batch_size * frames:
            timestep = timestep.reshape(batch_size, frames)
        elif timestep.shape[0] == 1:
            timestep = timestep.expand(batch_size)[:, None].expand(batch_size, frames)
        else:
            raise ValueError(
                f"Invalid 1D timestep shape {tuple(timestep.shape)} for batch={batch_size}, frames={frames}"
            )
    elif timestep.ndim == 2:
        if timestep.shape != (batch_size, frames):
            raise ValueError(f"Invalid 2D timestep shape {tuple(timestep.shape)}, expected {(batch_size, frames)}")
    else:
        raise ValueError(f"Unsupported timestep ndim {timestep.ndim}")

    timestep = timestep.to(device=device, dtype=torch.long)
    return timestep.clamp(min=0, max=num_train_timesteps - 1)


def _flatten_bt_for_scheduler(
    a: torch.Tensor,
    b: torch.Tensor,
    timestep: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int] | None]:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")

    if a.ndim == 4:
        bt = a.shape[0]
        t = _normalize_timestep(
            timestep=timestep,
            batch_size=bt,
            frames=1,
            device=a.device,
            num_train_timesteps=10**9,  # clipped later by concrete scheduler
        )[:, 0]
        return a, b, t, None

    if a.ndim != 5:
        raise ValueError(f"Input must be 4D or 5D tensor, got shape={tuple(a.shape)}")

    batch, channels, frames, height, width = a.shape
    a_bt = a.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
    b_bt = b.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
    t_bt = _normalize_timestep(
        timestep=timestep,
        batch_size=batch,
        frames=frames,
        device=a.device,
        num_train_timesteps=10**9,  # clipped later by concrete scheduler
    ).reshape(batch * frames)
    return a_bt, b_bt, t_bt, (batch, frames)


def _unflatten_bt_from_scheduler(x: torch.Tensor, pack_shape: tuple[int, int] | None) -> torch.Tensor:
    if pack_shape is None:
        return x
    batch, frames = pack_shape
    bt, channels, height, width = x.shape
    if bt != batch * frames:
        raise ValueError(f"Invalid packed shape: got bt={bt}, expected {batch * frames}")
    return x.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)


class GaussianDiffusionScheduler(SchedulerInterface):
    """
    DDPM-style Gaussian scheduler in causal-forcing scheduler style.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: torch.device | None = None,
    ) -> None:
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.timesteps = torch.arange(num_train_timesteps - 1, -1, -1, dtype=torch.long)
        if device is not None:
            self.to(device)

    def to(self, device: torch.device) -> "GaussianDiffusionScheduler":
        self.betas = self.betas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.timesteps = self.timesteps.to(device)
        return self

    def add_noise(
        self,
        clean_latent: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        clean_bt, noise_bt, t_bt, pack_shape = _flatten_bt_for_scheduler(clean_latent, noise, timestep)
        t_bt = t_bt.clamp(min=0, max=self.num_train_timesteps - 1)

        alphas_cumprod = self.alphas_cumprod.to(device=clean_bt.device, dtype=clean_bt.dtype)
        alpha_prod_t = alphas_cumprod[t_bt].reshape(-1, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t
        noisy_bt = alpha_prod_t.sqrt() * clean_bt + beta_prod_t.sqrt() * noise_bt
        return _unflatten_bt_from_scheduler(noisy_bt.type_as(clean_latent), pack_shape)

    def set_inference_timesteps(
        self,
        sampler: Literal["ddpm", "ddim"] = "ddpm",
        num_inference_steps: int = 50,
        min_timestep: int = 0,
        max_timestep: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        min_t = int(min_timestep)
        max_t = int(self.num_train_timesteps - 1 if max_timestep is None else max_timestep)
        if min_t < 0 or max_t >= self.num_train_timesteps or min_t > max_t:
            raise ValueError(
                f"Invalid inference timestep range: min_timestep={min_t}, max_timestep={max_t}, "
                f"num_train_timesteps={self.num_train_timesteps}"
            )

        step_device = self.timesteps.device if device is None else device
        if sampler == "ddpm":
            timesteps = torch.arange(max_t, min_t - 1, -1, device=step_device, dtype=torch.long)
        elif sampler == "ddim":
            if num_inference_steps <= 0:
                raise ValueError(f"num_inference_steps must be > 0 for DDIM, got {num_inference_steps}")
            timesteps = torch.linspace(
                float(max_t),
                float(min_t),
                steps=num_inference_steps,
                device=step_device,
            ).round().to(torch.long)
            timesteps = torch.unique_consecutive(timesteps)
        else:
            raise ValueError(f"Unsupported sampler: {sampler}")

        if timesteps.numel() == 0:
            timesteps = torch.tensor([max_t], device=step_device, dtype=torch.long)
        self.timesteps = timesteps
        return timesteps

    @torch.no_grad()
    def denoise(
        self,
        latents: torch.Tensor,
        predict_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        prediction_type: Literal["epsilon", "x0"] = "epsilon",
        sampler: Literal["ddpm", "ddim"] = "ddpm",
        num_inference_steps: int = 50,
        min_timestep: int = 0,
        max_timestep: int | None = None,
        clip_x0: bool = False,
    ) -> torch.Tensor:
        if latents.ndim != 5:
            raise ValueError(f"latents must be [B,C,T,H,W], got {tuple(latents.shape)}")
        if prediction_type not in {"epsilon", "x0"}:
            raise ValueError(f"prediction_type must be 'epsilon' or 'x0', got {prediction_type}")

        batch_size, _, target_frames, _, _ = latents.shape
        timesteps = self.set_inference_timesteps(
            sampler=sampler,
            num_inference_steps=num_inference_steps,
            min_timestep=min_timestep,
            max_timestep=max_timestep,
            device=latents.device,
        )

        betas = self.betas.to(device=latents.device, dtype=latents.dtype)
        alphas_cumprod = self.alphas_cumprod.to(device=latents.device, dtype=latents.dtype)
        for idx in range(timesteps.numel()):
            current_t = int(timesteps[idx].item())
            target_t = torch.full((batch_size, target_frames), current_t, device=latents.device, dtype=torch.long)
            pred = predict_fn(latents, target_t)
            if pred.shape != latents.shape:
                raise ValueError(f"predict_fn output shape must match latents, got {tuple(pred.shape)} vs {tuple(latents.shape)}")

            if prediction_type == "epsilon":
                eps_pred = pred
                x0_pred = self.convert_noise_to_x0(noise=eps_pred, xt=latents, timestep=target_t)
            else:
                x0_pred = pred
                eps_pred = self.convert_x0_to_noise(x0=x0_pred, xt=latents, timestep=target_t)

            if clip_x0:
                x0_pred = x0_pred.clamp(0.0, 1.0)

            if idx == timesteps.numel() - 1:
                latents = x0_pred
                break

            if sampler == "ddim":
                next_t = int(timesteps[idx + 1].item())
                alpha_next = alphas_cumprod[next_t]
                latents = alpha_next.sqrt() * x0_pred + (1.0 - alpha_next).clamp(min=0.0).sqrt() * eps_pred
                continue

            if current_t == 0:
                latents = x0_pred
                continue

            alpha_t = 1.0 - betas[current_t]
            alpha_bar_t = alphas_cumprod[current_t]
            alpha_bar_prev = alphas_cumprod[current_t - 1]
            beta_t = betas[current_t]

            mean = (latents - (beta_t / (1.0 - alpha_bar_t).clamp(min=1e-8).sqrt()) * eps_pred) / alpha_t.sqrt()
            posterior_var = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t).clamp(min=1e-8)
            posterior_var = posterior_var.clamp(min=1e-12)
            latents = mean + posterior_var.sqrt() * torch.randn_like(latents)

        return latents
