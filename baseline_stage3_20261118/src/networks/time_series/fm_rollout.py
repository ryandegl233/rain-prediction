from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from einops import rearrange

from src.networks.time_series.causal_patch_transformer_diffusion import (
    RainCausalPatchTransformerDiffusion,
)
from src.networks.time_series.diffusion.fm_scheduler import FlowMatchScheduler
from src.networks.time_series.diffusion.fm_solvers import FlowDPMSolverMultistepScheduler
from src.networks.time_series.diffusion.fm_solvers_unipc import FlowUniPCMultistepScheduler


ModalityDict = dict[str, torch.Tensor]
FlowRolloutScheduler = FlowMatchScheduler | FlowDPMSolverMultistepScheduler | FlowUniPCMultistepScheduler


@dataclass
class ModalitySpec:
    radar_channels: int = 1
    satellite_channels: int = 10
    rain_channels: int = 1

    @property
    def total_channels(self) -> int:
        return self.radar_channels + self.satellite_channels + self.rain_channels


def split_modalities(x: torch.Tensor, spec: ModalitySpec) -> ModalityDict:
    if x.ndim != 5:
        raise ValueError(f"x must be [B,C,T,H,W], got {tuple(x.shape)}")
    c = x.shape[1]
    if c != spec.total_channels:
        raise ValueError(f"channel mismatch: got {c}, expected {spec.total_channels}")
    r = spec.radar_channels
    s = spec.satellite_channels
    return {
        "radar": x[:, :r],
        "satellite": x[:, r : r + s],
        "rain": x[:, r + s :],
    }


def merge_modalities(modalities: ModalityDict) -> torch.Tensor:
    return torch.cat([modalities["radar"], modalities["satellite"], modalities["rain"]], dim=1)


def _x0_to_flow_pred(
    xt: torch.Tensor,
    x0: torch.Tensor,
    scheduler: FlowRolloutScheduler,
    timestep: torch.Tensor,
) -> torch.Tensor:
    """
    For flow matching:
        x_t = (1 - sigma_t) * x0 + sigma_t * noise
        flow_pred = noise - x0 = (x_t - x0) / sigma_t
    """
    if xt.shape != x0.shape:
        raise ValueError(f"xt/x0 shape mismatch: {tuple(xt.shape)} vs {tuple(x0.shape)}")
    if xt.ndim != 5:
        raise ValueError(f"xt must be [B,C,T,H,W], got {tuple(xt.shape)}")

    b, _, t, _, _ = xt.shape
    xt_bt = rearrange(xt, "b c t h w -> (b t) c h w")
    x0_bt = rearrange(x0, "b c t h w -> (b t) c h w")
    sigma_bt = _scheduler_sigma_from_timestep(scheduler=scheduler, timestep=timestep, device=xt.device)
    flow_bt = (xt_bt - x0_bt) / sigma_bt
    return rearrange(flow_bt, "(b t) c h w -> b c t h w", b=b, t=t)


def _scheduler_sigma_from_timestep(
    scheduler: FlowRolloutScheduler,
    timestep: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(scheduler, FlowMatchScheduler):
        return scheduler.sigma_from_timestep(timestep=timestep, device=device)

    if not hasattr(scheduler, "sigmas") or not hasattr(scheduler, "timesteps"):
        raise TypeError(f"Unsupported scheduler type: {type(scheduler).__name__}")

    timestep_flat = timestep.reshape(-1).to(device=device, dtype=torch.float32)
    solver_timesteps = scheduler.timesteps.to(device=device, dtype=torch.float32)
    timestep_id = torch.argmin((solver_timesteps.unsqueeze(0) - timestep_flat.unsqueeze(1)).abs(), dim=1)
    sigmas = scheduler.sigmas.to(device=device, dtype=torch.float32)
    return sigmas[timestep_id].reshape(-1, 1, 1, 1)


def _set_fm_inference_timesteps(
    scheduler: FlowRolloutScheduler,
    num_inference_steps: int,
    device: torch.device,
) -> None:
    if isinstance(scheduler, FlowMatchScheduler):
        scheduler.set_timesteps(num_inference_steps=num_inference_steps, training=False)
        return
    scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device)


def _extract_solver_sample(step_output: Any) -> torch.Tensor:
    if isinstance(step_output, tuple):
        return step_output[0]
    if hasattr(step_output, "prev_sample"):
        return step_output.prev_sample
    if torch.is_tensor(step_output):
        return step_output
    raise TypeError(f"Unsupported scheduler step output type: {type(step_output).__name__}")


@torch.no_grad()
def fm_denoise_target_with_kv_cache(
    model: RainCausalPatchTransformerDiffusion,
    scheduler: FlowRolloutScheduler,
    target_noisy: torch.Tensor,
    target_frames: int,
    num_inference_steps: int,
    strict_target_isolation: bool = True,
    solver_to_final: bool = False,
) -> ModalityDict:
    """
    Denoise target block using already-built context cache.
    The target branch itself is never cached across different timesteps.
    """
    if target_noisy.ndim != 5:
        raise ValueError(f"target_noisy must be [B,C,T,H,W], got {tuple(target_noisy.shape)}")
    b, c, t, h, w = target_noisy.shape
    if target_frames != t:
        raise ValueError(f"target_frames mismatch: got {target_frames}, input has T={t}")

    _set_fm_inference_timesteps(
        scheduler=scheduler,
        num_inference_steps=num_inference_steps,
        device=target_noisy.device,
    )
    latents = target_noisy

    for t_idx, current_timestep in enumerate(scheduler.timesteps):
        timestep = torch.full(
            (b, target_frames),
            fill_value=float(current_timestep),
            device=latents.device,
            dtype=torch.float32,
        )
        x0_pred = model.forward_with_context_cache(
            target_x=latents,
            target_timestep=timestep,
            predict_frames=target_frames,
            strict_target_isolation=strict_target_isolation,
            return_modality_dict=True,
        )
        x0_tensor = merge_modalities(x0_pred)
        flow_pred = _x0_to_flow_pred(xt=latents, x0=x0_tensor, scheduler=scheduler, timestep=timestep)
        if isinstance(scheduler, FlowMatchScheduler):
            flow_bt = rearrange(flow_pred, "b c t h w -> (b t) c h w")
            latents_bt = rearrange(latents, "b c t h w -> (b t) c h w")
            next_latents = scheduler.step(
                model_output=flow_bt,
                timestep=timestep,
                sample=latents_bt,
                to_final=(solver_to_final and (t_idx == len(scheduler.timesteps) - 1)),
            )
            latents = rearrange(next_latents, "(b t) c h w -> b c t h w", b=b, t=target_frames)
            continue

        step_output = scheduler.step(
            model_output=flow_pred,
            timestep=current_timestep,
            sample=latents,
            return_dict=False,
        )
        latents = _extract_solver_sample(step_output)

    return split_modalities(latents, spec=ModalitySpec())


@torch.no_grad()
def rollout_with_kv_cache(
    model: RainCausalPatchTransformerDiffusion,
    scheduler: FlowRolloutScheduler,
    context_clean: torch.Tensor,
    horizon_blocks: int,
    block_frames: int = 1,
    num_inference_steps: int = 4,
    strict_target_isolation: bool = True,
    seed: int | None = None,
) -> ModalityDict:
    """
    Autoregressive rollout with context-only KV cache:
      1) build cache from clean context
      2) denoise one target block from noise with FM
      3) append generated clean block into cache
      4) repeat
    """
    if seed is not None:
        g = torch.Generator(device=context_clean.device)
        g.manual_seed(seed)
    else:
        g = None

    spec = ModalitySpec()
    if context_clean.shape[1] != spec.total_channels:
        raise ValueError(f"context channels mismatch: got {context_clean.shape[1]}, expected {spec.total_channels}")

    model.clear_context_cache()
    b, _, _, h, w = context_clean.shape
    context_timestep = torch.zeros((b, context_clean.shape[2]), device=context_clean.device, dtype=torch.float32)
    model.build_context_cache(context_clean, context_timestep=context_timestep)

    radar_out = []
    satellite_out = []
    rain_out = []

    for _ in range(horizon_blocks):
        if g is None:
            target_noisy = torch.randn(b, spec.total_channels, block_frames, h, w, device=context_clean.device)
        else:
            target_noisy = torch.randn(
                b,
                spec.total_channels,
                block_frames,
                h,
                w,
                device=context_clean.device,
                generator=g,
            )

        denoised = fm_denoise_target_with_kv_cache(
            model=model,
            scheduler=scheduler,
            target_noisy=target_noisy,
            target_frames=block_frames,
            num_inference_steps=num_inference_steps,
            strict_target_isolation=strict_target_isolation,
        )
        radar_out.append(denoised["radar"])
        satellite_out.append(denoised["satellite"])
        rain_out.append(denoised["rain"])

        # append generated clean block as new context (t=0 cache branch)
        model.append_to_context_cache(
            new_context_x=merge_modalities(denoised),
            context_timestep=torch.zeros((b, block_frames), device=context_clean.device, dtype=torch.float32),
        )

    return {
        "radar": torch.cat(radar_out, dim=2) if radar_out else torch.empty(0, device=context_clean.device),
        "satellite": torch.cat(satellite_out, dim=2) if satellite_out else torch.empty(0, device=context_clean.device),
        "rain": torch.cat(rain_out, dim=2) if rain_out else torch.empty(0, device=context_clean.device),
    }
