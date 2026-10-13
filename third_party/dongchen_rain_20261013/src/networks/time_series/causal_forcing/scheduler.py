from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import torch


class SchedulerInterface(ABC):
    """
    Base interface for diffusion noise schedules.
    """

    alphas_cumprod: torch.Tensor
    num_train_timesteps: int

    @abstractmethod
    def add_noise(self, clean_latent: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        pass

    def _flatten_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        if timestep.ndim > 1:
            timestep = timestep.reshape(-1)
        return timestep.long()

    def _reshape_coeff(self, coeff: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        while coeff.ndim < sample.ndim:
            coeff = coeff.unsqueeze(-1)
        return coeff

    def _alpha_beta_prod(self, timestep: torch.Tensor, sample: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        timestep = self._flatten_timestep(timestep).to(self.alphas_cumprod.device)
        alpha_prod_t = self.alphas_cumprod[timestep]
        beta_prod_t = 1.0 - alpha_prod_t
        alpha_prod_t = self._reshape_coeff(alpha_prod_t, sample)
        beta_prod_t = self._reshape_coeff(beta_prod_t, sample)
        return alpha_prod_t, beta_prod_t

    def convert_x0_to_noise(self, x0: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        noise = (xt - sqrt(alpha_t) * x0) / sqrt(beta_t)
        """
        original_dtype = x0.dtype
        x0 = x0.double()
        xt = xt.double()
        alpha_prod_t, beta_prod_t = self._alpha_beta_prod(timestep, x0)
        noise_pred = (xt - alpha_prod_t.sqrt() * x0) / beta_prod_t.sqrt()
        return noise_pred.to(original_dtype)

    def convert_noise_to_x0(self, noise: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        x0 = (xt - sqrt(beta_t) * noise) / sqrt(alpha_t)
        """
        original_dtype = noise.dtype
        noise = noise.double()
        xt = xt.double()
        alpha_prod_t, beta_prod_t = self._alpha_beta_prod(timestep, noise)
        x0_pred = (xt - beta_prod_t.sqrt() * noise) / alpha_prod_t.sqrt()
        return x0_pred.to(original_dtype)

    def convert_velocity_to_x0(self, velocity: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        x0 = sqrt(alpha_t) * x_t - sqrt(beta_t) * v
        """
        original_dtype = velocity.dtype
        velocity = velocity.double()
        xt = xt.double()
        alpha_prod_t, beta_prod_t = self._alpha_beta_prod(timestep, velocity)
        x0_pred = alpha_prod_t.sqrt() * xt - beta_prod_t.sqrt() * velocity
        return x0_pred.to(original_dtype)

    def convert_x0_to_velocity(self, x0: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        v = (sqrt(alpha_t) * x_t - x0) / sqrt(beta_t)
        """
        original_dtype = x0.dtype
        x0 = x0.double()
        xt = xt.double()
        alpha_prod_t, beta_prod_t = self._alpha_beta_prod(timestep, x0)
        velocity = (alpha_prod_t.sqrt() * xt - x0) / beta_prod_t.sqrt()
        return velocity.to(original_dtype)

    def convert_noise_to_velocity(self, noise: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        v = sqrt(alpha_t) * noise - sqrt(beta_t) * x0
        """
        x0 = self.convert_noise_to_x0(noise=noise, xt=xt, timestep=timestep)
        original_dtype = noise.dtype
        noise = noise.double()
        x0 = x0.double()
        alpha_prod_t, beta_prod_t = self._alpha_beta_prod(timestep, noise)
        velocity = alpha_prod_t.sqrt() * noise - beta_prod_t.sqrt() * x0
        return velocity.to(original_dtype)

    def convert_velocity_to_noise(self, velocity: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        noise = sqrt(beta_t) * x_t + sqrt(alpha_t) * v
        """
        original_dtype = velocity.dtype
        velocity = velocity.double()
        xt = xt.double()
        alpha_prod_t, beta_prod_t = self._alpha_beta_prod(timestep, velocity)
        noise = beta_prod_t.sqrt() * xt + alpha_prod_t.sqrt() * velocity
        return noise.to(original_dtype)

    def training_target(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
        prediction_type: Literal["noise", "x0", "velocity"] = "noise",
    ) -> torch.Tensor:
        if prediction_type == "noise":
            return noise
        if prediction_type == "x0":
            return sample
        if prediction_type == "velocity":
            # v = sqrt(alpha_t) * noise - sqrt(beta_t) * x0
            alpha_prod_t, beta_prod_t = self._alpha_beta_prod(timestep, sample)
            return alpha_prod_t.sqrt().to(noise.dtype) * noise - beta_prod_t.sqrt().to(sample.dtype) * sample
        raise ValueError(f"Unsupported prediction_type={prediction_type}")


class DiffusionScheduler(SchedulerInterface):
    """
    DDPM-style Gaussian schedule.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ) -> None:
        self.num_train_timesteps = int(num_train_timesteps)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)

        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_train_timesteps, dtype=torch.float32)
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.timesteps = torch.arange(self.num_train_timesteps - 1, -1, -1, dtype=torch.long)

    def add_noise(self, clean_latent: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        timestep = self._flatten_timestep(timestep).to(self.alphas_cumprod.device)
        alpha_prod_t = self.alphas_cumprod[timestep].to(clean_latent.device)
        alpha_prod_t = self._reshape_coeff(alpha_prod_t, clean_latent)
        beta_prod_t = 1.0 - alpha_prod_t
        sample = alpha_prod_t.sqrt().to(clean_latent.dtype) * clean_latent + beta_prod_t.sqrt().to(noise.dtype) * noise
        return sample


class FlowMatchScheduler:
    """
    Copied/adapted from third_party causal_forcing utils/scheduler.py.
    """

    def __init__(
        self,
        num_inference_steps: int = 100,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        sigma_max: float = 1.0,
        sigma_min: float = 0.003 / 1.002,
        inverse_timesteps: bool = False,
        extra_one_step: bool = False,
        reverse_sigmas: bool = False,
    ) -> None:
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.linear_timesteps_weights = None
        self.set_timesteps(num_inference_steps=num_inference_steps)

    def set_timesteps(self, num_inference_steps: int = 100, denoising_strength: float = 1.0, training: bool = False):
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)
            y_shifted = y - y.min()
            self.linear_timesteps_weights = y_shifted * (num_inference_steps / y_shifted.sum())

    def step(self, model_output, timestep, sample, to_final: bool = False):
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        self.sigmas = self.sigmas.to(model_output.device)
        self.timesteps = self.timesteps.to(model_output.device)
        timestep_id = torch.argmin((self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma = self.sigmas[timestep_id].reshape(-1, 1, 1, 1)
        if to_final or (timestep_id + 1 >= len(self.timesteps)).any():
            sigma_ = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1].reshape(-1, 1, 1, 1)
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample

    def add_noise(self, original_samples, noise, timestep):
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        self.sigmas = self.sigmas.to(noise.device)
        self.timesteps = self.timesteps.to(noise.device)
        timestep_id = torch.argmin((self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma = self.sigmas[timestep_id].reshape(-1, 1, 1, 1)
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample.type_as(noise)

    def training_target(self, sample, noise, timestep):
        _ = timestep
        return noise - sample

    def training_weight(self, timestep):
        if self.linear_timesteps_weights is None:
            raise RuntimeError("training_weight requested before set_timesteps(training=True)")
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        self.linear_timesteps_weights = self.linear_timesteps_weights.to(timestep.device)
        timestep_id = torch.argmin((self.timesteps.unsqueeze(1) - timestep.unsqueeze(0)).abs(), dim=0)
        return self.linear_timesteps_weights[timestep_id]

    def sigma_from_timestep(self, timestep: torch.Tensor, device: torch.device) -> torch.Tensor:
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        sigmas = self.sigmas.to(device)
        timesteps = self.timesteps.to(device)
        timestep_id = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        return sigmas[timestep_id].reshape(-1, 1, 1, 1)

    def flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        if flow_pred.shape != xt.shape:
            raise ValueError(f"flow_pred/xt shape mismatch: {tuple(flow_pred.shape)} vs {tuple(xt.shape)}")
        original_dtype = flow_pred.dtype
        flow_pred = flow_pred.double()
        xt = xt.double()
        sigma_t = self.sigma_from_timestep(timestep=timestep, device=xt.device).double()
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    def x0_to_flow_pred(self, x0: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        if x0.shape != xt.shape:
            raise ValueError(f"x0/xt shape mismatch: {tuple(x0.shape)} vs {tuple(xt.shape)}")
        original_dtype = x0.dtype
        x0 = x0.double()
        xt = xt.double()
        sigma_t = self.sigma_from_timestep(timestep=timestep, device=xt.device).double()
        flow_pred = (xt - x0) / sigma_t.clamp(min=1e-12)
        return flow_pred.to(original_dtype)

    def convert_x0_to_noise(self, x0: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        if x0.shape != xt.shape:
            raise ValueError(f"x0/xt shape mismatch: {tuple(x0.shape)} vs {tuple(xt.shape)}")
        original_dtype = x0.dtype
        x0 = x0.double()
        xt = xt.double()
        sigma_t = self.sigma_from_timestep(timestep=timestep, device=xt.device).double()
        noise = (xt - (1.0 - sigma_t) * x0) / sigma_t.clamp(min=1e-12)
        return noise.to(original_dtype)

    def convert_noise_to_x0(self, noise: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        if noise.shape != xt.shape:
            raise ValueError(f"noise/xt shape mismatch: {tuple(noise.shape)} vs {tuple(xt.shape)}")
        original_dtype = noise.dtype
        noise = noise.double()
        xt = xt.double()
        sigma_t = self.sigma_from_timestep(timestep=timestep, device=xt.device).double()
        x0 = (xt - sigma_t * noise) / (1.0 - sigma_t).clamp(min=1e-12)
        return x0.to(original_dtype)
