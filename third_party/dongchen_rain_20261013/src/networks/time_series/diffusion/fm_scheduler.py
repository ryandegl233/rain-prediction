import torch


class FlowMatchScheduler:
    """
    Minimal Flow-Matching scheduler copied and adapted from causal_forcing.
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
        self.set_timesteps(num_inference_steps=num_inference_steps, training=False)

    def set_timesteps(
        self,
        num_inference_steps: int = 100,
        denoising_strength: float = 1.0,
        training: bool = False,
    ) -> None:
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

    def _lookup_sigma(self, timestep: torch.Tensor, device: torch.device) -> torch.Tensor:
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        sigmas = self.sigmas.to(device)
        timesteps = self.timesteps.to(device)
        timestep_id = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        return sigma

    def step(self, model_output: torch.Tensor, timestep: torch.Tensor, sample: torch.Tensor, to_final: bool = False) -> torch.Tensor:
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        sigmas = self.sigmas.to(model_output.device)
        timesteps = self.timesteps.to(model_output.device)
        timestep_id = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        if to_final or (timestep_id + 1 >= len(timesteps)).any():
            sigma_next = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_next = sigmas[timestep_id + 1].reshape(-1, 1, 1, 1)
        prev_sample = sample + model_output * (sigma_next - sigma)
        return prev_sample

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        sigma = self._lookup_sigma(timestep=timestep, device=noise.device)
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample.type_as(noise)

    def training_target(self, sample: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        _ = timestep
        return noise - sample

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        if self.linear_timesteps_weights is None:
            raise RuntimeError("training_weight requested before set_timesteps(training=True)")
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        timesteps = self.timesteps.to(timestep.device)
        weights = self.linear_timesteps_weights.to(timestep.device)
        timestep_id = torch.argmin((timesteps.unsqueeze(1) - timestep.unsqueeze(0)).abs(), dim=0)
        return weights[timestep_id]

    def sigma_from_timestep(self, timestep: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Return sigma_t for shape-compatible conversion:
            x_t = (1 - sigma_t) * x0 + sigma_t * noise
            flow_pred = noise - x0 = (x_t - x0) / sigma_t
        """
        return self._lookup_sigma(timestep=timestep, device=device)
