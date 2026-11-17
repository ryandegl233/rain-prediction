import torch
from torch import nn

from src.baselines.contracts import HistoryBatch


class NextFrameForecaster(nn.Module):
    required_modalities: tuple[str, ...] = ()

    def __init__(
        self,
        network: nn.Module,
        residual: bool = True,
        clamp_output: bool = False,
        min_value: float = 0.0,
        strict_target_isolation: bool = False,
        time_interval: int = 30,
    ) -> None:
        super().__init__()
        if network.frame_patch_size != 1:
            raise ValueError("Baseline evaluation requires frame_patch_size=1.")
        self.network = network
        self.residual = residual
        self.clamp_output = clamp_output
        self.min_value = min_value
        self.strict_target_isolation = strict_target_isolation
        self.time_interval = time_interval

    def forward(self, history: HistoryBatch) -> torch.Tensor:
        device = next(self.network.parameters()).device
        context = history.values.to(device)
        available = history.available.to(device)
        time = history.time.to(device) if history.time is not None else None
        if self.network.use_time_embedding and time is None:
            raise ValueError("Reference network requires time_past when time embedding is enabled.")
        seed = context[:, :, -1:]
        context = self.network._apply_context_modality_mask_token(
            context_x=context, context_modality_available=available
        )
        rain_predictions = []
        for _ in range(4):
            output = self.network.forward_ar(
                context_x=context,
                target_x=seed,
                predict_frames=1,
                strict_target_isolation=self.strict_target_isolation,
                return_modality_dict=True,
                context_time=time,
                target_time=time[:, -1:] if time is not None else None,
            )
            rain = output["rain"]
            if self.residual:
                rain = rain + seed[:, 11:12]
                if self.clamp_output:
                    rain = rain.clamp_min(self.min_value)
            rain_predictions.append(rain)
            generated = torch.cat([output["radar"], output["satellite"], output.get("rain_rollout", rain)], dim=1)
            context = torch.cat([context, generated], dim=2)
            seed = generated
            if time is not None:
                time = torch.cat([time, (time[:, -1:] + self.time_interval / 1440) % 1], dim=1)
        return torch.cat(rain_predictions, dim=2)
