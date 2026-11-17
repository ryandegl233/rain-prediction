import numpy as np
import torch
from torch import nn

from src.baselines.contracts import HistoryBatch


class Persistence(nn.Module):
    required_modalities = ("rain",)

    def forward(self, history: HistoryBatch) -> torch.Tensor:
        if not history.available[:, 2].all():
            raise ValueError("Persistence requires observed rain history.")
        return history.values[:, 11:12, -1:].expand(-1, -1, 4, -1, -1).clone()


class LucasKanade(nn.Module):
    required_modalities = ("rain",)

    def __init__(self, rain_threshold: float = 0.02, size_opening: int = 3) -> None:
        super().__init__()
        if not np.isfinite(rain_threshold) or rain_threshold <= 0 or size_opening < 0:
            raise ValueError("LK requires a positive rain threshold and nonnegative opening size.")
        self.rain_threshold = rain_threshold
        self.size_opening = size_opening

    def forward(self, history: HistoryBatch) -> torch.Tensor:
        if not history.available[:, 2].all():
            raise ValueError("Lucas-Kanade requires observed rain history.")
        from pysteps.extrapolation.semilagrangian import extrapolate
        from pysteps.motion.lucaskanade import dense_lucaskanade

        rain = history.values[:, 11].detach().float().cpu().numpy()
        if not np.isfinite(rain).all() or np.any(rain < 0):
            raise ValueError("LK requires finite nonnegative rain in dataset units.")
        forecasts = []
        for frames in rain:
            motion_images = 10 * np.log10(np.maximum(frames, self.rain_threshold))
            velocity = dense_lucaskanade(motion_images, size_opening=self.size_opening)
            prediction = extrapolate(frames[-1], velocity, 4, outval=0.0, interp_order=1)
            if prediction.shape != (4, *frames.shape[-2:]) or not np.isfinite(prediction).all():
                raise ValueError("LK returned invalid forecast fields.")
            forecasts.append(prediction)
        output = torch.from_numpy(np.stack(forecasts).astype(np.float32)).unsqueeze(1)
        return output.to(history.values.device)
