import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class TemporalRainGate(nn.Module):
    def __init__(self, channels: int, hidden_channels: int) -> None:
        super().__init__()
        if channels <= 0 or hidden_channels <= 0:
            raise ValueError("Gate channels must be positive.")
        self.net = nn.Sequential(
            nn.Conv2d(6 * channels + 13, hidden_channels, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, groups=hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, 2, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def condition_features(
        self,
        radar: torch.Tensor,
        satellite: torch.Tensor,
        rain: torch.Tensor,
        source: torch.Tensor,
        steps: torch.Tensor,
    ) -> torch.Tensor:
        b, c, t, h, w = radar.shape
        if satellite.shape != radar.shape or rain.shape != radar.shape:
            raise ValueError("Gate feature shapes must match.")
        if source.shape != (b, 3, t) or steps.shape != (b, t):
            raise ValueError("Gate metadata must match [B,3,T] sources and [B,T] steps.")
        if not torch.isfinite(source).all() or not ((source == 0) | (source == 1) | (source == 2)).all():
            raise ValueError("Source codes must be missing=0, truth-fed=1, or generated=2.")
        if not torch.isfinite(steps).all() or (steps[:, 1:] < steps[:, :-1]).any():
            raise ValueError("Input step indices must be finite and nondecreasing.")
        source = source.to(device=radar.device)
        steps = steps.to(device=radar.device)
        positions = torch.arange(t, device=radar.device)
        earlier = positions[None, :] < positions[:, None]
        adjacent = (steps[:, :, None] - steps[:, None, :]) == 1
        candidates = torch.where(adjacent & earlier, positions, -1)
        previous = candidates.amax(dim=-1)
        available = source > 0
        previous_available = available.gather(2, previous.clamp_min(0)[:, None, :].expand(-1, 3, -1))
        valid = available & previous_available & (previous[:, None, :] >= 0)
        current: list[torch.Tensor] = []
        changes: list[torch.Tensor] = []
        for index, value in enumerate((radar, satellite, rain)):
            normalized = F.layer_norm(value.permute(0, 2, 3, 4, 1).float(), (c,)).permute(0, 4, 1, 2, 3).to(value.dtype)
            prev = normalized.gather(2, previous.clamp_min(0)[:, None, :, None, None].expand(b, c, t, h, w))
            current.append(normalized)
            changes.append((normalized - prev) * valid[:, index : index + 1, :, None, None])
        one_hot = F.one_hot(source.long(), num_classes=3).permute(0, 1, 3, 2).reshape(b, 9, t)
        metadata = torch.cat([one_hot, valid, steps[:, None] / 4], dim=1).to(radar.dtype)
        return torch.cat([*current, *changes, metadata[:, :, :, None, None].expand(-1, -1, -1, h, w)], dim=1)

    def forward(
        self,
        radar: torch.Tensor,
        satellite: torch.Tensor,
        rain: torch.Tensor,
        source: torch.Tensor,
        steps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.condition_features(radar, satellite, rain, source, steps)
        b, _, t, _, _ = features.shape
        logits = self.net(rearrange(features, "b c t h w -> (b t) c h w"))
        gates = 1 + torch.tanh(rearrange(logits, "(b t) c h w -> b c t h w", b=b, t=t))
        return gates[:, :1], gates[:, 1:2]
