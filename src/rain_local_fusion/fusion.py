import torch
from torch import nn
from einops import rearrange


class RainLocalResidual(nn.Module):
    def __init__(self, channels: int, auxiliary_channels: int, hidden_channels: int = 32, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size not in {1, 3} or min(channels, auxiliary_channels, hidden_channels) <= 0:
            raise ValueError("Use positive channels and kernel_size=1 or 3 for the controlled comparison.")
        self.query = nn.Conv2d(channels, hidden_channels, 1)
        self.projections = nn.ModuleList([nn.Conv2d(auxiliary_channels, hidden_channels, 1) for _ in range(2)])
        self.mix = nn.Sequential(nn.Conv2d(3 * hidden_channels, hidden_channels, 1), nn.SiLU())
        self.output = nn.Conv2d(hidden_channels, channels, 1)
        self.spatial = nn.ModuleList(
            [
                nn.Conv2d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size,
                    groups=hidden_channels,
                    padding=kernel_size // 2,
                    padding_mode="replicate",
                )
                for _ in range(2)
            ]
        )
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self, query: torch.Tensor, radar: torch.Tensor, satellite: torch.Tensor, available: torch.Tensor
    ) -> torch.Tensor:
        if query.ndim != 5:
            raise ValueError("query must be [B,C,T,H,W].")
        b, _, t, h, w = query.shape
        if available.shape != (b, 2, t) or available.dtype != torch.bool:
            raise ValueError("available must be a boolean [B,2,T] tensor.")
        features = [self.query(rearrange(query, "b c t h w -> (b t) c h w"))]
        masks = available.to(query.device)
        for index, (value, projection, spatial) in enumerate(
            zip((radar, satellite), self.projections, self.spatial, strict=True)
        ):
            if value.ndim != 5 or (value.shape[0], *value.shape[2:]) != (b, t, h, w):
                raise ValueError("Auxiliary features must match query batch, time and spatial dimensions.")
            mask = masks[:, index, :, None, None, None].reshape(b * t, 1, 1, 1)
            value = rearrange(value, "b c t h w -> (b t) c h w")
            value = torch.where(mask, value, 0)
            feature = spatial(projection(value))
            features.append(torch.where(mask, feature, 0))
        residual = self.output(self.mix(torch.cat(features, dim=1)))
        residual = rearrange(residual, "(b t) c h w -> b c t h w", b=b, t=t)
        return torch.where(masks.any(dim=1)[:, None, :, None, None], residual, 0)
