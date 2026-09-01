import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def _group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _autoencoder_block(c_in: int, c_out: int, stride: tuple[int, int, int] = (1, 1, 1)) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(c_in, c_out, 3, stride=stride, padding=1),
        nn.GroupNorm(_group_count(c_out), c_out),
        nn.SiLU(),
        nn.Conv3d(c_out, c_out, 3, padding=1),
        nn.GroupNorm(_group_count(c_out), c_out),
        nn.SiLU(),
    )


class ResidualAutoencoder(nn.Module):
    """Spatial-only 4x residual compression with preserved time length."""

    def __init__(self, latent_channels: int = 8) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.enc1 = _autoencoder_block(1, 32, (1, 2, 2))
        self.enc2 = _autoencoder_block(32, 64, (1, 2, 2))
        self.to_latent = nn.Conv3d(64, self.latent_channels, 1)
        self.dec1 = _autoencoder_block(self.latent_channels, 64)
        self.dec2 = _autoencoder_block(64, 32)
        self.out = nn.Conv3d(32, 1, 3, padding=1)

    def encode(self, residual: torch.Tensor) -> torch.Tensor:
        return self.to_latent(self.enc2(self.enc1(residual)))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        x = self.dec1(latent)
        x = F.interpolate(x, scale_factor=(1, 2, 2), mode="trilinear", align_corners=False)
        x = self.dec2(x)
        x = F.interpolate(x, scale_factor=(1, 2, 2), mode="trilinear", align_corners=False)
        return self.out(x)

    def forward(self, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(residual)
        return self.decode(latent), latent


class LatentNormalizer(nn.Module):
    """Fixed per-channel affine normalization calibrated from the trained AE."""

    def __init__(self, channels: int, minimum_std: float = 1.0e-4) -> None:
        super().__init__()
        self.minimum_std = float(minimum_std)
        self.register_buffer("mean", torch.zeros(1, channels, 1, 1, 1))
        self.register_buffer("std", torch.ones(1, channels, 1, 1, 1))

    def set_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        expected = self.mean.shape
        mean = mean.reshape(expected).to(device=self.mean.device, dtype=self.mean.dtype)
        std = std.reshape(expected).to(device=self.std.device, dtype=self.std.dtype)
        self.mean.copy_(mean)
        self.std.copy_(std.clamp_min(self.minimum_std))

    def normalize(self, latent: torch.Tensor) -> torch.Tensor:
        return (latent - self.mean.to(dtype=latent.dtype)) / self.std.to(dtype=latent.dtype)

    def denormalize(self, latent: torch.Tensor) -> torch.Tensor:
        return latent * self.std.to(dtype=latent.dtype) + self.mean.to(dtype=latent.dtype)


def _timestep_embedding(timestep: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    frequency = torch.exp(
        -math.log(10_000) * torch.arange(half, device=timestep.device) / max(half - 1, 1)
    )
    args = timestep.float()[:, None] * frequency[None]
    embedding = torch.cat([args.sin(), args.cos()], dim=-1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class HistoryConditionEncoder(nn.Module):
    """Convert historical multimodal fields into condition tokens."""

    def __init__(self, in_channels: int, dim: int, patch_size: int) -> None:
        super().__init__()
        self.projection = nn.Conv3d(
            in_channels,
            dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        encoded = self.projection(history)
        return encoded.permute(0, 2, 3, 4, 1).flatten(2, 3)


class ConditionCompressor(nn.Module):
    def __init__(self, dim: int, heads: int, tokens: int) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(tokens, dim) * 0.02)
        self.cross = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        batch, time, patches, dim = history.shape
        memory = history.reshape(batch, time * patches, dim)
        query = self.queries[None].expand(batch, -1, -1)
        return query + self.cross(query, memory, memory, need_weights=False)[0]


class CoarseConditionEncoder(nn.Module):
    def __init__(self, latent_channels: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, 3, stride=(1, 2, 2), padding=1),
            nn.GroupNorm(_group_count(32), 32),
            nn.SiLU(),
            nn.Conv3d(32, latent_channels, 3, stride=(1, 2, 2), padding=1),
        )

    def forward(self, coarse_rain: torch.Tensor) -> torch.Tensor:
        return self.encoder(coarse_rain)


class ResBlock3D(nn.Module):
    def __init__(self, c_in: int, c_out: int, emb_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(c_in), c_in)
        self.conv1 = nn.Conv3d(c_in, c_out, (1, 3, 3), padding=(0, 1, 1))
        self.emb = nn.Linear(emb_dim, 2 * c_out)
        self.norm2 = nn.GroupNorm(_group_count(c_out), c_out)
        self.conv2 = nn.Conv3d(c_out, c_out, (1, 3, 3), padding=(0, 1, 1))
        self.skip = nn.Conv3d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(embedding).chunk(2, dim=-1)
        hidden = self.norm2(hidden) * (1 + scale[:, :, None, None, None]) + shift[:, :, None, None, None]
        return self.skip(x) + self.conv2(F.silu(hidden))


class LowResolutionCrossAttention(nn.Module):
    def __init__(self, channels: int, condition_dim: int, heads: int) -> None:
        super().__init__()
        self.to_query = nn.Linear(channels, condition_dim)
        self.attention = nn.MultiheadAttention(condition_dim, heads, batch_first=True)
        self.out = nn.Linear(condition_dim, channels)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        batch, channels, time, height, width = x.shape
        sequence = x.permute(0, 2, 3, 4, 1).reshape(batch, time * height * width, channels)
        query = self.to_query(sequence)
        attended = self.attention(query, condition, condition, need_weights=False)[0]
        attended = self.out(attended).reshape(batch, time, height, width, channels)
        return x + attended.permute(0, 4, 1, 2, 3)


class ConditionalUNet3D(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        latent_channels = int(cfg["latent_channels"])
        base_channels = int(cfg["base_channels"])
        widths = [base_channels * int(multiplier) for multiplier in cfg["channel_mult"]]
        if len(widths) != 4:
            raise ValueError(f"channel_mult must contain four values, got {cfg['channel_mult']}")

        condition_dim = int(cfg["condition_dim"])
        condition_heads = int(cfg["condition_heads"])
        self.checkpointing = bool(cfg["gradient_checkpointing"])
        self.time_mlp = nn.Sequential(nn.Linear(128, 512), nn.SiLU(), nn.Linear(512, 512))
        self.input = nn.Conv3d(latent_channels * 2, widths[0], 3, padding=1)
        self.down = nn.ModuleList()
        previous = widths[0]
        for width in widths:
            self.down.append(ResBlock3D(previous, width, 512))
            previous = width
        self.downsample = nn.ModuleList(
            [
                nn.Conv3d(widths[index], widths[index], (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
                for index in range(3)
            ]
        )
        self.attention_1 = LowResolutionCrossAttention(widths[1], condition_dim, condition_heads)
        self.attention_2 = LowResolutionCrossAttention(widths[2], condition_dim, condition_heads)
        self.mid_attention = LowResolutionCrossAttention(widths[3], condition_dim, condition_heads)
        self.up_projection = nn.ModuleList(
            [nn.Conv3d(widths[index + 1], widths[index], 1) for index in reversed(range(3))]
        )
        self.up_blocks = nn.ModuleList(
            [ResBlock3D(widths[index] * 2, widths[index], 512) for index in reversed(range(3))]
        )
        self.out = nn.Conv3d(widths[0], latent_channels, 3, padding=1)

    def _run(self, module: nn.Module, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        if self.checkpointing and self.training and x.requires_grad:
            return checkpoint(module, x, embedding, use_reentrant=False, preserve_rng_state=True)
        return module(x, embedding)

    def forward(
        self,
        noisy: torch.Tensor,
        coarse_condition: torch.Tensor,
        timestep: torch.Tensor,
        condition_tokens: torch.Tensor,
    ) -> torch.Tensor:
        embedding = self.time_mlp(_timestep_embedding(timestep, 128))
        x = self.input(torch.cat([noisy, coarse_condition], dim=1))
        skips: list[torch.Tensor] = []
        for index, layer in enumerate(self.down):
            x = self._run(layer, x, embedding)
            skips.append(x)
            if index < 3:
                x = self.downsample[index](x)
                if index == 1:
                    x = self.attention_1(x, condition_tokens)
                elif index == 2:
                    x = self.attention_2(x, condition_tokens)
        x = self.mid_attention(x, condition_tokens)
        for projection, layer, skip in zip(self.up_projection, self.up_blocks, reversed(skips[:-1]), strict=True):
            x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            x = projection(x)
            x = self._run(layer, torch.cat([x, skip], dim=1), embedding)
        return self.out(x)


class ConditionalDiffusion(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.steps = int(cfg["train_steps"])
        self.sample_steps = int(cfg["sample_steps"])
        self.history_encoder = HistoryConditionEncoder(
            int(cfg["history_channels"]),
            int(cfg["condition_dim"]),
            int(cfg["history_patch_size"]),
        )
        self.condition = ConditionCompressor(
            int(cfg["condition_dim"]),
            int(cfg["condition_heads"]),
            int(cfg["condition_tokens"]),
        )
        self.coarse_encoder = CoarseConditionEncoder(int(cfg["latent_channels"]))
        self.unet = ConditionalUNet3D(cfg)
        betas = torch.linspace(1.0e-4, 0.02, self.steps)
        self.register_buffer("alphas_bar", torch.cumprod(1.0 - betas, dim=0))

    def training_outputs(
        self,
        normalized_residual_latent: torch.Tensor,
        coarse_rain: torch.Tensor,
        history: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch = normalized_residual_latent.shape[0]
        timestep = torch.randint(0, self.steps, (batch,), device=normalized_residual_latent.device)
        alpha = self.alphas_bar[timestep].view(batch, 1, 1, 1, 1)
        noise = torch.randn_like(normalized_residual_latent)
        noisy = alpha.sqrt() * normalized_residual_latent + (1 - alpha).sqrt() * noise
        coarse_condition = self.coarse_encoder(coarse_rain)
        condition = self.condition(self.history_encoder(history))
        prediction = self.unet(noisy, coarse_condition, timestep, condition)
        predicted_x0 = (noisy - (1 - alpha).sqrt() * prediction) / alpha.sqrt()
        return {
            "noise_loss": F.mse_loss(prediction, noise),
            "predicted_x0": predicted_x0,
            "alpha_bar": alpha,
            "timestep": timestep,
        }

    def training_loss(
        self,
        normalized_residual_latent: torch.Tensor,
        coarse_rain: torch.Tensor,
        history: torch.Tensor,
    ) -> torch.Tensor:
        return self.training_outputs(normalized_residual_latent, coarse_rain, history)["noise_loss"]

    def forward(
        self,
        normalized_residual_latent: torch.Tensor,
        coarse_rain: torch.Tensor,
        history: torch.Tensor,
        return_outputs: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        outputs = self.training_outputs(normalized_residual_latent, coarse_rain, history)
        return outputs if return_outputs else outputs["noise_loss"]

    def _ddim_loop(
        self,
        x: torch.Tensor,
        coarse_condition: torch.Tensor,
        condition: torch.Tensor,
        steps: int,
    ) -> torch.Tensor:
        schedule = torch.linspace(self.steps - 1, 0, steps, device=x.device).long()
        for index, timestep_scalar in enumerate(schedule):
            timestep = timestep_scalar.expand(x.shape[0])
            alpha = self.alphas_bar[timestep_scalar]
            eps = self.unet(x, coarse_condition, timestep, condition)
            x0 = (x - (1 - alpha).sqrt() * eps) / alpha.sqrt()
            if index + 1 == len(schedule):
                x = x0
            else:
                alpha_next = self.alphas_bar[schedule[index + 1]]
                x = alpha_next.sqrt() * x0 + (1 - alpha_next).sqrt() * eps
        return x

    @torch.no_grad()
    def ddim_sample(
        self,
        shape: tuple[int, ...],
        coarse_rain: torch.Tensor,
        history: torch.Tensor,
        steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        x = torch.randn(shape, device=coarse_rain.device, dtype=coarse_rain.dtype, generator=generator)
        condition = self.condition(self.history_encoder(history))
        coarse_condition = self.coarse_encoder(coarse_rain)
        return self._ddim_loop(x, coarse_condition, condition, self.sample_steps if steps is None else int(steps))
