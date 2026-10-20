import torch
import torch.nn.functional as F
from einops import rearrange

from src.rain_local_fusion.fusion import RainLocalResidual
from src.rain_priority.model import RainPriorityModel


class RainLocalFusionModel(RainPriorityModel):
    def __init__(
        self,
        local_fusion_enabled: bool = True,
        local_fusion_kernel_size: int = 3,
        local_fusion_hidden_channels: int = 32,
        spatial_modality_gate_enabled: bool = False,
        **model_kwargs: object,
    ) -> None:
        if spatial_modality_gate_enabled:
            raise ValueError("The local-fusion comparison disables the shared input gate.")
        for name, expected in (
            ("decoder_type", "3d"),
            ("decoder_condition_mode", "none"),
            ("decoder_output_mode", "pixels"),
        ):
            if model_kwargs.get(name, expected) != expected:
                raise ValueError(f"Keep {name}={expected} in the local-fusion experiment.")
        if model_kwargs.get("cross_modal_adapter_enabled", False) or model_kwargs.get(
            "local_window_refiner_enabled", False
        ):
            raise ValueError("Additional adapters/refiners are outside the local-fusion comparison.")
        if local_fusion_kernel_size not in {1, 3} or local_fusion_hidden_channels <= 0:
            raise ValueError("Use kernel size 1/3 and positive hidden channels.")
        super().__init__(spatial_modality_gate_enabled=False, **model_kwargs)
        self.local_fusion: RainLocalResidual | None = None
        if local_fusion_enabled:
            with torch.random.fork_rng(devices=[]):
                self.local_fusion = RainLocalResidual(
                    self.dim,
                    self.patch_embed.out_channels,
                    hidden_channels=local_fusion_hidden_channels,
                    kernel_size=local_fusion_kernel_size,
                )

    def forward(
        self,
        x: torch.Tensor,
        predict_frames: int = 1,
        strict_target_isolation: bool = False,
        return_modality_dict: bool = True,
        frame_times: torch.Tensor | None = None,
        gate_source: torch.Tensor | None = None,
        gate_steps: torch.Tensor | None = None,
        return_gate_stats: bool = False,
    ) -> dict[str, torch.Tensor] | torch.Tensor | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if self.local_fusion is None:
            return super().forward(
                x,
                predict_frames,
                strict_target_isolation,
                return_modality_dict,
                frame_times,
                gate_source,
                gate_steps,
                return_gate_stats,
            )
        if x.ndim != 5 or x.shape[1] != self.in_channels:
            raise ValueError(f"x must be [B,{self.in_channels},T,H,W].")
        b, _, t, h, w = x.shape
        if not 0 < predict_frames <= t <= self.max_frames:
            raise ValueError("Expected 0 < predict_frames <= input frames <= max_frames.")
        if return_gate_stats and not return_modality_dict:
            raise ValueError("Statistics require modality dictionary output.")
        if gate_source is None or gate_source.shape != (b, 3, t):
            raise ValueError("Explicit gate_source with shape [B,3,T] is required.")
        if not ((gate_source == 0) | (gate_source == 1) | (gate_source == 2)).all():
            raise ValueError("gate_source must contain only missing=0, truth=1, generated=2.")

        tokens, hp, wp, _, _, temporal_positions, _ = self._encode_tokens(
            x=x,
            frame_offset=0,
            frame_times=frame_times,
        )
        context_frames = t - predict_frames
        for block in self.blocks:
            tokens = block(
                tokens,
                context_frames=context_frames,
                strict_target_isolation=strict_target_isolation,
                hp=hp,
                wp=wp,
                temporal_positions=temporal_positions,
            )
        tokens = self.norm(tokens)[:, -predict_frames:]
        features = rearrange(tokens, "b t (h w) c -> b c t h w", h=hp, w=wp)

        available = (gate_source[:, :2, -predict_frames:] != 0).to(x.device)
        seed = x[:, :, -predict_frames:]
        contributions: list[torch.Tensor] = []
        start = 0
        for index, name in enumerate(("radar", "satellite")):
            end = start + self.modality_channels[name]
            mask = available[:, index : index + 1, :, None, None]
            value = torch.where(mask, seed[:, start:end], 0)
            contribution = F.conv3d(
                value,
                self.patch_embed.weight[:, start:end],
                bias=None,
                stride=self.patch_embed.stride,
                padding=self.patch_embed.padding,
                dilation=self.patch_embed.dilation,
            )
            contributions.append(torch.where(mask, contribution, 0))
            start = end
        residual = self.local_fusion(features, contributions[0], contributions[1], available)
        rain_features = torch.empty_strided(
            features.shape, features.stride(), device=features.device, dtype=features.dtype
        ).copy_(features + residual)  # Preserve the baseline decoder layout, including sliced batch strides.
        target_size = (predict_frames, h, w)
        result = {
            "radar": self.radar_decoder(features, target_thw=target_size, cond=None),
            "satellite": self.satellite_decoder(features, target_thw=target_size, cond=None),
            "rain": self.rain_decoder(rain_features, target_thw=target_size, cond=None),
        }
        if return_gate_stats:
            base_rms = features.detach().float().square().mean().sqrt()
            delta_rms = residual.detach().float().square().mean().sqrt()
            stats = {
                "fusion/base_rms": base_rms,
                "fusion/delta_rms": delta_rms,
                "fusion/relative_rms": delta_rms / base_rms.clamp_min(1e-6),
                "fusion/radar_available": available[:, 0].float().mean(),
                "fusion/satellite_available": available[:, 1].float().mean(),
            }
            return result, stats
        return result if return_modality_dict else result["rain"]
