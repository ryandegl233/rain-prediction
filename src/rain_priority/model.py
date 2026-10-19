import torch
import torch.nn.functional as F
from einops import rearrange

from src.gated_modality_rain.model import GatedModalityRainModel, _SpatialModalityGate
from src.rain_priority.gate import TemporalRainGate


class RainPriorityModel(GatedModalityRainModel):
    def __init__(
        self,
        gate_mode: str = "temporal",
        spatial_modality_gate_enabled: bool = True,
        spatial_modality_gate_hidden_channels: int = 32,
        **model_kwargs: object,
    ) -> None:
        if gate_mode not in {"spatial", "temporal"}:
            raise ValueError("gate_mode must be spatial or temporal.")
        if model_kwargs.get("encoder_type", "patch") != "patch" or model_kwargs.get("frame_patch_size", 1) != 1:
            raise ValueError("Rain-priority training requires the original patch encoder and frame_patch_size=1.")
        super().__init__(spatial_modality_gate_enabled=False, **model_kwargs)
        self.gate_mode = gate_mode
        if spatial_modality_gate_enabled:
            channels = self.patch_embed.out_channels
            gate_type = TemporalRainGate if gate_mode == "temporal" else _SpatialModalityGate
            self.spatial_modality_gate = gate_type(channels, spatial_modality_gate_hidden_channels)

    def _encode_tokens(
        self,
        x: torch.Tensor,
        frame_offset: int = 0,
        frame_times: torch.Tensor | None = None,
        gate_source: torch.Tensor | None = None,
        gate_steps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int, int, int, int, torch.Tensor, dict[str, torch.Tensor]]:
        b, c, t, h, w = x.shape
        stats: dict[str, torch.Tensor] = {}
        if t % self.frame_patch_size != 0:
            raise ValueError(f"Input T must be divisible by frame_patch_size={self.frame_patch_size}, got T={t}")
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError(f"Input H/W must be divisible by patch_size={self.patch_size}, got H={h}, W={w}")

        token_t = t // self.frame_patch_size
        if frame_offset + token_t > self.max_token_frames:
            raise ValueError(
                f"frame_offset({frame_offset}) + token_frames({token_t}) exceeds max_token_frames({self.max_token_frames})"
            )

        if self.encoder_type == "patch":
            if self.patch_embed is None or self.stem is None:
                raise RuntimeError("patch encoder is not initialized.")
            encoded = self.patch_embed(x)
            if self.spatial_modality_gate is not None:
                modality_contributions: list[torch.Tensor] = []
                channel_start = 0
                for channels in self.modality_channels.values():
                    channel_end = channel_start + channels
                    modality_contributions.append(
                        F.conv3d(
                            x[:, channel_start:channel_end],
                            self.patch_embed.weight[:, channel_start:channel_end],
                            bias=None,
                            stride=self.patch_embed.stride,
                            padding=self.patch_embed.padding,
                            dilation=self.patch_embed.dilation,
                        )
                    )
                    channel_start = channel_end
                z_radar, z_satellite, z_rain = modality_contributions
                if gate_source is not None:
                    gate_source = gate_source.to(x.device)
                if self.gate_mode == "temporal":
                    if gate_source is None or gate_steps is None:
                        raise ValueError("Temporal gates require explicit gate_source and gate_steps.")
                    gate_radar, gate_satellite = self.spatial_modality_gate(
                        z_radar, z_satellite, z_rain, gate_source, gate_steps
                    )
                else:
                    gate_radar, gate_satellite = self.spatial_modality_gate(z_radar, z_satellite, z_rain)
                input_wet = (
                    F.max_pool3d(
                        (x[:, -self.modality_channels["rain"] :].detach().float() >= 0.1).float(),
                        kernel_size=(1, self.patch_size, self.patch_size),
                    )
                    > 0
                )
                for modality_index, (name, gate, contribution) in enumerate(
                    (("radar", gate_radar, z_radar), ("satellite", gate_satellite, z_satellite))
                ):
                    value = gate.detach().float()
                    stats[f"gate/{name}/mean"] = value.mean()
                    stats[f"gate/{name}/saturated"] = ((value < 0.05) | (value > 1.95)).float().mean()
                    stats[f"gate/{name}/contribution_rms"] = (
                        (value * contribution.detach().float()).square().mean().sqrt()
                    )
                    groups = {"input_wet": input_wet, "input_dry": ~input_wet}
                    if gate_source is not None:
                        for code, label in enumerate(("missing", "truth", "generated")):
                            groups[f"source_{label}"] = (
                                gate_source[:, modality_index : modality_index + 1, :, None, None] == code
                            ).expand_as(value)
                    for label, mask in groups.items():
                        count = mask.sum().float()
                        mean = (value * mask).sum() / count.clamp_min(1)
                        prefix = f"gate/{name}/{label}"
                        stats[f"{prefix}/count"] = count
                        stats[f"{prefix}/mean"] = mean
                        stats[f"{prefix}/std"] = (((value - mean).square() * mask).sum() / count.clamp_min(1)).sqrt()
                        stats[f"{prefix}/saturated"] = (
                            ((value < 0.05) | (value > 1.95)) & mask
                        ).sum() / count.clamp_min(1)
                encoded = encoded + (gate_radar - 1) * z_radar + (gate_satellite - 1) * z_satellite
            _, _, tp, hp, wp = encoded.shape
            encoded = rearrange(encoded, "b d tp hp wp -> (b tp) d hp wp")
            encoded = self.stem(encoded)
            tokens = rearrange(encoded, "(b tp) d hp wp -> b tp (hp wp) d", b=b, tp=tp, hp=hp, wp=wp)
        else:
            if self.resnet_encoder is None:
                raise RuntimeError("resnet encoder is not initialized.")
            encoded = self.resnet_encoder(x)
            _, _, tp, hp, wp = encoded.shape
            tokens = rearrange(encoded, "b d tp hp wp -> b tp (hp wp) d")

        expected_hp = h // self.patch_size
        expected_wp = w // self.patch_size
        if tp != token_t or hp != expected_hp or wp != expected_wp:
            raise ValueError(
                "encoder output size mismatch with patch/frame scales: "
                f"got (tp={tp}, hp={hp}, wp={wp}), expected (tp={token_t}, hp={expected_hp}, wp={expected_wp})"
            )
        tokens = self.token_proj(tokens)

        spatial_pe = self._resize_spatial_pos_embed(hp=hp, wp=wp).to(tokens.dtype).to(tokens.device)
        temporal_pe = self.temporal_pos_embed[:, frame_offset : frame_offset + tp, :].to(tokens.dtype).to(tokens.device)
        tokens = tokens + spatial_pe[:, None, :, :] + temporal_pe[:, :, None, :]
        if self.use_time_embedding and frame_times is not None:
            if frame_times.ndim != 2:
                raise ValueError(f"frame_times must be [B,T], got shape={tuple(frame_times.shape)}")
            if int(frame_times.shape[0]) != b or int(frame_times.shape[1]) != t:
                raise ValueError(
                    f"frame_times shape mismatch with x: expected ({b}, {t}), got {tuple(frame_times.shape)}"
                )
            if self.time_embed_proj is None:
                raise RuntimeError("time embedding is enabled but time_embed_proj is not initialized.")
            grouped_time = rearrange(frame_times, "b (tp fp) -> b tp fp", fp=self.frame_patch_size)
            pooled_time = grouped_time.mean(dim=-1)
            time_phase = pooled_time.to(tokens.dtype) * (2.0 * torch.pi)
            time_features = torch.stack([torch.sin(time_phase), torch.cos(time_phase)], dim=-1)
            time_embed = self.time_embed_proj(time_features).unsqueeze(2)
            tokens = tokens + time_embed.to(tokens.dtype).to(tokens.device)

        temporal_positions = torch.arange(frame_offset, frame_offset + tp, device=x.device, dtype=torch.long)
        return tokens, hp, wp, h, w, temporal_positions, stats

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
        if x.ndim != 5:
            raise ValueError(f"Expected x to be 5D [B,C,T,H,W], got shape={tuple(x.shape)}")

        b, c, t, h, w = x.shape
        if c != self.in_channels:
            raise ValueError(f"in_channels mismatch: model expects {self.in_channels}, got {c}")
        if t > self.max_frames:
            raise ValueError(f"Input frames {t} exceed max_frames {self.max_frames}")
        if predict_frames <= 0 or predict_frames > t:
            raise ValueError(f"predict_frames should be in [1, {t}], got {predict_frames}")
        if t % self.frame_patch_size != 0:
            raise ValueError(f"Input T must be divisible by frame_patch_size={self.frame_patch_size}, got T={t}")
        if predict_frames % self.frame_patch_size != 0:
            raise ValueError(
                "predict_frames must be divisible by "
                f"frame_patch_size={self.frame_patch_size}, got predict_frames={predict_frames}"
            )

        tokens, hp, wp, _, _, temporal_positions, stats = self._encode_tokens(
            x=x, frame_offset=0, frame_times=frame_times, gate_source=gate_source, gate_steps=gate_steps
        )
        predict_token_frames = predict_frames // self.frame_patch_size
        total_token_frames = int(tokens.shape[1])
        if predict_token_frames <= 0 or predict_token_frames > total_token_frames:
            raise ValueError(f"predict token frames should be in [1, {total_token_frames}], got {predict_token_frames}")

        context_frames = total_token_frames - predict_token_frames
        context_frames_raw = t - predict_frames
        for block in self.blocks:
            tokens = block(
                tokens,
                context_frames=context_frames,
                strict_target_isolation=strict_target_isolation,
                hp=hp,
                wp=wp,
                temporal_positions=temporal_positions,
            )
        tokens = self.norm(tokens)

        tokens = tokens[:, -predict_token_frames:, :, :]
        context_anchor = self._last_context_anchor(x=x, context_frames=context_frames_raw)
        if self.cross_modal_adapter is not None:
            tokens = self.cross_modal_adapter(tokens, context_x=x[:, :, :context_frames_raw])
        if self.local_window_refiner is not None:
            tokens = self.local_window_refiner(tokens, height=hp, width=wp)
        film_condition = self._build_film_condition_sequence(
            x=x,
            context_frames=context_frames_raw,
            predict_frames=predict_frames,
        )
        result = self._decode_target_tokens(
            target_tokens=tokens,
            hp=hp,
            wp=wp,
            h=h,
            w=w,
            batch_size=b,
            predict_frames=predict_token_frames,
            context_anchor=context_anchor,
            film_condition=film_condition,
            return_modality_dict=return_modality_dict,
        )
        if return_gate_stats:
            if not return_modality_dict:
                raise ValueError("Gate statistics require modality dictionary output.")
            return result, stats
        return result

    def forward_ar(
        self,
        context_x: torch.Tensor | None,
        target_x: torch.Tensor,
        predict_frames: int | None = None,
        strict_target_isolation: bool = False,
        return_modality_dict: bool = True,
        context_modality_available: torch.Tensor | None = None,
        context_time: torch.Tensor | None = None,
        target_time: torch.Tensor | None = None,
        gate_source: torch.Tensor | None = None,
        gate_steps: torch.Tensor | None = None,
        return_gate_stats: bool = False,
    ) -> dict[str, torch.Tensor] | torch.Tensor | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if target_x.ndim != 5:
            raise ValueError(f"target_x must be [B,C,T,H,W], got {tuple(target_x.shape)}")

        b, c, tt, h, w = target_x.shape
        if tt <= 0:
            raise ValueError("target_x must contain at least 1 frame.")

        frame_times: torch.Tensor | None = None
        if context_x is None:
            x = target_x
            frame_times = target_time
        else:
            if context_x.ndim != 5:
                raise ValueError(f"context_x must be [B,C,T,H,W], got {tuple(context_x.shape)}")
            bc, cc, _tc, hc, wc = context_x.shape
            if bc != b or cc != c or hc != h or wc != w:
                raise ValueError(
                    "context_x and target_x shape mismatch: "
                    f"context={tuple(context_x.shape)}, target={tuple(target_x.shape)}"
                )
            masked_context_x = self._apply_context_modality_mask_token(
                context_x=context_x,
                context_modality_available=context_modality_available,
            )
            x = torch.cat([masked_context_x, target_x], dim=2)
            if context_time is not None or target_time is not None:
                if context_time is None or target_time is None:
                    raise ValueError("context_time and target_time should be both set or both None.")
                if context_time.ndim != 2 or target_time.ndim != 2:
                    raise ValueError(
                        "context_time and target_time must be [B,T], "
                        f"got context={tuple(context_time.shape)}, target={tuple(target_time.shape)}"
                    )
                if int(context_time.shape[0]) != b or int(target_time.shape[0]) != b:
                    raise ValueError(
                        "batch size mismatch between x and time tensors: "
                        f"x={b}, context_time={int(context_time.shape[0])}, target_time={int(target_time.shape[0])}"
                    )
                if int(context_time.shape[1]) != int(context_x.shape[2]) or int(target_time.shape[1]) != int(
                    target_x.shape[2]
                ):
                    raise ValueError(
                        "time length mismatch with frames: "
                        f"context_time={int(context_time.shape[1])}, context_frames={int(context_x.shape[2])}, "
                        f"target_time={int(target_time.shape[1])}, target_frames={int(target_x.shape[2])}"
                    )
                frame_times = torch.cat([context_time, target_time], dim=1)

        target_frames = tt if predict_frames is None else int(predict_frames)
        if target_frames <= 0 or target_frames > tt:
            raise ValueError(f"predict_frames should be in [1, {tt}], got {target_frames}")

        return self.forward(
            x=x,
            predict_frames=target_frames,
            strict_target_isolation=strict_target_isolation,
            return_modality_dict=return_modality_dict,
            frame_times=frame_times,
            gate_source=gate_source,
            gate_steps=gate_steps,
            return_gate_stats=return_gate_stats,
        )
