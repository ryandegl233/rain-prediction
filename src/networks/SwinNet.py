import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.networks.modules.SwinTransformer import StageModule, StageModule_up, StageModule_up_final
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
from src.networks.modules.ttt_memory import LoGerStyleMemory

"""
SwinNet v4:
- keep AR rain_window rollout
- use external LoGeR-style global TTT memory module (apply then update)
- no direct dependency on official LoGeR code
"""


class SwinNet(nn.Module):
    def __init__(
        self,
        input_channel=12,
        hidden_dim=64,
        downscaling_factors=(4, 2, 1, 1),
        layers=(2, 2, 2, 2),
        heads=(4, 4, 4, 4),
        head_dim=64,
        window_size=8,
        relative_pos_embedding=True,
        input_resolution=(256, 256),
        num_classes=6,
        n_past=None,
        lstm_layers=1,
        output_frames=1,
    ):
        super().__init__()
        self.input_channel = input_channel
        self.hidden_dim = hidden_dim
        self.downscaling_factors = downscaling_factors
        self.layers = layers
        self.heads = heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.input_resolution = input_resolution
        self.num_classes = num_classes
        # keep config compatibility when hydra passes n_past
        self.n_past = n_past
        self.lstm_layers = lstm_layers
        self.output_frames = int(output_frames)
        if self.output_frames < 1:
            raise ValueError(f"output_frames must be >= 1, got {self.output_frames}")

        H, W = input_resolution
        s1_h = H // downscaling_factors[0]
        s2_h = s1_h // downscaling_factors[1]
        s3_h = s2_h // downscaling_factors[2]
        s4_h = s3_h // downscaling_factors[3]

        # Encoder
        self.stage1 = StageModule(
            in_channels=input_channel,
            hidden_dimension=hidden_dim,
            layers=layers[0],
            downscaling_factor=downscaling_factors[0],
            num_heads=heads[0],
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding,
            h_w=(s1_h, s1_h),
        )
        self.stage2 = StageModule(
            in_channels=hidden_dim,
            hidden_dimension=hidden_dim * 2,
            layers=layers[1],
            downscaling_factor=downscaling_factors[1],
            num_heads=heads[1],
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding,
            h_w=(s2_h, s2_h),
        )
        self.stage3 = StageModule(
            in_channels=hidden_dim * 2,
            hidden_dimension=hidden_dim * 4,
            layers=layers[2],
            downscaling_factor=downscaling_factors[2],
            num_heads=heads[2],
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding,
            h_w=(s3_h, s3_h),
        )
        self.stage4 = StageModule(
            in_channels=hidden_dim * 4,
            hidden_dimension=hidden_dim * 8,
            layers=layers[3],
            downscaling_factor=downscaling_factors[3],
            num_heads=heads[3],
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding,
            h_w=(s4_h, s4_h),
        )

        # Decoder
        self.stage5 = StageModule_up(
            in_channels=hidden_dim * 8,
            hidden_dimension=hidden_dim * 4,
            layers=layers[3],
            upscaling_factor=downscaling_factors[3],
            num_heads=heads[3],
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding,
            h_w=(s3_h, s3_h),
        )
        self.stage6 = StageModule_up(
            in_channels=hidden_dim * 8,
            hidden_dimension=hidden_dim * 2,
            layers=layers[2],
            upscaling_factor=downscaling_factors[2],
            num_heads=heads[2],
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding,
            h_w=(s2_h, s2_h),
        )
        self.stage7 = StageModule_up(
            in_channels=hidden_dim * 4,
            hidden_dimension=hidden_dim,
            layers=layers[1],
            upscaling_factor=downscaling_factors[1],
            num_heads=heads[1],
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding,
            h_w=(s1_h, s1_h),
        )
        self.stage8 = StageModule_up_final(
            in_channels=hidden_dim * 2,
            hidden_dimension=input_channel,
            layers=layers[0],
            upscaling_factor=downscaling_factors[0],
            num_heads=heads[0],
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding,
            h_w=(H, W),
        )

        self.class_head = nn.Conv2d(input_channel, num_classes, kernel_size=1)

        # whole rain window encoder (non-lazy)
        self.rain_window_in = nn.Conv3d(num_classes, hidden_dim, kernel_size=1)
        self.rain_window_encoder = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(hidden_dim * 2, hidden_dim * 8, kernel_size=3, padding=1),
        )

        # initial memory build + external LoGeR-style TTT memory module
        self.memory_init_proj = nn.Conv2d(hidden_dim * 8, hidden_dim * 8, kernel_size=1)
        memory_init_kwargs = {
            "channels": hidden_dim * 8,
            "apply_scale": 0.1,
            "update_momentum": 0.1,
        }
        # Backward-compatible with older LoGerStyleMemory implementations
        # that do not expose `pred_channels`.
        init_sig = inspect.signature(LoGerStyleMemory.__init__)
        if "pred_channels" in init_sig.parameters:
            memory_init_kwargs["pred_channels"] = num_classes
        self.ttt_memory = LoGerStyleMemory(**memory_init_kwargs)
        update_sig = inspect.signature(self.ttt_memory.update)
        # Bound method excludes `self` from parameters.
        self._update_accepts_pred = len(update_sig.parameters) >= 3

    def forward_encoder_single_frame(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        return x4, (x1, x2, x3)

    def _decode_single(self, bottleneck, skips):
        skip1, skip2, skip3 = skips
        x5 = self.stage5(bottleneck, skip3)
        x6 = self.stage6(x5, skip2)
        x7 = self.stage7(x6, skip1)
        x8 = self.stage8(x7)
        return self.class_head(x8)

    def _rain_to_prob(self, rain_cls_like):
        if rain_cls_like.shape[1] != 1:
            raise ValueError(
                f"rain tensor must have one class-index-like channel, got {rain_cls_like.shape[1]}"
            )
        rain_cls = rain_cls_like.squeeze(1).round().long().clamp_(0, self.num_classes - 1)
        rain_prob = F.one_hot(rain_cls, num_classes=self.num_classes).permute(0, 3, 1, 2)
        return rain_prob.float()

    def init_rain_window(self, rain):
        if rain.ndim != 5:
            raise ValueError(f"rain must be 5D [B,1,T,H,W], got shape {tuple(rain.shape)}")
        if rain.shape[1] != 1:
            raise ValueError(f"rain must have one channel, got {rain.shape[1]}")

        frames = [self._rain_to_prob(rain[:, :, t]) for t in range(rain.shape[2])]
        return torch.stack(frames, dim=2)

    def encode_rain_window(self, rain_window):
        if rain_window.ndim != 5:
            raise ValueError(
                f"rain_window must be 5D [B,C,T,H,W], got shape {tuple(rain_window.shape)}"
            )
        x = self.rain_window_in(rain_window)
        x = self.rain_window_encoder(x)
        return x.mean(dim=2)

    def update_rain_window(self, rain_window, next_rain_prob):
        if next_rain_prob.ndim != 4:
            raise ValueError(
                f"next_rain_prob must be 4D [B,C,H,W], got shape {tuple(next_rain_prob.shape)}"
            )
        return torch.cat([rain_window[:, :, 1:], next_rain_prob.unsqueeze(2)], dim=2)

    def build_memory(self, bottlenecks):
        z = torch.stack(bottlenecks, dim=1).mean(dim=1)
        return self.memory_init_proj(z)

    def forward(self, radar, sat, rain):
        # radar/satellite are only used at past stage to initialize global memory once
        x = torch.cat([radar, sat, rain], dim=1)
        _, _, T, _, _ = x.shape

        bottlenecks = []
        last_skips = None
        for t in range(T):
            frame_t = x[:, :, t]
            feat_t, skips_t = self.forward_encoder_single_frame(frame_t)
            bottlenecks.append(feat_t)
            if t == T - 1:
                last_skips = skips_t

        if last_skips is None:
            raise RuntimeError("No past frames were provided; last_skips is None.")

        initial_memory = self.build_memory(bottlenecks)
        memory_state = self.ttt_memory.init_memory(initial_memory)
        rain_window = self.init_rain_window(rain)

        outputs = []
        for _ in range(self.output_frames):
            rain_context_feat = self.encode_rain_window(rain_window)
            step_state = self.ttt_memory.apply(memory_state, rain_context_feat)
            logits_t = self._decode_single(step_state, last_skips)
            outputs.append(logits_t)

            next_rain_prob = torch.softmax(logits_t, dim=1)
            if self._update_accepts_pred:
                memory_state = self.ttt_memory.update(
                    memory_state,
                    rain_context_feat,
                    next_rain_prob,
                )
            else:
                memory_state = self.ttt_memory.update(
                    memory_state,
                    rain_context_feat,
                )
            rain_window = self.update_rain_window(rain_window, next_rain_prob)

        return torch.stack(outputs, dim=2)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def _n_params(module: nn.Module) -> int:
        return sum(p.numel() for p in module.parameters())
    def _fmt(n: int) -> str:
        return f"{n:,} ({n / 1e6:.2f}M)"

    for size in [256]:
        B, T = 2, 10
        model = SwinNet(
            input_channel=12,
            hidden_dim=64,
            downscaling_factors=(4, 2, 1, 1),
            layers=(4, 4, 4, 4),
            heads=(4, 4, 4, 4),
            head_dim=64,
            window_size=8,
            relative_pos_embedding=True,
            input_resolution=(size, size),
            num_classes=6,
            output_frames=5,
        ).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"Model params -> total: {total_params:,} ({total_params / 1e6:.2f}M), "
            f"trainable: {trainable_params:,} ({trainable_params / 1e6:.2f}M)"
        )
        encoder_params = _n_params(model.stage1) + _n_params(model.stage2) + _n_params(model.stage3) + _n_params(model.stage4)
        decoder_params = _n_params(model.stage5) + _n_params(model.stage6) + _n_params(model.stage7) + _n_params(model.stage8)
        print("[Param Breakdown]")
        print(f"  encoder(stage1-4): {_fmt(encoder_params)}")
        print(f"  decoder(stage5-8): {_fmt(decoder_params)}")
        print(f"  class_head: {_fmt(_n_params(model.class_head))}")
        print(f"  rain_window_in: {_fmt(_n_params(model.rain_window_in))}")
        print(f"  rain_window_encoder: {_fmt(_n_params(model.rain_window_encoder))}")
        print(f"  memory_init_proj: {_fmt(_n_params(model.memory_init_proj))}")
        print(f"  ttt_memory: {_fmt(_n_params(model.ttt_memory))}")

        radar = torch.randn(B, 1, T, size, size, device=device)
        sat = torch.randn(B, 10, T, size, size, device=device)
        rain = torch.randn(B, 1, T, size, size, device=device)

        y = model(radar, sat, rain)
        print(f"Input size {size} -> Output: {y.shape}")
