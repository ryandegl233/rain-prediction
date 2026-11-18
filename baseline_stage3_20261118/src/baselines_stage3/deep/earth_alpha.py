import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .earth_alpha_port.alphapre.models.alphapre import AlphaPre
from .earth_alpha_port.earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel


class EarthformerAdapter(nn.Module):
    def __init__(self, options: dict) -> None:
        super().__init__()
        size = int(options.get("image_size", 448))
        channels = int(options.get("in_channels", 12))
        context = int(options.get("context_frames", 7))
        width = int(options.get("base_units", 288))
        depths = list(options.get("depths", [1, 1]))
        stem_dims = list(options.get("stem_dims", [max(4, width // 8), width // 2, width]))
        self.model_options = dict(
            input_shape=(context, size, size, channels),
            target_shape=(1, size, size, channels),
            base_units=width,
            enc_depth=depths,
            dec_depth=depths,
            num_heads=int(options.get("num_heads", 4)),
            enc_attn_patterns="axial",
            dec_self_attn_patterns="axial",
            dec_cross_attn_patterns="cross_1x1",
            num_global_vectors=int(options.get("num_global_vectors", 8)),
            use_dec_self_global=False,
            use_dec_cross_global=False,
            use_global_vector_ffn=False,
            use_global_self_attn=True,
            separate_global_qkv=True,
            dec_use_first_self_attn=False,
            initial_downsample_type="stack_conv",
            initial_downsample_stack_conv_num_layers=len(stem_dims),
            initial_downsample_stack_conv_dim_list=stem_dims,
            initial_downsample_stack_conv_downscale_list=list(options.get("stem_scales", [4, 2, 2])),
            initial_downsample_stack_conv_num_conv_list=[2] * len(stem_dims),
            ffn_activation="gelu",
            padding_type="zeros",
            z_init_method="zeros",
            attn_drop=float(options.get("dropout", 0.1)),
            proj_drop=float(options.get("dropout", 0.1)),
            ffn_drop=float(options.get("dropout", 0.1)),
            checkpoint_level=int(options.get("checkpoint_level", 0)),
        )
        self.network = CuboidTransformerModel(**self.model_options)

    def forward(self, history: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        prediction = self.network(history.permute(0, 2, 3, 4, 1).contiguous())
        return prediction.permute(0, 4, 1, 2, 3).contiguous(), {}


class AlphaPreAdapter(nn.Module):
    def __init__(self, options: dict) -> None:
        super().__init__()
        size = int(options.get("image_size", 448))
        self.rain_linear_amplitude = bool(options.get("rain_linear_amplitude", False))
        self.model_options = dict(
            pre_seq_length=int(options.get("context_frames", 7)),
            aft_seq_length=1,
            input_shape=(size, size),
            input_dim=int(options.get("in_channels", 12)),
            hidden_dim=int(options.get("hidden_dim", 496)),
            n_layers=3,
            spec_num=int(options.get("spec_num", min(20, size // 2))),
            pha_weight=float(options.get("pha_weight", 0.01)),
            anet_weight=float(options.get("anet_weight", 0.1)),
            amp_weight=float(options.get("amp_weight", 0.01)),
            aweight_stop_steps=int(options.get("aweight_stop_steps", 10000)),
        )
        self.network = AlphaPre(**self.model_options)

    def forward(self, history: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        with torch.autocast(device_type=history.device.type, enabled=False):
            frames = history.float().transpose(1, 2)
            if self.rain_linear_amplitude:
                if frames.shape[2] != 12:
                    raise ValueError("Linear rain amplitude assumes radar/satellite/rain channels 1/10/1.")
                raw_amplitude = self.network.amplinet(frames)
                amplitude = torch.cat([raw_amplitude[:, :, :11].sigmoid(), raw_amplitude[:, :, 11:12]], dim=2)
                phase_pixels, phase, _ = self.network.phasenet(frames)
                prediction = self.network.alphamixer(amplitude, phase_pixels, phase)
            else:
                prediction, _, amplitude, phase, _ = self.network(frames, None)
        auxiliary = {"amplitude_prediction": amplitude, "phase_prediction": phase}
        return prediction.transpose(1, 2).contiguous(), auxiliary

    def supervised_losses(self, auxiliary: dict[str, Tensor], target: Tensor, step: int) -> dict[str, Tensor]:
        with torch.autocast(device_type=target.device.type, enabled=False):
            target = target.float().transpose(1, 2)
            spectrum = torch.fft.rfft2(target)
            mask = self.network.spec_mask
            phase = auxiliary["phase_prediction"]
            amplitude = auxiliary["amplitude_prediction"]
            phase_error = 1 - torch.cos(torch.angle(spectrum) * mask - phase * mask)
            phase_loss = phase_error.sum() / (mask.sum() * target.shape[0] * target.shape[1] * target.shape[2])
            amplitude_loss = F.mse_loss(torch.fft.rfft2(amplitude).abs(), spectrum.abs())
            branch_loss = F.mse_loss(amplitude, target)
            amplitude_weight = self.network.amp_weight * max(0.0, 1 - (step + 1) / self.network.aweight_stop_steps)
        return {
            "phase_loss": self.network.pha_weight * phase_loss,
            "amplitude_loss": amplitude_weight * amplitude_loss,
            "anet_loss": self.network.anet_weight * branch_loss,
        }


def build_earth_alpha(method: str, options: dict) -> nn.Module:
    if method == "earthformer":
        return EarthformerAdapter(options)
    if method == "alphapre":
        return AlphaPreAdapter(options)
    raise ValueError(f"Unsupported Earthformer/AlphaPre method: {method}")
