from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .openstl_port.convlstm_cell import ConvLSTMCell
from .openstl_port.predrnnv2_cell import SpatioTemporalLSTMCellv2
from .openstl_port.simvp_incepu import Decoder, Encoder, MidIncepNet


DEFAULT_OPENSTL_OPTIONS: dict[str, dict[str, object]] = {
    "convlstm": {"num_hidden": [210, 210, 210, 210], "patch_size": 4, "filter_size": 5, "layer_norm": False},
    "predrnn_v2": {"num_hidden": [164, 164, 164, 164], "patch_size": 4, "filter_size": 5, "layer_norm": False},
    "simvp": {"hid_S": 64, "hid_T": 448, "N_S": 4, "N_T": 8},
}


def _patchify(history: Tensor, patch_size: int) -> Tensor:
    batch, channels, frames, height, width = history.shape
    return (
        history.permute(0, 2, 3, 4, 1)
        .reshape(batch, frames, height // patch_size, patch_size, width // patch_size, patch_size, channels)
        .permute(0, 1, 2, 4, 3, 5, 6)
        .reshape(batch, frames, height // patch_size, width // patch_size, patch_size**2 * channels)
        .permute(0, 1, 4, 2, 3)
        .contiguous()
    )


def _unpatchify(frame: Tensor, patch_size: int, channels: int) -> Tensor:
    batch, _, height, width = frame.shape
    return (
        frame.permute(0, 2, 3, 1)
        .reshape(batch, height, width, patch_size, patch_size, channels)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(batch, height * patch_size, width * patch_size, channels)
        .permute(0, 3, 1, 2)
        .unsqueeze(2)
        .contiguous()
    )


class RecurrentOpenSTL(nn.Module):
    def __init__(self, method: str, options: Mapping[str, object]) -> None:
        super().__init__()
        self.method = method
        self.in_channels = int(options.get("in_channels", 12))
        self.context_frames = int(options.get("context_frames", 7))
        self.image_size = int(options.get("image_size", 448))
        self.patch_size = int(options["patch_size"])
        self.num_hidden = list(options["num_hidden"])
        self.num_layers = len(self.num_hidden)
        filter_size = int(options["filter_size"])
        layer_norm = bool(options["layer_norm"])
        if self.patch_size < 1 or self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by a positive patch_size")
        if not self.num_hidden or any(width < 1 for width in self.num_hidden):
            raise ValueError("num_hidden must contain positive layer widths")
        if method == "predrnn_v2" and len(set(self.num_hidden)) != 1:
            raise ValueError("PredRNN-V2 shared memory and adapter require equal layer widths")
        if filter_size < 1 or filter_size % 2 != 1:
            raise ValueError("filter_size must be a positive odd integer")
        if self.context_frames < 1 or self.in_channels < 1:
            raise ValueError("context_frames and in_channels must be positive")

        self.frame_channel = self.patch_size**2 * self.in_channels
        cell_type = ConvLSTMCell if method == "convlstm" else SpatioTemporalLSTMCellv2
        spatial_size = self.image_size // self.patch_size
        self.cell_list = nn.ModuleList(
            cell_type(
                self.frame_channel if index == 0 else self.num_hidden[index - 1],
                width,
                spatial_size,
                spatial_size,
                filter_size,
                1,
                layer_norm,
            )
            for index, width in enumerate(self.num_hidden)
        )
        self.conv_last = nn.Conv2d(self.num_hidden[-1], self.frame_channel, 1, bias=False)
        if method == "predrnn_v2":
            self.adapter = nn.Conv2d(self.num_hidden[0], self.num_hidden[0], 1, bias=False)

    def forward(self, history: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        expected = (self.in_channels, self.context_frames, self.image_size, self.image_size)
        if history.ndim != 5 or tuple(history.shape[1:]) != expected:
            raise ValueError(f"Expected [B, {expected}], received {tuple(history.shape)}")
        frames = _patchify(history, self.patch_size)
        batch, _, _, height, width = frames.shape
        hidden = [frames.new_zeros(batch, channels, height, width) for channels in self.num_hidden]
        cells = [torch.zeros_like(state) for state in hidden]
        memory = torch.zeros_like(hidden[0])
        decoupling: list[Tensor] = []

        for frame in frames.unbind(dim=1):
            latent = frame
            for layer, cell in enumerate(self.cell_list):
                if self.method == "convlstm":
                    hidden[layer], cells[layer] = cell(latent, hidden[layer], cells[layer])
                else:
                    hidden[layer], cells[layer], memory, delta_c, delta_m = cell(
                        latent, hidden[layer], cells[layer], memory
                    )
                    delta_c = F.normalize(self.adapter(delta_c).flatten(2), dim=2)
                    delta_m = F.normalize(self.adapter(delta_m).flatten(2), dim=2)
                    decoupling.append(torch.cosine_similarity(delta_c, delta_m, dim=2).abs().mean())
                latent = hidden[layer]

        prediction = _unpatchify(self.conv_last(hidden[-1]), self.patch_size, self.in_channels)
        auxiliary = {"decoupling": torch.stack(decoupling).mean()} if decoupling else {}
        return prediction, auxiliary


class SimVPIncepU(nn.Module):
    def __init__(self, options: Mapping[str, object]) -> None:
        super().__init__()
        self.in_channels = int(options.get("in_channels", 12))
        self.context_frames = int(options.get("context_frames", 7))
        self.image_size = int(options.get("image_size", 448))
        spatial_width = int(options["hid_S"])
        temporal_width = int(options["hid_T"])
        spatial_layers = int(options["N_S"])
        temporal_layers = int(options["N_T"])
        kernel_enc = int(options.get("spatio_kernel_enc", 3))
        kernel_dec = int(options.get("spatio_kernel_dec", 3))
        if spatial_layers < 2 or spatial_layers % 2:
            raise ValueError("N_S must be a positive even integer of at least two")
        if self.image_size % (2 ** (spatial_layers // 2)):
            raise ValueError("image_size must be divisible by the SimVP spatial downsampling factor")
        if spatial_width < 2 or spatial_width % 2:
            raise ValueError("hid_S must be a positive even integer")
        if temporal_width < 16 or temporal_width % 16:
            raise ValueError("hid_T must be a positive multiple of 16 for the official IncepU groups")
        if self.context_frames * spatial_width % 8:
            raise ValueError("context_frames * hid_S must be divisible by the official IncepU group count eight")
        if temporal_layers < 2 or self.context_frames < 1 or self.in_channels < 1:
            raise ValueError("N_T must be at least two; context_frames and in_channels must be positive")
        if kernel_enc < 1 or kernel_enc % 2 != 1 or kernel_dec < 1 or kernel_dec % 2 != 1:
            raise ValueError("Spatial kernel sizes must be positive odd integers")
        if str(options.get("model_type", "IncepU")).lower() != "incepu":
            raise ValueError("This adapter implements the original SimVP IncepU architecture")

        self.enc = Encoder(self.in_channels, spatial_width, spatial_layers, kernel_enc, act_inplace=False)
        self.dec = Decoder(spatial_width, self.in_channels, spatial_layers, kernel_dec, act_inplace=False)
        self.hid = MidIncepNet(self.context_frames * spatial_width, temporal_width, temporal_layers)

    def forward(self, history: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        expected = (self.in_channels, self.context_frames, self.image_size, self.image_size)
        if history.ndim != 5 or tuple(history.shape[1:]) != expected:
            raise ValueError(f"Expected [B, {expected}], received {tuple(history.shape)}")
        batch, channels, frames, height, width = history.shape
        inputs = history.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        embedding, skip = self.enc(inputs)
        _, spatial_channels, spatial_height, spatial_width = embedding.shape
        embedding = embedding.reshape(batch, frames, spatial_channels, spatial_height, spatial_width)
        translated = self.hid(embedding).reshape(batch * frames, spatial_channels, spatial_height, spatial_width)
        prediction = self.dec(translated, skip).reshape(batch, frames, channels, height, width)
        return prediction[:, :1].permute(0, 2, 1, 3, 4).contiguous(), {}


def build_openstl(method: str, options: dict[str, object]) -> nn.Module:
    if method not in DEFAULT_OPENSTL_OPTIONS:
        raise ValueError(f"Unsupported OpenSTL method: {method}")
    resolved = DEFAULT_OPENSTL_OPTIONS[method] | options
    return SimVPIncepU(resolved) if method == "simvp" else RecurrentOpenSTL(method, resolved)
