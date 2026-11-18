import accelerate
import torch
import torch.nn as nn
from beartype import beartype
from jaxtyping import Float
from torch import Tensor

from src.networks.modules.wan22_blocks import (
    Decoder3d,
    Encoder3d,
    count_conv3d,
    patchify,
    unpatchify,
)

ModalityTensor = Float[Tensor, "b c t h w"]
TimestepTensor = Float[Tensor, "b t"]


class TimeCausalUnet(nn.Module):
    def __init__(
        self,
        in_chan: int,
        out_chan: int,
        dim=128,
        dec_dim=128,
        z_dim=16,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False, False],
        dropout=0.0,
        patch_size: int = 2,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]
        self.patch_size = patch_size
        
        # modules
        self.encoder = Encoder3d(
            in_chan * patch_size**2,
            dim,
            z_dim,
            dim_mult,
            num_res_blocks,
            attn_scales,
            self.temperal_downsample,
            dropout,
            collect_features=True,
        )
        self.decoder = Decoder3d(
            out_chan * patch_size**2,
            dec_dim,
            z_dim,
            dim_mult,
            num_res_blocks,
            attn_scales,
            self.temperal_upsample,
            dropout,
            get_enc_features=True,
        )
        # self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        # self.conv2 = CausalConv3d(z_dim, z_dim, 1)

    @beartype
    def forward(
        self,
        radar: ModalityTensor,
        satellite: ModalityTensor,
        rain: ModalityTensor,
        timesteps: TimestepTensor | None = None,
        *,
        scale: list[Tensor] | None = None,
    ):
        nframes = radar.shape[2]
        assert (nframes - 1) % 4 == 0

        # Concat along the channel dimension
        x = torch.cat([radar, satellite, rain], dim=1)
        z, feat1, feat2 = self.encoder_forward(x, scale)
        y = self.decoder_forward(z, feat1, feat2, scale)

        return y

    def encoder_forward(self, x, scale=None):
        self.clear_cache()
        x = patchify(x, patch_size=self.patch_size)
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out, feat1 = self.encoder(
                    x[:, :, :1, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
                pass
            else:
                out_, feat2 = self.encoder(
                    x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
                out = torch.cat([out, out_], 2)
        # mu, log_var = self.conv1(out).chunk(2, dim=1)
        mu = out
        if scale is not None:
            if isinstance(scale[0], torch.Tensor):
                mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                    1, self.z_dim, 1, 1, 1
                )
            else:
                mu = (mu - scale[0]) * scale[1]
        self.clear_cache()
        return mu, feat1, feat2

    def decoder_forward(self, z, feat1, feat2, scale=None):
        self.clear_cache()
        if scale is not None:
            if isinstance(scale[0], torch.Tensor):
                z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                    1, self.z_dim, 1, 1, 1
                )
            else:
                z = z / scale[1] + scale[0]
        iter_ = z.shape[2]
        # x = self.conv2(z)
        x = z
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(
                    x[:, :, i : i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                    first_chunk=True,
                    enc_features=feat1,
                )
            else:
                out_ = self.decoder(
                    x[:, :, i : i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                    enc_features=feat2,
                )
                out = torch.cat([out, out_], 2)
        out = unpatchify(out, patch_size=self.patch_size)
        self.clear_cache()
        return out

    def clear_cache(self):
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        # cache encode
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num


# * --- tests --- #


@beartype
def test_typing(a: ModalityTensor, b: TimestepTensor):
    """
    Test function to check typing.
    """
    assert isinstance(a, Tensor)
    assert isinstance(b, Tensor)
    return a, b


def test_time_unet():
    from fvcore.nn import parameter_count_table

    unet = TimeCausalUnet(
        9,
        3,
        dim_mult=[1, 2, 2],
        temperal_downsample=[True, True, False],
        z_dim=128,
        dim=128,
        dec_dim=128,
        patch_size=2,
    ).cuda()
    modality_shape = (4, 3, 9, 256, 256)
    radar = torch.randn(*modality_shape).cuda()
    satellite = torch.randn(*modality_shape).cuda()
    rain = torch.randn(*modality_shape).cuda()
    print(parameter_count_table(unet))

    optimizer = torch.optim.Adam(unet.parameters())
    y = unet(radar, satellite, rain)
    print(y.shape)
    y.mean().backward()
    optimizer.zero_grad()


if __name__ == "__main__":
    # a = torch.randn(2, 3, 64, 64)  # Example tensor
    # b = torch.randn(2, 10)  # Example tensor
    # test_typing(a, b)

    ## Unet test
    test_time_unet()
