import math
from abc import abstractmethod
from typing import cast

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.networks.modules.unet_nn import (
    SiLU,
    avg_pool_nd,
    # checkpoint,
    conv_nd,
    linear,
    normalization,
    timestep_embedding,
    zero_module,
)


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, channels, channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, channels, channels, 3, stride=stride, padding=1)
        else:
            self.op = avg_pool_nd(stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # return checkpoint(
        #     self._forward, (x, emb), self.parameters(), self.use_checkpoint
        # )
        if self.use_checkpoint:
            return checkpoint(self._forward, x, emb, use_reentrant=False)
        else:
            return self._forward(x, emb)

    # @th.compile
    def _forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self, channels, num_heads=1, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint

        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention()
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        # return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)
        if self.use_checkpoint:
            return checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.attention(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """

    # @th.compile
    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x C x T] tensor after attention.
        """
        ch = qkv.shape[1] // 3
        q, k, v = th.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        return th.einsum("bts,bcs->bct", weight, v)

    @staticmethod
    def count_flops(model, _x, y):
        """
        A counter for the `thop` package to count the operations in an
        attention operation.

        Meant to be used like:

            macs, params = thop.profile(
                model,
                inputs=(inputs, timestamps),
                custom_ops={QKVAttention: QKVAttention.count_flops},
            )

        """
        b, c, *spatial = y[0].shape
        num_spatial = int(np.prod(spatial))
        # We perform two matmuls with the same number of ops.
        # The first computes the weight matrix, the second computes
        # the combination of the value vectors.
        matmul_ops = 2 * b * (num_spatial**2) * c
        model.total_ops += th.DoubleTensor([matmul_ops])


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    """

    def __init__(
        self,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        n_past=2,
        n_future=1,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample
        self.n_past = n_past
        self.n_future = n_future

        # future pred noise
        # self.pred_rain_learnable = nn.Parameter(
        #     th.randn(model_channels, n_future, 1, 1)
        # )

        time_embed_dim = model_channels * 2
        self.time_embed_dim = time_embed_dim
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        self.uni_time_embed = nn.Sequential(
            linear(time_embed_dim * (n_past + n_future), time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        # modality embedding
        self.modalities_block = nn.ModuleDict(
            {
                # todo: refering the wan2.2 time-causal conv layer
                "radar": conv_nd(dims, 1, model_channels, 3, padding=1),
                "satellite": conv_nd(dims, 10, model_channels, 3, padding=1),
                "rain": conv_nd(dims, 1, model_channels, 3, padding=1),
            }
        )
        self.fused_conv = conv_nd(
            dims, model_channels * n_past * 3, model_channels, 1, 1
        )

        self.encoder = nn.ModuleDict({})
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult, 1):
            layer_in = nn.Module()
            layer = []
            for i in range(num_res_blocks):
                res_block = ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    out_channels=mult * model_channels,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                )
                layer.append(res_block)
                ch = mult * model_channels
                if ds in attention_resolutions:
                    attn = AttentionBlock(
                        ch, use_checkpoint=use_checkpoint, num_heads=num_heads
                    )
                    layer.append(attn)
            layer_in.resblock = TimestepEmbedSequential(*layer)
            input_block_chans.append(ch)

            downsample = Downsample(ch, conv_resample, dims=dims)
            layer_in.downsample = downsample
            ds *= 2
            self.encoder[f"lvl_{level}"] = layer_in

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            # AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self.decoder = nn.ModuleDict({})
        for level, mult in list(enumerate(channel_mult, 1))[::-1]:
            layer_in = nn.Module()
            layer = []
            ich = input_block_chans.pop()
            for i in range(num_res_blocks + 1):
                res_block = ResBlock(
                    ch + ich if i == 0 else model_channels * mult,
                    time_embed_dim,
                    dropout,
                    out_channels=model_channels * mult,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                )
                layer.append(res_block)

                ch = model_channels * mult
                if ds in attention_resolutions:
                    layer.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                        )
                    )
            layer_in.resblock = TimestepEmbedSequential(*layer)

            upsample = Upsample(ch, conv_resample, dims=dims)
            layer_in.upsample = upsample
            ds //= 2
            self.decoder[f"lvl_{level}"] = layer_in

        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(
                conv_nd(dims, model_channels, out_channels * n_future, 3, padding=1)
            ),
        )

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.parameters()).dtype

    def modality_embedding(self, modality: th.Tensor, m_name: str):
        lst = modality.unbind(dim=-3)
        assert len(lst) == self.n_past, (
            f"Expected {self.n_past} past frames, got {len(lst)}"
        )

        new_lst = []
        for x in lst:
            patch_embedder = self.modalities_block[m_name]
            xp = patch_embedder(x)  # (bs, c, h, w)
            new_lst.append(xp)

        return th.cat(new_lst, dim=1)

    def unify_time_embd(self, past_embd, future_embd):
        # [bs, n_past, c]
        bs = past_embd.shape[0]
        past_t = past_embd.view(bs, -1)
        future_t = future_embd.view(bs, -1)
        uni_t = th.cat([past_t, future_t], dim=-1)
        return self.uni_time_embed(uni_t)

    def forward(self, radar, satellite, rain, timesteps=None, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (self.num_classes is not None), (
            "must specify y if and only if the model is class-conditional"
        )

        hs = []
        if timesteps:
            # timesteps: tuple, (bs, n_past), (bs, n_future)
            past_t, future_t = timesteps
            assert past_t.shape[-1] == self.n_past
            assert future_t.shape[-1] == self.n_future

            past_t = past_t.view(-1)
            emb_past = self.time_embed(timestep_embedding(past_t, self.model_channels))
            emb_past = emb_past.view(-1, self.n_past, self.time_embed_dim)

            future_t = future_t.view(-1)
            emb_future = self.time_embed(
                timestep_embedding(future_t, self.model_channels)
            )
            emb_future = emb_future.view(-1, self.n_future, self.time_embed_dim)

        else:
            emb_past = [
                th.zeros(radar.shape[0], self.n_past, self.time_embed_dim).to(radar)
            ] * self.n_past
            emb_future = [
                th.zeros(radar.shape[0], self.n_future, self.time_embed_dim).to(radar)
            ] * self.n_future

        emb = self.unify_time_embd(emb_past, emb_future)

        # if self.num_classes is not None and y:
        #     assert y.shape == (rain.shape[0],)
        #     emb = emb + self.label_emb(y)

        # modality embedding
        radar_embd = self.modality_embedding(radar, "radar")
        sat_embd = self.modality_embedding(satellite, "satellite")
        rain_embd = self.modality_embedding(rain, "rain")

        # cat all
        h = th.cat([radar_embd, sat_embd, rain_embd], dim=1)
        h = self.fused_conv(h)
        h = h.type(self.inner_dtype)

        for module in self.encoder.values():
            h = module.resblock(h, emb)
            if hasattr(module, "downsample"):
                h = module.downsample(h)
            hs.append(h)

        h = self.middle_block(h, emb)
        for module in self.decoder.values():
            h_enc = hs.pop()
            h = th.cat([h, h_enc], dim=1)
            h = module.resblock(h, emb)
            if hasattr(module, "upsample"):
                h = module.upsample(h)

        h = h.type(radar.dtype)
        y = self.out(h)
        y = y.reshape(radar.shape[0], self.out_channels, self.n_future, *y.shape[2:])
        return y


class SuperResModel(UNetModel):
    """
    A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(in_channels * 2, *args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)

    def get_feature_vectors(self, x, timesteps, low_res=None, **kwargs):
        _, new_height, new_width, _ = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().get_feature_vectors(x, timesteps, **kwargs)


# * --- test --- #


def test_model():
    th.cuda.set_device("cuda:0")
    dtype = th.bfloat16
    model = (
        UNetModel(
            model_channels=64,
            out_channels=1,
            num_res_blocks=2,
            attention_resolutions=[],
            channel_mult=(1, 2, 4),
            use_scale_shift_norm=True,
            use_checkpoint=False,
        )
        .cuda()
        .to(dtype)
    )

    bs = 16
    radar = th.randn(bs, 1, 2, 384, 384).cuda().to(dtype)
    satellite = th.randn(bs, 10, 2, 384, 384).cuda().to(dtype)
    rain = th.randn(bs, 1, 2, 384, 384).cuda().to(dtype)

    times = [th.ones(bs, 2).cuda().to(dtype), th.ones(bs, 1).cuda().to(dtype)]

    optimizer = th.optim.Adam(model.parameters())
    with th.autocast("cuda", dtype=dtype):
        y = model(radar, satellite, rain, timesteps=times)
        optimizer.zero_grad()
        y.mean().backward()
        optimizer.step()

    print("done.")

    import time

    time.sleep(10)


if __name__ == "__main__":
    test_model()
