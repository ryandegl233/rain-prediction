import torch

from src.networks.modules.reconstruction import SpatialPadConv3d


def test_spatial_pad_conv3d_causal_blocks_future_leakage() -> None:
    conv = SpatialPadConv3d(
        in_channels=1,
        out_channels=1,
        kernel_size=(3, 1, 1),
        padding_mode="zeros",
        causal=True,
    )
    with torch.no_grad():
        conv.conv.weight.fill_(1.0)
        conv.conv.bias.zero_()

    x_base = torch.zeros(1, 1, 5, 1, 1)
    x_future_changed = x_base.clone()
    x_future_changed[:, :, 4, :, :] = 1.0

    y_base = conv(x_base)
    y_future_changed = conv(x_future_changed)

    assert torch.allclose(y_base[:, :, :4], y_future_changed[:, :, :4])


def test_spatial_pad_conv3d_non_causal_uses_future_context() -> None:
    conv = SpatialPadConv3d(
        in_channels=1,
        out_channels=1,
        kernel_size=(3, 1, 1),
        padding_mode="zeros",
        causal=False,
    )
    with torch.no_grad():
        conv.conv.weight.fill_(1.0)
        conv.conv.bias.zero_()

    x_base = torch.zeros(1, 1, 5, 1, 1)
    x_future_changed = x_base.clone()
    x_future_changed[:, :, 4, :, :] = 1.0

    y_base = conv(x_base)
    y_future_changed = conv(x_future_changed)

    assert torch.any(torch.ne(y_base[:, :, :4], y_future_changed[:, :, :4]))
