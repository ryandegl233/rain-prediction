import pytest
import torch

from src.networks.GAN.patchgan3d import PatchGAN3D


def _build_model(**kwargs) -> PatchGAN3D:
    config = {
        "context_channels": 12,
        "target_channels": 12,
        "base_channels": 16,
        "channel_multipliers": (1, 2, 4),
        "norm_layer": "groupnorm",
        "norm_max_groups": 32,
        "act_negative_slope": 0.2,
    }
    config.update(kwargs)
    return PatchGAN3D(**config)


def test_patchgan3d_forward_output_shape() -> None:
    model = _build_model()
    context = torch.randn(2, 12, 4, 64, 64)
    target = torch.randn(2, 12, 4, 64, 64)
    logits = model(context=context, target=target)

    assert logits.ndim == 5
    assert logits.shape[0] == 2
    assert logits.shape[1] == 1
    assert logits.shape[2] > 0 and logits.shape[2] <= context.shape[2]
    assert logits.shape[3] > 0 and logits.shape[3] <= context.shape[3]
    assert logits.shape[4] > 0 and logits.shape[4] <= context.shape[4]


def test_patchgan3d_raises_for_context_target_shape_mismatch() -> None:
    model = _build_model()
    context = torch.randn(2, 12, 4, 64, 64)
    target = torch.randn(2, 12, 5, 64, 64)

    with pytest.raises(ValueError, match="shape mismatch"):
        _ = model(context=context, target=target)


def test_patchgan3d_groupnorm_dynamic_groups_with_non_divisible_32_channels() -> None:
    model = _build_model(base_channels=18, channel_multipliers=(1, 3))
    context = torch.randn(2, 12, 4, 64, 64)
    target = torch.randn(2, 12, 4, 64, 64)

    logits = model(context=context, target=target)
    assert logits.shape[0] == 2
    assert logits.shape[1] == 1


def test_patchgan3d_backward_runs_and_has_gradients() -> None:
    model = _build_model()
    context = torch.randn(2, 12, 4, 64, 64)
    target = torch.randn(2, 12, 4, 64, 64)

    logits = model(context=context, target=target)
    loss = logits.mean()
    loss.backward()

    has_non_none_grad = any(param.grad is not None for param in model.parameters() if param.requires_grad)
    assert has_non_none_grad
