import pytest
import torch

from src.networks.losses.gan import (
    gan_critic_total_loss,
    gan_discriminator_loss,
    gan_generator_loss,
    r1_regularization,
    r2_regularization,
)


@pytest.mark.parametrize("loss_type", ["ns", "rel_ns", "hinge"])
def test_generator_and_discriminator_losses_run_for_all_types(loss_type: str) -> None:
    real_logits = torch.randn(2, 1, 4, 8, 8)
    fake_logits = torch.randn(2, 1, 4, 8, 8)

    g_loss, g_logs = gan_generator_loss(fake_logits=fake_logits, real_logits=real_logits, loss_type=loss_type)
    d_loss, d_logs = gan_discriminator_loss(real_logits=real_logits, fake_logits=fake_logits, loss_type=loss_type)

    assert g_loss.ndim == 0
    assert d_loss.ndim == 0
    assert "gan/g_loss" in g_logs
    assert "gan/d_loss" in d_logs


def test_patch_logits_shape_is_supported_without_reshape() -> None:
    real_logits = torch.randn(3, 1, 2, 11, 13)
    fake_logits = torch.randn(3, 1, 2, 11, 13)

    g_loss, _ = gan_generator_loss(fake_logits=fake_logits, loss_type="ns")
    d_loss, _ = gan_discriminator_loss(real_logits=real_logits, fake_logits=fake_logits, loss_type="ns")

    assert torch.isfinite(g_loss)
    assert torch.isfinite(d_loss)


def test_rel_ns_requires_real_logits_in_generator_loss() -> None:
    fake_logits = torch.randn(2, 1, 4, 8, 8)

    with pytest.raises(ValueError, match="requires real_logits"):
        _ = gan_generator_loss(fake_logits=fake_logits, loss_type="rel_ns")


def test_r1_r2_regularization_backward_and_non_negative() -> None:
    real_input = torch.randn(2, 12, 3, 8, 8, requires_grad=True)
    fake_input = torch.randn(2, 12, 3, 8, 8, requires_grad=True)

    real_logits = real_input.pow(2).mean(dim=1, keepdim=True)
    fake_logits = fake_input.pow(2).mean(dim=1, keepdim=True)

    r1_loss, r1_logs = r1_regularization(real_logits=real_logits, real_input=real_input, weight=5.0)
    r2_loss, r2_logs = r2_regularization(fake_logits=fake_logits, fake_input=fake_input, weight=3.0)

    total = r1_loss + r2_loss + real_logits.mean() + fake_logits.mean()
    total.backward()

    assert float(r1_loss.item()) >= 0.0
    assert float(r2_loss.item()) >= 0.0
    assert real_input.grad is not None
    assert fake_input.grad is not None
    assert "gan/r1" in r1_logs
    assert "gan/r2" in r2_logs


def test_r1_r2_weight_zero_returns_zero_without_grad_requirements() -> None:
    real_input = torch.randn(2, 12, 3, 8, 8)
    fake_input = torch.randn(2, 12, 3, 8, 8)
    real_logits = torch.randn(2, 1, 3, 8, 8)
    fake_logits = torch.randn(2, 1, 3, 8, 8)

    r1_loss, _ = r1_regularization(real_logits=real_logits, real_input=real_input, weight=0.0)
    r2_loss, _ = r2_regularization(fake_logits=fake_logits, fake_input=fake_input, weight=0.0)

    assert float(r1_loss.item()) == 0.0
    assert float(r2_loss.item()) == 0.0


@pytest.mark.parametrize("bad_loss_type", ["", "abc", "wgan"])
def test_invalid_loss_type_raises_value_error(bad_loss_type: str) -> None:
    real_logits = torch.randn(2, 1, 4, 8, 8)
    fake_logits = torch.randn(2, 1, 4, 8, 8)

    with pytest.raises(ValueError, match="Unsupported GAN loss_type"):
        _ = gan_generator_loss(fake_logits=fake_logits, real_logits=real_logits, loss_type=bad_loss_type)
    with pytest.raises(ValueError, match="Unsupported GAN loss_type"):
        _ = gan_discriminator_loss(real_logits=real_logits, fake_logits=fake_logits, loss_type=bad_loss_type)


def test_critic_total_loss_aggregates_components() -> None:
    real_input = torch.randn(2, 12, 3, 8, 8, requires_grad=True)
    fake_input = torch.randn(2, 12, 3, 8, 8, requires_grad=True)
    real_logits = real_input.pow(2).mean(dim=1, keepdim=True)
    fake_logits = fake_input.pow(2).mean(dim=1, keepdim=True)

    total_loss, logs = gan_critic_total_loss(
        real_logits=real_logits,
        fake_logits=fake_logits,
        loss_type="ns",
        d_weight=1.0,
        real_input=real_input,
        fake_input=fake_input,
        r1_weight=1.0,
        r2_weight=1.0,
    )
    total_loss.backward()

    assert total_loss.ndim == 0
    assert "gan/critic_total_loss" in logs
    assert "gan/r1" in logs
    assert "gan/r2" in logs
