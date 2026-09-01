import torch

from src.residual_diffusion_stage.losses import (
    ChannelMoments,
    diffusion_residual_loss,
    residual_autoencoder_loss,
)
from src.residual_diffusion_stage.models import ConditionalDiffusion, LatentNormalizer, ResidualAutoencoder


def _diffusion_config() -> dict:
    return {
        "latent_channels": 4,
        "history_channels": 12,
        "history_patch_size": 8,
        "condition_dim": 16,
        "condition_heads": 4,
        "condition_tokens": 4,
        "base_channels": 8,
        "channel_mult": [1, 2, 4, 4],
        "train_steps": 20,
        "sample_steps": 3,
        "gradient_checkpointing": False,
    }


def test_residual_autoencoder_preserves_shape_and_has_spatial_latent() -> None:
    model = ResidualAutoencoder(latent_channels=4)
    residual = torch.randn(1, 1, 2, 32, 32)
    reconstruction, latent = model(residual)
    assert reconstruction.shape == residual.shape
    assert latent.shape == (1, 4, 2, 8, 8)


def test_latent_normalizer_round_trip() -> None:
    normalizer = LatentNormalizer(channels=2)
    mean = torch.tensor([0.25, -0.5])
    std = torch.tensor([0.1, 2.0])
    normalizer.set_stats(mean, std)
    latent = torch.randn(2, 2, 3, 4, 4)
    restored = normalizer.denormalize(normalizer.normalize(latent))
    torch.testing.assert_close(restored, latent)


def test_channel_moments_match_direct_statistics() -> None:
    first = torch.randn(2, 3, 2, 4, 4)
    second = torch.randn(1, 3, 2, 4, 4)
    moments = ChannelMoments(channels=3)
    moments.update(first)
    moments.update(second)
    mean, std = moments.compute()
    combined = torch.cat([first, second], dim=0)
    expected_mean = combined.mean(dim=(0, 2, 3, 4))
    expected_std = combined.std(dim=(0, 2, 3, 4), unbiased=False)
    torch.testing.assert_close(mean, expected_mean)
    torch.testing.assert_close(std, expected_std)


def test_autoencoder_loss_penalizes_active_residual_and_bias() -> None:
    target = torch.zeros(1, 1, 1, 8, 8)
    target[..., 2:4, 2:4] = 0.5
    reconstruction = torch.zeros_like(target)
    loss, terms = residual_autoencoder_loss(
        reconstruction,
        target,
        {
            "global_l1_weight": 1.0,
            "active_l1_weight": 2.0,
            "bias_weight": 1.0,
            "std_weight": 0.5,
            "active_threshold": 0.01,
        },
    )
    assert loss.item() > terms["global_l1"].item()
    assert terms["active_l1"].item() == 0.5
    assert terms["bias"].item() > 0


def test_diffusion_training_and_sampling_shapes() -> None:
    model = ConditionalDiffusion(_diffusion_config())
    latent = torch.randn(1, 4, 2, 8, 8)
    coarse = torch.randn(1, 1, 2, 32, 32)
    history = torch.randn(1, 12, 3, 32, 32)
    loss = model.training_loss(latent, coarse, history)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    outputs = model(latent, coarse, history, return_outputs=True)
    assert outputs["predicted_x0"].shape == latent.shape
    assert outputs["alpha_bar"].shape == (1, 1, 1, 1, 1)
    assert torch.isfinite(outputs["predicted_x0"]).all()
    sampled = model.ddim_sample(tuple(latent.shape), coarse, history)
    assert sampled.shape == latent.shape
    assert torch.isfinite(sampled).all()


def test_diffusion_residual_loss_rewards_correct_amplitude_and_direction() -> None:
    target = torch.randn(2, 1, 2, 8, 8)
    noise_loss = torch.tensor(0.2)
    alpha_bar = torch.tensor([0.81, 0.25]).view(2, 1, 1, 1, 1)
    cfg = {
        "noise_weight": 1.0,
        "rainfall_l1_weight": 1.0,
        "active_l1_weight": 1.0,
        "bias_weight": 0.5,
        "std_weight": 0.25,
        "direction_weight": 0.05,
        "active_threshold": 0.01,
    }
    correct_loss, correct_terms = diffusion_residual_loss(noise_loss, target, target, alpha_bar, cfg)
    wrong_loss, wrong_terms = diffusion_residual_loss(noise_loss, -target + 0.1, target, alpha_bar, cfg)
    torch.testing.assert_close(correct_loss, noise_loss)
    torch.testing.assert_close(correct_terms["correlation"], torch.tensor(1.0))
    assert wrong_loss > correct_loss
    assert wrong_terms["bias"] > 0
    assert wrong_terms["correlation"] < 0
