import torch
from torch.nn import functional as F


def residual_autoencoder_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    cfg: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    global_l1 = F.l1_loss(reconstruction, target)
    active_threshold = float(cfg.get("active_threshold", 0.01))
    active_mask = target.abs() >= active_threshold
    if active_mask.any():
        active_l1 = (reconstruction - target).abs()[active_mask].mean()
    else:
        active_l1 = torch.zeros((), device=target.device, dtype=target.dtype)
    bias = (reconstruction.mean() - target.mean()).abs()
    std = (reconstruction.std(unbiased=False) - target.std(unbiased=False)).abs()
    loss = (
        float(cfg.get("global_l1_weight", 1.0)) * global_l1
        + float(cfg.get("active_l1_weight", 2.0)) * active_l1
        + float(cfg.get("bias_weight", 1.0)) * bias
        + float(cfg.get("std_weight", 0.5)) * std
    )
    return loss, {"global_l1": global_l1, "active_l1": active_l1, "bias": bias, "std": std}


def diffusion_residual_loss(
    noise_loss: torch.Tensor,
    predicted_residual: torch.Tensor,
    target_residual: torch.Tensor,
    alpha_bar: torch.Tensor,
    cfg: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Align one-step x0 predictions with the decoded rainfall residual."""
    difference = predicted_residual - target_residual
    rainfall_l1 = difference.abs().mean()

    active_threshold = float(cfg.get("active_threshold", 0.01))
    active_mask = target_residual.abs() >= active_threshold
    if active_mask.any():
        active_l1 = difference.abs()[active_mask].mean()
    else:
        active_l1 = torch.zeros((), device=target_residual.device, dtype=target_residual.dtype)

    reduce_dims = tuple(range(1, target_residual.ndim))
    predicted_mean = predicted_residual.mean(dim=reduce_dims)
    target_mean = target_residual.mean(dim=reduce_dims)
    bias = (predicted_mean - target_mean).abs().mean()
    predicted_std = predicted_residual.flatten(1).std(dim=1, unbiased=False)
    target_std = target_residual.flatten(1).std(dim=1, unbiased=False)
    std = (predicted_std - target_std).abs().mean()

    predicted_centered = predicted_residual - predicted_mean.view(-1, 1, 1, 1, 1)
    target_centered = target_residual - target_mean.view(-1, 1, 1, 1, 1)
    correlation = F.cosine_similarity(
        predicted_centered.flatten(1),
        target_centered.flatten(1),
        dim=1,
        eps=1.0e-6,
    ).mean()
    direction = 1.0 - correlation

    # A one-step x0 estimate is trustworthy mainly at lower-noise timesteps.
    # Keep epsilon-MSE active everywhere while fading decoded-space supervision
    # according to the retained signal amplitude sqrt(alpha_bar).
    signal_weight = alpha_bar.flatten(1).mean(dim=1).sqrt().mean()
    decoded_loss = (
        float(cfg.get("rainfall_l1_weight", 1.0)) * rainfall_l1
        + float(cfg.get("active_l1_weight", 1.0)) * active_l1
        + float(cfg.get("bias_weight", 0.5)) * bias
        + float(cfg.get("std_weight", 0.25)) * std
        + float(cfg.get("direction_weight", 0.05)) * direction
    )
    total = float(cfg.get("noise_weight", 1.0)) * noise_loss + signal_weight * decoded_loss
    return total, {
        "noise": noise_loss,
        "rainfall_l1": rainfall_l1,
        "active_l1": active_l1,
        "bias": bias,
        "std": std,
        "correlation": correlation,
        "direction": direction,
        "signal_weight": signal_weight,
        "decoded": decoded_loss,
    }


class ChannelMoments:
    def __init__(self, channels: int) -> None:
        self.sum = torch.zeros(channels, dtype=torch.float64)
        self.square_sum = torch.zeros(channels, dtype=torch.float64)
        self.count = 0

    def update(self, latent: torch.Tensor) -> None:
        reduce_dims = (0, 2, 3, 4)
        values = latent.detach().double().cpu()
        self.sum += values.sum(dim=reduce_dims)
        self.square_sum += values.square().sum(dim=reduce_dims)
        self.count += values.shape[0] * values.shape[2] * values.shape[3] * values.shape[4]

    def compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.count <= 0:
            raise ValueError("Cannot compute latent statistics without observations")
        mean = self.sum / self.count
        variance = self.square_sum / self.count - mean.square()
        return mean.float(), variance.clamp_min(0).sqrt().float()
