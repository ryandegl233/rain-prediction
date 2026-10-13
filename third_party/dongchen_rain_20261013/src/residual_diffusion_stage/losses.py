import torch
from torch.nn import functional as F


def deterministic_correction_loss(
    center: torch.Tensor,
    target: torch.Tensor,
    coarse: torch.Tensor,
    mu: torch.Tensor,
    gate: torch.Tensor,
    cfg: dict,
    occurrence_logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train the deterministic correction while explicitly controlling false alarms."""
    global_l1 = F.l1_loss(center, target)
    active_threshold = float(cfg.get("active_threshold", 0.01))
    active_mask = target >= active_threshold
    dry_mask = ~active_mask
    zero = torch.zeros((), device=target.device, dtype=target.dtype)
    active_l1 = (center - target).abs()[active_mask].mean() if active_mask.any() else zero
    dry_false_alarm = F.relu(center[dry_mask] - active_threshold).mean() if dry_mask.any() else zero

    target_residual = target - coarse
    residual_l1 = F.l1_loss(gate * mu, target_residual)
    correction_bias = ((gate * mu).mean() - target_residual.mean()).abs()
    dry_gate = gate[dry_mask].mean() if dry_mask.any() else zero

    absolute_error = (center - target).abs()
    graded_l1 = zero
    graded_weight_sum = 0.0
    graded_thresholds = [float(value) for value in cfg.get("graded_thresholds", [0.1, 0.5, 1.0])]
    graded_weights = [float(value) for value in cfg.get("graded_weights", [1.0, 2.0, 4.0])]
    if len(graded_thresholds) != len(graded_weights):
        raise ValueError("graded_thresholds and graded_weights must have the same length")
    for threshold, weight in zip(graded_thresholds, graded_weights):
        threshold_mask = target >= threshold
        if threshold_mask.any():
            graded_l1 = graded_l1 + weight * absolute_error[threshold_mask].mean()
            graded_weight_sum += weight
    if graded_weight_sum > 0.0:
        graded_l1 = graded_l1 / graded_weight_sum

    target_occurrence = active_mask.to(target.dtype)
    if occurrence_logits is None:
        occurrence_temperature = float(cfg.get("occurrence_temperature", 0.02))
        occurrence_logits = (center - active_threshold) / occurrence_temperature
    occurrence_probability = torch.sigmoid(occurrence_logits)
    occurrence_bce = F.binary_cross_entropy_with_logits(
        occurrence_logits,
        target_occurrence,
        reduction="none",
        pos_weight=torch.as_tensor(
            float(cfg.get("occurrence_positive_weight", 4.0)),
            device=target.device,
            dtype=target.dtype,
        ),
    )
    focal_gamma = float(cfg.get("occurrence_focal_gamma", 2.0))
    occurrence_pt = torch.where(active_mask, occurrence_probability, 1.0 - occurrence_probability)
    occurrence_loss = (occurrence_bce * (1.0 - occurrence_pt).pow(focal_gamma)).mean()

    local_support_loss = zero
    for kernel_size in [int(value) for value in cfg.get("local_support_scales", [3, 5])]:
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"local support scale must be a positive odd integer, got {kernel_size}")
        padding = kernel_size // 2
        predicted_fraction = F.avg_pool3d(
            occurrence_probability,
            kernel_size=(1, kernel_size, kernel_size),
            stride=1,
            padding=(0, padding, padding),
        )
        target_fraction = F.avg_pool3d(
            target_occurrence,
            kernel_size=(1, kernel_size, kernel_size),
            stride=1,
            padding=(0, padding, padding),
        )
        local_support_loss = local_support_loss + F.mse_loss(predicted_fraction, target_fraction)
    local_support_loss = local_support_loss / max(len(cfg.get("local_support_scales", [3, 5])), 1)

    support_kernel = int(cfg.get("false_alarm_support_scale", 5))
    if support_kernel <= 0 or support_kernel % 2 == 0:
        raise ValueError(f"false_alarm_support_scale must be a positive odd integer, got {support_kernel}")
    target_support = F.max_pool3d(
        target_occurrence,
        kernel_size=(1, support_kernel, support_kernel),
        stride=1,
        padding=(0, support_kernel // 2, support_kernel // 2),
    ).bool()
    outside_support = ~target_support
    outside_support_false_alarm = (
        F.relu(center[outside_support] - active_threshold).mean()
        if outside_support.any()
        else zero
    )

    soft_csi_loss = zero
    temperature = float(cfg.get("soft_csi_temperature", 0.02))
    thresholds = [float(value) for value in cfg.get("soft_csi_thresholds", [0.01, 0.1, 0.2, 0.5])]
    for threshold in thresholds:
        predicted_event = torch.sigmoid((center - threshold) / temperature)
        target_event = torch.sigmoid((target - threshold) / temperature)
        intersection = (predicted_event * target_event).sum()
        union = predicted_event.sum() + target_event.sum() - intersection
        soft_csi_loss = soft_csi_loss + 1.0 - intersection / union.clamp_min(1.0)
    soft_csi_loss = soft_csi_loss / max(len(thresholds), 1)

    total = (
        float(cfg.get("global_l1_weight", 1.0)) * global_l1
        + float(cfg.get("active_l1_weight", 2.0)) * active_l1
        + float(cfg.get("dry_false_alarm_weight", 1.0)) * dry_false_alarm
        + float(cfg.get("residual_l1_weight", 0.5)) * residual_l1
        + float(cfg.get("bias_weight", 0.5)) * correction_bias
        + float(cfg.get("dry_gate_weight", 0.05)) * dry_gate
        + float(cfg.get("soft_csi_weight", 0.25)) * soft_csi_loss
        + float(cfg.get("graded_l1_weight", 0.0)) * graded_l1
        + float(cfg.get("occurrence_loss_weight", 0.0)) * occurrence_loss
        + float(cfg.get("local_support_loss_weight", 0.0)) * local_support_loss
        + float(cfg.get("outside_support_false_alarm_weight", 0.0))
        * outside_support_false_alarm
    )
    return total, {
        "global_l1": global_l1,
        "active_l1": active_l1,
        "dry_false_alarm": dry_false_alarm,
        "residual_l1": residual_l1,
        "bias": correction_bias,
        "dry_gate": dry_gate,
        "soft_csi": 1.0 - soft_csi_loss,
        "gate_mean": gate.mean(),
        "mu_abs_mean": mu.abs().mean(),
        "graded_l1": graded_l1,
        "occurrence_loss": occurrence_loss,
        "occurrence_probability_mean": occurrence_probability.mean(),
        "local_support_loss": local_support_loss,
        "outside_support_false_alarm": outside_support_false_alarm,
    }


def residual_autoencoder_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    cfg: dict,
    rainfall_target: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Reconstruct residuals while preserving the first lead's heavy-rain detail.

    ``target`` is the residual, while ``rainfall_target`` is the corresponding
    physical rain field used only to identify important t+1 pixels.
    """
    global_l1 = F.l1_loss(reconstruction, target)
    active_threshold = float(cfg.get("active_threshold", 0.01))
    active_mask = target.abs() >= active_threshold
    if active_mask.any():
        active_l1 = (reconstruction - target).abs()[active_mask].mean()
    else:
        active_l1 = torch.zeros((), device=target.device, dtype=target.dtype)
    bias = (reconstruction.mean() - target.mean()).abs()
    std = (reconstruction.std(unbiased=False) - target.std(unbiased=False)).abs()

    zero = torch.zeros((), device=target.device, dtype=target.dtype)
    lead0_peak_l1 = zero
    lead0_gradient_l1 = zero
    if rainfall_target is not None and target.shape[2] > 0:
        peak_threshold = float(cfg.get("lead0_peak_threshold", 0.1))
        peak_weight = float(cfg.get("lead0_peak_weight", 0.0))
        gradient_loss_weight = float(cfg.get("lead0_gradient_loss_weight", 0.0))
        if peak_weight > 0.0 or gradient_loss_weight > 0.0:
            lead0_rain = rainfall_target[:, :, :1]
            lead0_target = target[:, :, :1]
            lead0_reconstruction = reconstruction[:, :, :1]
            # Weight continuously by rain intensity, while retaining a hard
            # lower bound so isolated heavy-rain pixels receive supervision.
            intensity = (lead0_rain / max(peak_threshold, 1.0e-6)).clamp(
                min=0.0,
                max=float(cfg.get("lead0_peak_intensity_cap", 4.0)),
            )
            peak_mask = lead0_rain >= peak_threshold
            weights = 1.0 + peak_weight * intensity * peak_mask.to(target.dtype)
            lead0_peak_l1 = (
                (lead0_reconstruction - lead0_target).abs() * weights
            ).sum() / weights.sum().clamp_min(1.0)

            if gradient_loss_weight > 0.0:
                # Spatial gradients preserve compact-cell boundaries that are
                # easily smoothed by the AE's 4x down/up-sampling path.
                pred_dx = lead0_reconstruction[..., 1:, :] - lead0_reconstruction[..., :-1, :]
                pred_dy = lead0_reconstruction[..., :, 1:] - lead0_reconstruction[..., :, :-1]
                true_dx = lead0_target[..., 1:, :] - lead0_target[..., :-1, :]
                true_dy = lead0_target[..., :, 1:] - lead0_target[..., :, :-1]
                mask_x = peak_mask[..., 1:, :] | peak_mask[..., :-1, :]
                mask_y = peak_mask[..., :, 1:] | peak_mask[..., :, :-1]
                errors = []
                if mask_x.any():
                    errors.append((pred_dx - true_dx).abs()[mask_x])
                if mask_y.any():
                    errors.append((pred_dy - true_dy).abs()[mask_y])
                if errors:
                    lead0_gradient_l1 = torch.cat(errors).mean()
    loss = (
        float(cfg.get("global_l1_weight", 1.0)) * global_l1
        + float(cfg.get("active_l1_weight", 2.0)) * active_l1
        + float(cfg.get("bias_weight", 1.0)) * bias
        + float(cfg.get("std_weight", 0.5)) * std
        + float(cfg.get("lead0_peak_loss_weight", 0.0)) * lead0_peak_l1
        + float(cfg.get("lead0_gradient_loss_weight", 0.0)) * lead0_gradient_l1
    )
    return loss, {
        "global_l1": global_l1,
        "active_l1": active_l1,
        "bias": bias,
        "std": std,
        "lead0_peak_l1": lead0_peak_l1,
        "lead0_gradient_l1": lead0_gradient_l1,
    }


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

    # Add a bounded tiered weight for large residual amplitudes.  The regular
    # active-region term remains in place so this only adds emphasis to the
    # compact/heavy-rain corrections instead of changing the full objective.
    intensity_l1 = torch.zeros((), device=target_residual.device, dtype=target_residual.dtype)
    intensity_thresholds = [
        float(value) for value in cfg.get("intensity_thresholds", [0.1, 0.5, 1.0])
    ]
    intensity_weights = [
        float(value) for value in cfg.get("intensity_weights", [1.0, 1.5, 2.5])
    ]
    if len(intensity_thresholds) != len(intensity_weights):
        raise ValueError("intensity_thresholds and intensity_weights must have the same length")
    if intensity_thresholds:
        residual_magnitude = target_residual.abs()
        weights = torch.ones_like(residual_magnitude)
        for threshold, weight in zip(intensity_thresholds, intensity_weights, strict=True):
            weights = torch.where(
                residual_magnitude >= threshold,
                torch.as_tensor(weight, device=weights.device, dtype=weights.dtype),
                weights,
            )
        intensity_l1 = (difference.abs() * weights).sum() / weights.sum().clamp_min(1.0)

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
    zero_mean = predicted_mean.abs().mean()

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
        + float(cfg.get("zero_mean_weight", 0.0)) * zero_mean
    )
    total = float(cfg.get("noise_weight", 1.0)) * noise_loss + signal_weight * decoded_loss
    total = total + signal_weight * float(cfg.get("intensity_l1_weight", 0.0)) * intensity_l1
    return total, {
        "noise": noise_loss,
        "rainfall_l1": rainfall_l1,
        "active_l1": active_l1,
        "intensity_l1": intensity_l1,
        "bias": bias,
        "std": std,
        "correlation": correlation,
        "direction": direction,
        "zero_mean": zero_mean,
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
