import torch


def lead_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.ndim != 5 or pred.shape != target.shape:
        raise ValueError("Prediction and target must have matching [B,C,T,H,W] shapes.")
    return (pred.float() - target.float()).square().mean(dim=(0, 1, 3, 4))


def modality_losses(
    teacher_pred: dict[str, torch.Tensor],
    teacher_target: dict[str, torch.Tensor],
    rollout_pred: dict[str, torch.Tensor],
    rollout_target: dict[str, torch.Tensor],
    horizon_weights: torch.Tensor,
    rollout_weight: float,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    if horizon_weights.shape != (4,) or not torch.isfinite(horizon_weights).all():
        raise ValueError("Four finite horizon weights are required.")
    if (horizon_weights < 0).any() or not torch.isclose(horizon_weights.sum(), horizon_weights.new_tensor(1.0)):
        raise ValueError("Horizon weights must be nonnegative and sum to one.")
    if rollout_weight < 0:
        raise ValueError("rollout_weight must be nonnegative.")
    losses: dict[str, torch.Tensor] = {}
    rain_errors = None
    for name in ("rain", "radar", "satellite"):
        teacher = lead_mse(teacher_pred[name], teacher_target[name])
        future = lead_mse(rollout_pred[name], rollout_target[name])
        if teacher.shape != (7,) or future.shape != (4,):
            raise ValueError("The protocol requires seven shifted targets and four rollout targets.")
        teacher_loss = 0.5 * teacher[:3].mean() + 0.5 * teacher[3:].mean()
        future_loss = (future * horizon_weights.detach().to(future)).sum() if name == "rain" else future.mean()
        losses[name] = teacher_loss + rollout_weight * future_loss
        if name == "rain":
            rain_errors = future.detach()
    return losses, rain_errors
