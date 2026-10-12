"""Loss and scheduling utilities for soft rain-occurrence gating."""

import torch
from torch.nn import functional as F


def resolve_gate_strength(step: int, start_step: int, ramp_steps: int) -> float:
    if step < 0 or start_step < 0 or ramp_steps < 0:
        raise ValueError("Gate schedule values must be non-negative")
    if step <= start_step:
        return 0.0
    if ramp_steps == 0:
        return 1.0
    return min((step - start_step) / ramp_steps, 1.0)


def soft_occurrence_loss(
    logits: torch.Tensor,
    target_mm_min: torch.Tensor,
    *,
    threshold_mm_min: float,
    false_positive_weight: float,
    strong_threshold_mm_min: float,
    strong_weight: float,
    pod_floor: float,
    pod_floor_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if logits.shape != target_mm_min.shape:
        raise ValueError(f"Occurrence shape mismatch: {tuple(logits.shape)} vs {tuple(target_mm_min.shape)}")
    if threshold_mm_min <= 0 or strong_threshold_mm_min < threshold_mm_min:
        raise ValueError("Occurrence thresholds are invalid")
    if false_positive_weight <= 0 or strong_weight < 0 or pod_floor_weight < 0:
        raise ValueError("Occurrence weights are invalid")
    if not 0 <= pod_floor <= 1:
        raise ValueError("pod_floor must be in [0, 1]")

    wet = target_mm_min >= threshold_mm_min
    dry = ~wet
    labels = wet.to(logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits.float(), labels.float(), reduction="none")

    def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_float = mask.to(values.dtype)
        return (values * mask_float).sum() / mask_float.sum().clamp_min(1.0)

    wet_loss = masked_mean(bce, wet)
    dry_loss = masked_mean(bce, dry)
    strong = target_mm_min >= strong_threshold_mm_min
    weak = wet & ~strong
    strong_loss = masked_mean(bce, strong)
    probability = logits.float().sigmoid()
    soft_pod = masked_mean(probability, wet)
    pod_penalty = F.relu(probability.new_tensor(pod_floor) - soft_pod).square()
    balanced = (wet_loss + false_positive_weight * dry_loss) / (1.0 + false_positive_weight)
    total = balanced + strong_weight * strong_loss + pod_floor_weight * pod_penalty
    return total, {
        "wet_bce": wet_loss,
        "dry_bce": dry_loss,
        "strong_bce": strong_loss,
        "dry_probability_mean": masked_mean(probability, dry),
        "weak_probability_mean": masked_mean(probability, weak),
        "strong_probability_mean": masked_mean(probability, strong),
        "soft_pod": soft_pod,
        "pod_penalty": pod_penalty,
        "probability_mean": probability.mean(),
    }
