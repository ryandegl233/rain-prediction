"""Physical-unit rollout constraints and validation acceptance checks."""

import math

import torch
from torch.nn import functional as F


def dry_rollout_constraint(
    prediction: torch.Tensor, target: torch.Tensor, config: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError("Expected matching [B,C,T,H,W] rainfall tensors")
    prediction, target = prediction.float(), target.detach().float()
    weights = prediction.new_tensor(config["lead_weights"])
    if len(weights) != prediction.shape[2] or (weights <= 0).any():
        raise ValueError("One positive lead weight is required per forecast frame")
    weights = weights / weights.sum()
    reduce_dims = (0, 1, 3, 4)

    def masked_mean(error: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        counts = mask.sum(dim=reduce_dims)
        per_lead = (error * mask).sum(dim=reduce_dims) / counts.clamp_min(1)
        valid_weights = weights * (counts > 0)
        return (per_lead * valid_weights).sum() / valid_weights.sum().clamp_min(1e-8)

    dry = target < float(config["dry_threshold_mm_min"])
    excess = F.relu(prediction - target) / float(config["error_scale_mm_min"])
    dry_loss = masked_mean(F.smooth_l1_loss(excess, torch.zeros_like(excess), beta=0.2, reduction="none"), dry)
    coverage = []
    for threshold in config["coverage_thresholds_mm_min"]:
        threshold = float(threshold)
        coverage.append(masked_mean(F.relu(threshold - prediction) / threshold, target >= threshold))
    coverage_loss = torch.stack(coverage).mean()
    strong = target >= float(config["strong_threshold_mm_min"])
    relative_shortfall = F.relu(target - prediction) / target.clamp_min(float(config["strong_threshold_mm_min"]))
    strong_loss = masked_mean(relative_shortfall, strong)
    combined = dry_loss + float(config["coverage_weight"]) * coverage_loss + float(config["strong_weight"]) * strong_loss
    return combined, {"dry": dry_loss, "coverage": coverage_loss, "strong": strong_loss}


def bounded_auxiliary(base_loss: torch.Tensor, auxiliary: torch.Tensor, step: int, config: dict) -> tuple[torch.Tensor, float, torch.Tensor]:
    ramp = min((step + 1) / max(int(config["warmup_steps"]), 1), 1.0)
    weight = float(config["weight"]) * ramp
    budget = base_loss.detach().clamp_min(0) * float(config["max_base_loss_ratio"])
    requested = weight * auxiliary
    scale = (budget / requested.detach().clamp_min(1e-12)).clamp(max=1.0)
    return requested * scale, ramp, scale


def assess_candidate(current: dict, reference: dict, config: dict) -> dict:
    failures = []
    if len(current["deterministic"]["by_lead"]) != 4 or len(reference["deterministic"]["by_lead"]) != 4:
        raise ValueError("Selection requires all four forecast leads")
    for now, before in zip(current["deterministic"]["by_lead"], reference["deterministic"]["by_lead"], strict=True):
        if now["lead_index"] != before["lead_index"]:
            raise ValueError("Lead order differs")
        lead = now["lead_index"] + 1
        for threshold in ("0.01", "0.1", "0.2", "0.5"):
            for metric in ("csi", "pod", "far"):
                if not all(math.isfinite(row[f"threshold@{threshold}"][metric]) for row in (now, before)):
                    failures.append(f"t+{lead} nonfinite {metric}@{threshold}")
            for metric in ("csi", "pod"):
                if now[f"threshold@{threshold}"][metric] < before[f"threshold@{threshold}"][metric] - float(config[f"{metric}_tolerance"]):
                    failures.append(f"t+{lead} {metric}@{threshold}")
        a, b = current["evolution"][str(lead)], reference["evolution"][str(lead)]
        for threshold in ("0.1", "0.5", "1.0"):
            mae_key, ratio_key = f"core_{threshold}_mae", f"core_{threshold}_intensity_ratio"
            for key in (mae_key, ratio_key):
                if any(value is not None and not math.isfinite(value) for value in (a.get(key), b.get(key))):
                    failures.append(f"t+{lead} nonfinite {key}")
            if a.get(mae_key) is not None and b.get(mae_key) is not None:
                if a[mae_key] > b[mae_key] * (1 + float(config["strong_mae_relative_tolerance"])):
                    failures.append(f"t+{lead} {mae_key}")
            if a.get(ratio_key) is not None and b.get(ratio_key) is not None:
                if a[ratio_key] < b[ratio_key] * (1 - float(config["strong_intensity_relative_tolerance"])):
                    failures.append(f"t+{lead} {ratio_key}")
    def long_far(report: dict) -> float:
        return sum(row["threshold@0.01"]["far"] for row in report["deterministic"]["by_lead"][2:4]) / 2
    score, baseline = long_far(current), long_far(reference)
    if score >= baseline:
        failures.append("long-lead FAR@0.01 did not decrease")
    return {"eligible": not failures, "failed_checks": failures, "long_lead_far": score, "reference_long_lead_far": baseline}
