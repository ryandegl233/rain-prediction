import math

import torch


PRIMARY_THRESHOLDS_MM_PER_MIN = [0.01, 0.1, 0.2, 0.5]
EXTENDED_THRESHOLDS_MM_PER_MIN = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]


def _event_metrics(hits: float, misses: float, false_alarms: float, correct_negatives: float) -> dict[str, float]:
    csi = hits / max(hits + misses + false_alarms, 1.0)
    pod = hits / max(hits + misses, 1.0)
    far = false_alarms / max(hits + false_alarms, 1.0)
    f1 = 2.0 * hits / max(2.0 * hits + misses + false_alarms, 1.0)
    hss_numerator = 2.0 * (hits * correct_negatives - misses * false_alarms)
    hss_denominator = (hits + misses) * (misses + correct_negatives) + (
        hits + false_alarms
    ) * (false_alarms + correct_negatives)
    return {
        "csi": csi,
        "pod": pod,
        "far": far,
        "f1": f1,
        "hss": hss_numerator / max(hss_denominator, 1.0),
    }


class DeterministicMetricAccumulator:
    """Pixel-wise deterministic metrics separated by forecast lead time."""

    def __init__(self, thresholds: list[float], lead_times: int, device: torch.device) -> None:
        self.thresholds = [float(value) for value in thresholds]
        error_shape = (lead_times,)
        event_shape = (lead_times, len(thresholds))
        self.absolute_error = torch.zeros(error_shape, device=device, dtype=torch.float64)
        self.squared_error = torch.zeros(error_shape, device=device, dtype=torch.float64)
        self.signed_error = torch.zeros(error_shape, device=device, dtype=torch.float64)
        self.pixel_count = torch.zeros(error_shape, device=device, dtype=torch.float64)
        self.hits = torch.zeros(event_shape, device=device, dtype=torch.float64)
        self.misses = torch.zeros(event_shape, device=device, dtype=torch.float64)
        self.false_alarms = torch.zeros(event_shape, device=device, dtype=torch.float64)
        self.correct_negatives = torch.zeros(event_shape, device=device, dtype=torch.float64)

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        if prediction.shape != target.shape or prediction.ndim != 5:
            raise ValueError(
                f"prediction and target must have the same [B,C,T,H,W] shape, got "
                f"{tuple(prediction.shape)} and {tuple(target.shape)}"
            )
        reduce_dims = (0, 1, 3, 4)
        error = (prediction - target).double()
        self.absolute_error += error.abs().sum(dim=reduce_dims)
        self.squared_error += error.square().sum(dim=reduce_dims)
        self.signed_error += error.sum(dim=reduce_dims)
        pixels = prediction.shape[0] * prediction.shape[1] * prediction.shape[3] * prediction.shape[4]
        self.pixel_count += pixels
        for index, threshold in enumerate(self.thresholds):
            predicted_event = prediction >= threshold
            target_event = target >= threshold
            self.hits[:, index] += (predicted_event & target_event).sum(dim=reduce_dims)
            self.misses[:, index] += ((~predicted_event) & target_event).sum(dim=reduce_dims)
            self.false_alarms[:, index] += (predicted_event & (~target_event)).sum(dim=reduce_dims)
            self.correct_negatives[:, index] += ((~predicted_event) & (~target_event)).sum(dim=reduce_dims)

    def reduce(self, accelerator) -> None:
        for name in (
            "absolute_error",
            "squared_error",
            "signed_error",
            "pixel_count",
            "hits",
            "misses",
            "false_alarms",
            "correct_negatives",
        ):
            setattr(self, name, accelerator.reduce(getattr(self, name), reduction="sum"))

    def _one_result(self, lead_index: int | None) -> dict:
        selector = slice(None) if lead_index is None else lead_index
        count = float(self.pixel_count[selector].sum().item())
        result = {
            "mae": float(self.absolute_error[selector].sum().item()) / max(count, 1.0),
            "rmse": math.sqrt(float(self.squared_error[selector].sum().item()) / max(count, 1.0)),
            "bias": float(self.signed_error[selector].sum().item()) / max(count, 1.0),
        }
        csi_values = []
        for threshold_index, threshold in enumerate(self.thresholds):
            event_selector = (selector, threshold_index)
            event = _event_metrics(
                float(self.hits[event_selector].sum().item()),
                float(self.misses[event_selector].sum().item()),
                float(self.false_alarms[event_selector].sum().item()),
                float(self.correct_negatives[event_selector].sum().item()),
            )
            result[f"threshold@{threshold:g}"] = event
            csi_values.append(event["csi"])
        result["mean_csi"] = sum(csi_values) / max(len(csi_values), 1)
        return result

    def result(self) -> dict:
        return {
            "overall": self._one_result(None),
            "by_lead": [
                {"lead_index": lead_index, **self._one_result(lead_index)}
                for lead_index in range(self.pixel_count.shape[0])
            ],
        }


def center_ensemble_residuals(residuals: torch.Tensor, variance_correction: bool = True) -> torch.Tensor:
    """Center members without silently shrinking their ensemble variance."""
    if residuals.ndim != 6:
        raise ValueError(f"residuals must be [B,K,C,T,H,W], got {tuple(residuals.shape)}")
    centered = residuals - residuals.mean(dim=1, keepdim=True)
    members = residuals.shape[1]
    if variance_correction and members > 1:
        centered = centered * math.sqrt(members / (members - 1))
    return centered


def _crps_field(ensemble: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    members = ensemble.shape[1]
    first = (ensemble - target[:, None]).abs().mean(dim=1)
    if members == 1:
        return first
    sorted_ensemble = ensemble.sort(dim=1).values
    ranks = torch.arange(1, members + 1, device=ensemble.device, dtype=ensemble.dtype)
    weights = 2.0 * ranks - members - 1.0
    pair_term = (sorted_ensemble * weights.view(1, members, 1, 1, 1, 1)).sum(dim=1) / (members**2)
    return first - pair_term


class EnsembleMetricAccumulator:
    """CRPS, threshold-weighted CRPS, Brier skill, reliability and spread-skill."""

    def __init__(
        self,
        thresholds: list[float],
        lead_times: int,
        device: torch.device,
        reliability_bins: int = 10,
    ) -> None:
        self.thresholds = [float(value) for value in thresholds]
        self.reliability_bins = int(reliability_bins)
        lead_shape = (lead_times,)
        threshold_shape = (lead_times, len(thresholds))
        reliability_shape = (lead_times, len(thresholds), reliability_bins)
        self.crps_sum = torch.zeros(lead_shape, device=device, dtype=torch.float64)
        self.reference_crps_sum = torch.zeros_like(self.crps_sum)
        self.twcrps_sum = torch.zeros(threshold_shape, device=device, dtype=torch.float64)
        self.reference_twcrps_sum = torch.zeros_like(self.twcrps_sum)
        self.brier_sum = torch.zeros(threshold_shape, device=device, dtype=torch.float64)
        self.reference_brier_sum = torch.zeros_like(self.brier_sum)
        self.pixel_count = torch.zeros(lead_shape, device=device, dtype=torch.float64)
        self.spread_square_sum = torch.zeros_like(self.crps_sum)
        self.mean_error_square_sum = torch.zeros_like(self.crps_sum)
        self.spread_sum = torch.zeros_like(self.crps_sum)
        self.absolute_error_sum = torch.zeros_like(self.crps_sum)
        self.spread_absolute_error_sum = torch.zeros_like(self.crps_sum)
        self.absolute_error_square_sum = torch.zeros_like(self.crps_sum)
        self.reliability_count = torch.zeros(reliability_shape, device=device, dtype=torch.float64)
        self.reliability_probability = torch.zeros_like(self.reliability_count)
        self.reliability_observed = torch.zeros_like(self.reliability_count)

    def update(self, ensemble: torch.Tensor, target: torch.Tensor, reference: torch.Tensor) -> None:
        if ensemble.ndim != 6 or ensemble.shape[0] != target.shape[0] or ensemble.shape[2:] != target.shape[1:]:
            raise ValueError("ensemble must be [B,K,C,T,H,W] and match target [B,C,T,H,W]")
        if target.shape != reference.shape:
            raise ValueError("target and reference must have identical shapes")
        reduce_dims = (0, 1, 3, 4)
        pixels = target.shape[0] * target.shape[1] * target.shape[3] * target.shape[4]
        self.pixel_count += pixels
        crps = _crps_field(ensemble.double(), target.double())
        self.crps_sum += crps.sum(dim=reduce_dims)
        self.reference_crps_sum += (reference - target).double().abs().sum(dim=reduce_dims)

        ensemble_mean = ensemble.mean(dim=1)
        spread = ensemble.double().std(dim=1, unbiased=ensemble.shape[1] > 1)
        absolute_error = (ensemble_mean - target).double().abs()
        self.spread_square_sum += spread.square().sum(dim=reduce_dims)
        self.mean_error_square_sum += (ensemble_mean - target).double().square().sum(dim=reduce_dims)
        self.spread_sum += spread.sum(dim=reduce_dims)
        self.absolute_error_sum += absolute_error.sum(dim=reduce_dims)
        self.spread_absolute_error_sum += (spread * absolute_error).sum(dim=reduce_dims)
        self.absolute_error_square_sum += absolute_error.square().sum(dim=reduce_dims)

        for threshold_index, threshold in enumerate(self.thresholds):
            clipped_ensemble = ensemble.double().clamp_min(threshold)
            clipped_target = target.double().clamp_min(threshold)
            self.twcrps_sum[:, threshold_index] += _crps_field(clipped_ensemble, clipped_target).sum(
                dim=reduce_dims
            )
            self.reference_twcrps_sum[:, threshold_index] += (
                reference.double().clamp_min(threshold) - clipped_target
            ).abs().sum(dim=reduce_dims)
            probability = (ensemble >= threshold).double().mean(dim=1)
            observed = (target >= threshold).double()
            reference_probability = (reference >= threshold).double()
            self.brier_sum[:, threshold_index] += (probability - observed).square().sum(dim=reduce_dims)
            self.reference_brier_sum[:, threshold_index] += (
                reference_probability - observed
            ).square().sum(dim=reduce_dims)
            bin_index = (probability * self.reliability_bins).long().clamp_max(self.reliability_bins - 1)
            for lead_index in range(target.shape[2]):
                lead_bins = bin_index[:, :, lead_index].reshape(-1)
                lead_probability = probability[:, :, lead_index].reshape(-1)
                lead_observed = observed[:, :, lead_index].reshape(-1)
                for bin_value in range(self.reliability_bins):
                    mask = lead_bins == bin_value
                    if mask.any():
                        self.reliability_count[lead_index, threshold_index, bin_value] += mask.sum()
                        self.reliability_probability[lead_index, threshold_index, bin_value] += lead_probability[
                            mask
                        ].sum()
                        self.reliability_observed[lead_index, threshold_index, bin_value] += lead_observed[mask].sum()

    def reduce(self, accelerator) -> None:
        for name in (
            "crps_sum",
            "reference_crps_sum",
            "twcrps_sum",
            "reference_twcrps_sum",
            "brier_sum",
            "reference_brier_sum",
            "pixel_count",
            "spread_square_sum",
            "mean_error_square_sum",
            "spread_sum",
            "absolute_error_sum",
            "spread_absolute_error_sum",
            "absolute_error_square_sum",
            "reliability_count",
            "reliability_probability",
            "reliability_observed",
        ):
            setattr(self, name, accelerator.reduce(getattr(self, name), reduction="sum"))

    def _lead_result(self, lead_index: int | None) -> dict:
        selector = slice(None) if lead_index is None else lead_index
        count = float(self.pixel_count[selector].sum().item())
        crps = float(self.crps_sum[selector].sum().item()) / max(count, 1.0)
        reference_crps = float(self.reference_crps_sum[selector].sum().item()) / max(count, 1.0)
        spread_rms = math.sqrt(float(self.spread_square_sum[selector].sum().item()) / max(count, 1.0))
        error_rmse = math.sqrt(float(self.mean_error_square_sum[selector].sum().item()) / max(count, 1.0))
        spread_sum = float(self.spread_sum[selector].sum().item())
        error_sum = float(self.absolute_error_sum[selector].sum().item())
        cross_sum = float(self.spread_absolute_error_sum[selector].sum().item())
        spread_square_sum = float(self.spread_square_sum[selector].sum().item())
        error_square_sum = float(self.absolute_error_square_sum[selector].sum().item())
        covariance = cross_sum - spread_sum * error_sum / max(count, 1.0)
        spread_variance = spread_square_sum - spread_sum**2 / max(count, 1.0)
        error_variance = error_square_sum - error_sum**2 / max(count, 1.0)
        result = {
            "crps": crps,
            "crps_reference": reference_crps,
            "crpss": 1.0 - crps / max(reference_crps, 1.0e-12),
            "spread_rms": spread_rms,
            "ensemble_mean_rmse": error_rmse,
            "spread_skill_ratio": spread_rms / max(error_rmse, 1.0e-12),
            "spread_error_correlation": covariance / max(math.sqrt(spread_variance * error_variance), 1.0e-12),
        }
        for threshold_index, threshold in enumerate(self.thresholds):
            event_selector = (selector, threshold_index)
            brier = float(self.brier_sum[event_selector].sum().item()) / max(count, 1.0)
            reference_brier = float(self.reference_brier_sum[event_selector].sum().item()) / max(count, 1.0)
            twcrps = float(self.twcrps_sum[event_selector].sum().item()) / max(count, 1.0)
            reference_twcrps = float(self.reference_twcrps_sum[event_selector].sum().item()) / max(count, 1.0)
            result[f"threshold@{threshold:g}"] = {
                "brier": brier,
                "brier_reference": reference_brier,
                "bss": 1.0 - brier / max(reference_brier, 1.0e-12),
                "twcrps": twcrps,
                "twcrps_reference": reference_twcrps,
                "twcrpss": 1.0 - twcrps / max(reference_twcrps, 1.0e-12),
            }
        return result

    def _reliability_result(self) -> dict:
        result = {}
        for threshold_index, threshold in enumerate(self.thresholds):
            by_lead = []
            for lead_index in range(self.pixel_count.shape[0]):
                bins = []
                for bin_index in range(self.reliability_bins):
                    count = float(self.reliability_count[lead_index, threshold_index, bin_index].item())
                    bins.append(
                        {
                            "lower": bin_index / self.reliability_bins,
                            "upper": (bin_index + 1) / self.reliability_bins,
                            "count": int(count),
                            "mean_probability": float(
                                self.reliability_probability[lead_index, threshold_index, bin_index].item()
                            )
                            / max(count, 1.0),
                            "observed_frequency": float(
                                self.reliability_observed[lead_index, threshold_index, bin_index].item()
                            )
                            / max(count, 1.0),
                        }
                    )
                by_lead.append({"lead_index": lead_index, "bins": bins})
            result[f"threshold@{threshold:g}"] = by_lead
        return result

    def result(self) -> dict:
        return {
            "overall": self._lead_result(None),
            "by_lead": [
                {"lead_index": lead_index, **self._lead_result(lead_index)}
                for lead_index in range(self.pixel_count.shape[0])
            ],
            "reliability": self._reliability_result(),
        }
