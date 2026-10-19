from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


class RainLeadMetrics:
    def __init__(self, thresholds: list[float]) -> None:
        if not thresholds or not all(torch.isfinite(torch.tensor(thresholds))):
            raise ValueError("Finite event thresholds are required.")
        self.thresholds = thresholds
        self.counts = torch.zeros(4, len(thresholds), 3, dtype=torch.float64)
        self.errors = torch.zeros(4, 2, dtype=torch.float64)
        self.pixels = 0
        self.wet = torch.zeros(4, 2, dtype=torch.float64)
        self.dry = torch.zeros(4, 2, dtype=torch.float64)
        self.dry_edges = torch.linspace(-1, 2, 3001, dtype=torch.float64)
        self.dry_hist = torch.zeros(4, 3002, dtype=torch.int64)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        if pred.shape != target.shape or pred.ndim != 5 or pred.shape[2] != 4:
            raise ValueError("Metrics require matching four-frame predictions and targets.")
        if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
            raise ValueError("Nonfinite prediction or target in validation.")
        pred, target = pred.detach().float(), target.detach().float()
        error = pred - target
        axes = (0, 1, 3, 4)
        self.errors[:, 0] += error.abs().sum(dim=axes, dtype=torch.float64).cpu()
        self.errors[:, 1] += error.square().sum(dim=axes, dtype=torch.float64).cpu()
        self.pixels += pred.numel() // 4
        for index, threshold in enumerate(self.thresholds):
            p, y = pred >= threshold, target >= threshold
            for column, event in enumerate((p & y, p & ~y, ~p & y)):
                self.counts[:, index, column] += event.sum(dim=axes).cpu()
        threshold = min(self.thresholds)
        p, y = pred >= threshold, target >= threshold
        self.wet[:, 0] += p.sum(dim=axes).cpu()
        self.wet[:, 1] += y.sum(dim=axes).cpu()
        self.dry[:, 0] += torch.where(~y, pred, 0).sum(dim=axes, dtype=torch.float64).cpu()
        self.dry[:, 1] += (~y).sum(dim=axes).cpu()
        for horizon in range(4):
            values = pred[:, :, horizon][~y[:, :, horizon]].double()
            buckets = torch.bucketize(values, self.dry_edges.to(values.device))
            self.dry_hist[horizon] += torch.bincount(buckets, minlength=3002).cpu()

    def compute(self) -> dict[str, float | None]:
        if self.pixels == 0:
            raise ValueError("No validation pixels were observed.")
        result: dict[str, float | None] = {}
        for horizon in range(4):
            prefix = f"h{horizon + 1}"
            result[f"{prefix}/mae"] = float(self.errors[horizon, 0] / self.pixels)
            result[f"{prefix}/mse"] = float(self.errors[horizon, 1] / self.pixels)
            result[f"{prefix}/pred_wet_fraction"] = float(self.wet[horizon, 0] / self.pixels)
            result[f"{prefix}/target_wet_fraction"] = float(self.wet[horizon, 1] / self.pixels)
            result[f"{prefix}/dry_pred_mean"] = (
                float(self.dry[horizon, 0] / self.dry[horizon, 1]) if self.dry[horizon, 1] else None
            )
            histogram = self.dry_hist[horizon]
            count = int(histogram.sum())
            for label, value in (("underflow", histogram[0]), ("overflow", histogram[-1])):
                result[f"{prefix}/dry_hist_{label}_fraction"] = float(value / count) if count else None
            cumulative = histogram.cumsum(0)
            for percentile in (50, 90, 99):
                value = None
                if count:
                    bucket = int(torch.searchsorted(cumulative, torch.tensor(max(1.0, count * percentile / 100))))
                    if 0 < bucket < len(self.dry_edges):
                        value = float((self.dry_edges[bucket - 1] + self.dry_edges[bucket]) / 2)
                result[f"{prefix}/dry_pred_p{percentile}_hist"] = value
        for label, counts in [*((f"h{i + 1}", self.counts[i]) for i in range(4)), ("all", self.counts.sum(dim=0))]:
            for index, threshold in enumerate(self.thresholds):
                tp, fp, fn = (float(v) for v in counts[index])
                for name, numerator, denominator in (
                    ("csi", tp, tp + fp + fn),
                    ("far", fp, tp + fp),
                    ("pod", tp, tp + fn),
                ):
                    result[f"{label}/{name}@{threshold:g}"] = numerator / denominator if denominator else None
                for name, value in (("hits", tp), ("false_alarms", fp), ("misses", fn)):
                    result[f"{label}/{name}@{threshold:g}"] = value
        return result


def save_raw_forecast(
    path: Path,
    history: dict[str, torch.Tensor],
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
) -> None:
    fig, axes = plt.subplots(3, 12, figsize=(24, 6), squeeze=False)
    for row, modality in enumerate(("radar", "satellite", "rain")):
        arrays = [value[modality][0, 0].detach().float().cpu().numpy() for value in (history, prediction, target)]
        vmin = 0.0
        vmax = 1.0 if modality != "rain" else max(float(arrays[0].max()), float(arrays[2].max()), 0.5)
        for group, (name, frames) in enumerate(zip(("history", "prediction", "truth"), arrays, strict=True)):
            for step in range(4):
                ax = axes[row, group * 4 + step]
                ax.imshow(
                    frames[step],
                    vmin=vmin,
                    vmax=vmax,
                    cmap="gray" if modality == "satellite" else "turbo",
                    interpolation="nearest",
                )
                ax.set_title(f"{name} {step + 1}", fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
        axes[row, 0].set_ylabel(f"{modality} c0 raw", fontsize=9)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=180)
    plt.close(fig)
