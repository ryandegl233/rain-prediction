import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def transition_masks(
    previous: NDArray, target: NDArray, threshold: float
) -> dict[str, NDArray]:
    if previous.shape != target.shape or threshold <= 0:
        raise ValueError("Matching shapes and a positive threshold are required.")
    valid = np.isfinite(previous) & np.isfinite(target)
    before = previous >= threshold
    after = target >= threshold
    return {
        "dry": valid & ~before & ~after,
        "onset": valid & ~before & after,
        "cessation": valid & before & ~after,
        "persistent_wet": valid & before & after,
    }


def region_metrics(
    previous: NDArray,
    target: NDArray,
    prediction: NDArray,
    prediction_previous: NDArray,
    threshold: float,
) -> dict[str, dict[str, float | int | None]]:
    if not (previous.shape == target.shape == prediction.shape == prediction_previous.shape):
        raise ValueError("All rain arrays must have matching shapes.")
    masks = transition_masks(previous, target, threshold)
    valid = np.isfinite(prediction) & np.isfinite(prediction_previous)
    if any(np.any(mask & ~valid) for mask in masks.values()):
        raise ValueError("Prediction contains non-finite values at valid target pixels.")
    error = prediction.astype(np.float64) - target
    true_delta = target.astype(np.float64) - previous
    predicted_delta = prediction.astype(np.float64) - prediction_previous
    delta_error = predicted_delta - true_delta
    total = sum(int(mask.sum()) for mask in masks.values())
    result = {}
    for name, mask in masks.items():
        count = int(mask.sum())
        metrics = {
            "count": count,
            "fraction": count / total if total else None,
            "mae": float(np.abs(error[mask]).mean()) if count else None,
            "mse": float(np.square(error[mask]).mean()) if count else None,
            "bias": float(error[mask].mean()) if count else None,
            "delta_mae": float(np.abs(delta_error[mask]).mean()) if count else None,
            "delta_mse": float(np.square(delta_error[mask]).mean()) if count else None,
            "target_abs_delta": float(np.abs(true_delta[mask]).mean()) if count else None,
            "predicted_abs_delta": float(np.abs(predicted_delta[mask]).mean()) if count else None,
            "prediction_mean": float(prediction[mask].mean()) if count else None,
            "predicted_wet_fraction": float((prediction[mask] >= threshold).mean()) if count else None,
            "prediction_negative_fraction": float((prediction[mask] < 0).mean()) if count else None,
        }
        result[name] = metrics
    return result


def field_summary(values: NDArray) -> dict:
    finite = values[np.isfinite(values)].astype(np.float64)
    positive = finite[finite > 0]
    neighbors = np.concatenate([
        np.diff(values, axis=-1).reshape(-1),
        np.diff(values, axis=-2).reshape(-1),
    ])
    neighbors = np.abs(neighbors[np.isfinite(neighbors)]).astype(np.float64)
    std = float(finite.std())
    return {
        "shape": list(values.shape),
        "nonfinite_fraction": float(1 - finite.size / values.size),
        "zero_fraction": float((finite == 0).mean()),
        "negative_fraction": float((finite < 0).mean()),
        "mean": float(finite.mean()),
        "std": std,
        "max": float(finite.max()),
        "positive_quantiles_p10_p50_p90_p99": np.quantile(positive, [0.1, 0.5, 0.9, 0.99]).tolist() if positive.size else None,
        "neighbor_abs_difference_mean_over_std": float(neighbors.mean() / std) if std else None,
        "neighbor_abs_difference_p99_over_std": float(np.quantile(neighbors, 0.99) / std) if std else None,
    }


def temporal_summary(previous: NDArray, target: NDArray) -> dict:
    valid = np.isfinite(previous) & np.isfinite(target)
    left = previous[valid].astype(np.float64)
    right = target[valid].astype(np.float64)
    delta = right - left
    energy = np.square(delta)
    count = max(1, int(np.ceil(energy.size * 0.01)))
    top = np.partition(energy, energy.size - count)[-count:]
    std = float(np.concatenate([left, right]).std())
    return {
        "mae": float(np.abs(delta).mean()),
        "mse": float(energy.mean()),
        "rmse_over_field_std": float(np.sqrt(energy.mean()) / std) if std else None,
        "top_one_percent_pixels_delta_energy_share": float(top.sum() / energy.sum()) if energy.sum() else None,
        "pearson": float(np.corrcoef(left, right)[0, 1]) if left.std() and right.std() else None,
    }


def evaluate_predictions(path: Path, thresholds: list[float]) -> dict:
    with np.load(path, allow_pickle=False) as arrays:
        prediction = arrays["prediction"]
        target = arrays["target"]
        anchor = arrays["anchor"]
    if prediction.ndim != 4 or prediction.shape != target.shape:
        raise ValueError("prediction and target must both be [N,T,H,W].")
    if anchor.shape != (target.shape[0], *target.shape[2:]):
        raise ValueError("anchor must be [N,H,W], the last observed rain frame.")
    previous = np.concatenate([anchor[:, None], target[:, :-1]], axis=1)
    prediction_previous = np.concatenate([anchor[:, None], prediction[:, :-1]], axis=1)
    persistence = np.broadcast_to(anchor[:, None], target.shape)
    return {
        str(threshold): {
            "prediction_by_lead": [
                region_metrics(previous[:, lead], target[:, lead], prediction[:, lead], prediction_previous[:, lead], threshold)
                for lead in range(target.shape[1])
            ],
            "fixed_anchor_persistence_by_lead": [
                region_metrics(previous[:, lead], target[:, lead], persistence[:, lead], persistence[:, lead], threshold)
                for lead in range(target.shape[1])
            ],
        }
        for threshold in thresholds
    }


def save_example(
    output: Path, radar: NDArray, rain: NDArray, pair: tuple[int, int], times: list[str]
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    before, after = pair
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for row, (name, fields) in enumerate((("Radar / 60", radar), ("Rain", rain))):
        selected = fields[[before, after]]
        positive = selected[np.isfinite(selected) & (selected > 0)]
        upper = float(np.quantile(positive, 0.995)) if positive.size else 1.0
        upper = max(upper, 1e-8)
        for column, index in enumerate(pair):
            artist = axes[row, column].imshow(fields[index], vmin=0, vmax=upper, cmap="viridis", interpolation="nearest")
            axes[row, column].set_title(f"{name}\n{times[index]}")
            fig.colorbar(artist, ax=axes[row, column], shrink=0.75)
        delta = fields[after] - fields[before]
        limit = max(float(np.nanquantile(np.abs(delta), 0.995)), 1e-8)
        artist = axes[row, 2].imshow(delta, vmin=-limit, vmax=limit, cmap="RdBu_r", interpolation="nearest")
        axes[row, 2].set_title("Next minus previous")
        fig.colorbar(artist, ax=axes[row, 2], shrink=0.75)
    for axis in axes.flat:
        axis.set_axis_off()
    fig.suptitle("No display smoothing; shared scale within each modality; colors capped at p99.5")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    import pandas as pd
    import torch
    from kornia.augmentation import Resize
    from litdata import StreamingDataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.01, 0.1, 0.3])
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--interval-minutes", type=int, default=30)
    parser.add_argument("--predictions", type=Path)
    args = parser.parse_args()
    if min(args.thresholds) <= 0 or args.image_size <= 0 or args.interval_minutes <= 0:
        raise ValueError("Thresholds, image size and interval must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    chunk_path = args.chunk.resolve()
    index = json.loads((chunk_path.parent / "index.json").read_text())
    start = 0
    for chunk in index["chunks"]:
        if chunk["filename"] == chunk_path.name:
            stop = start + chunk["chunk_size"]
            break
        start += chunk["chunk_size"]
    else:
        raise ValueError("Chunk is not present in index.json.")
    metadata = pd.read_parquet(chunk_path.parent.parent / "metadata.parquet")
    dataset = StreamingDataset(input_dir=str(chunk_path.parent), shuffle=False)
    resizer = Resize((args.image_size, args.image_size), align_corners=False, keepdim=True)
    times = []
    raw = {name: [] for name in ("radar", "rain", "rain_interpolated")}
    prepared = {name: [] for name in ("radar", "rain_interpolated")}
    for row in range(start, stop):
        sample = dataset[row]
        time = datetime.strptime(sample["__key__"], "modalities_%Y%m%d_%H%M%S")
        if pd.Timestamp(metadata.iloc[row]["time"]) != time:
            raise ValueError(f"Sample/metadata time mismatch at index {row}.")
        times.append(time.isoformat(sep=" "))
        for name in raw:
            field = np.asarray(sample[name], dtype=np.float32)
            raw[name].append(field)
            if name in prepared:
                resized = resizer(torch.from_numpy(field)).squeeze()
                resized = torch.nan_to_num(resized, nan=0.0, posinf=60.0 if name == "radar" else 0.0, neginf=0.0)
                resized = resized.clamp(0, 60) / 60 if name == "radar" else resized.clamp_min(0)
                prepared[name].append(resized.numpy())
    raw = {name: np.stack(fields) for name, fields in raw.items()}
    prepared = {name: np.stack(fields) for name, fields in prepared.items()}
    order = sorted(range(len(times)), key=lambda i: times[i])
    pairs = [(a, b) for a, b in zip(order[:-1], order[1:]) if (datetime.fromisoformat(times[b]) - datetime.fromisoformat(times[a])).total_seconds() == args.interval_minutes * 60]
    if not pairs:
        raise ValueError("No exact-interval adjacent frames in this chunk.")
    left, right = np.asarray(pairs).T
    rain = prepared["rain_interpolated"]
    report = {
        "chunk": str(chunk_path),
        "sample_index_start_inclusive": start,
        "sample_index_stop_exclusive": stop,
        "frame_count": len(times),
        "exact_interval_pair_count": len(pairs),
        "time_start": min(times),
        "time_end": max(times),
        "interval_minutes": args.interval_minutes,
        "image_size": args.image_size,
        "raw_fields": {name: field_summary(fields) for name, fields in raw.items()},
        "training_space_fields": {name: field_summary(fields) for name, fields in prepared.items()},
        "adjacent_frame_changes": {name: temporal_summary(fields[left], fields[right]) for name, fields in prepared.items()},
        "rain_regions_one_step_persistence": {
            str(threshold): region_metrics(rain[left], rain[right], rain[left], rain[left], threshold)
            for threshold in args.thresholds
        },
        "notes": [
            "Training-space arrays match resize, clipping and radar / 60 with modality_zero_centering=false; no augmentation.",
            "No display Gaussian expansion or peak restoration is used.",
            "All frames in the chunk are included, without the training window filter.",
            "Thresholds are in stored rain units; dry means below threshold, not necessarily exactly zero.",
            "Regions use adjacent GT frames; persistence is a baseline, not a model prediction.",
            "Transitions outside the selected chunk and across time gaps are excluded.",
        ],
    }
    filter_path = chunk_path.parent.parent / "metadata_rain_delta_filter.parquet"
    if filter_path.exists():
        filtered = pd.read_parquet(filter_path)
        if len(filtered) != len(metadata) or not filtered["time"].equals(metadata["time"]):
            raise ValueError("Filter metadata does not align with dataset metadata.")
        report["delta_filter"] = {
            "chunk_selected_frames": int(filtered.iloc[start:stop]["rain_delta_selected"].sum()),
            "month_selected_frames": int(filtered["rain_delta_selected"].sum()),
            "month_total_frames": len(filtered),
        }
    records = []
    for a, b in pairs:
        for threshold in args.thresholds:
            for region, metrics in region_metrics(rain[a], rain[b], rain[a], rain[a], threshold).items():
                records.append({"previous_time": times[a], "target_time": times[b], "threshold": threshold, "region": region, **metrics})
    pd.DataFrame(records).to_csv(args.output_dir / "regions_by_pair.csv", index=False)
    chosen = max(pairs, key=lambda pair: float(np.square(rain[pair[1]] - rain[pair[0]]).mean()))
    report["example_pair"] = [times[chosen[0]], times[chosen[1]]]
    report["example_selection"] = "Largest rain change MSE among valid pairs; not a random or typical pair."
    save_example(args.output_dir / "raw_transition.png", prepared["radar"], rain, chosen, times)
    if args.predictions is not None:
        report["model_predictions"] = evaluate_predictions(args.predictions, args.thresholds)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("frame_count", "exact_interval_pair_count", "time_start", "time_end", "delta_filter") if key in report}, indent=2))
    print(f"Report: {args.output_dir / 'report.json'}")


if __name__ == "__main__":
    main()
