import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from kornia.augmentation import Resize
from litdata import StreamingDataset
from skimage.filters import gaussian

from src.tools.diagnose_rain_transitions import transition_masks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--month-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metadata = pd.read_parquet(args.month_dir / "metadata.parquet")
    dataset = StreamingDataset(input_dir=str(args.month_dir / "pairs"), shuffle=False)
    if len(dataset) != len(metadata):
        raise ValueError("Dataset and metadata counts differ.")
    times = pd.to_datetime(metadata["time"])
    if not times.is_monotonic_increasing or times.duplicated().any():
        raise ValueError("Expected strictly increasing, unique times.")
    torch.set_num_threads(2)
    resize = Resize((448, 448), align_corners=False, keepdim=True)
    sums = {name: np.zeros(12, dtype=np.float64) for name in ("radar", "rain")}
    regions = {name: np.zeros(3, dtype=np.float64) for name in ("dry", "onset", "cessation", "persistent_wet")}
    previous = None
    raw_union = None
    raw_positive_count = 0
    raw_total = 0
    nonfinite = {name: 0 for name in ("radar", "rain", "rain_interpolated")}
    pairs = 0
    reconstruction = []
    rows = []
    probe_indices = set(np.linspace(0, len(dataset) - 1, 32, dtype=int).tolist())
    for index in range(len(dataset)):
        sample = dataset[index]
        sample_time = pd.to_datetime(sample["__key__"], format="modalities_%Y%m%d_%H%M%S")
        if sample_time != times.iloc[index]:
            raise ValueError(f"Sample time differs from metadata at {index}.")
        for name in nonfinite:
            nonfinite[name] += int((~np.isfinite(sample[name])).sum())
        raw_rain = np.asarray(sample["rain"], dtype=np.float32)
        raw_total += raw_rain.size
        raw_positive_count += int((raw_rain > 0).sum())
        raw_union = (raw_rain > 0) if raw_union is None else raw_union | (raw_rain > 0)
        fields = {}
        for name, key in (("radar", "radar"), ("rain", "rain_interpolated")):
            field = resize(torch.as_tensor(sample[key], dtype=torch.float32)).squeeze()
            field = torch.nan_to_num(field, nan=0.0, posinf=60.0 if name == "radar" else 0.0, neginf=0.0)
            field = field.clamp(0, 60) / 60 if name == "radar" else field.clamp_min(0)
            fields[name] = field.numpy().astype(np.float64)
            value = fields[name]
            sums[name][:4] += [value.size, value.sum(), np.square(value).sum(), (value == 0).sum()]

        if index in probe_indices and np.isfinite(raw_rain).all() and raw_rain.max() > 0:
            expanded = gaussian(gaussian(raw_rain, sigma=1.5), sigma=1.5)
            scale = float(raw_rain.max() / expanded.max())
            reconstructed = expanded * scale
            stored = np.asarray(sample["rain_interpolated"])
            reconstruction.append({
                "index": index,
                "time": str(sample_time),
                "peak_restore_factor": scale,
                "max_abs_error": float(np.abs(reconstructed - stored).max()),
                "relative_max_error": float(np.abs(reconstructed - stored).max() / stored.max()),
            })

        if previous is not None and sample_time - times.iloc[index - 1] == pd.Timedelta(minutes=30):
            pairs += 1
            row = {"time": str(sample_time)}
            for name, current in fields.items():
                before = previous[name]
                delta = current - before
                energy = np.square(delta)
                top_count = int(np.ceil(energy.size * 0.01))
                top_energy = float(np.partition(energy.reshape(-1), energy.size - top_count)[-top_count:].sum())
                sums[name][4:] += [
                    energy.sum(), np.abs(delta).sum(), top_energy, current.size,
                    before.sum(), current.sum(), np.square(before).sum(), (before * current).sum(),
                ]
                row[f"{name}_mse"] = float(energy.mean())
                row[f"{name}_mae"] = float(np.abs(delta).mean())
                row[f"{name}_top_1pct_energy_share"] = top_energy / energy.sum() if energy.sum() else None
                row[f"{name}_unchanged_fraction"] = float((delta == 0).mean())
                row[f"{name}_target_energy"] = float(np.square(current).mean())
                row[f"{name}_previous_energy"] = float(np.square(before).mean())
            rain_delta = fields["rain"] - previous["rain"]
            for name, mask in transition_masks(previous["rain"], fields["rain"], 0.01).items():
                regions[name] += [mask.sum(), np.abs(rain_delta[mask]).sum(), np.square(rain_delta[mask]).sum()]
            rows.append(row)
        previous = fields
        if (index + 1) % 200 == 0:
            print(f"Scanned {index + 1}/{len(dataset)} frames", flush=True)

    frame = pd.DataFrame(rows)
    report = {
        "month_dir": str(args.month_dir.resolve()),
        "frames": len(dataset),
        "pairs_30_minutes": pairs,
        "time_start": str(times.min()),
        "time_end": str(times.max()),
        "selection": "All stored frames, no rain/delta filter; only exact 30-minute pairs.",
        "raw_rain_positive_fraction": raw_positive_count / raw_total,
        "raw_rain_ever_positive_spatial_fraction": float(raw_union.mean()),
        "raw_nonfinite_counts": nonfinite,
        "fields": {},
        "regions_threshold_0p01": {},
        "gaussian_reconstruction_probes": reconstruction,
    }
    for name, values in sums.items():
        n, total, energy, zeros, delta_energy, delta_abs, top_energy, pair_n, left_sum, right_sum, left_energy, cross = values
        target_energy = float(frame[f"{name}_target_energy"].sum() * 448 * 448)
        var_left = left_energy / pair_n - (left_sum / pair_n) ** 2
        var_right = target_energy / pair_n - (right_sum / pair_n) ** 2
        denominator = np.sqrt(var_left * var_right)
        report["fields"][name] = {
            "zero_fraction": zeros / n,
            "mean": total / n,
            "std": np.sqrt(energy / n - (total / n) ** 2),
            "persistence_mse": delta_energy / pair_n,
            "persistence_mae": delta_abs / pair_n,
            "zero_prediction_mse": target_energy / pair_n,
            "persistence_mse_over_zero_prediction_mse": delta_energy / target_energy,
            "top_1pct_per_pair_energy_share_pooled": top_energy / delta_energy,
            "adjacent_pearson_pooled": (cross / pair_n - left_sum * right_sum / pair_n ** 2) / denominator,
            "exact_repeat_pairs": int((frame[f"{name}_mse"] == 0).sum()),
        }
    region_total = sum(value[0] for value in regions.values())
    for name, (count, abs_sum, sq_sum) in regions.items():
        report["regions_threshold_0p01"][name] = {
            "fraction": count / region_total,
            "persistence_mae": abs_sum / count if count else None,
            "persistence_mse": sq_sum / count if count else None,
        }
    report["modality_time_offsets_minutes"] = {}
    for name in ("radar", "satellite"):
        offsets = (pd.to_datetime(metadata[f"raw_{name}_time"]) - times).dt.total_seconds() / 60
        report["modality_time_offsets_minutes"][name] = {
            "min": float(offsets.min()), "max": float(offsets.max()),
            "median": float(offsets.median()), "exact_fraction": float((offsets == 0).mean()),
        }
    frame.to_csv(args.output_dir / "pairs.csv", index=False)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "gaussian_reconstruction_probes"}, indent=2))


if __name__ == "__main__":
    main()
