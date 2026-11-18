import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from litdata import StreamingDataset
from skimage.filters import gaussian


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--month-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=64)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    metadata = pd.read_parquet(args.month_dir / "metadata.parquet")
    times = pd.to_datetime(metadata["time"])
    indices = np.flatnonzero((times.diff() == pd.Timedelta(minutes=30)).to_numpy())
    selected = indices[np.linspace(0, len(indices) - 1, args.pairs, dtype=int)]
    dataset = StreamingDataset(input_dir=str(args.month_dir / "pairs"), shuffle=False)
    records = []
    for index in selected:
        fields = []
        for frame_index in (index - 1, index):
            sample = dataset[int(frame_index)]
            if pd.to_datetime(sample["__key__"], format="modalities_%Y%m%d_%H%M%S") != times.iloc[frame_index]:
                raise ValueError("Sample time differs from metadata.")
            raw = np.asarray(sample["rain"], dtype=np.float64)
            blurred = gaussian(gaussian(raw, sigma=1.5), sigma=1.5)
            if not np.isfinite(raw).all() or blurred.max() <= 0:
                raise ValueError("This decomposition requires finite, nonzero rain frames.")
            factor = float(raw.max() / blurred.max())
            target = np.asarray(sample["rain_interpolated"], dtype=np.float64)
            if not np.allclose(factor * blurred, target, atol=1e-6, rtol=1e-5):
                raise ValueError("Stored target does not match Gaussian peak restoration.")
            fields.append((blurred, factor, target))
        before, after = fields
        local = after[1] * (after[0] - before[0])
        gain = (after[1] - before[1]) * before[0]
        delta = after[2] - before[2]
        if not np.allclose(local + gain, delta, atol=2e-6, rtol=2e-5):
            raise ValueError("Temporal change decomposition failed.")
        records.append({
            "time": str(times.iloc[index]),
            "previous_factor": before[1],
            "factor": after[1],
            "factor_relative_change": abs(after[1] - before[1]) / before[1],
            "delta_energy": float(np.square(delta).sum()),
            "local_term_energy": float(np.square(local).sum()),
            "gain_term_energy": float(np.square(gain).sum()),
            "twice_cross_term": float(2 * (local * gain).sum()),
        })
    total = sum(row["delta_energy"] for row in records)
    summary = {
        "pairs": len(records),
        "sampling": "Uniformly spaced exact-30-minute pairs, no intensity or change filtering.",
        "median_relative_gain_change": float(np.median([row["factor_relative_change"] for row in records])),
        "gain_change_over_10_percent_pair_fraction": float(np.mean([row["factor_relative_change"] > 0.1 for row in records])),
        "local_energy_over_total_delta_energy": sum(row["local_term_energy"] for row in records) / total,
        "gain_energy_over_total_delta_energy": sum(row["gain_term_energy"] for row in records) / total,
        "twice_cross_over_total_delta_energy": sum(row["twice_cross_term"] for row in records) / total,
        "interpretation": "Energy terms are not independent percentages: local+gain+cross equals total; terms can cancel. This is a label decomposition, not a causal attribution of model error.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "records": records}, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
