import argparse
import csv
import hashlib
import json
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils.visualization.plot_ragion import _draw_geojson_boundary, _load_geojson, _rain_colormap


@dataclass(frozen=True)
class ForecastCase:
    path: Path
    metadata: dict
    prediction: np.ndarray
    truth: np.ndarray
    history: np.ndarray
    future_times: list[str]
    history_times: list[str]
    raw_sha256: str


def frame_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float | int | None]:
    if prediction.shape != truth.shape or prediction.ndim != 2:
        raise ValueError("Prediction and truth must be matching two-dimensional frames.")
    if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise ValueError("Only finite raw values can be scored.")
    error = prediction.astype(np.float64) - truth.astype(np.float64)
    result: dict[str, float | int | None] = {
        "mse": float(np.square(error).mean()),
        "mae": float(np.abs(error).mean()),
    }
    for threshold in (0.1, 0.3, 0.5):
        predicted, observed = prediction >= threshold, truth >= threshold
        tp = int(np.count_nonzero(predicted & observed))
        fp = int(np.count_nonzero(predicted & ~observed))
        fn = int(np.count_nonzero(~predicted & observed))
        for name, numerator, denominator in (
            ("csi", tp, tp + fp + fn),
            ("pod", tp, tp + fn),
            ("far", fp, tp + fp),
        ):
            result[f"{name}@{threshold}"] = numerator / denominator if denominator else None
        result[f"tp@{threshold}"] = tp
        result[f"fp@{threshold}"] = fp
        result[f"fn@{threshold}"] = fn
    return result


def load_cases(paths: list[Path]) -> list[ForecastCase]:
    cases: list[ForecastCase] = []
    for path in dict.fromkeys(p.resolve() for p in paths):
        required = [path / name for name in ("forecast_raw.npz", "case.json", "sichuan_boundary.geojson")]
        if not all(file.is_file() for file in required):
            print(f"Skipping incomplete export: {path}", flush=True)
            continue
        metadata = json.loads(required[1].read_text(encoding="utf-8"))
        with np.load(required[0], allow_pickle=False) as archive:
            prediction = archive["prediction_rain"]
            truth = archive["truth_rain"]
            history = archive["history_rain"]
            future_times = archive["future_times"].tolist()
            history_times = archive["history_times"].tolist()
        if prediction.ndim != 3 or prediction.shape != truth.shape or prediction.shape[0] != len(future_times):
            raise ValueError(f"Invalid forecast shape or timestamps: {path}")
        if history.ndim != 3 or history.shape[0] != len(history_times) or history.shape[1:] != truth.shape[1:]:
            raise ValueError(f"Invalid history shape: {path}")
        if future_times != metadata["future_times"] or history_times != metadata["history_times"]:
            raise ValueError(f"Array timestamps differ from case metadata: {path}")
        if not all(np.isfinite(array).all() for array in (prediction, truth, history)):
            raise ValueError(f"Nonfinite values in {path}")
        if not future_times or not history_times or metadata["origin"] != "upper":
            raise ValueError(f"Missing frames or unsupported geographic orientation: {path}")
        with required[0].open("rb") as stream:
            raw_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        case = ForecastCase(path, metadata, prediction, truth, history, future_times, history_times, raw_sha256)
        if cases:
            reference = cases[0]
            if future_times != reference.future_times or history_times != reference.history_times:
                raise ValueError("Case times differ; compare the same origin and forecast window.")
            if metadata["geo_bounds"] != reference.metadata["geo_bounds"]:
                raise ValueError("Case geographic extents differ.")
            if not np.array_equal(truth, reference.truth) or not np.array_equal(history, reference.history):
                raise ValueError("Case truth/history differs; check preprocessing before ranking.")
        cases.append(case)
    if not cases:
        raise ValueError(
            "No complete case exports found. Supply directories containing forecast_raw.npz and case.json."
        )
    return cases


def select_frames(rows: list[dict], top_k: int) -> list[dict]:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    eligible = [row for row in rows if row["csi@0.5"] is not None]
    ranked = sorted(
        eligible,
        key=lambda row: (-row["csi@0.5"], -(row["csi@0.3"] or 0), row["mse"], row["valid_time"], row["case"]),
    )
    selected: list[dict] = []
    seen_times: set[str] = set()
    for row in ranked:
        if row["valid_time"] not in seen_times:
            selected.append(row)
            seen_times.add(row["valid_time"])
            if len(selected) == top_k:
                break
    return selected


def save_selected_panel(cases: list[ForecastCase], rows: list[dict], destination: Path, dpi: int) -> None:
    by_path = {str(case.path): case for case in cases}
    count = len(rows)
    fig, axes = plt.subplots(2, count, figsize=(5 * count, 9.5), squeeze=False)
    cmap, norm = _rain_colormap()
    for col, row in enumerate(rows):
        case = by_path[row["case"]]
        boundary = _load_geojson(case.path / "sichuan_boundary.geojson")
        extent = case.metadata["geo_bounds"]
        frame = row["frame_index"]
        valid = datetime.fromisoformat(row["valid_time"])
        model_label = "Baseline A" if "priority_A" in case.metadata["checkpoint"] else "Old gated model"
        for position, (label, array) in enumerate((("Prediction", case.prediction), ("Truth", case.truth))):
            ax = axes[position, col]
            ax.imshow(
                array[frame],
                cmap=cmap,
                norm=norm,
                extent=extent,
                origin="upper",
                interpolation="nearest",
                alpha=1.0,
            )
            _draw_geojson_boundary(ax, boundary, color="#cf5f64", linewidth=0.65)
            ax.set_xlim(extent[:2])
            ax.set_ylim(extent[2:])
            ax.set_aspect("equal")
            ax.axis("off")
            title = f"{label} | {valid:%m-%d %H:%M} | +{row['lead_minutes']} min"
            if position == 0:
                title = f"{model_label}, step {row['step']}\n{title}\nCSI@0.5 = {row['csi@0.5']:.3f}"
            ax.set_title(title, fontsize=11, pad=8)
    origin = datetime.fromisoformat(cases[0].history_times[-1])
    fig.suptitle(f"Sichuan rainfall case\nLast observed nominal time: {origin:%Y-%m-%d %H:%M}", fontsize=12)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.22, top=0.85, hspace=0.25, wspace=0.08)
    color_ax = fig.add_axes((0.16, 0.13, 0.68, 0.018))
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=color_ax,
        orientation="horizontal",
        ticks=[0, 0.1, 0.5, 5, 15],
    )
    color_ax.set_xlabel("Rain value (dataset scale)\nCommon bins; unmodified predictions", fontsize=9)
    footer_width = 59 if count == 1 else 100
    footer = "\n".join(
        textwrap.fill(line, width=footer_width)
        for line in (
            "Examples selected using ground truth by CSI@0.5; not an overall performance summary.",
            "Nominal Beijing times; leads beyond +120 min extend past the training forecast horizon.",
        )
    )
    fig.text(
        0.5,
        0.015,
        footer,
        ha="center",
        fontsize=8,
    )
    fig.savefig(destination.with_suffix(".png"), dpi=dpi)
    fig.savefig(destination.with_suffix(".pdf"), dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score raw Sichuan case exports and plot labelled selected examples.")
    parser.add_argument("--cases", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.top_k <= 0 or args.dpi <= 0:
        parser.error("--top-k and --dpi must be positive.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {args.output_dir}")
    cases = load_cases(args.cases)
    rows = []
    for case in cases:
        origin = datetime.fromisoformat(case.history_times[-1])
        for index, valid_time in enumerate(case.future_times):
            row = {
                "case": str(case.path),
                "step": case.metadata["step"],
                "frame_index": index,
                "valid_time": valid_time,
                "lead_minutes": int((datetime.fromisoformat(valid_time) - origin).total_seconds() / 60),
                **frame_metrics(case.prediction[index], case.truth[index]),
            }
            persistence = frame_metrics(case.history[-1], case.truth[index])
            row["persistence_csi@0.5"] = persistence["csi@0.5"]
            row["csi05_gain_vs_persistence"] = (
                row["csi@0.5"] - persistence["csi@0.5"]
                if row["csi@0.5"] is not None and persistence["csi@0.5"] is not None
                else None
            )
            rows.append(row)
    selected = select_frames(rows, args.top_k)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "all_frame_metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "selection_uses_ground_truth": True,
        "criterion": "CSI@0.5 descending; ties CSI@0.3 descending then MSE ascending; distinct valid times.",
        "requested_top_k": args.top_k,
        "candidate_frames": len(rows),
        "selected": selected,
        "inputs": [
            {
                "case": str(case.path),
                "raw_sha256": case.raw_sha256,
                "checkpoint": case.metadata["checkpoint"],
            }
            for case in cases
        ],
        "skipped_incomplete_inputs": [
            str(path) for path in args.cases if path.resolve() not in {c.path for c in cases}
        ],
        "interpretation": "Selected retrospective examples, not a validation score or proof of overall skill.",
        "rendering": "Same original bins and nearest rendering for prediction/truth; no smoothing or intensity changes.",
    }
    (args.output_dir / "selection.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    if not selected:
        print("No frames with defined CSI@0.5. Metrics exported; no best-frame claim or selected plot produced.")
        return
    save_selected_panel(cases, selected, args.output_dir / "selected_comparison", args.dpi)
    for rank, row in enumerate(selected, 1):
        save_selected_panel(
            cases,
            [row],
            args.output_dir / f"selected_{rank:02d}_step{row['step']}_lead{row['lead_minutes']:03d}min",
            args.dpi,
        )
        print(
            f"Selected {rank}: step={row['step']}, valid={row['valid_time']}, "
            f"lead={row['lead_minutes']} min, CSI@0.5={row['csi@0.5']:.4f}"
        )
    print(f"Compared {len(cases)} cases / {len(rows)} frames. Saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
