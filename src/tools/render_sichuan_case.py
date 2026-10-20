import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from PIL import Image

from src.tools.select_sichuan_cases import load_cases
from src.utils.visualization.plot_ragion import _draw_geojson_boundary, _load_geojson, _rain_colormap


def comparison_figure(
    truth: np.ndarray,
    prediction: np.ndarray,
    boundary: dict,
    extent: list[float],
    frame_index: int,
    valid_time: str,
) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.9), facecolor="white")
    cmap, norm = _rain_colormap()
    for ax, label, array in zip(axes, ("GT", "Pred"), (truth, prediction), strict=True):
        ax.imshow(array, cmap=cmap, norm=norm, extent=extent, origin="upper", interpolation="nearest")
        _draw_geojson_boundary(ax, boundary, color="#ff4040", linewidth=0.45)
        ax.set_xlim(extent[:2])
        ax.set_ylim(extent[2:])
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(label, fontsize=11, pad=8)

    timestamp = datetime.fromisoformat(valid_time)
    fig.suptitle(f"frame={frame_index:02d} | time={timestamp:%Y-%m-%d %H:%M:%S}", fontsize=12, y=0.94)
    fig.subplots_adjust(left=0.025, right=0.975, bottom=0.025, top=0.80, wspace=0.08)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Redraw raw Sichuan forecasts as compact GT/Pred pairs.")
    parser.add_argument("--cases", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-lead-minutes", type=int, default=180)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    if args.max_lead_minutes <= 0 or args.dpi <= 0:
        parser.error("--max-lead-minutes and --dpi must be positive.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {args.output_dir}")

    paths = list(dict.fromkeys(path.resolve() for path in args.cases))
    paths = [path for path in paths if (path / "forecast_raw.npz").is_file()]
    if not paths:
        parser.error("No forecast_raw.npz found in --cases. Supply the original inference export directories.")
    if len({path.name for path in paths}) != len(paths):
        parser.error("Case directory names must be unique within one export.")

    for path in paths:
        case = load_cases([path])[0]
        boundary = _load_geojson(path / "sichuan_boundary.geojson")
        origin = datetime.fromisoformat(case.history_times[-1])
        selected = []
        for index, value in enumerate(case.future_times):
            lead = (datetime.fromisoformat(value) - origin).total_seconds() / 60
            if 0 < lead <= args.max_lead_minutes:
                selected.append({"frame_index": index, "valid_time": value, "lead_minutes": lead})
        if not selected:
            raise ValueError(f"No frames within +{args.max_lead_minutes} minutes: {path}")

        output = args.output_dir / path.name
        output.mkdir(parents=True, exist_ok=True)
        images: list[Image.Image] = []
        for record in selected:
            index = record["frame_index"]
            timestamp = datetime.fromisoformat(record["valid_time"])
            name = f"frame_{index:02d}_{timestamp:%Y%m%d_%H%M}"
            fig = comparison_figure(
                case.truth[index],
                case.prediction[index],
                boundary,
                case.metadata["geo_bounds"],
                index,
                record["valid_time"],
            )
            fig.savefig(output / f"{name}.png", dpi=args.dpi, facecolor="white")
            fig.savefig(output / f"{name}.pdf", facecolor="white")
            plt.close(fig)
            with Image.open(output / f"{name}.png") as image:
                images.append(image.convert("RGB"))

        images[0].save(
            output / "comparison.gif",
            save_all=True,
            append_images=images[1:],
            duration=800,
            loop=0,
            disposal=2,
        )
        for image in images:
            image.close()
        report = {
            "source_case": str(path),
            "source_metadata": case.metadata,
            "raw_sha256": case.raw_sha256,
            "last_observed_time": case.history_times[-1],
            "max_lead_minutes": args.max_lead_minutes,
            "frames": selected,
            "rendering": "GT left, Pred right; original shared bins; nearest rendering; unchanged rain arrays.",
            "selection": "Every exported frame within the requested lead limit; no skill-based selection.",
        }
        (output / "render.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
        print(f"Saved {len(selected)} pairs and comparison.gif: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
