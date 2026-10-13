#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import tifffile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.ai_model_invoke import _Runtime
from src.utils.visualization.plot import plot_any_modality


def _load_frame(frame_dir: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    radar = torch.as_tensor(tifffile.imread(frame_dir / "radar.tiff"), dtype=torch.float32)
    sat = torch.as_tensor(tifffile.imread(frame_dir / "satellite.tiff"), dtype=torch.float32)
    rain = torch.as_tensor(tifffile.imread(frame_dir / "rain_interpolated.tiff"), dtype=torch.float32)

    if sat.ndim == 3 and sat.shape[0] != 10 and sat.shape[-1] == 10:
        sat = sat.permute(2, 0, 1)

    radar = radar.unsqueeze(0)
    rain = rain.unsqueeze(0)
    return radar, sat, rain


def _save_vis(
    out_path: Path,
    radar_past: torch.Tensor,
    sat_past: torch.Tensor,
    rain_past: torch.Tensor,
    pred_cls: torch.Tensor,
    n_show: int = 5,
) -> None:
    num_classes = int(pred_cls.max().item()) + 1
    colors = ["white", "lightblue", "blue", "green", "yellow", "orange", "red", "purple", "black"][:num_classes]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(np.arange(num_classes + 1), cmap.N)

    radar_img = plot_any_modality(radar_past[0:1, :, -1], "radar", False)
    sat_img = plot_any_modality(sat_past[0:1, :, -1], "satellite", False)
    rain_img = plot_any_modality(rain_past[0:1, :, -1], "rain", False)

    t_pred = int(pred_cls.shape[1])
    n_show = max(1, min(int(n_show), t_pred))
    ncols = 3 + n_show

    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
    axes[0].imshow(radar_img)
    axes[0].set_title("Radar Past Last")
    axes[1].imshow(sat_img)
    axes[1].set_title("Satellite Past Last")
    axes[2].imshow(rain_img)
    axes[2].set_title("Rain Past Last")

    for t in range(n_show):
        pred_img = (cmap(norm(pred_cls[0, t].cpu().numpy()))[..., :3] * 255).astype(np.uint8)
        axes[3 + t].imshow(pred_img)
        axes[3 + t].set_title(f"Pred Rain Class t={t}")

    for ax in axes:
        ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime stream consumer for SwinNet inference")
    parser.add_argument("--stream-dir", type=str, default=DEFAULT_CONFIG["stream_dir"])
    parser.add_argument("--out-dir", type=str, default=DEFAULT_CONFIG["out_dir"])
    parser.add_argument("--poll-seconds", type=int, default=DEFAULT_CONFIG["poll_seconds"])
    args = parser.parse_args()

    runtime = _Runtime.get()
    stream_dir = Path(args.stream_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seen_windows: set[str] = set()

    while True:
        frames = sorted([p for p in stream_dir.iterdir() if p.is_dir() and p.name[:1].isdigit()])
        if len(frames) < runtime.n_past:
            time.sleep(max(1, int(args.poll_seconds)))
            continue

        window = frames[-runtime.n_past :]
        window_key = "__".join([p.name for p in window])
        if window_key in seen_windows:
            time.sleep(max(1, int(args.poll_seconds)))
            continue

        radar_frames: list[torch.Tensor] = []
        sat_frames: list[torch.Tensor] = []
        rain_frames: list[torch.Tensor] = []

        for frame in window:
            radar_raw, sat_raw, rain_raw = _load_frame(frame)
            radar, sat, rain = runtime.preprocessor.process_frame(radar_raw=radar_raw, sat_raw=sat_raw, rain_raw=rain_raw)
            radar_frames.append(radar)
            sat_frames.append(sat)
            rain_frames.append(rain)

        radar_past = torch.stack(radar_frames, dim=1).unsqueeze(0)
        sat_past = torch.stack(sat_frames, dim=1).unsqueeze(0)
        rain_past = torch.stack(rain_frames, dim=1).unsqueeze(0)

        with torch.no_grad():
            pred = runtime.model(
                radar_past.to(runtime.device),
                sat_past.to(runtime.device),
                rain_past.to(runtime.device),
            )
            pred_cls = torch.argmax(pred, dim=1).cpu()

        out = {
            "window": [p.name for p in window],
            "pred_shape": list(pred.shape),
            "pred_distribution": torch.bincount(pred_cls.reshape(-1), minlength=runtime.num_classes).tolist(),
        }
        out_path = out_dir / f"{window[-1].name}.json"
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

        vis_path = out_dir / f"{window[-1].name}.jpg"
        _save_vis(vis_path, radar_past, sat_past, rain_past, pred_cls, n_show=5)
        print(f"[consumer] inferred {window[-1].name} -> {out_path} | vis -> {vis_path}")

        seen_windows.add(window_key)
        time.sleep(max(1, int(args.poll_seconds)))





# Example:
# python src/tools/realtime_stream_infer.py \
#   --stream-dir runtime_stream \
#   --out-dir results/realtime_infer \
#   --poll-seconds 5
DEFAULT_CONFIG = {
    "stream_dir": "runtime_stream",
    "out_dir": "results/realtime_infer",
    "poll_seconds": 5,
}

if __name__ == "__main__":
    main()
