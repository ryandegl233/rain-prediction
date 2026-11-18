#!/usr/bin/env python3
import argparse
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class RegionGifFrame:
    frame_idx: int
    gt_path: Path
    pred_path: Path
    valid_time: datetime | None


def _extract_future_idx(path: Path) -> int | None:
    match = re.search(r"future(\d+)", path.stem)
    if match is None:
        return None
    return int(match.group(1))


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        font_path = Path(candidate)
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def _collect_frame_pairs(
    input_dir: Path,
    *,
    sample_idx: int,
    gt_prefix: str,
    pred_prefix: str,
    start_time: datetime | None,
    end_time: datetime | None,
    interval_minutes: int,
    max_frame_idx: int | None,
) -> list[RegionGifFrame]:
    gt_files = input_dir.glob(f"{gt_prefix}_sample{sample_idx}_future*.jpg")
    pred_files = input_dir.glob(f"{pred_prefix}_sample{sample_idx}_future*.jpg")

    gt_by_idx = {idx: path for path in gt_files if (idx := _extract_future_idx(path)) is not None}
    pred_by_idx = {idx: path for path in pred_files if (idx := _extract_future_idx(path)) is not None}
    common_indices = sorted(set(gt_by_idx) & set(pred_by_idx))
    if not common_indices:
        raise FileNotFoundError(
            f"No paired gt/pred frames found in {input_dir} for sample_idx={sample_idx}. "
            f"Expected patterns {gt_prefix}_sample{sample_idx}_future*.jpg and "
            f"{pred_prefix}_sample{sample_idx}_future*.jpg."
        )

    frames: list[RegionGifFrame] = []
    for frame_idx in common_indices:
        valid_time = None
        if start_time is not None:
            valid_time = start_time + timedelta(minutes=interval_minutes * frame_idx)
        if max_frame_idx is not None and frame_idx > max_frame_idx:
            continue
        if end_time is not None and valid_time is not None and valid_time > end_time:
            continue
        frames.append(
            RegionGifFrame(
                frame_idx=frame_idx,
                gt_path=gt_by_idx[frame_idx],
                pred_path=pred_by_idx[frame_idx],
                valid_time=valid_time,
            )
        )
    if not frames:
        raise FileNotFoundError("No frames left after applying time/frame filters.")
    return frames


def _crop_footer(image: Image.Image, crop_bottom_ratio: float) -> Image.Image:
    ratio = max(0.0, min(float(crop_bottom_ratio), 0.5))
    if ratio <= 0.0:
        return image
    keep_h = max(1, int(round(image.height * (1.0 - ratio))))
    return image.crop((0, 0, image.width, keep_h))


def _resize_to_height(image: Image.Image, target_h: int) -> Image.Image:
    if image.height == target_h:
        return image
    target_w = max(1, int(round(image.width * target_h / image.height)))
    return image.resize((target_w, target_h), Image.Resampling.LANCZOS)


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    *,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = (20, 20, 20),
) -> None:
    left, top, right, bottom = box
    text_box = draw.textbbox((0, 0), text, font=font)
    text_w = text_box[2] - text_box[0]
    text_h = text_box[3] - text_box[1]
    x = left + max(0, (right - left - text_w) // 2)
    y = top + max(0, (bottom - top - text_h) // 2)
    draw.text((x, y), text, font=font, fill=fill)


def _compose_comparison_frame(
    frame: RegionGifFrame,
    *,
    panel_height: int,
    crop_bottom_ratio: float,
    gt_label: str,
    pred_label: str,
) -> Image.Image:
    gt_img = _crop_footer(Image.open(frame.gt_path).convert("RGB"), crop_bottom_ratio)
    pred_img = _crop_footer(Image.open(frame.pred_path).convert("RGB"), crop_bottom_ratio)
    gt_img = _resize_to_height(gt_img, panel_height)
    pred_img = _resize_to_height(pred_img, panel_height)

    title_font = _load_font(26)
    label_font = _load_font(24)

    title_h = 54
    label_h = 42
    margin = 26
    gap = 34

    panel_w = max(gt_img.width, pred_img.width)
    gt_img = gt_img.resize((panel_w, panel_height), Image.Resampling.LANCZOS)
    pred_img = pred_img.resize((panel_w, panel_height), Image.Resampling.LANCZOS)

    group_w = panel_w
    canvas_w = margin * 2 + group_w * 2 + gap
    canvas_h = margin + title_h + label_h + panel_height + margin
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    time_label = ""
    if frame.valid_time is not None:
        time_label = f" | time={frame.valid_time.strftime('%Y-%m-%d %H:%M:%S')}"
    title = f"frame={frame.frame_idx:02d}{time_label}"
    _draw_centered_text(draw, (0, margin, canvas_w, margin + title_h), title, font=title_font)

    label_top = margin + title_h
    img_top = label_top + label_h
    gt_x = margin
    pred_x = margin + group_w + gap

    _draw_centered_text(draw, (gt_x, label_top, gt_x + panel_w, label_top + label_h), gt_label, font=label_font)
    _draw_centered_text(draw, (pred_x, label_top, pred_x + panel_w, label_top + label_h), pred_label, font=label_font)

    canvas.paste(gt_img, (gt_x, img_top))
    canvas.paste(pred_img, (pred_x, img_top))
    return canvas


def create_region_prediction_gif(
    input_dir: str | Path,
    output_path: str | Path,
    *,
    sample_idx: int = 0,
    gt_prefix: str = "gt_rain",
    pred_prefix: str = "pred_rain",
    gt_label: str = "GT",
    pred_label: str = "Pred",
    start_time: str | None = None,
    end_time: str | None = None,
    interval_minutes: int = 30,
    max_frame_idx: int | None = None,
    panel_height: int = 420,
    crop_bottom_ratio: float = 0.14,
    duration_ms: int = 500,
    loop: int = 0,
) -> Path:
    input_path = Path(input_dir)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    parsed_start_time = None
    if start_time:
        parsed_start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    parsed_end_time = None
    if end_time:
        parsed_end_time = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")

    frame_pairs = _collect_frame_pairs(
        input_path,
        sample_idx=sample_idx,
        gt_prefix=gt_prefix,
        pred_prefix=pred_prefix,
        start_time=parsed_start_time,
        end_time=parsed_end_time,
        interval_minutes=int(interval_minutes),
        max_frame_idx=max_frame_idx,
    )
    frames = [
        _compose_comparison_frame(
            frame,
            panel_height=int(panel_height),
            crop_bottom_ratio=float(crop_bottom_ratio),
            gt_label=gt_label,
            pred_label=pred_label,
        )
        for frame in frame_pairs
    ]
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=int(duration_ms),
        loop=int(loop),
        optimize=False,
        disposal=2,
    )
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a side-by-side GT/PRED rain-region GIF.")
    parser.add_argument("--input-dir", type=str, default="vis_show/ema8")
    parser.add_argument("--output-path", type=str, default="vis_show/ema8/gt_pred_region_compare.gif")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--gt-prefix", type=str, default="gt_rain")
    parser.add_argument("--pred-prefix", type=str, default="pred_rain")
    parser.add_argument("--gt-label", type=str, default="GT")
    parser.add_argument("--pred-label", type=str, default="Pred")
    parser.add_argument("--start-time", type=str, default="")
    parser.add_argument("--end-time", type=str, default="")
    parser.add_argument("--interval-minutes", type=int, default=30)
    parser.add_argument("--max-frame-idx", type=int, default=None)
    parser.add_argument("--panel-height", type=int, default=420)
    parser.add_argument("--crop-bottom-ratio", type=float, default=0.14)
    parser.add_argument("--duration-ms", type=int, default=500)
    parser.add_argument("--loop", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = create_region_prediction_gif(
        input_dir=args.input_dir,
        output_path=args.output_path,
        sample_idx=int(args.sample_idx),
        gt_prefix=args.gt_prefix,
        pred_prefix=args.pred_prefix,
        gt_label=args.gt_label,
        pred_label=args.pred_label,
        start_time=args.start_time or None,
        end_time=args.end_time or None,
        interval_minutes=int(args.interval_minutes),
        max_frame_idx=args.max_frame_idx,
        panel_height=int(args.panel_height),
        crop_bottom_ratio=float(args.crop_bottom_ratio),
        duration_ms=int(args.duration_ms),
        loop=int(args.loop),
    )
    print(f"Saved GIF: {output}")


if __name__ == "__main__":
    main()
