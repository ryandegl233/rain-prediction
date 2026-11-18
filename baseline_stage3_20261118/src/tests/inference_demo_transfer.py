"""Rain-only inference demo for transfer-trainer checkpoints.

Usage:
    python src/tests/inference_demo_transfer.py \
        --checkpoint-path /home/dongchenwang/RainPrediction/runs/next_frame_light_rain/2026-10-27/23-43-31_light_rain_occurrence_finetune/checkpoints/step_00019000 \
        --predict-frames 12 \
        --target-valid-time "2025-08-10 23:30:00" \
        --history-window-frames 7 \
        --reference-dir vis_reference/fixed_20250810_2330_transfer_12f_stage1_next_frame_block
"""

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import accelerate
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import default_collate

from src.tests.inference_demo import (
    _analyze_input_reasonability,
    _build_demo_cfg_for_trainer,
    _build_reference_sequence,
    _prepare_analysis_modalities,
)
from src.tests.inference_demo_transfer_visualization import (
    save_conditioned_timeline,
    save_conditioning_diff_analysis,
    save_modality_diff_analysis,
    save_rain_residual_diagnostic_analysis,
)
from src.trainer.rain_trainer_ts_next_frame_transfer import RainTSNextFrameTransferTrainer


@dataclass(frozen=True)
class CheckpointFiles:
    directory: Path
    config: Path
    metadata: Path | None
    ema: Path | None
    model: Path | None
    explicit_weight: Path | None


def select_inference_window(
    dataset: object,
    *,
    dataset_index: int | None,
    target_valid_time: str | None,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    if dataset_index is not None and target_valid_time is not None:
        raise ValueError("dataset_index and target_valid_time are mutually exclusive")

    times_pairs = getattr(dataset, "times_pairs", None)
    indices_pairs = getattr(dataset, "indices_pairs", None)
    get_item = getattr(dataset, "__getitem__", None)
    if not isinstance(times_pairs, list) or not isinstance(indices_pairs, list) or get_item is None:
        raise TypeError("Validation dataset does not expose indexed time-series windows")

    resolved_index = 0 if dataset_index is None else dataset_index
    if target_valid_time is not None:
        try:
            requested_time = datetime.fromisoformat(target_valid_time)
        except ValueError as exc:
            raise ValueError(
                "target_valid_time must use ISO format, for example 2025-08-10 23:30:00"
            ) from exc
        matches = [
            index
            for index, (_, future_times) in enumerate(times_pairs)
            if len(future_times) > 0 and future_times[0] == requested_time
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one window with target valid time {requested_time.isoformat(sep=' ')}, "
                f"found {len(matches)}"
            )
        resolved_index = matches[0]

    if not 0 <= resolved_index < len(times_pairs):
        raise IndexError(f"dataset_index={resolved_index} is outside [0, {len(times_pairs)})")

    sample = get_item(resolved_index)
    if not isinstance(sample, Mapping):
        raise TypeError(f"Expected a mapping from validation dataset, got {type(sample).__name__}")
    batch = default_collate([sample])

    past_times, future_times = times_pairs[resolved_index]
    past_indices, future_indices = indices_pairs[resolved_index]
    selection = {
        "dataset_index": resolved_index,
        "window_start_time": past_times[0].isoformat(sep=" "),
        "target_valid_time": future_times[0].isoformat(sep=" "),
        "context_times": [value.isoformat(sep=" ") for value in past_times],
        "future_times": [value.isoformat(sep=" ") for value in future_times],
        "context_metadata_indices": [int(value) for value in past_indices],
        "future_metadata_indices": [int(value) for value in future_indices],
    }
    metadata = getattr(dataset, "metadata", None)
    if metadata is not None:
        selection["source_frames"] = [
            {
                "metadata_index": int(index),
                "time": str(metadata.iloc[int(index)]["time"]),
                "raw_radar_file": str(metadata.iloc[int(index)]["raw_radar_file"]),
                "raw_satellite_file": str(metadata.iloc[int(index)]["raw_satellite_file"]),
            }
            for index in [*past_indices, *future_indices]
        ]
    return batch, selection


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _select_latest_checkpoint(checkpoint_root: Path) -> Path:
    latest_file = checkpoint_root / "latest.json"
    if latest_file.is_file():
        checkpoint_name = _read_json(latest_file).get("checkpoint")
        if not isinstance(checkpoint_name, str) or not checkpoint_name:
            raise ValueError(f"Missing checkpoint name in {latest_file}")
        checkpoint_dir = checkpoint_root / checkpoint_name
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(f"Latest checkpoint directory does not exist: {checkpoint_dir}")
        return checkpoint_dir

    checkpoint_dirs = [
        path
        for pattern in ("step_*", "checkpoint_*")
        for path in checkpoint_root.glob(pattern)
        if path.is_dir()
        and any((path / filename).is_file() for filename in ("ema.pt", "model.safetensors"))
        and (
            not (path / "metadata.json").is_file()
            or _read_json(path / "metadata.json").get("status", "complete") == "complete"
        )
    ]
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint directories found under {checkpoint_root}")

    def checkpoint_step(path: Path) -> int:
        metadata_file = path / "metadata.json"
        if metadata_file.is_file():
            step = _read_json(metadata_file).get("global_step")
            if isinstance(step, int):
                return step
        suffix = path.name.rsplit("_", maxsplit=1)[-1]
        return int(suffix) if suffix.isdigit() else -1

    return max(checkpoint_dirs, key=checkpoint_step)


def _find_run_config(checkpoint_dir: Path, metadata_file: Path | None) -> Path:
    if metadata_file is not None:
        config_ref = _read_json(metadata_file).get("config")
        if isinstance(config_ref, str) and config_ref:
            config_path = (checkpoint_dir / config_ref).resolve()
            if config_path.is_file():
                return config_path

    for parent in (checkpoint_dir, *checkpoint_dir.parents):
        config_path = parent / "config" / "config_total.yaml"
        if config_path.is_file():
            return config_path
    raise FileNotFoundError(f"Cannot find config/config_total.yaml for checkpoint {checkpoint_dir}")


def resolve_checkpoint_files(checkpoint_path: str | Path) -> CheckpointFiles:
    requested = Path(checkpoint_path).expanduser().resolve()
    if not requested.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {requested}")

    explicit_weight = requested if requested.is_file() and requested.name in {"ema.pt", "model.safetensors"} else None
    if requested.is_file() and requested.name == "latest.json":
        checkpoint_dir = _select_latest_checkpoint(requested.parent)
    elif explicit_weight is not None:
        checkpoint_dir = requested.parent
    elif requested.is_dir() and (requested / "checkpoints").is_dir():
        checkpoint_dir = _select_latest_checkpoint(requested / "checkpoints")
    elif requested.is_dir() and any(
        (requested / filename).is_file()
        for filename in ("metadata.json", "ema.pt", "model.safetensors")
    ):
        checkpoint_dir = requested
    elif requested.is_dir():
        checkpoint_dir = _select_latest_checkpoint(requested)
    else:
        raise ValueError(
            "checkpoint_path must be a run directory, checkpoints directory, step directory, "
            "latest.json, ema.pt, or model.safetensors"
        )

    metadata_file = checkpoint_dir / "metadata.json"
    metadata = metadata_file if metadata_file.is_file() else None
    if metadata is not None and _read_json(metadata).get("status", "complete") != "complete":
        raise ValueError(f"Checkpoint is not marked complete: {checkpoint_dir}")

    ema_file = checkpoint_dir / "ema.pt"
    model_file = checkpoint_dir / "model.safetensors"
    ema = ema_file if ema_file.is_file() else None
    model = model_file if model_file.is_file() else None
    if explicit_weight is None and ema is None and model is None:
        raise FileNotFoundError(f"No ema.pt or model.safetensors found for {checkpoint_dir}")

    config = _find_run_config(checkpoint_dir, metadata)
    return CheckpointFiles(checkpoint_dir, config, metadata, ema, model, explicit_weight)


def load_checkpoint_weights(
    trainer: RainTSNextFrameTransferTrainer,
    files: CheckpointFiles,
    weight_source: str,
) -> tuple[str, Path]:
    if weight_source not in {"auto", "ema", "model"}:
        raise ValueError(f"Unsupported weight source: {weight_source}")

    explicit_kind = None
    if files.explicit_weight is not None:
        explicit_kind = "ema" if files.explicit_weight.name == "ema.pt" else "model"
        if weight_source != "auto" and weight_source != explicit_kind:
            raise ValueError(
                f"Explicit {files.explicit_weight.name} conflicts with --weight-source={weight_source}"
            )
    selected_kind = explicit_kind or ("ema" if weight_source == "auto" and files.ema is not None else weight_source)
    if selected_kind == "auto":
        selected_kind = "model"

    checkpoint_step: int | None = None
    if selected_kind == "ema":
        ema_file = files.explicit_weight if explicit_kind == "ema" else files.ema
        if ema_file is None:
            raise FileNotFoundError(f"EMA weights are unavailable in {files.directory}")
        state = torch.load(ema_file, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(state, Mapping):
            raise ValueError(f"Expected an EMA state mapping in {ema_file}")
        initted = state.get("initted")
        if initted is not None:
            initted_value = bool(initted.item()) if torch.is_tensor(initted) else bool(initted)
            if not initted_value:
                raise ValueError(f"EMA weights are not initialized in {ema_file}")
        ema_step = state.get("step")
        if isinstance(ema_step, int):
            checkpoint_step = ema_step
        elif torch.is_tensor(ema_step) and ema_step.numel() == 1:
            checkpoint_step = int(ema_step.item())
        ema_state = {
            key.removeprefix("ema_model."): value
            for key, value in state.items()
            if isinstance(key, str) and key.startswith("ema_model.")
        }
        if not ema_state:
            raise ValueError(f"No ema_model.* parameters found in {ema_file}")
        model = trainer.accelerator.unwrap_model(trainer.model)
        model.load_state_dict(ema_state, strict=True)
        del ema_state, state
        selected_file = ema_file
    else:
        model_file = files.explicit_weight if explicit_kind == "model" else files.model
        if model_file is None:
            raise FileNotFoundError(f"Model weights are unavailable in {files.directory}")
        model = trainer.accelerator.unwrap_model(trainer.model)
        accelerate.load_checkpoint_in_model(model, str(model_file), strict=True)
        selected_file = model_file

    if files.metadata is not None:
        global_step = _read_json(files.metadata).get("global_step", 0)
        trainer.global_step = int(global_step) if isinstance(global_step, int) else 0
    else:
        if checkpoint_step is None:
            suffix = files.directory.name.rsplit("_", maxsplit=1)[-1]
            checkpoint_step = int(suffix) if suffix.isdigit() else 0
        trainer.global_step = checkpoint_step
    return selected_kind, selected_file


def _build_residual_diagnostics(
    trainer: RainTSNextFrameTransferTrainer,
    context: torch.Tensor,
    target: dict[str, torch.Tensor],
    prediction: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    context_rain = trainer._split_modalities(context)["rain"]
    predicted_rain = prediction["rain"]
    rain_ref = torch.cat([context_rain[:, :, -1:], predicted_rain[:, :, :-1]], dim=2)
    return {
        "rain_ref": rain_ref,
        "pred_delta": prediction["rain_delta"],
        "gt_delta": target["rain"] - rain_ref,
        "final_rain": predicted_rain,
    }


def _rollout_predict_sliding_window(
    trainer: RainTSNextFrameTransferTrainer,
    context: torch.Tensor,
    target: dict[str, torch.Tensor],
    *,
    context_time: torch.Tensor | None,
    future_time: torch.Tensor | None,
    history_window_frames: int,
) -> tuple[dict[str, torch.Tensor], list[int]]:
    model = trainer.accelerator.unwrap_model(trainer.model)
    frame_patch_size = int(model.frame_patch_size)
    total_future_frames = int(target["rain"].shape[2])
    if history_window_frames <= 0 or history_window_frames % frame_patch_size != 0:
        raise ValueError(
            "history_window_frames must be positive and divisible by frame_patch_size, "
            f"got history_window_frames={history_window_frames}, frame_patch_size={frame_patch_size}"
        )
    if history_window_frames + frame_patch_size > int(model.max_frames):
        raise ValueError(
            "Sliding-window model input exceeds max_frames: "
            f"history={history_window_frames}, seed={frame_patch_size}, max_frames={int(model.max_frames)}"
        )
    if total_future_frames <= 0 or total_future_frames % frame_patch_size != 0:
        raise ValueError(
            "Future frames must be positive and divisible by frame_patch_size, "
            f"got future_frames={total_future_frames}, frame_patch_size={frame_patch_size}"
        )
    if context_time is not None or future_time is not None:
        if context_time is None or future_time is None:
            raise ValueError("context_time and future_time must both be provided for sliding rollout")
        if int(context_time.shape[1]) != int(context.shape[2]):
            raise ValueError("context_time length must match context frames")
        if int(future_time.shape[1]) != total_future_frames:
            raise ValueError("future_time length must match future frames")

    context_cur = context[:, :, -history_window_frames:]
    context_time_cur = None if context_time is None else context_time[:, -history_window_frames:]
    rain_predictions: list[torch.Tensor] = []
    rain_deltas: list[torch.Tensor] = []
    model_input_frames: list[int] = []
    residual_enabled = trainer._resolve_rain_residual_cfg()[0]

    for start in range(0, total_future_frames, frame_patch_size):
        end = start + frame_patch_size
        seed = context_cur[:, :, -frame_patch_size:]
        model_input_frames.append(int(context_cur.shape[2]) + frame_patch_size)
        rain_delta = trainer.model.forward_ar(
            context_x=context_cur,
            target_x=seed,
            predict_frames=frame_patch_size,
            strict_target_isolation=bool(trainer.train_cfg.strict_target_isolation),
            return_modality_dict=False,
            context_time=context_time_cur,
            target_time=None if context_time_cur is None else context_time_cur[:, -frame_patch_size:],
        )
        if not torch.is_tensor(rain_delta):
            raise TypeError(f"Expected rain tensor from rain-only decoder, got {type(rain_delta).__name__}")

        rain = rain_delta
        if residual_enabled:
            rain = trainer._apply_rain_residual_output(rain_delta, seed[:, -trainer.rain_c :])
        rain_predictions.append(rain)
        rain_deltas.append(rain_delta)

        if end >= total_future_frames:
            continue

        history = trainer._merge_modalities(
            target["radar"][:, :, start:end],
            target["satellite"][:, :, start:end],
            rain.detach(),
        )
        context_cur = torch.cat([context_cur, history], dim=2)[:, :, -history_window_frames:]
        if context_time_cur is not None:
            if future_time is None:
                raise ValueError("future_time is required when context_time is provided")
            context_time_cur = torch.cat([context_time_cur, future_time[:, start:end]], dim=1)
            context_time_cur = context_time_cur[:, -history_window_frames:]

    return {
        "rain": torch.cat(rain_predictions, dim=2),
        "rain_delta": torch.cat(rain_deltas, dim=2),
    }, model_input_frames


def _parse_frame_stats(path: Path) -> dict[int, dict[str, float]]:
    frame_stats: dict[int, dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("future["):
            continue
        fields = [field.strip() for field in line.split("|")]
        frame_index = int(fields[0].removeprefix("future[").removesuffix("]"))
        values: dict[str, float] = {}
        for field in fields[1:]:
            name, raw_value = field.split("=", maxsplit=1)
            values[name] = float(raw_value)
        frame_stats[frame_index] = values
    if not frame_stats:
        raise ValueError(f"No future frame statistics found in {path}")
    return frame_stats


def _compare_frame_stats(reference_path: Path, sliding_path: Path) -> dict[str, object]:
    reference_stats = _parse_frame_stats(reference_path)
    sliding_stats = _parse_frame_stats(sliding_path)
    frame_indices = sorted(set(reference_stats) & set(sliding_stats))
    if not frame_indices:
        raise ValueError(f"No common future frames between {reference_path} and {sliding_path}")

    metric_names = sorted(
        set.intersection(
            *(set(reference_stats[index]) & set(sliding_stats[index]) for index in frame_indices)
        )
    )
    frames: list[dict[str, object]] = []
    for frame_index in frame_indices:
        frames.append(
            {
                "future_index": frame_index,
                "metrics": {
                    metric_name: {
                        "reference": reference_stats[frame_index][metric_name],
                        "sliding": sliding_stats[frame_index][metric_name],
                        "delta_sliding_minus_reference": (
                            sliding_stats[frame_index][metric_name] - reference_stats[frame_index][metric_name]
                        ),
                    }
                    for metric_name in metric_names
                },
            }
        )

    aggregate: dict[str, dict[str, float]] = {}
    for metric_name in metric_names:
        reference_mean = sum(reference_stats[index][metric_name] for index in frame_indices) / len(frame_indices)
        sliding_mean = sum(sliding_stats[index][metric_name] for index in frame_indices) / len(frame_indices)
        aggregate[metric_name] = {
            "reference_mean": reference_mean,
            "sliding_mean": sliding_mean,
            "delta_sliding_minus_reference": sliding_mean - reference_mean,
        }
    return {
        "reference_stats": str(reference_path),
        "sliding_stats": str(sliding_path),
        "frame_indices": frame_indices,
        "aggregate": aggregate,
        "frames": frames,
    }


def _find_reference_manifest(reference_dir: Path, sample_tag: str) -> tuple[Path, dict[str, object]]:
    candidates = sorted(reference_dir.rglob(f"inference_{sample_tag}_manifest.json"))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one reference manifest for {sample_tag} under {reference_dir}, found {len(candidates)}"
        )
    return candidates[0], _read_json(candidates[0])


def _write_comparison_report(
    output_dir: Path,
    output_prefix: str,
    *,
    reference_label: str,
    reference_manifest: dict[str, object],
    current_manifest: dict[str, object],
    analyses: dict[str, dict[str, object]],
) -> Path:
    metadata_keys = (
        "dataset_index",
        "target_valid_time",
        "context_frames",
        "predict_frames",
        "checkpoint_step",
        "weight_source",
    )
    metadata_comparison = {
        key: {
            "reference": reference_manifest.get(key),
            "sliding": current_manifest.get(key),
            "match": reference_manifest.get(key) == current_manifest.get(key),
        }
        for key in metadata_keys
    }
    report = {
        "reference_label": reference_label,
        "candidate_label": "sliding_window",
        "metadata": metadata_comparison,
        "analyses": analyses,
    }
    json_path = output_dir / f"{output_prefix}_comparison.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    text_lines = [
        f"reference={reference_label}",
        "candidate=sliding_window",
        "",
        "metadata:",
    ]
    for key, values in metadata_comparison.items():
        text_lines.append(
            f"{key} | reference={values['reference']} | sliding={values['sliding']} | match={values['match']}"
        )
    for analysis_name, analysis in analyses.items():
        text_lines.extend(["", f"[{analysis_name}]", "metric | reference_mean | sliding_mean | delta"])
        aggregate = analysis.get("aggregate", {})
        if not isinstance(aggregate, dict):
            continue
        for metric_name, values in aggregate.items():
            if not isinstance(values, dict):
                continue
            text_lines.append(
                f"{metric_name} | {float(values['reference_mean']):.6f} | "
                f"{float(values['sliding_mean']):.6f} | "
                f"{float(values['delta_sliding_minus_reference']):+.6f}"
            )
    text_path = output_dir / f"{output_prefix}_comparison.txt"
    text_path.write_text("\n".join(text_lines), encoding="utf-8")
    print(f"Saved comparison report: {json_path}")
    print(f"Saved comparison summary: {text_path}")
    return json_path


def _save_prediction_visuals(
    trainer: RainTSNextFrameTransferTrainer,
    context: torch.Tensor,
    target: dict[str, torch.Tensor],
    prediction: dict[str, torch.Tensor],
    *,
    future_start: int,
    future_end: int,
    sample_idx: int,
    output_prefix: str,
    rain_diff_vmax: float | None,
    radar_diff_vmax: float | None,
    residual_delta_vmax: float | None,
    save_residual_diagnostic: bool,
    save_radar_conditioning: bool,
) -> None:
    context_modalities, pred_modalities, target_modalities = _prepare_analysis_modalities(
        trainer, context, {"rain": prediction["rain"]}, target
    )
    save_conditioned_timeline(
        trainer=trainer,
        context=context_modalities,
        predicted_rain=pred_modalities["rain"],
        target=target_modalities,
        future_start=future_start,
        future_end=future_end,
        sample_idx=sample_idx,
        output_prefix=f"{output_prefix}_future_{future_start}_{future_end}",
    )
    save_modality_diff_analysis(
        trainer=trainer,
        context_modality=context_modalities["rain"],
        pred_modality=pred_modalities["rain"],
        target_modality=target_modalities["rain"],
        modality_name="rain",
        future_start=future_start,
        future_end=future_end,
        sample_idx=sample_idx,
        output_prefix=output_prefix,
        diff_vmax=rain_diff_vmax,
    )
    if save_radar_conditioning:
        save_conditioning_diff_analysis(
            trainer=trainer,
            context_modality=context_modalities["radar"],
            condition_modality=target_modalities["radar"],
            modality_name="radar",
            future_start=future_start,
            future_end=future_end,
            sample_idx=sample_idx,
            output_prefix=output_prefix,
            diff_vmax=radar_diff_vmax,
        )

    if not save_residual_diagnostic:
        return

    residual = _build_residual_diagnostics(trainer, context, target, prediction)
    radar_ref = _build_reference_sequence(context_modalities["radar"], target_modalities["radar"])
    save_rain_residual_diagnostic_analysis(
        trainer=trainer,
        residual_diag=residual,
        target_rain=target["rain"],
        radar_ref=radar_ref,
        radar_target=target_modalities["radar"],
        future_start=future_start,
        future_end=future_end,
        sample_idx=sample_idx,
        output_prefix=output_prefix,
        delta_vmax=residual_delta_vmax,
        radar_diff_vmax=radar_diff_vmax,
    )


def run_demo(cfg: DictConfig, demo_cfg: dict[str, object], files: CheckpointFiles) -> None:
    _build_demo_cfg_for_trainer(cfg, demo_cfg)
    cfg.train.resume_path = None
    cfg.ema.beta = 0

    print(f"Checkpoint config: {files.config}")
    trainer = RainTSNextFrameTransferTrainer(cfg)
    history_window_frames = int(demo_cfg["history_window_frames"])
    frame_patch_size = trainer._resolve_frame_patch_size()
    _analyze_input_reasonability(
        trainer,
        context_frames=history_window_frames,
        predict_frames=frame_patch_size,
    )
    print(
        "Sliding rollout request: "
        f"initial_context={int(demo_cfg['context_frames'])}, "
        f"history_window={history_window_frames}, total_future={int(demo_cfg['predict_frames'])}"
    )
    source, weight_file = load_checkpoint_weights(trainer, files, str(demo_cfg["weight_source"]))
    print(f"Loaded {source} weights: {weight_file}")

    trainer.model.eval()
    vis_start = int(demo_cfg["vis_future_start"])
    vis_end = int(demo_cfg["vis_future_end"])
    sample_idx = int(demo_cfg["sample_index"])
    dataset_index_value = demo_cfg.get("dataset_index")
    target_valid_time_value = demo_cfg.get("target_valid_time")
    batch, sample_selection = select_inference_window(
        trainer.val_dataset,
        dataset_index=None if dataset_index_value is None else int(dataset_index_value),
        target_valid_time=None if target_valid_time_value is None else str(target_valid_time_value),
    )
    print(
        "Selected validation window: "
        f"dataset_index={sample_selection['dataset_index']}, "
        f"target_valid_time={sample_selection['target_valid_time']}"
    )
    target_time_token = (
        str(sample_selection["target_valid_time"])
        .replace("-", "")
        .replace(":", "")
        .replace(" ", "T")
    )
    sample_tag = f"dataset{int(sample_selection['dataset_index']):06d}_target{target_time_token}"
    compare_growing_context = bool(demo_cfg["compare_growing_context"])
    rain_diff_vmax_value = demo_cfg.get("rain_diff_vmax")
    radar_diff_vmax_value = demo_cfg.get("radar_diff_vmax")
    residual_delta_vmax_value = demo_cfg.get("residual_delta_vmax")
    rain_diff_vmax = None if rain_diff_vmax_value is None else float(rain_diff_vmax_value)
    radar_diff_vmax = None if radar_diff_vmax_value is None else float(radar_diff_vmax_value)
    residual_delta_vmax = None if residual_delta_vmax_value is None else float(residual_delta_vmax_value)

    with torch.no_grad(), trainer.accelerator.autocast():
        context, target, context_time, future_time = trainer._prepare_val_inference_batch(batch)
        prediction, model_input_frames = _rollout_predict_sliding_window(
            trainer,
            context,
            target,
            context_time=context_time,
            future_time=future_time,
            history_window_frames=history_window_frames,
        )
        growing_prediction = None
        if compare_growing_context:
            growing_prediction = trainer._rollout_predict_with_settings(
                context=context,
                total_future_frames=int(target["rain"].shape[2]),
                mode="frame",
                rollout_block_size=trainer._resolve_frame_patch_size(),
                detach_history=True,
                context_time=context_time,
                future_time=future_time,
                future_modalities=target,
                use_gt_future_modalities=True,
                return_rain_delta=True,
                force_rain_only_decode=True,
            )

    total_future_frames = int(prediction["rain"].shape[2])
    if not 0 <= vis_start < vis_end <= total_future_frames:
        raise ValueError(f"Visualization range [{vis_start}, {vis_end}) exceeds {total_future_frames} future frames")
    if not 0 <= sample_idx < int(prediction["rain"].shape[0]):
        raise ValueError(f"sample_index={sample_idx} is outside the inference batch")
    print(f"Sliding-window model input frames: {model_input_frames}")

    sliding_prefix = f"inference_sliding_w{history_window_frames}_{sample_tag}"
    _save_prediction_visuals(
        trainer=trainer,
        context=context,
        target=target,
        prediction=prediction,
        future_start=vis_start,
        future_end=vis_end,
        sample_idx=sample_idx,
        output_prefix=sliding_prefix,
        rain_diff_vmax=rain_diff_vmax,
        radar_diff_vmax=radar_diff_vmax,
        residual_delta_vmax=residual_delta_vmax,
        save_residual_diagnostic=bool(demo_cfg["save_residual_diagnostic"]),
        save_radar_conditioning=True,
    )

    growing_prefix = f"inference_growing_{sample_tag}"
    if growing_prediction is not None:
        _save_prediction_visuals(
            trainer=trainer,
            context=context,
            target=target,
            prediction=growing_prediction,
            future_start=vis_start,
            future_end=vis_end,
            sample_idx=sample_idx,
            output_prefix=growing_prefix,
            rain_diff_vmax=rain_diff_vmax,
            radar_diff_vmax=radar_diff_vmax,
            residual_delta_vmax=residual_delta_vmax,
            save_residual_diagnostic=bool(demo_cfg["save_residual_diagnostic"]),
            save_radar_conditioning=False,
        )

    output_dir = trainer.proj_dir / "val_viz" / f"step_{trainer.global_step:08d}"
    repo_root = Path(__file__).resolve().parents[2]
    resolved_data_dirs: list[str] = []
    for value in demo_cfg["data_dirs"]:
        data_path = Path(str(value)).expanduser()
        if not data_path.is_absolute() and not data_path.exists():
            repo_data_path = repo_root / data_path
            if repo_data_path.exists():
                data_path = repo_data_path
        resolved_data_dirs.append(str(data_path.resolve()))

    manifest = {
        **sample_selection,
        "sample_tag": sample_tag,
        "data_dirs": resolved_data_dirs,
        "context_frames": int(demo_cfg["context_frames"]),
        "predict_frames": int(demo_cfg["predict_frames"]),
        "visualized_future_range": [vis_start, vis_end],
        "checkpoint_config": str(files.config),
        "checkpoint_file": str(weight_file),
        "checkpoint_step": int(trainer.global_step),
        "weight_source": source,
        "rollout_policy": "sliding_window",
        "history_window_frames": history_window_frames,
        "frame_patch_size": trainer._resolve_frame_patch_size(),
        "model_input_frames": model_input_frames,
        "compare_growing_context": compare_growing_context,
    }
    manifest_path = output_dir / f"{sliding_prefix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Saved sample manifest: {manifest_path}")

    sliding_rain_stats = (
        output_dir / f"{sliding_prefix}_rain_sample{sample_idx}_frames_{vis_start}_{vis_end}_diff_stats.txt"
    )
    sliding_residual_stats = (
        output_dir / f"{sliding_prefix}_rain_residual_sample{sample_idx}_frames_{vis_start}_{vis_end}_stats.txt"
    )
    if growing_prediction is not None:
        growing_analyses = {
            "rain_temporal_change": _compare_frame_stats(
                output_dir / f"{growing_prefix}_rain_sample{sample_idx}_frames_{vis_start}_{vis_end}_diff_stats.txt",
                sliding_rain_stats,
            )
        }
        growing_residual_stats = (
            output_dir / f"{growing_prefix}_rain_residual_sample{sample_idx}_frames_{vis_start}_{vis_end}_stats.txt"
        )
        if growing_residual_stats.is_file() and sliding_residual_stats.is_file():
            growing_analyses["rain_residual"] = _compare_frame_stats(
                growing_residual_stats,
                sliding_residual_stats,
            )
        _write_comparison_report(
            output_dir,
            f"{sliding_prefix}_vs_growing",
            reference_label="growing_context_same_checkpoint",
            reference_manifest=manifest,
            current_manifest=manifest,
            analyses=growing_analyses,
        )

    reference_dir_value = demo_cfg.get("reference_dir")
    if reference_dir_value is not None:
        reference_dir = Path(str(reference_dir_value)).expanduser().resolve()
        reference_manifest_path, reference_manifest = _find_reference_manifest(reference_dir, sample_tag)
        for key in ("dataset_index", "target_valid_time", "context_frames", "predict_frames"):
            if reference_manifest.get(key) != manifest.get(key):
                raise ValueError(
                    f"Reference {key} mismatch: reference={reference_manifest.get(key)}, current={manifest.get(key)}"
                )
        if reference_manifest.get("checkpoint_step") != manifest.get("checkpoint_step"):
            print(
                "Reference checkpoint differs from current inference: "
                f"reference_step={reference_manifest.get('checkpoint_step')}, "
                f"current_step={manifest.get('checkpoint_step')}. "
                "Use --compare-growing-context for a same-checkpoint policy comparison."
            )
        reference_prefix = reference_manifest_path.name.removesuffix("_manifest.json")
        reference_rain_stats = (
            reference_manifest_path.parent
            / f"{reference_prefix}_rain_sample{sample_idx}_frames_{vis_start}_{vis_end}_diff_stats.txt"
        )
        reference_analyses = {
            "rain_temporal_change": _compare_frame_stats(reference_rain_stats, sliding_rain_stats)
        }
        reference_residual_stats = (
            reference_manifest_path.parent
            / f"{reference_prefix}_rain_residual_sample{sample_idx}_frames_{vis_start}_{vis_end}_stats.txt"
        )
        if reference_residual_stats.is_file() and sliding_residual_stats.is_file():
            reference_analyses["rain_residual"] = _compare_frame_stats(
                reference_residual_stats,
                sliding_residual_stats,
            )
        _write_comparison_report(
            output_dir,
            f"{sliding_prefix}_vs_reference",
            reference_label=str(reference_manifest_path),
            reference_manifest=reference_manifest,
            current_manifest=manifest,
            analyses=reference_analyses,
        )
    print(f"Rain-only inference complete: {output_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run rain-only sliding-window inference with GT radar and satellite inputs"
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--weight-source", choices=("auto", "ema", "model"), default="auto")
    parser.add_argument("--data-dir", action="append", dest="data_dirs")
    parser.add_argument("--output-dir", default="vis/inference_demo_transfer_sliding")
    parser.add_argument("--context-frames", type=int, default=None)
    parser.add_argument("--predict-frames", type=int, default=None)
    parser.add_argument(
        "--history-window-frames",
        type=int,
        default=7,
        help="Maximum unique history frames retained before adding the autoregressive seed",
    )
    parser.add_argument(
        "--compare-growing-context",
        action="store_true",
        help="Also run the original growing-context rollout with the same checkpoint",
    )
    parser.add_argument(
        "--reference-dir",
        default=None,
        help="Previous result root containing the matching manifest and frame statistics",
    )
    parser.add_argument("--vis-future-start", type=int, default=0)
    parser.add_argument("--vis-future-end", type=int, default=None)
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Index inside the selected batch; direct window selection produces a one-sample batch",
    )
    selection_group = parser.add_mutually_exclusive_group()
    selection_group.add_argument(
        "--dataset-index",
        type=int,
        default=None,
        help="Validation window index; defaults to 0 for deterministic inference",
    )
    selection_group.add_argument(
        "--target-valid-time",
        default=None,
        help="First future frame time in ISO format, for example '2025-08-10 23:30:00'",
    )
    parser.add_argument("--no-residual-diagnostic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    files = resolve_checkpoint_files(args.checkpoint_path)
    cfg = OmegaConf.load(files.config)

    context_frames = int(cfg.dataset.n_past) if args.context_frames is None else args.context_frames
    predict_frames = int(cfg.dataset.n_futures) if args.predict_frames is None else args.predict_frames
    data_dirs = list(cfg.dataset.val_inp_dirs) if args.data_dirs is None else args.data_dirs
    vis_end = predict_frames if args.vis_future_end is None else args.vis_future_end
    demo_cfg: dict[str, object] = {
        "data_dirs": data_dirs,
        "context_frames": context_frames,
        "predict_frames": predict_frames,
        "val_batch_size": 1,
        "val_num_workers": 1,
        "val_persistent_workers": False,
        "val_pin_memory": False,
        "output_dir": args.output_dir,
        "vis_future_start": args.vis_future_start,
        "vis_future_end": vis_end,
        "sample_index": args.sample_index,
        "dataset_index": args.dataset_index,
        "target_valid_time": args.target_valid_time,
        "weight_source": args.weight_source,
        "history_window_frames": args.history_window_frames,
        "compare_growing_context": args.compare_growing_context,
        "reference_dir": args.reference_dir,
        "rain_diff_vmax": None,
        "radar_diff_vmax": None,
        "save_residual_diagnostic": not args.no_residual_diagnostic,
        "residual_delta_vmax": None,
    }
    run_demo(cfg, demo_cfg, files)


if __name__ == "__main__":
    main()
