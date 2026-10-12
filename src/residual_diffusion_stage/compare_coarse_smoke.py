"""Paired fixed-sample evaluation of baseline and CCC smoke checkpoints."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap
from omegaconf import OmegaConf, open_dict
from safetensors.torch import load_model
from torch.utils.data import default_collate

from src.residual_diffusion_stage.metrics import DeterministicMetricAccumulator
from src.trainer.rain_trainer_ts_next_frame import RainTSNextFrameTrainer


def build_evaluator(config_path: Path, checkpoint: Path, runtime: Path) -> RainTSNextFrameTrainer:
    cfg = OmegaConf.load(config_path)
    with open_dict(cfg):
        cfg.train.init_model_path = None
        cfg.train.resume_path = None
        cfg.train.proj_dir = str(runtime)
        cfg.train.debug = True
        cfg.train.strict_target_isolation = True
        cfg.train.log.log_with_time = False
        cfg.train.log.run_comment = ""
        cfg.train.gan.enabled = False
        cfg.ema.beta = 0.0
        cfg.accelerator.project_config.project_dir = str(runtime)
        cfg.accelerator.project_config.logging_dir = str(runtime / "tensorboard")
        cfg.accelerator.gradient_accumulation_plugin.num_steps = 1
        for split in (cfg.dataset.train, cfg.dataset.val):
            split.batch_size = 1
            split.num_workers = 0
            split.persistent_workers = False
        cfg.dataset.val.shuffle = False
        cfg.dataset.val.iter_index_mode = "sequential"
        cfg.dataset.val.aug_enabled = False
        cfg.dataset.val.drop_last = False
        cfg.val.rollout_mode = "frame"
        cfg.val.rollout_block_size = 1
        cfg.val.rollout_history_detach = True
        cfg.val.rollout_use_gt_future_modalities = False
        cfg.val.save_visuals = False
        cfg.val.after_roll_next.enabled = False
    trainer = RainTSNextFrameTrainer(cfg)
    model = trainer.accelerator.unwrap_model(trainer.model)
    load_model(model, str(checkpoint / "model.safetensors"), strict=True)
    trainer.model.eval().requires_grad_(False)
    return trainer


def save_preview(path: Path, fields: dict[str, np.ndarray], sample_index: int) -> None:
    colors = ["white", "#add8e6", "#0000ff", "#008000", "#ffff00"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([0, 0.01, 0.2, 0.5, 1.0, 1000.0], cmap.N)
    fig, axes = plt.subplots(3, 4, figsize=(14, 10), constrained_layout=True)
    for row, (name, rain) in enumerate(fields.items()):
        for lead in range(4):
            axes[row, lead].imshow(rain[lead], cmap=cmap, norm=norm, interpolation="nearest")
            axes[row, lead].set_title(f"{name} | t+{lead + 1}")
            axes[row, lead].set_axis_off()
    fig.suptitle(
        f"Fixed validation sample {sample_index} | mm/min\n"
        "white <0.01; light blue 0.01-0.2; blue 0.2-0.5; green 0.5-1; yellow >=1"
    )
    fig.savefig(path, dpi=140)
    plt.close(fig)
    np.savez_compressed(path.with_suffix(".npz"), **fields)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--smoke-config", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--smoke-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()
    if args.samples < 1:
        raise ValueError("samples must be positive")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    baseline = build_evaluator(args.baseline_config, args.baseline_checkpoint, out / "baseline_runtime")
    smoke = build_evaluator(args.smoke_config, args.smoke_checkpoint, out / "smoke_runtime")
    base_data = OmegaConf.to_container(baseline.dataset_cfg.val, resolve=True)
    smoke_data = OmegaConf.to_container(smoke.dataset_cfg.val, resolve=True)
    if base_data != smoke_data:
        raise ValueError("Validation data configurations differ; paired comparison requires identical preprocessing")
    population = len(baseline.val_dataset.indices_pairs)
    indices = torch.randperm(population, generator=torch.Generator().manual_seed(args.seed))[
        : min(args.samples, population)
    ].tolist()
    thresholds = [0.01, 0.1, 0.2, 0.5]
    accumulators = {
        name: DeterministicMetricAccumulator(thresholds, 4, torch.device("cpu"))
        for name in ("baseline", "smoke")
    }
    manifest = []
    per_sample = []
    strongest_mass = -1.0
    strongest = None
    for ordinal, index in enumerate(indices):
        batch = default_collate([baseline.val_dataset[index]])
        digest = hashlib.sha256()
        for key in sorted(batch):
            if torch.is_tensor(batch[key]):
                digest.update(key.encode())
                digest.update(batch[key].contiguous().numpy().tobytes())
        manifest.append({
            "index": index,
            "sha256": digest.hexdigest(),
            "timestamps": batch["time_future_timestamp"].tolist(),
        })
        predictions = {}
        target_reference = None
        for name, trainer in (("baseline", baseline), ("smoke", smoke)):
            with trainer.accelerator.autocast():
                context, target, context_time, target_time = trainer._prepare_val_inference_batch(batch)
                prediction = trainer._rollout_predict_with_settings(
                    context=context, total_future_frames=target["rain"].shape[2],
                    mode="frame", rollout_block_size=1, detach_history=True,
                    context_time=context_time, future_time=target_time,
                    future_modalities=None, use_gt_future_modalities=False,
                )["rain"]
            rain = trainer._denormalize_rain_for_metrics(prediction.float()).clamp_min(0).cpu()
            truth = trainer._denormalize_rain_for_metrics(target["rain"].float()).clamp_min(0).cpu()
            if not torch.isfinite(rain).all():
                raise RuntimeError(f"Nonfinite prediction: {name}, sample={index}")
            if target_reference is not None:
                torch.testing.assert_close(truth, target_reference, rtol=0, atol=0)
            target_reference = truth
            accumulators[name].update(rain, truth)
            predictions[name] = rain
        per_sample.append({
            "index": index,
            "target_mean": truth.mean().item(),
            **{f"{name}_mae": (rain - truth).abs().mean().item() for name, rain in predictions.items()},
        })
        fields = {"GT": truth[0, 0].numpy(), **{name: rain[0, 0].numpy() for name, rain in predictions.items()}}
        if ordinal < 3:
            save_preview(out / f"sample_{ordinal:02d}_index_{index}.png", fields, index)
        mass = truth.sum().item()
        if mass > strongest_mass:
            strongest_mass = mass
            strongest = (index, fields)
        if (ordinal + 1) % 10 == 0:
            print(f"Paired samples: {ordinal + 1}/{len(indices)}", flush=True)
    if strongest is not None:
        save_preview(out / "largest_target_rain_mass.png", strongest[1], strongest[0])
    report = {
        "protocol": {
            "unit": "mm/min", "thresholds": thresholds, "seed": args.seed,
            "samples": len(indices), "eligible_validation_windows": population,
            "selection": "seeded uniform window subset; existing rain-area filter retained",
            "aggregation": "pixel counts pooled across samples; overall and by lead",
            "rollout": "free rollout; no observed future radar, satellite or rain",
            "weights": "raw checkpoint weights, EMA disabled",
            "preview_selection": "first 3 seeded samples plus greatest observed total rainfall in evaluated subset",
        },
        "checkpoints": {"baseline": str(args.baseline_checkpoint), "smoke": str(args.smoke_checkpoint)},
        "manifest": manifest, "per_sample": per_sample,
        **{name: accumulator.result() for name, accumulator in accumulators.items()},
    }
    (out / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for trainer in (baseline, smoke):
        trainer._close_tensorboard_writer()
    print(json.dumps({name: report[name]["overall"] for name in accumulators}, indent=2), flush=True)
    print(f"Report: {out / 'comparison.json'}", flush=True)


if __name__ == "__main__":
    main()
