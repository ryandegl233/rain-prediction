"""Read-only checkpoint diagnostics: false-alarm origins and branch interventions."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import default_collate

from src.residual_diffusion_stage.compare_coarse_smoke import build_evaluator
from src.residual_diffusion_stage.metrics import DeterministicMetricAccumulator


def origin_masks(pred: torch.Tensor, truth: torch.Tensor, history: torch.Tensor, threshold: float) -> list[dict[str, torch.Tensor]]:
    past_wet = history >= threshold
    last_wet = past_wet[:, :, -1]
    ever_wet = past_wet.any(dim=2)
    near = F.max_pool2d(ever_wet.float(), 17, stride=1, padding=8).bool()
    near_small = F.max_pool2d(ever_wet.float(), 5, stride=1, padding=2).bool()
    rows = []
    for lead in range(pred.shape[2]):
        current = pred[:, :, lead]
        false = (current >= threshold) & (truth[:, :, lead] < threshold)
        previous_truth = last_wet if lead == 0 else truth[:, :, lead - 1] >= threshold
        previous_pred = last_wet if lead == 0 else pred[:, :, lead - 1] >= threshold
        rows.append({
            "false_pixels": false,
            "last_observed_wet": false & last_wet,
            "earlier_history_wet_only": false & ~last_wet & ever_wet,
            "all_history_dry_near_8px": false & ~ever_wet & near,
            "all_history_dry_far_8px": false & ~ever_wet & ~near,
            "all_history_dry_far_2px": false & ~ever_wet & ~near_small,
            "previous_truth_wet": false & previous_truth,
            "persistent_previous_false_alarm": false & ~previous_truth & previous_pred,
            "new_from_previous_pred_dry": false & ~previous_truth & ~previous_pred,
            "weak_under_0.03": false & (current < 0.03),
            "weak_under_0.05": false & (current < 0.05),
            "weak_under_0.1": false & (current < 0.1),
        })
    return rows


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=60)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest_path = Path("runs/ccc_fixed_comparison_20261005/comparison.json")
    full_manifest = json.loads(manifest_path.read_text())["manifest"]
    if not 0 < args.samples <= len(full_manifest):
        raise ValueError("samples exceeds the fixed manifest")
    manifest = full_manifest[:args.samples]
    thresholds = [0.01, 0.03, 0.1, 0.2, 0.5]
    results = {}
    trainer = None
    try:
        for arm in ("without_ccc", "with_ccc"):
            trainer = build_evaluator(args.run / arm / "config/config_total.yaml",
                                      args.run / arm / "checkpoints/checkpoint_4", args.output / f"{arm}_runtime")
            model = trainer.accelerator.unwrap_model(trainer.model)
            cross_gate = model.cross_modal_adapter.gate.clone()
            local_gate = model.local_window_refiner.gate.clone()
            variants = ["normal", "cross_off", "local_off", "both_off"] if arm == "without_ccc" else ["normal"]
            accumulators = {name: DeterministicMetricAccumulator(thresholds, 4, torch.device("cpu")) for name in variants}
            counts = {f"{threshold:g}": [{} for _ in range(4)] for threshold in thresholds}
            feature_sums = {name: {"input_sq": 0.0, "delta_sq": 0.0, "calls": 0} for name in ("cross", "local")}
            record_features = False
            handles = []
            for name, module in (("cross", model.cross_modal_adapter), ("local", model.local_window_refiner)):
                def measure(module: torch.nn.Module, inputs: tuple, output: torch.Tensor, label: str = name) -> None:
                    if record_features:
                        values = feature_sums[label]
                        values["input_sq"] += float(inputs[0].float().square().sum())
                        values["delta_sq"] += float((output.float() - inputs[0].float()).square().sum())
                        values["calls"] += 1
                handles.append(module.register_forward_hook(measure))
            per_sample = []
            for ordinal, item in enumerate(manifest):
                batch = default_collate([trainer.val_dataset[item["index"]]])
                digest = hashlib.sha256()
                for key in sorted(batch):
                    if torch.is_tensor(batch[key]):
                        digest.update(key.encode())
                        digest.update(batch[key].contiguous().numpy().tobytes())
                if digest.hexdigest() != item["sha256"]:
                    raise ValueError(f"Input checksum mismatch: {item['index']}")
                context, target, context_time, future_time = trainer._prepare_val_inference_batch(batch)
                truth = trainer._denormalize_rain_for_metrics(target["rain"].float()).clamp_min(0).cpu()
                history = trainer._denormalize_rain_for_metrics(context[:, -trainer.rain_c:].float()).clamp_min(0).cpu()
                fields = {"GT": truth[0, 0].numpy(), "history": history[0, 0].numpy()}
                row = {"index": item["index"]}
                for variant in variants:
                    model.cross_modal_adapter.gate.copy_(cross_gate if variant not in ("cross_off", "both_off") else torch.zeros_like(cross_gate))
                    model.local_window_refiner.gate.copy_(local_gate if variant not in ("local_off", "both_off") else torch.zeros_like(local_gate))
                    record_features = variant == "normal"
                    with trainer.accelerator.autocast():
                        prediction = trainer._rollout_predict_with_settings(
                            context=context, total_future_frames=4, mode="frame", rollout_block_size=1,
                            detach_history=True, context_time=context_time, future_time=future_time,
                            future_modalities=None, use_gt_future_modalities=False,
                        )["rain"]
                    pred = trainer._denormalize_rain_for_metrics(prediction.float()).clamp_min(0).cpu()
                    if not torch.isfinite(pred).all():
                        raise ValueError("Nonfinite prediction")
                    accumulators[variant].update(pred, truth)
                    fields[variant] = pred[0, 0].numpy()
                    row[variant] = {"mae": float((pred - truth).abs().mean())}
                    if variant == "normal":
                        for threshold in thresholds:
                            masks = origin_masks(pred, truth, history, threshold)
                            for lead, values in enumerate(masks):
                                total = counts[f"{threshold:g}"][lead]
                                for key, value in values.items():
                                    total[key] = total.get(key, 0) + int(value.sum())
                    else:
                        delta = np.abs(fields[variant] - fields["normal"])
                        row[variant]["difference_from_normal_mae"] = float(delta.mean())
                if ordinal < 3:
                    np.savez_compressed(args.output / f"{arm}_index_{item['index']}.npz", **fields)
                per_sample.append(row)
                if (ordinal + 1) % 10 == 0:
                    print(f"{arm}: {ordinal + 1}/{len(manifest)}", flush=True)
            for handle in handles:
                handle.remove()
            model.cross_modal_adapter.gate.copy_(cross_gate)
            model.local_window_refiner.gate.copy_(local_gate)
            results[arm] = {
                "metrics": {key: accumulator.result() for key, accumulator in accumulators.items()},
                "false_alarm_origins": counts,
                "feature_effects": {key: {**value, "delta_input_rms_ratio": (value["delta_sq"] / max(value["input_sq"], 1e-30)) ** 0.5} for key, value in feature_sums.items()},
                "gates": {"cross": float(cross_gate), "local": float(local_gate)}, "per_sample": per_sample,
            }
            (args.output / "diagnosis.json").write_text(json.dumps({"samples": len(manifest), "results": results}, indent=2))
            trainer._close_tensorboard_writer()
            del trainer, model, module, prediction, context, target
            trainer = None
            torch.cuda.empty_cache()
        print(f"Completed: {args.output / 'diagnosis.json'}", flush=True)
    finally:
        if trainer is not None:
            trainer._close_tensorboard_writer()


if __name__ == "__main__":
    main()
