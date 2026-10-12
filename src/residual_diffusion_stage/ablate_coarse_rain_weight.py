"""Run paired short rainfall-weight ablations and export an auditable comparison."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
from omegaconf import OmegaConf


def compact_metrics(report: dict) -> dict:
    overall = report["deterministic"]["overall"]
    output = {key: overall[key] for key in ("mae", "rmse", "bias")}
    for threshold in ("0.01", "0.1", "0.2", "0.5"):
        for metric in ("csi", "pod", "far"):
            output[f"{metric}@{threshold}"] = overall[f"threshold@{threshold}"][metric]
    for lead, metrics in report["evolution"].items():
        for key in ("active_delta_mae", "core_0.5_mae", "change_amplitude_ratio",
                    "decay_direction_recall", "fss_0.1_scale_8"):
            output[f"t{lead}_{key}"] = metrics[key]
    return output


def export_report(output: Path, source: Path) -> None:
    locations = {
        "initial": source / "fixed_validation/step_00000000",
        "alpha_2": source / "fixed_validation/step_00000250",
        "alpha_0p5": output / "alpha_0p5/fixed_validation/step_00000250",
        "alpha_0": output / "alpha_0/fixed_validation/step_00000250",
    }
    reports = {name: json.loads((path / "metrics.json").read_text()) for name, path in locations.items()}
    reference = reports["initial"]
    for name, report in reports.items():
        if report["samples"] != 60 or report["manifest_source"] != reference["manifest_source"]:
            raise ValueError(f"Validation protocol mismatch: {name}")
    rows = {name: compact_metrics(report) for name, report in reports.items()}
    (output / "comparison.json").write_text(json.dumps({
        "protocol": {
            "steps": 250, "scheduler_horizon": 1000, "seed": 2025,
            "variable": "train.loss.rain_region_weight.alpha",
            "reference": str(source), "scope": "Single seed; 60 fixed rain-filtered validation windows",
            "limitation": "Aggregate development-set comparison, not a significance test or full generalization test",
        }, "metrics": rows, "full_reports": reports,
    }, indent=2), encoding="utf-8")
    columns = ["mae", "rmse", "bias", "csi@0.01", "csi@0.1", "csi@0.5", "far@0.01", "far@0.5",
               "t4_core_0.5_mae", "t4_active_delta_mae", "t4_decay_direction_recall"]
    lines = ["# 雨强加权单项消融：250 步", "", "同一初始化、种子、训练日程和验证样本；仅改变 alpha。",
             "", "| 设置 | " + " | ".join(columns) + " |", "|---|" + "---|" * len(columns)]
    for name, row in rows.items():
        lines.append(f"| {name} | " + " | ".join(f"{row[key]:.5f}" for key in columns) + " |")
    candidates = [name for name in ("alpha_0p5", "alpha_0") if
                  rows[name]["mae"] < rows["alpha_2"]["mae"] and
                  rows[name]["far@0.01"] < rows["alpha_2"]["far@0.01"] and
                  rows[name]["csi@0.5"] >= rows["initial"]["csi@0.5"]]
    lines += ["", "## 初筛提示", "",
              "降低 MAE/弱雨 FAR 且强雨 CSI 不低于训练前的候选：" + (", ".join(candidates) or "无"),
              "", "以上是显式规则筛选，不代表图像质量或泛化已经达标。仍需结合边界、消散和位置误差人工判断。",
              "未自动修改下一轮训练配置，也未启动 corrector/diffusion。"]
    for index in (856, 1115, 1170):
        fields = {}
        truth = None
        for name, path in locations.items():
            with np.load(path / f"rain_index_{index}.npz") as data:
                if truth is None:
                    truth = data["GT"].copy()
                else:
                    np.testing.assert_array_equal(truth, data["GT"])
                fields[name] = data["Coarse"].copy()
        cmap = ListedColormap(["white", "#add8e6", "#0000ff", "#008000", "#ffff00"])
        norm = BoundaryNorm([0, 0.01, 0.2, 0.5, 1, 1000], cmap.N)
        fig, axes = plt.subplots(5, 4, figsize=(14, 16), constrained_layout=True)
        for row, (name, rain) in enumerate({"GT": truth, **fields}.items()):
            for lead in range(4):
                axes[row, lead].imshow(rain[lead], cmap=cmap, norm=norm, interpolation="nearest")
                axes[row, lead].set_title(f"{name} | t+{lead + 1}")
                axes[row, lead].set_axis_off()
        fig.suptitle(f"Paired rainfall-weight ablation | sample {index} | mm/min | white <0.01")
        fig.savefig(output / f"comparison_{index}.png", dpi=130)
        plt.close(fig)
        lines += ["", f"![样本 {index}](comparison_{index}.png)"]
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("runs/coarse_evolution_20261005"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if args.report_only:
        export_report(output, source)
        return
    output.mkdir(parents=True, exist_ok=False)
    config_path = source / "config/config_total.yaml"
    cfg = OmegaConf.load(config_path)
    assert cfg.train.seed == 2025 and cfg.train.loss.rain_region_weight.alpha == 2.0
    assert cfg.train.scheduler.num_training_steps == 1000
    assert cfg.train.resume_path is None
    assert (source / "fixed_validation/step_00000250/metrics.json").is_file()
    initial = json.loads((source / "fixed_validation/step_00000000/metrics.json").read_text())
    manifest = Path(initial["manifest_source"])
    provenance = {
        "source_config": str(config_path), "source_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "training_budget": "250 optimizer steps per arm; 1000-step LR schedule preserved",
        "arms": {"alpha_2": "existing control at step 250", "alpha_0p5": 0.5, "alpha_0": 0.0},
    }
    (output / "protocol.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    for name, alpha in (("alpha_0p5", 0.5), ("alpha_0", 0.0)):
        command = [
            "bash", "src/residual_diffusion_stage/train_coarse_evolution_4090.sh",
            "--config-path", str(config_path.parent), "--config-name", "config_total",
            "train.max_steps=250", "train.scheduler.num_training_steps=1000",
            f"train.loss.rain_region_weight.alpha={alpha}", f"hydra.run.dir={output / name}",
        ]
        polls = 0
        while True:
            memory = subprocess.check_output([
                "nvidia-smi", "-i", "GPU-53d5627c-0a70-b40d-031b-51c20452acb6",
                "--query-gpu=memory.free", "--format=csv,noheader,nounits",
            ], text=True)
            free_mib = float(memory.strip())
            if free_mib >= 14.5 * 1024:
                break
            (output / "status.json").write_text(json.dumps({
                "state": "waiting_for_gpu", "arm": name, "free_mib": free_mib,
                "required_free_mib": 14.5 * 1024,
            }), encoding="utf-8")
            if polls % 10 == 0:
                print(f"Waiting for RTX 4090: free={free_mib / 1024:.2f} GiB; need 14.5 GiB", flush=True)
            polls += 1
            time.sleep(30)
        (output / "status.json").write_text(json.dumps({"state": "running", "arm": name}), encoding="utf-8")
        print(f"Starting {name}: {' '.join(command)}", flush=True)
        with (output / f"{name}.launch.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
        if result.returncode:
            (output / "status.json").write_text(
                json.dumps({"state": "failed", "arm": name, "returncode": result.returncode}), encoding="utf-8",
            )
            raise RuntimeError(f"Training failed; inspect {output / f'{name}.launch.log'}")
        before = json.loads((output / name / "fixed_validation/step_00000000/metrics.json").read_text())
        np.testing.assert_allclose(
            list(compact_metrics(before).values()), list(compact_metrics(initial).values()), rtol=1e-6, atol=1e-8,
        )
        final = json.loads((output / name / "fixed_validation/step_00000250/metrics.json").read_text())
        if final["step"] != 250:
            raise ValueError(f"Incomplete arm: {name}")
        print(f"Finished {name}", flush=True)
    export_report(output, source)
    (output / "status.json").write_text(json.dumps({"state": "complete"}), encoding="utf-8")
    print(f"Complete: {output / 'comparison.md'}", flush=True)


if __name__ == "__main__":
    main()
