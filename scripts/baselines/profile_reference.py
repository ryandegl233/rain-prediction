import argparse
import hashlib
import json
from pathlib import Path

import hydra
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="rain_trainer_ts_next_frame")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    config_dir = root / "src/config/ts_rain_train"

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name=args.config_name)
    torch.manual_seed(2025)
    model = hydra.utils.instantiate(config.rain_prediction_model).cpu().eval()
    parameter_counts = {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "direct_parameters": sum(parameter.numel() for parameter in model.parameters(recurse=False)),
        "modules": {
            name: sum(parameter.numel() for parameter in module.parameters()) for name, module in model.named_children()
        },
    }
    source_paths = [
        config_dir / f"{args.config_name}.yaml",
        config_dir / "rain_prediction_model/causal_patch_transformer_next_frame.yaml",
        root / "src/networks/time_series/causal_patch_transformer_next_frame.py",
        root / "src/networks/time_series/causal_patch_transformer_diffusion.py",
        root / "src/networks/modules/reconstruction.py",
    ]
    report = {
        "config_name": args.config_name,
        "torch_version": torch.__version__,
        "measurement": "CPU construction; registered parameters; no forward or training executed",
        "model": OmegaConf.to_container(config.rain_prediction_model, resolve=True),
        "data": {
            key: config.dataset[key]
            for key in ("n_past", "n_futures", "img_size", "time_interval", "modality_zero_centering")
        },
        "configured_train_future_observations": config.train.next_pred.rollout_branch.use_gt_future_modalities,
        "configured_val_future_observations": config.val.rollout_use_gt_future_modalities,
        "agreed_baseline_protocol": "fixed origin; history only; future observation forcing disabled",
        "parameters": parameter_counts,
        "source_sha256": {
            path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(parameter_counts, indent=2))
    print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
