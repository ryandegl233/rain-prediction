import json
from pathlib import Path

import hydra
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(root / "src/config/ts_rain_train"), version_base=None):
        cfg = compose(config_name="baseline_stage3_train")
    model = hydra.utils.instantiate(cfg.rain_prediction_model).cpu().eval()
    report = {
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "modules": {name: sum(parameter.numel() for parameter in module.parameters()) for name, module in model.named_children()},
        "model_config": OmegaConf.to_container(cfg.rain_prediction_model, resolve=True),
        "training_steps": cfg.train.max_steps,
        "loss_weights": OmegaConf.to_container(cfg.train.loss_weights, resolve=True),
        "rollout": OmegaConf.to_container(cfg.train.next_pred.rollout_branch, resolve=True),
        "initialization": cfg.train.init_model_path,
        "torch_version": torch.__version__,
        "measurement": "CPU model construction and parameter count; no full-size forward or GPU benchmark",
    }
    destination = root / "artifacts/reference_profile.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{report['model_class']}: {report['parameters']:,} parameters; saved to {destination}")


if __name__ == "__main__":
    main()
