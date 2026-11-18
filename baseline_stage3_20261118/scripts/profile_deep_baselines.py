import json
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.baselines_stage3.deep.common import DeepForecaster


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    rows = []
    for method in ("convlstm", "predrnn_v2", "simvp", "earthformer", "alphapre", "npm"):
        with initialize_config_dir(config_dir=str(root / "src/config/ts_rain_train"), version_base=None):
            cfg = compose(config_name="baseline_deep_train", overrides=[f"baseline_model={method}"])
        with torch.device("meta"):
            model = DeepForecaster(cfg)
        parameters = sum(value.numel() for value in model.parameters())
        inactive = sum(
            value.numel()
            for name, value in model.named_parameters()
            if method == "convlstm" and (".conv_o." in name or ".cell_list." in name and ".conv_last." in name)
        )
        row = {
            "method": method,
            "parameters": parameters,
            "inactive_upstream_parameters": inactive,
            "active_parameters": parameters - inactive,
            "ratio_to_reference": parameters / 44_162_136,
            "model": OmegaConf.to_container(cfg.baseline_model, resolve=True),
            "training_steps": cfg.train.max_steps,
            "protocol": "fixed_origin_satellite_only" if method == "npm" else "progressively_observed_radar_satellite",
            "measurement": "Exact instantiated tensor dimensions on meta device; no full-size training memory estimate.",
        }
        assert abs(row["ratio_to_reference"] - 1) <= 0.1
        rows.append(row)
        print(f"{method}: {parameters:,} ({row['ratio_to_reference']:.4f} x reference)")
    (root / "artifacts").mkdir(exist_ok=True)
    (root / "artifacts/deep_parameter_profiles.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
