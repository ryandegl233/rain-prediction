import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from src.baselines_stage3.models import LucasKanade, Persistence, Stage3Reference

ROOT = Path(__file__).resolve().parents[2]


def validate_config(cfg: DictConfig) -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Use a single process and a single GPU for this entry.")
    if (cfg.dataset.n_past, cfg.dataset.n_futures, cfg.dataset.time_interval) != (4, 4, 30):
        raise ValueError("Stage 3 comparison requires 4 history frames, 4 leads, and 30-minute intervals.")
    if cfg.dataset.get("modality_zero_centering", False) or any(
        cfg.dataset[split].get("modality_zero_centering", False) for split in ("train", "val")
    ):
        raise ValueError("Metrics and LK require uncentered dataset rain units.")
    val = cfg.dataset.val
    if (
        any(val.get(key, False) for key in ("shuffle", "drop_last", "aug_enabled", "is_cycled"))
        or val.iter_index_mode != "sequential"
    ):
        raise ValueError("Evaluation requires sequential, complete, unaugmented validation windows.")
    if (
        not cfg.train.next_pred.missing_modality.anchor_first_frame
        and cfg.get("baseline_model", {}).get("name") != "npm"
    ):
        raise ValueError("This comparison requires a fully observed first historical frame.")
    if cfg.evaluation.checkpoint_kind not in ("model", "ema"):
        raise ValueError("checkpoint_kind must be model or ema.")


def check_data(cfg: DictConfig, splits: tuple[str, ...]) -> None:
    for split in splits:
        data = cfg.dataset[split]
        for name in data.get("inp_dirs", []):
            path = Path(name)
            if not path.is_absolute():
                path = ROOT / path
            if not path.is_dir():
                raise FileNotFoundError(f"Missing {split} data directory: {path}")
            if data.get("rain_ratio_filter_enabled", False):
                metadata = path / data.rain_ratio_filter_file_name
                if not metadata.is_file():
                    raise FileNotFoundError(
                        f"Required rain filter metadata is missing; refusing to silently disable filtering: {metadata}"
                    )


def classical_models(cfg: DictConfig) -> dict[str, nn.Module]:
    models = {}
    for name in cfg.baseline.methods:
        if name in models:
            raise ValueError(f"Duplicate method: {name}")
        if name == "persistence":
            models[name] = Persistence()
        elif name == "lk":
            if version("pysteps") != "1.21.5":
                raise ValueError("Install pysteps==1.21.5 from requirements-baselines.txt.")
            import pysteps

            models[name] = LucasKanade(cfg.baseline.lk.rain_threshold, cfg.baseline.lk.size_opening)
        else:
            raise ValueError(f"Unknown classical baseline: {name}")
    return models


def checkpoint_state(path: Path, kind: str) -> dict[str, torch.Tensor]:
    if path.is_dir():
        path = path / ("ema.pt" if kind == "ema" else "model.safetensors")
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path))
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    if kind == "ema":
        state = {key.removeprefix("ema_model."): value for key, value in state.items() if key.startswith("ema_model.")}
        if not state:
            raise ValueError("No ema_model.* weights found in EMA checkpoint.")
    return state


def load_reference(cfg: DictConfig) -> Stage3Reference:
    network = hydra.utils.instantiate(cfg.rain_prediction_model)
    network.load_state_dict(
        checkpoint_state(Path(cfg.evaluation.reference_checkpoint), cfg.evaluation.checkpoint_kind), strict=True
    )
    network.to(cfg.evaluation.device).eval()
    return Stage3Reference(network, cfg)


def complete_report(report: dict, cfg: DictConfig, expected: int) -> dict:
    complete = cfg.evaluation.max_batches == -1
    if complete and report["samples"] != expected:
        raise ValueError(f"Validation coverage mismatch: expected {expected}, observed {report['samples']}.")
    data = OmegaConf.to_container(cfg.dataset.val, resolve=True)
    report.update(
        complete=complete,
        expected_samples=expected,
        max_batches=cfg.evaluation.max_batches,
        config=OmegaConf.to_container(cfg, resolve=True),
        data_config_sha256=hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(),
        source_zip_sha256=json.loads((ROOT / "SOURCE_MANIFEST.json").read_text())["source_zip_sha256"],
        torch_version=torch.__version__,
        pysteps_version=version("pysteps") if "lk" in cfg.baseline.methods else None,
    )
    return report
