import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from src.baselines.classical import LucasKanade, Persistence
from src.baselines.reference import NextFrameForecaster


def validate_protocol(cfg: DictConfig) -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This baseline entry supports one process; launch on one GPU or CPU.")
    if (cfg.dataset.n_past, cfg.dataset.n_futures, cfg.dataset.time_interval) != (4, 4, 30):
        raise ValueError("Baseline protocol requires 4 history frames, 4 forecasts and 30-minute intervals.")
    if cfg.dataset.get("modality_zero_centering", False) or cfg.dataset.val.get("modality_zero_centering", False):
        raise ValueError("Baseline evaluation requires uncentered rain in dataset units.")
    if cfg.train.next_pred.rollout_branch.get("use_gt_future_modalities", False):
        raise ValueError("Training rollout must disable future observed modalities.")
    if cfg.val.get("rollout_use_gt_future_modalities", False):
        raise ValueError("Evaluation cannot use future observed modalities.")
    val = cfg.dataset.val
    if (
        val.get("shuffle", False)
        or val.get("drop_last", False)
        or val.get("aug_enabled", False)
        or val.get("is_cycled", False)
    ):
        raise ValueError("Evaluation requires unshuffled, unaugmented, noncycled windows without drop_last.")
    if val.get("iter_index_mode") != "sequential":
        raise ValueError("Evaluation requires iter_index_mode=sequential.")


def build_classical(cfg: DictConfig) -> dict[str, nn.Module]:
    models = {}
    for name in cfg.baseline.methods:
        if name in models:
            raise ValueError(f"Duplicate baseline: {name}")
        if name == "persistence":
            models[name] = Persistence()
        elif name == "lk":
            if version("pysteps") != "1.21.5":
                raise ValueError("Install the pinned runtime: pysteps==1.21.5.")
            import pysteps

            models[name] = LucasKanade(
                rain_threshold=cfg.baseline.lk.rain_threshold,
                size_opening=cfg.baseline.lk.size_opening,
            )
        else:
            raise ValueError(f"Unknown baseline: {name}")
    return models


def wrap_reference(network: nn.Module, cfg: DictConfig) -> NextFrameForecaster:
    residual = cfg.train.loss.get("rain_residual", {})
    wrapper = NextFrameForecaster(
        network,
        residual=residual.get("enabled", False),
        clamp_output=residual.get("clamp_output", False),
        min_value=residual.get("min_value", 0.0),
        strict_target_isolation=cfg.train.get("strict_target_isolation", False),
        time_interval=cfg.dataset.time_interval,
    )
    return wrapper.train(network.training)


def load_reference(cfg: DictConfig) -> nn.Module:
    path = Path(cfg.evaluation.reference_checkpoint)
    if path.is_dir():
        path = path / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path))
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    kind = cfg.evaluation.checkpoint_kind
    if kind == "ema":
        state = {key.removeprefix("ema_model."): value for key, value in state.items() if key.startswith("ema_model.")}
        if not state:
            raise ValueError("EMA checkpoint must contain ema_model.* parameters.")
    elif kind != "model":
        raise ValueError("checkpoint_kind must be model or ema.")
    network = hydra.utils.instantiate(cfg.rain_prediction_model)
    network.load_state_dict(state, strict=True)
    network.to(cfg.evaluation.device).eval()
    return wrap_reference(network, cfg)


def complete_report(report: dict, cfg: DictConfig, expected_samples: int) -> dict:
    complete = cfg.evaluation.max_batches == -1
    if complete and report["samples"] != expected_samples:
        raise ValueError(f"Expected {expected_samples} validation windows, observed {report['samples']}.")
    config = OmegaConf.to_container(cfg, resolve=True)
    data = OmegaConf.to_container(cfg.dataset.val, resolve=True)
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / "third_party/baselines/UPSTREAMS.json").read_text(encoding="utf-8-sig"))
    report.update(
        expected_samples=expected_samples,
        complete=complete,
        rain_units="dataset_native",
        target_field="rain_interpolated",
        data_config_sha256=hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(),
        config=config,
        upstreams=manifest,
        torch_version=torch.__version__,
        pysteps_runtime=version("pysteps") if "lk" in cfg.baseline.methods else None,
    )
    return report
