import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from accelerate import Accelerator
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import save_file

from src.gated_modality_rain.trainer import GatedModalityRainTrainer
from src.tests.time_series.test_spatial_modality_gate import build_baseline_model, build_model
from src.trainer.rain_trainer_ts_next_frame import RainTSNextFrameTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "src/config/ts_rain_train"


def compose_config(name: str) -> DictConfig:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name=name, return_hydra_config=True)


def test_fixed_origin_configs_preserve_baseline_training_contract() -> None:
    baseline = compose_config("rain_trainer_ts_next_frame")
    fixed = compose_config("rain_trainer_ts_next_frame_fixed_origin")
    gated = compose_config("gated_modality_rain_trainer")
    for section in ("dataset", "val", "ema"):
        assert OmegaConf.to_container(fixed[section], resolve=True) == OmegaConf.to_container(gated[section], resolve=True)
    expected_train = OmegaConf.to_container(baseline.train, resolve=True)
    expected_train["resume_path"] = None
    expected_train["next_pred"]["rollout_branch"]["use_gt_future_modalities"] = False
    expected_train["log"]["run_comment"] = "next_frame_fixed_origin"
    expected_val = OmegaConf.to_container(baseline.val, resolve=True)
    expected_val["rollout_use_gt_future_modalities"] = False
    assert OmegaConf.to_container(fixed.train, resolve=True) == expected_train
    expected_train["log"]["run_comment"] = "gated_modality_rain_trainer"
    assert OmegaConf.to_container(gated.train, resolve=True) == expected_train
    assert OmegaConf.to_container(fixed.val, resolve=True) == expected_val
    assert OmegaConf.to_container(fixed.dataset, resolve=True) == OmegaConf.to_container(baseline.dataset, resolve=True)
    assert fixed.rain_prediction_model.get("spatial_modality_gate_enabled", False) is False
    assert gated.rain_prediction_model.spatial_modality_gate_enabled is True
    assert gated.rain_prediction_model.spatial_modality_gate_hidden_channels == 32
    expected_model = OmegaConf.to_container(fixed.rain_prediction_model, resolve=True)
    expected_model.update(
        _target_="src.gated_modality_rain.model.GatedModalityRainModel",
        spatial_modality_gate_enabled=True,
        spatial_modality_gate_hidden_channels=32,
    )
    assert OmegaConf.to_container(gated.rain_prediction_model, resolve=True) == expected_model
    assert fixed.hydra.run.dir != gated.hydra.run.dir
    for cfg, name in ((fixed, "next_frame_fixed_origin"), (gated, "gated_modality_rain_trainer")):
        expected_accelerator = OmegaConf.to_container(baseline.accelerator, resolve=True)
        expected_accelerator["project_config"]["project_dir"] = f"runs/{name}/"
        expected_accelerator["project_config"]["logging_dir"] = f"runs/{name}/tensorboard"
        assert OmegaConf.to_container(cfg.accelerator, resolve=True) == expected_accelerator
    assert fixed.train.get("init_model_path") is None
    assert gated.train.get("init_model_path") is None


@pytest.mark.parametrize("module_mode", [False, True], ids=["direct", "module"])
def test_gated_entrypoint_resolves_config_without_training(module_mode: bool) -> None:
    command = [sys.executable]
    if module_mode:
        command += ["-m", "src.trainer.gated_modality_rain_trainer"]
    else:
        command += [str(PROJECT_ROOT / "src/trainer/gated_modality_rain_trainer.py")]
    result = subprocess.run(command + ["--cfg", "job"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    cfg = OmegaConf.create(result.stdout)
    assert cfg.train.log.run_comment == "gated_modality_rain_trainer"
    assert cfg.rain_prediction_model.spatial_modality_gate_enabled is True
    assert cfg.train.next_pred.rollout_branch.use_gt_future_modalities is False
    assert cfg.val.rollout_use_gt_future_modalities is False
    assert cfg.train.resume_path is None


def write_checkpoint(state: dict[str, torch.Tensor], directory: Path, checkpoint_format: str) -> Path:
    if "sharded" in checkpoint_format:
        keys = list(state)
        split = len(keys) // 2
        suffix = "safetensors" if "safe" in checkpoint_format else "bin"
        weight_map: dict[str, str] = {}
        for index, shard_keys in enumerate((keys[:split], keys[split:])):
            shard_path = directory / f"model-{index + 1:05d}-of-00002.{suffix}"
            shard = {key: state[key].contiguous() for key in shard_keys}
            if suffix == "safetensors":
                save_file(shard, str(shard_path), metadata={"format": "pt"})
            else:
                torch.save(shard, shard_path)
            weight_map.update({key: shard_path.name for key in shard_keys})
        index_path = directory / "model.index.json"
        index_path.write_text(json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8")
        return directory if checkpoint_format.startswith("directory") else index_path
    if "safe" in checkpoint_format:
        path = directory / "model.safetensors"
        save_file({key: value.contiguous() for key, value in state.items()}, str(path), metadata={"format": "pt"})
    else:
        path = directory / "pytorch_model.bin"
        torch.save(state, path)
    return directory if checkpoint_format.startswith("directory") else path


@pytest.mark.parametrize("source_gated", [False, True], ids=["baseline", "gated"])
@pytest.mark.parametrize(
    "checkpoint_format",
    ["file_bin", "file_safe", "directory_bin", "directory_safe", "sharded_bin", "sharded_safe", "directory_sharded_bin", "directory_sharded_safe"],
)
def test_checked_initialization_loads_complete_weights(
    tmp_path: Path, source_gated: bool, checkpoint_format: str
) -> None:
    from src.utils.gated_checkpoint import load_gated_model_initialization

    source = (build_model(spatial_modality_gate_enabled=True) if source_gated else build_baseline_model()).eval()
    if source_gated:
        with torch.no_grad():
            source.spatial_modality_gate.net[-1].bias.fill_(0.25)
    checkpoint = write_checkpoint(source.state_dict(), tmp_path, checkpoint_format)
    target = build_model(spatial_modality_gate_enabled=True).eval()
    load_gated_model_initialization(target, checkpoint)
    for key, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[key], value, rtol=0, atol=0)
    x = torch.randn(1, 12, 4, 16, 16)
    with torch.no_grad():
        expected, actual = source(x), target(x)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


@pytest.mark.parametrize("corruption", ["missing_base", "unexpected", "wrong_shape", "partial_gate"])
@pytest.mark.parametrize("checkpoint_format", ["file_bin", "directory_sharded_safe"])
def test_checked_initialization_rejects_corruption_before_loading(
    tmp_path: Path, corruption: str, checkpoint_format: str
) -> None:
    from src.utils.gated_checkpoint import load_gated_model_initialization

    source = build_model(spatial_modality_gate_enabled=True)
    state = dict(source.state_dict())
    if corruption == "missing_base":
        state.pop("patch_embed.bias")
    elif corruption == "unexpected":
        state["unexpected.weight"] = torch.zeros(1)
    elif corruption == "wrong_shape":
        state["patch_embed.weight"] = torch.zeros(1)
    else:
        state.pop("spatial_modality_gate.net.0.weight")
    checkpoint = write_checkpoint(state, tmp_path, checkpoint_format)
    target = build_model(spatial_modality_gate_enabled=True)
    original = {key: value.clone() for key, value in target.state_dict().items()}
    with pytest.raises(ValueError, match="checkpoint|Checkpoint"):
        load_gated_model_initialization(target, checkpoint)
    for key, value in target.state_dict().items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)


def test_trainer_rejects_incomplete_gated_initialization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = dict(build_model().state_dict())
    state.pop("patch_embed.bias")
    checkpoint = write_checkpoint(state, tmp_path, "file_bin")
    cfg = OmegaConf.create(
        {
            "train": {"init_model_path": str(checkpoint)},
            "val": {},
            "dataset": {"train": {"test_dataset": "train"}, "val": {"test_dataset": "val"}},
            "ema": {},
            "accelerator": {"test_accelerator": True},
            "rain_prediction_model": {"test_model": True},
        }
    )

    def instantiate_local(config: DictConfig) -> object:
        if config.get("test_accelerator"):
            return SimpleNamespace(device=torch.device("cpu"))
        if config.get("test_dataset"):
            return [], []
        if config.get("test_model"):
            return build_model(spatial_modality_gate_enabled=True)
        return instantiate(config)

    def configure_local_logger(self: GatedModalityRainTrainer) -> Path:
        self.proj_dir = tmp_path
        return tmp_path / "unused.log"

    monkeypatch.setattr("src.gated_modality_rain.trainer.hydra.utils.instantiate", instantiate_local)
    monkeypatch.setattr(GatedModalityRainTrainer, "_configure_logger", configure_local_logger)
    monkeypatch.setattr(GatedModalityRainTrainer, "log_msg", lambda *args: None)
    monkeypatch.setattr(GatedModalityRainTrainer, "_init_rain_norm_params", lambda self: None)
    with pytest.raises(ValueError, match="checkpoint|Checkpoint"):
        GatedModalityRainTrainer(cfg)


@pytest.mark.parametrize("config_name", ["rain_trainer_ts_next_frame_fixed_origin", "gated_modality_rain_trainer"])
@pytest.mark.parametrize("settings_source", ["train", "val"])
def test_fixed_origin_rollout_ignores_future_labels(config_name: str, settings_source: str) -> None:
    cfg = compose_config(config_name)
    trainer_class = RainTSNextFrameTrainer if config_name.endswith("fixed_origin") else GatedModalityRainTrainer
    trainer = object.__new__(trainer_class)
    trainer.train_cfg = cfg.train
    trainer.val_cfg = cfg.val
    trainer.radar_c, trainer.satellite_c, trainer.rain_c = 1, 10, 1
    trainer.accelerator = Accelerator(cpu=True)
    trainer.model = (
        build_baseline_model()
        if config_name.endswith("fixed_origin")
        else build_model(spatial_modality_gate_enabled=cfg.rain_prediction_model.spatial_modality_gate_enabled)
    ).eval()
    if getattr(trainer.model, "spatial_modality_gate", None) is not None:
        with torch.no_grad():
            trainer.model.spatial_modality_gate.net[-1].bias.copy_(torch.tensor([0.2, -0.3]))
    context = torch.randn(1, 12, 4, 16, 16)
    future = {
        "radar": torch.randn(1, 1, 4, 16, 16),
        "satellite": torch.randn(1, 10, 4, 16, 16),
        "rain": torch.randn(1, 1, 4, 16, 16),
    }
    changed_future = {name: value + 10.0 for name, value in future.items()}
    with torch.no_grad():
        if settings_source == "val":
            expected = trainer._rollout_predict(context, 4, future_modalities=future)
            actual = trainer._rollout_predict(context, 4, future_modalities=changed_future)
        else:
            settings = trainer._resolve_train_rollout_branch(total_future_frames=4)
            assert settings is not None
            rollout_args = {
                "context": context,
                "total_future_frames": settings["rollout_frames"],
                "mode": settings["mode"],
                "rollout_block_size": settings["rollout_block_size"],
                "detach_history": settings["detach_history"],
                "use_gt_future_modalities": settings["use_gt_future_modalities"],
            }
            expected = trainer._rollout_predict_with_settings(**rollout_args, future_modalities=future)
            actual = trainer._rollout_predict_with_settings(**rollout_args, future_modalities=changed_future)
    for name, channels in (("radar", 1), ("satellite", 10), ("rain", 1)):
        assert actual[name].shape == (1, channels, 4, 16, 16)
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
