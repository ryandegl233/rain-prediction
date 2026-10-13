import hashlib
import importlib
import json
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path

import pytest
import torch
from accelerate.data_loader import DataLoaderDispatcher, DataLoaderShard
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.data import DataLoader, IterableDataset

from src.gated_modality_rain.trainer import GatedModalityRainTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "src/config/ts_rain_train"
MODEL_FIELDS = {
    "radar_past",
    "radar_future",
    "satellite_past",
    "satellite_future",
    "rain_past",
    "rain_future",
    "time_past",
    "time_future",
}


class IterableSamples(IterableDataset):
    def __init__(self, samples: list[dict[str, torch.Tensor]]) -> None:
        self.samples = samples

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        yield from self.samples

    def __len__(self) -> int:
        return len(self.samples)


def make_test_loader(iterable: bool = False) -> tuple[list[dict[str, torch.Tensor]] | IterableSamples, DataLoader]:
    generator = torch.Generator().manual_seed(2025)
    sample = {
        f"{modality}_{period}": torch.rand(channels, 2, 8, 8, generator=generator)
        for modality, channels in (("radar", 1), ("satellite", 10), ("rain", 1))
        for period in ("past", "future")
    }
    sample["time_past"] = torch.tensor([0.1, 0.2])
    sample["time_future"] = torch.tensor([0.3, 0.4])
    sample["time_past_timestamp"] = torch.arange(2, dtype=torch.float64)
    dataset = IterableSamples([sample, sample]) if iterable else [sample, sample]
    return dataset, DataLoader(dataset, batch_size=1)


def make_config(directory: Path, gan_enabled: bool = False, use_cuda: bool = False) -> DictConfig:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="gated_modality_rain_trainer",
            overrides=[f"+accelerator.cpu={str(not use_cuda).lower()}"],
        )
    cfg.accelerator.mixed_precision = "no"
    cfg.accelerator.log_with = None
    cfg.accelerator.project_config.automatic_checkpoint_naming = False
    cfg.accelerator.project_config.total_limit = None
    cfg.ema.beta = 0
    cfg.train.proj_dir = str(directory)
    cfg.train.debug = True
    cfg.train.log.log_with_time = False
    cfg.train.log.run_comment = ""
    cfg.train.max_steps = 1
    cfg.train.gan.enabled = gan_enabled
    cfg.train.next_pred.missing_modality.enabled = False
    cfg.train.next_pred.rollout_branch.rollout_frames = 2
    cfg.rain_prediction_model = {"_target_": "src.tests.trainer.test_rain_trainer_ts_next_frame.DummyTrainModel"}
    cfg.train.gan.discriminator = {
        "_target_": "src.tests.trainer.test_rain_trainer_ts_next_frame.DummyTrainDiscriminator"
    }
    cfg.dataset.train = {"_target_": f"{__name__}.make_test_loader", "iterable": False}
    cfg.dataset.val = {"_target_": f"{__name__}.make_test_loader", "iterable": False}
    return cfg


class SourceCopyCounter(TorchDispatchMode):
    def __init__(self, batch: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.source_names = {id(tensor): name for name, tensor in batch.items()}
        self.copies: Counter[str] = Counter()

    def __torch_dispatch__(
        self,
        func: Callable[..., object],
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        if func is torch.ops.aten._to_copy.default and id(args[0]) in self.source_names:
            self.copies[self.source_names[id(args[0])]] += 1
        return func(*args, **(kwargs or {}))


def capture_gradient(gradient: torch.Tensor, *, name: str, gradients: dict[str, torch.Tensor]) -> None:
    gradients[name] = gradient.detach().clone()


def run_loader_case(directory: Path, gan_enabled: bool, iterable: bool) -> None:
    module = importlib.import_module("src.gated_modality_rain.io_trainer")
    cfg = make_config(directory, gan_enabled)
    if iterable:
        cfg.accelerator.dataloader_config.dispatch_batches = True
        cfg.dataset.train.iterable = True
        cfg.dataset.val.iterable = True
    trainer = module.GatedModalityRainIOTrainer(cfg)
    loader_type = DataLoaderDispatcher if iterable else DataLoaderShard
    assert isinstance(trainer.train_dataloader, loader_type)
    assert not hasattr(trainer, "train_dataloadaer")
    assert trainer.train_dataloader._non_blocking is True
    trainer.train()
    assert trainer.global_step == 1
    checkpoint = directory / "checkpoint-00000001"
    assert json.loads((checkpoint / "meta.json").read_text())["global_step"] == 1


def run_parity_case(directory: Path) -> None:
    module = importlib.import_module("src.gated_modality_rain.io_trainer")
    cfg = make_config(directory / "original")
    cfg.rain_prediction_model = {
        "_target_": "src.tests.time_series.test_spatial_modality_gate.build_model",
        "spatial_modality_gate_enabled": True,
    }
    original = GatedModalityRainTrainer(cfg)
    cfg.train.proj_dir = str(directory / "optimized")
    optimized = module.GatedModalityRainIOTrainer(cfg)
    batch = {name: tensor.double() for name, tensor in next(iter(original.train_dataloader)).items()}
    original_inputs = {name: tensor.clone() for name, tensor in batch.items()}
    expected_gradients: dict[str, torch.Tensor] = {}
    actual_gradients: dict[str, torch.Tensor] = {}
    for trainer, gradients in ((original, expected_gradients), (optimized, actual_gradients)):
        for name, parameter in trainer.model.named_parameters():
            parameter.register_hook(partial(capture_gradient, name=name, gradients=gradients))

    with SourceCopyCounter(batch) as original_copies:
        expected_logs, expected_step = original.train_step(batch)
    with SourceCopyCounter(batch) as optimized_copies:
        actual_logs, actual_step = optimized.train_step(batch)

    assert original_copies.copies == Counter({name: 2 for name in MODEL_FIELDS})
    assert optimized_copies.copies == Counter({name: 1 for name in MODEL_FIELDS})
    assert actual_step == expected_step
    assert actual_logs.keys() == expected_logs.keys()
    for name in expected_logs:
        torch.testing.assert_close(actual_logs[name], expected_logs[name], rtol=0, atol=0)
    assert actual_gradients.keys() == expected_gradients.keys()
    for name in expected_gradients:
        torch.testing.assert_close(actual_gradients[name], expected_gradients[name], rtol=0, atol=0)
    for expected, actual in zip(original.model.parameters(), optimized.model.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for name, tensor in batch.items():
        assert tensor.dtype == torch.float64
        torch.testing.assert_close(tensor, original_inputs[name], rtol=0, atol=0)

    prepared_batch = next(iter(optimized.train_dataloader))
    with SourceCopyCounter(prepared_batch) as prepared_copies:
        optimized.train_step(prepared_batch)
    assert not prepared_copies.copies


def run_gate_case(directory: Path, all_modules: bool) -> None:
    module = importlib.import_module("src.gated_modality_rain.io_trainer")
    cfg = make_config(directory)
    cfg.rain_prediction_model = {
        "_target_": "src.tests.time_series.test_spatial_modality_gate.build_model",
        "spatial_modality_gate_enabled": True,
        "cross_modal_adapter_enabled": all_modules,
        "local_window_refiner_enabled": all_modules,
        "local_window_size": 2,
        "consensus_calibration_enabled": all_modules,
        "consensus_calibration_bottleneck": 8,
    }
    trainer = module.GatedModalityRainIOTrainer(cfg)
    model = trainer.accelerator.unwrap_model(trainer.model)
    assert model.spatial_modality_gate is not None
    if all_modules:
        assert model.cross_modal_adapter.consensus_calibration is not None
        assert model.local_window_refiner is not None
    gradients: dict[str, torch.Tensor] = {}
    gate_before = {
        name: parameter.detach().clone() for name, parameter in model.spatial_modality_gate.named_parameters()
    }
    for name, parameter in model.spatial_modality_gate.named_parameters():
        parameter.register_hook(partial(capture_gradient, name=name, gradients=gradients))
    logs, did_step = trainer.train_step(next(iter(trainer.train_dataloader)))
    assert did_step
    assert torch.isfinite(logs["loss"]).item()
    assert gradients and all(torch.isfinite(gradient).all().item() for gradient in gradients.values())
    assert any(gradient.abs().sum().item() > 0 for gradient in gradients.values())
    assert any(
        not torch.equal(gate_before[name], parameter)
        for name, parameter in model.spatial_modality_gate.named_parameters()
    )


def run_cuda_case(directory: Path) -> None:
    module = importlib.import_module("src.gated_modality_rain.io_trainer")
    trainer = module.GatedModalityRainIOTrainer(make_config(directory, use_cuda=True))
    _, loader = make_test_loader()
    batch = {name: tensor.pin_memory() for name, tensor in next(iter(loader)).items()}
    with SourceCopyCounter(batch) as copies:
        trainer.train_step(batch)
    torch.cuda.synchronize()
    assert copies.copies == Counter({name: 1 for name in batch})
    assert all(tensor.device.type == "cpu" for tensor in batch.values())


def run_isolated_case(case: str, directory: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "src.tests.gated_modality_rain.test_io_trainer", case, str(directory)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("case", ["loader", "loader_gan", "loader_iterable"])
def test_training_loop_uses_prepared_loader(case: str, tmp_path: Path) -> None:
    run_isolated_case(case, tmp_path)


def test_batch_conversion_preserves_losses_gradients_and_updates(tmp_path: Path) -> None:
    run_isolated_case("parity", tmp_path)


@pytest.mark.parametrize("case", ["gate", "all_modules"])
def test_io_path_updates_gate_and_preserves_model_modules(case: str, tmp_path: Path) -> None:
    run_isolated_case(case, tmp_path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for host-to-device transfer checks")
def test_cuda_transfers_raw_batch_once(tmp_path: Path) -> None:
    run_isolated_case("cuda", tmp_path)


def test_entrypoint_resolves_preserved_training_contract() -> None:
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "src.trainer.gated_modality_rain_io_trainer", "--cfg", "job", "--resolve"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    cfg = OmegaConf.create(result.stdout)
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        original = compose(config_name="gated_modality_rain_trainer")
    for section in ("rain_prediction_model", "ema"):
        assert OmegaConf.to_container(cfg[section], resolve=True) == OmegaConf.to_container(
            original[section], resolve=True
        )
    for section in ("next_pred", "loss", "optim", "scheduler", "gan"):
        assert OmegaConf.to_container(cfg.train[section], resolve=True) == OmegaConf.to_container(
            original.train[section], resolve=True
        )
    assert cfg.train.resume_path is None
    assert cfg.train.init_model_path is None
    assert cfg.dataset.train.batch_size == cfg.dataset.val.batch_size == 1
    assert cfg.dataset.train.pin_memory and cfg.dataset.val.pin_memory
    assert cfg.accelerator.dataloader_config.non_blocking
    assert cfg.accelerator.project_config.total_limit is None
    assert not cfg.accelerator.project_config.automatic_checkpoint_naming
    assert not cfg.train.next_pred.rollout_branch.use_gt_future_modalities
    assert not cfg.val.rollout_use_gt_future_modalities


def test_peer_snapshot_is_unchanged() -> None:
    snapshot = PROJECT_ROOT / "third_party/dongchen_rain_20261013"
    manifest = json.loads((snapshot / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    for record in manifest["files"]:
        assert hashlib.sha256((snapshot / record["path"]).read_bytes()).hexdigest() == record["sha256"]


if __name__ == "__main__":
    torch.set_num_threads(2)
    selected_case = sys.argv[1]
    output_directory = Path(sys.argv[2])
    if selected_case in {"loader", "loader_gan", "loader_iterable"}:
        run_loader_case(output_directory, selected_case == "loader_gan", selected_case == "loader_iterable")
    elif selected_case == "parity":
        run_parity_case(output_directory)
    elif selected_case in {"gate", "all_modules"}:
        run_gate_case(output_directory, selected_case == "all_modules")
    elif selected_case == "cuda":
        run_cuda_case(output_directory)
    else:
        raise ValueError(selected_case)
