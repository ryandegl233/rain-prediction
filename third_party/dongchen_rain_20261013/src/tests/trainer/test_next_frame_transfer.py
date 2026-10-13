from collections import Counter
from collections.abc import Callable
from pathlib import Path

import hydra
import pytest
import torch
from accelerate.data_loader import DataLoaderShard
from omegaconf import DictConfig
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.data import DataLoader

from src.trainer.rain_trainer_ts_next_frame import RainTSNextFrameTrainer
from src.trainer.rain_trainer_ts_next_frame_transfer import RainTSNextFrameTransferTrainer


def make_test_loader() -> tuple[list[dict[str, torch.Tensor]], DataLoader]:
    generator = torch.Generator().manual_seed(2025)
    sample = {
        f"{modality}_{period}": torch.rand(channels, 4, 8, 8, generator=generator)
        for modality, channels in (("radar", 1), ("satellite", 10), ("rain", 1))
        for period in ("past", "future")
    }
    sample["time_past"] = torch.arange(4, dtype=torch.float32)
    sample["time_future"] = torch.arange(4, 8, dtype=torch.float32)
    sample["time_past_timestamp"] = torch.arange(4, dtype=torch.float64)
    dataset = [sample, sample]
    return dataset, DataLoader(dataset, batch_size=1)


def make_config(tmp_path: Path, gan_enabled: bool = False) -> DictConfig:
    config_dir = Path(__file__).resolve().parents[2] / "config/ts_rain_train"
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = hydra.compose(config_name="rain_trainer_ts_next_frame", overrides=["+accelerator.cpu=true"])
    cfg.accelerator.mixed_precision = "no"
    cfg.accelerator.log_with = None
    cfg.ema.beta = 0
    cfg.train.proj_dir = str(tmp_path)
    cfg.train.debug = True
    cfg.train.log.log_with_time = False
    cfg.train.log.run_comment = ""
    cfg.train.max_steps = 1
    cfg.train.gan.enabled = gan_enabled
    cfg.train.next_pred.missing_modality.enabled = False
    cfg.rain_prediction_model = {
        "_target_": "src.tests.trainer.test_rain_trainer_ts_next_frame.DummyTrainModel"
    }
    cfg.train.gan.discriminator = {
        "_target_": "src.tests.trainer.test_rain_trainer_ts_next_frame.DummyTrainDiscriminator"
    }
    cfg.dataset.train = {"_target_": f"{__name__}.make_test_loader"}
    cfg.dataset.val = {"_target_": f"{__name__}.make_test_loader"}
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


@pytest.mark.parametrize("gan_enabled", [False, True])
def test_training_loop_uses_prepared_loader(tmp_path: Path, gan_enabled: bool) -> None:
    trainer = RainTSNextFrameTransferTrainer(make_config(tmp_path, gan_enabled))
    assert isinstance(trainer.train_dataloader, DataLoaderShard)
    assert not hasattr(trainer, "train_dataloadaer")
    assert trainer.train_dataloader._non_blocking is True
    trainer.train()
    assert trainer.global_step == 1
    assert trainer.last_saved_checkpoint_step == 1


def test_batch_conversion_is_shared_and_preserves_training_result(tmp_path: Path) -> None:
    cfg = make_config(tmp_path / "baseline")
    baseline = RainTSNextFrameTrainer(cfg)
    cfg.train.proj_dir = str(tmp_path / "transfer")
    fixed = RainTSNextFrameTransferTrainer(cfg)
    batch = next(iter(baseline.train_dataloader))
    batch = {name: tensor.double() for name, tensor in batch.items()}

    with SourceCopyCounter(batch) as baseline_copies:
        expected_logs, expected_step = baseline.train_step(batch)
    with SourceCopyCounter(batch) as fixed_copies:
        actual_logs, actual_step = fixed.train_step(batch)
    model_fields = set(batch) - {"time_past_timestamp"}
    assert baseline_copies.copies == Counter({name: 2 for name in model_fields})
    assert fixed_copies.copies == Counter({name: 1 for name in model_fields})
    assert all(tensor.dtype == torch.float64 for tensor in batch.values())
    assert actual_step == expected_step
    assert actual_logs.keys() == expected_logs.keys()
    for name in expected_logs:
        torch.testing.assert_close(actual_logs[name], expected_logs[name], rtol=0, atol=0)
    for expected, actual in zip(baseline.model.parameters(), fixed.model.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    prepared_batch = next(iter(fixed.train_dataloader))
    with SourceCopyCounter(prepared_batch) as prepared_copies:
        fixed.train_step(prepared_batch)
    assert not prepared_copies.copies


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required to count host-to-device copies")
def test_cpu_batch_transfers_once_to_cuda(tmp_path: Path) -> None:
    trainer = RainTSNextFrameTransferTrainer(make_config(tmp_path))
    trainer.device = torch.device("cuda")
    trainer.model.to(trainer.device)
    batch = next(iter(trainer.train_dataloader))
    batch = {name: tensor.pin_memory() for name, tensor in batch.items()}
    with SourceCopyCounter(batch) as copies:
        trainer.train_step(batch)
    assert copies.copies == Counter({name: 1 for name in batch})
