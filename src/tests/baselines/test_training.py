import json
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from src.baselines.contracts import prepare_history
from src.baselines.runtime import load_reference, validate_protocol, wrap_reference


def config() -> DictConfig:
    directory = Path(__file__).resolve().parents[3] / "src/config/ts_rain_train"
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        return compose(config_name="baselines_train_reference")


def test_config_inherits_reference_and_enforces_history_only() -> None:
    cfg = config()
    validate_protocol(cfg)
    assert cfg.rain_prediction_model.dim == 384
    assert cfg.rain_prediction_model.depth == 6
    assert cfg.dataset.img_size == 448
    assert cfg.dataset.train.batch_size * cfg.accelerator.gradient_accumulation_plugin.num_steps == 8
    assert not cfg.dataset.val.drop_last
    cfg.val.rollout_use_gt_future_modalities = True
    with pytest.raises(ValueError, match="future"):
        validate_protocol(cfg)


def test_original_trainer_smoke_and_identical_full_modality_rollout(tmp_path: Path) -> None:
    from src.baselines.trainer import BaselineComparisonTrainer

    cfg = config()
    cfg.baseline.methods = ["persistence", "lk"]
    OmegaConf.update(cfg, "accelerator.cpu", True, force_add=True)
    cfg.accelerator.mixed_precision = "no"
    cfg.accelerator.log_with = None
    cfg.accelerator.gradient_accumulation_plugin.num_steps = 2
    cfg.ema.beta = 0
    cfg.train.proj_dir = str(tmp_path)
    cfg.train.max_steps = 1
    cfg.train.save_every = 1
    cfg.train.log.log_every = 1
    cfg.val.val_duration = 1
    cfg.evaluation.progress_every = 0
    cfg.dataset.img_size = 16
    for split in ("train", "val"):
        cfg.dataset[split] = {
            "_target_": "src.tests.baselines.fixtures.make_loader",
            "batch_size": 1 if split == "train" else 2,
            "count": 3,
            "shuffle": False,
            "drop_last": False,
            "iter_index_mode": "sequential",
        }
    for name, value in {
        "input_size": 16,
        "patch_size": 2,
        "stem_channels": 8,
        "dim": 8,
        "depth": 1,
        "num_heads": 2,
        "decoder_base_channels": 8,
        "decoder_k_size": 3,
        "activation_checkpoint": False,
    }.items():
        cfg.rain_prediction_model[name] = value
    trainer = BaselineComparisonTrainer(cfg)
    data = next(iter(trainer.val_dataloader))
    trainer.model.eval()
    history = prepare_history(data, ("radar", "satellite", "rain"))
    with torch.no_grad():
        expected = trainer._rollout_predict_with_settings(
            context=history.values,
            total_future_frames=4,
            mode="frame",
            rollout_block_size=1,
            detach_history=True,
            use_gt_future_modalities=False,
            context_time=history.time,
            future_time=data["time_future"],
        )["rain"]
        actual = wrap_reference(trainer.model, cfg)(history)
    torch.testing.assert_close(actual, expected)
    before = trainer.model.patch_embed.weight.detach().clone()
    trainer.run()
    assert trainer.global_step == 1
    assert not torch.equal(trainer.model.patch_embed.weight, before)
    reports = list(tmp_path.rglob("baseline_validation/step-00000001/metrics.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert report["samples"] == 3
    assert report["complete"]
    assert set(report["results"]["radar+satellite+rain"]) == {"reference", "persistence", "lk"}
    checkpoints = list(tmp_path.rglob("checkpoint-00000001/model.safetensors"))
    assert len(checkpoints) == 1
    assert (checkpoints[0].parent / "meta.json").is_file()
    cfg.evaluation.reference_checkpoint = str(checkpoints[0].parent)
    restored = load_reference(cfg)
    trainer.model.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(history), wrap_reference(trainer.model, cfg)(history))
