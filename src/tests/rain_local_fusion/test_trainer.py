import copy
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, open_dict

from src.rain_local_fusion.trainer import RainLocalFusionTrainer
from src.rain_priority.trainer import RainPriorityTrainer
from src.tests.rain_priority.test_trainer import tiny_config


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def local_config(path: Path, enabled: bool = True) -> DictConfig:
    cfg = tiny_config(path)
    cfg.train.priority.horizon.enabled = False
    cfg.rain_prediction_model.spatial_modality_gate_enabled = False
    with open_dict(cfg.rain_prediction_model):
        cfg.rain_prediction_model._target_ = "src.rain_local_fusion.model.RainLocalFusionModel"
        cfg.rain_prediction_model.local_fusion_enabled = enabled
        cfg.rain_prediction_model.local_fusion_hidden_channels = 4
        cfg.rain_prediction_model.local_fusion_kernel_size = 3
    return cfg


def test_three_configs_differ_only_in_controlled_branch_settings() -> None:
    path = Path(__file__).resolve().parents[3] / "src/config/ts_rain_train"
    with initialize_config_dir(config_dir=str(path), version_base=None):
        a0, r0, r1 = [
            compose(config_name=name) for name in ("rain_local_fusion_a0", "rain_local_fusion_r0", "rain_local_fusion")
        ]
    assert not a0.rain_prediction_model.local_fusion_enabled
    assert r0.rain_prediction_model.local_fusion_kernel_size == 1
    assert r1.rain_prediction_model.local_fusion_kernel_size == 3
    for cfg in (a0, r0, r1):
        assert cfg.dataset.n_past == cfg.dataset.n_futures == 4
        assert cfg.rain_prediction_model.decoder_type == "3d"
        assert not cfg.rain_prediction_model.spatial_modality_gate_enabled
        assert not cfg.train.priority.horizon.enabled
        assert cfg.train.max_steps == 6000
        assert cfg.train.scheduler.num_training_steps == 100000
        assert cfg.train.save_every == cfg.val.val_duration == 2000
        assert cfg.val.max_val_iters == -1
        assert cfg.dataset.train.batch_size * cfg.accelerator.gradient_accumulation_plugin.num_steps == 2
    for cfg in (a0, r0):
        cfg.rain_prediction_model = copy.deepcopy(r1.rain_prediction_model)
        cfg.train.log.run_comment = r1.train.log.run_comment
        assert cfg == r1


def test_disabled_branch_matches_A_loss_gradients_and_updates(tmp_path: Path) -> None:
    cfg = tiny_config(tmp_path / "A")
    cfg.train.priority.horizon.enabled = False
    cfg.rain_prediction_model.spatial_modality_gate_enabled = False
    baseline = RainPriorityTrainer(cfg)
    local = RainLocalFusionTrainer(local_config(tmp_path / "A0", enabled=False))
    batch = next(iter(baseline.train_dataloader))
    for _ in range(2):
        expected, stepped_a = baseline.train_step(batch)
        actual, stepped_b = local.train_step(batch)
        assert stepped_a == stepped_b
        for name in ("loss", "loss/rain", "loss/radar", "loss/satellite"):
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
        for p, q in zip(baseline.model.parameters(), local.model.parameters(), strict=True):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    baseline._close_tensorboard_writer()
    local._close_tensorboard_writer()


def test_real_accumulation_resume_validation_and_fixed_origin(tmp_path: Path) -> None:
    cfg = local_config(tmp_path / "run")
    trainer = RainLocalFusionTrainer(cfg)
    batch = next(iter(trainer.train_dataloader))
    before = trainer.model.patch_embed.weight.detach().clone()
    for _ in range(4):
        logs, stepped = trainer.train_step(batch)
        assert all(torch.isfinite(value).all() for value in logs.values())
        assert all(logs[f"horizon/h{h}/weight_used"] == 0.25 for h in range(1, 5))
    assert stepped and trainer.global_step == 2
    assert torch.count_nonzero(trainer.model.local_fusion.output.weight) > 0
    assert not torch.equal(before, trainer.model.patch_embed.weight)
    trainer._save_checkpoint()
    cfg2 = local_config(tmp_path / "resumed")
    cfg2.train.resume_path = str(trainer.proj_dir / "checkpoint-00000002")
    resumed = RainLocalFusionTrainer(cfg2)
    resumed._resume_if_needed()
    assert resumed.global_step == resumed.controller.step == 2
    resumed.model.eval()
    changed = {key: value + 100 if key.endswith("future") else value for key, value in batch.items()}
    first = resumed._val_inference_step(batch)[0]
    second = resumed._val_inference_step(changed)[0]
    assert all(value.shape[2] == 4 for value in first.values())
    for name in first:
        torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)
    resumed._run_val()
    assert (resumed.proj_dir / "validation/step-00000002.json").is_file()
    assert (resumed.proj_dir / "validation/step-00000002-raw.png").is_file()
    resumed.model.train()
    for _ in range(2):
        trainer.train_step(batch)
        resumed.train_step(batch)
    for key, value in trainer.model.state_dict().items():
        torch.testing.assert_close(value, resumed.model.state_dict()[key], rtol=0, atol=0)
    trainer._close_tensorboard_writer()
    resumed._close_tensorboard_writer()


def test_confounded_loss_or_controller_settings_are_rejected(tmp_path: Path) -> None:
    cfg = local_config(tmp_path)
    cfg.train.priority.horizon.enabled = True
    with pytest.raises(ValueError, match="horizon"):
        RainLocalFusionTrainer(cfg)
    cfg.train.priority.horizon.enabled = False
    cfg.train.loss_weights.radar = 1.0
    with pytest.raises(ValueError, match="coefficients"):
        RainLocalFusionTrainer(cfg)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_bf16_four_step_rollout_backward() -> None:
    from src.rain_local_fusion.model import RainLocalFusionModel
    from src.tests.rain_local_fusion.test_model import settings

    model = RainLocalFusionModel(**settings(), local_fusion_hidden_channels=4).cuda().train()
    history = torch.rand(1, 12, 4, 8, 8, device="cuda")
    loss = torch.zeros((), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for horizon in range(4):
            source = torch.ones(1, 3, history.shape[2] + 1, device="cuda")
            if horizon:
                source[:, :, 4:] = 2
            pred = model.forward_ar(history, history[:, :, -1:], gate_source=source)
            rain = pred["rain"] + history[:, -1:, -1:]
            loss = loss + rain.float().square().mean()
            history = torch.cat([history, torch.cat([pred["radar"], pred["satellite"], rain], dim=1).detach()], dim=2)
    loss.backward()
    grad = model.local_fusion.output.weight.grad
    assert torch.isfinite(loss) and grad is not None
    assert torch.isfinite(grad).all() and torch.count_nonzero(grad)
