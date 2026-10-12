import copy

import torch
from omegaconf import OmegaConf

from src.residual_diffusion_stage.diagnose_coarse_evolution import evolution_statistics, summarize
from src.tests.trainer.test_rain_trainer_ts_next_frame import (
    DummyAccelerator,
    DummyTrainModel,
    _make_batch,
    _make_trainer_for_batch,
)
from src.trainer.rain_trainer_coarse_evolution import CoarseEvolutionTrainer


def test_sequential_backward_matches_combined_with_accumulation() -> None:
    original = _make_trainer_for_batch("block", 1)
    original.accelerator = DummyAccelerator()
    original.accelerator.num_processes = 1
    original.model = DummyTrainModel()
    original.optim = torch.optim.SGD(original.model.parameters(), lr=0.001)
    original.sched = torch.optim.lr_scheduler.LambdaLR(original.optim, lr_lambda=lambda _: 1.0)
    original.use_gan = False
    original.ema_model = None
    original.global_step = 0
    original.train_cfg.next_pred.rollout_branch = {
        "enabled": True, "weight": 0.4, "mode": "frame", "rollout_block_size": 1,
        "rollout_frames": 4, "detach_history": True, "use_gt_future_modalities": False, "loss_on": "rain",
    }
    original.train_cfg.loss = OmegaConf.create({
        "mode": "enhanced", "regression_type": "mse",
        "rain_region_weight": {"enabled": True, "alpha": 2.0, "r0": 0.05, "gamma": 0.7},
        "temporal_diff": {"enabled": True, "weight_init": 0.2, "weight_max": 0.2},
        "spatial_gradient": {"enabled": True, "weight": 0.05},
    })
    sequential = object.__new__(CoarseEvolutionTrainer)
    sequential.__dict__.update(copy.deepcopy(original.__dict__))
    initial = sequential.model.scale.detach().clone()
    for sync in (False, True):
        batch = _make_batch(batch=1, n_past=4, n_future=4)
        original.accelerator.sync_gradients = sync
        sequential.accelerator.sync_gradients = sync
        a, a_step = original.train_step(batch)
        b, b_step = sequential.train_step(batch)
        assert a_step == b_step == sync
        torch.testing.assert_close(a["loss"], b["loss"])
        torch.testing.assert_close(original.model.scale, sequential.model.scale)
        if not sync:
            torch.testing.assert_close(original.model.scale.grad, sequential.model.scale.grad)
            torch.testing.assert_close(sequential.model.scale, initial)
    assert sequential.global_step == 1
    assert not torch.equal(sequential.model.scale, initial)


def test_motion_diagnostic_detects_persistence_and_perfect_forecast() -> None:
    truth = torch.zeros(1, 1, 4, 12, 12)
    for lead in range(4):
        truth[:, :, lead, 3:6, 4 + lead:6 + lead] = 0.7
    anchor = torch.zeros(1, 1, 1, 12, 12)
    anchor[:, :, :, 3:6, 3:5] = 0.7
    perfect = summarize(evolution_statistics(truth, truth, anchor))
    copied = summarize(evolution_statistics(anchor.expand_as(truth), truth, anchor))
    for lead in perfect:
        assert perfect[lead]["active_delta_mae"] == 0
        assert perfect[lead]["change_amplitude_ratio"] == 1
        assert perfect[lead]["growth_direction_recall"] == 1
        assert perfect[lead]["decay_direction_recall"] == 1
        assert perfect[lead]["fss_0.1_scale_4"] == 1
        assert copied[lead]["change_amplitude_ratio"] == 0
        assert copied[lead]["active_delta_mae"] > 0


def test_dry_diagnostic_reports_undefined_core_not_perfect_score() -> None:
    truth = torch.zeros(1, 1, 4, 12, 12)
    values = summarize(evolution_statistics(truth, truth, truth[:, :, :1]))
    assert values["1"]["core_0.5_intensity_ratio"] is None
    assert values["1"]["growth_direction_recall"] is None
    assert values["1"]["fss_0.1_scale_4"] is None
