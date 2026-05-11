from contextlib import nullcontext

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.trainer.rain_trainer_ts_next_frame import RainTSNextFrameTrainer, apply_context_modality_dropout


class DummyAccelerator:
    def __init__(self) -> None:
        self.sync_gradients = True

    def autocast(self):
        return nullcontext()

    def accumulate(self, _model):
        return nullcontext()

    @staticmethod
    def backward(loss: torch.Tensor) -> None:
        loss.backward()

    @staticmethod
    def clip_grad_norm_(parameters, max_grad_norm: float) -> None:
        torch.nn.utils.clip_grad_norm_(list(parameters), max_grad_norm)

    @staticmethod
    def unwrap_model(model):
        return model


class DummyARModel:
    def forward_ar(
        self,
        target_x: torch.Tensor,
        context_x: torch.Tensor | None,
        predict_frames: int,
        strict_target_isolation: bool,
        return_modality_dict: bool,
        context_time: torch.Tensor | None = None,
        target_time: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        _ = context_x, predict_frames, strict_target_isolation, return_modality_dict, context_time, target_time
        out = target_x + 1.0
        return {
            "radar": out[:, :1],
            "satellite": out[:, 1:11],
            "rain": out[:, 11:12],
        }


class DummyARModelTimeProbe:
    def __init__(self) -> None:
        self.time_call_count = 0

    def forward_ar(
        self,
        target_x: torch.Tensor,
        context_x: torch.Tensor | None,
        predict_frames: int,
        strict_target_isolation: bool,
        return_modality_dict: bool,
        context_time: torch.Tensor | None = None,
        target_time: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        _ = strict_target_isolation, return_modality_dict
        if context_x is None:
            raise AssertionError("rollout probe expects non-empty context.")
        if context_time is None or target_time is None:
            raise AssertionError("context_time and target_time must be passed in rollout with time inputs.")
        if context_time.shape != (int(context_x.shape[0]), int(context_x.shape[2])):
            raise AssertionError("context_time shape mismatch.")
        if target_time.shape != (int(target_x.shape[0]), int(target_x.shape[2])):
            raise AssertionError("target_time shape mismatch.")
        if int(target_x.shape[2]) != int(predict_frames):
            raise AssertionError("predict_frames mismatch with target_x length.")
        self.time_call_count += 1

        out = target_x + 1.0
        return {
            "radar": out[:, :1],
            "satellite": out[:, 1:11],
            "rain": out[:, 11:12],
        }


class DummyTrainModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))
        self.frame_patch_size = 1

    def forward_ar(
        self,
        target_x: torch.Tensor,
        context_x: torch.Tensor | None,
        predict_frames: int,
        strict_target_isolation: bool,
        return_modality_dict: bool,
        context_modality_available: torch.Tensor | None = None,
        context_time: torch.Tensor | None = None,
        target_time: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        _ = (
            context_x,
            predict_frames,
            strict_target_isolation,
            return_modality_dict,
            context_modality_available,
            context_time,
            target_time,
        )
        out = target_x * (1.0 + self.scale)
        return {
            "radar": out[:, :1],
            "satellite": out[:, 1:11],
            "rain": out[:, 11:12],
        }


class DummyTrainDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, context: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = torch.cat([context, target], dim=1)
        return x.mean(dim=1, keepdim=True) * self.scale


def _make_batch(batch: int = 2, n_past: int = 4, n_future: int = 3) -> dict[str, torch.Tensor]:
    h, w = 8, 8
    radar_past = torch.arange(batch * 1 * n_past * h * w, dtype=torch.float32).reshape(batch, 1, n_past, h, w)
    sat_past = torch.arange(batch * 10 * n_past * h * w, dtype=torch.float32).reshape(batch, 10, n_past, h, w)
    rain_past = torch.arange(batch * 1 * n_past * h * w, dtype=torch.float32).reshape(batch, 1, n_past, h, w)

    radar_future = torch.arange(batch * 1 * n_future * h * w, dtype=torch.float32).reshape(batch, 1, n_future, h, w)
    sat_future = torch.arange(batch * 10 * n_future * h * w, dtype=torch.float32).reshape(batch, 10, n_future, h, w)
    rain_future = torch.arange(batch * 1 * n_future * h * w, dtype=torch.float32).reshape(batch, 1, n_future, h, w)

    return {
        "radar_past": radar_past,
        "satellite_past": sat_past,
        "rain_past": rain_past,
        "radar_future": radar_future,
        "satellite_future": sat_future,
        "rain_future": rain_future,
    }


def _attach_time(batch_data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    batch = int(batch_data["radar_past"].shape[0])
    n_past = int(batch_data["radar_past"].shape[2])
    n_future = int(batch_data["radar_future"].shape[2])
    batch_data["time_past"] = torch.linspace(0.1, 0.1 * n_past, steps=n_past, dtype=torch.float32).unsqueeze(0).repeat(batch, 1)
    batch_data["time_future"] = (
        torch.linspace(0.1 * (n_past + 1), 0.1 * (n_past + n_future), steps=n_future, dtype=torch.float32)
        .unsqueeze(0)
        .repeat(batch, 1)
    )
    return batch_data


def _make_trainer_for_batch(target_mode: str, block_size: int = 1) -> RainTSNextFrameTrainer:
    trainer = object.__new__(RainTSNextFrameTrainer)
    trainer.device = torch.device("cpu")
    trainer.radar_c = 1
    trainer.satellite_c = 10
    trainer.rain_c = 1
    trainer.train_cfg = OmegaConf.create(
        {
            "next_pred": {
                "target_mode": target_mode,
                "block_size": block_size,
                "missing_modality": {"enabled": False},
            },
            "loss_weights": {"radar": 1.0, "satellite": 1.0, "rain": 1.0},
            "strict_target_isolation": True,
        }
    )
    return trainer


def test_build_next_pred_batch_block_seed_shifted() -> None:
    trainer = _make_trainer_for_batch(target_mode="block", block_size=3)
    batch = _make_batch(n_past=4, n_future=3)

    context, target_seed, target_gt, aux = trainer._build_next_pred_batch(batch)
    seed_dict = trainer._split_modalities(target_seed)

    assert aux["target_mode"] == "block"
    assert aux["target_frames"] == 3

    assert torch.allclose(seed_dict["radar"][:, :, :1], batch["radar_past"][:, :, -1:])
    assert torch.allclose(seed_dict["satellite"][:, :, :1], batch["satellite_past"][:, :, -1:])
    assert torch.allclose(seed_dict["rain"][:, :, :1], batch["rain_past"][:, :, -1:])

    assert torch.allclose(seed_dict["radar"][:, :, 1:], target_gt["radar"][:, :, :-1])
    assert torch.allclose(seed_dict["satellite"][:, :, 1:], target_gt["satellite"][:, :, :-1])
    assert torch.allclose(seed_dict["rain"][:, :, 1:], target_gt["rain"][:, :, :-1])

    assert context.shape[2] == 4


def test_build_next_pred_batch_next_frame_uses_anchor_only() -> None:
    trainer = _make_trainer_for_batch(target_mode="next_frame", block_size=2)
    batch = _make_batch(n_past=4, n_future=3)

    _context, target_seed, _target_gt, aux = trainer._build_next_pred_batch(batch)
    seed_dict = trainer._split_modalities(target_seed)

    assert aux["target_mode"] == "next_frame"
    assert aux["target_frames"] == 1
    assert torch.allclose(seed_dict["radar"], batch["radar_past"][:, :, -1:])
    assert torch.allclose(seed_dict["satellite"], batch["satellite_past"][:, :, -1:])
    assert torch.allclose(seed_dict["rain"], batch["rain_past"][:, :, -1:])


def test_next_prediction_loss_runs() -> None:
    trainer = _make_trainer_for_batch(target_mode="next_frame", block_size=1)
    pred = {
        "radar": torch.zeros(2, 1, 1, 8, 8),
        "satellite": torch.zeros(2, 10, 1, 8, 8),
        "rain": torch.zeros(2, 1, 1, 8, 8),
    }
    target = {
        "radar": torch.ones(2, 1, 1, 8, 8),
        "satellite": torch.ones(2, 10, 1, 8, 8),
        "rain": torch.ones(2, 1, 1, 8, 8),
    }

    loss, logs = trainer._next_prediction_loss(pred=pred, target_gt=target)
    assert torch.isfinite(loss)
    assert set(logs.keys()) == {"loss", "loss/radar", "loss/satellite", "loss/rain"}


def test_rain_weighted_loss_can_exceed_plain_mse() -> None:
    trainer_mse = _make_trainer_for_batch(target_mode="next_frame", block_size=1)
    trainer_enhanced = _make_trainer_for_batch(target_mode="next_frame", block_size=1)
    trainer_enhanced.train_cfg.loss = {
        "mode": "enhanced",
        "rain_region_weight": {"enabled": True, "alpha": 2.0, "r0": 0.0, "gamma": 1.0},
        "rain_event_aux": {"enabled": False},
    }

    pred = {
        "radar": torch.zeros(1, 1, 1, 4, 4),
        "satellite": torch.zeros(1, 10, 1, 4, 4),
        "rain": torch.zeros(1, 1, 1, 4, 4),
    }
    target = {
        "radar": torch.zeros(1, 1, 1, 4, 4),
        "satellite": torch.zeros(1, 10, 1, 4, 4),
        "rain": torch.ones(1, 1, 1, 4, 4),
    }

    loss_mse, _ = trainer_mse._next_prediction_loss(pred=pred, target_gt=target)
    loss_enhanced, logs_enhanced = trainer_enhanced._next_prediction_loss(pred=pred, target_gt=target)

    assert torch.isfinite(loss_mse)
    assert torch.isfinite(loss_enhanced)
    assert float(loss_enhanced.item()) > float(loss_mse.item())
    assert "loss/rain_weighted_reg" in logs_enhanced


def test_rain_event_aux_loss_adds_extra_term() -> None:
    trainer_base = _make_trainer_for_batch(target_mode="next_frame", block_size=1)
    trainer_base.train_cfg.loss = {
        "mode": "enhanced",
        "rain_region_weight": {"enabled": False},
        "rain_event_aux": {"enabled": False},
    }
    trainer_event = _make_trainer_for_batch(target_mode="next_frame", block_size=1)
    trainer_event.train_cfg.loss = {
        "mode": "enhanced",
        "rain_region_weight": {"enabled": False},
        "rain_event_aux": {
            "enabled": True,
            "type": "bce",
            "thresholds": [0.5],
            "weight": 0.5,
            "logit_scale": 10.0,
        },
    }

    pred = {
        "radar": torch.zeros(1, 1, 1, 4, 4),
        "satellite": torch.zeros(1, 10, 1, 4, 4),
        "rain": torch.zeros(1, 1, 1, 4, 4),
    }
    target = {
        "radar": torch.zeros(1, 1, 1, 4, 4),
        "satellite": torch.zeros(1, 10, 1, 4, 4),
        "rain": torch.ones(1, 1, 1, 4, 4),
    }

    loss_base, _ = trainer_base._next_prediction_loss(pred=pred, target_gt=target)
    loss_event, logs_event = trainer_event._next_prediction_loss(pred=pred, target_gt=target)

    assert torch.isfinite(loss_base)
    assert torch.isfinite(loss_event)
    assert float(loss_event.item()) > float(loss_base.item())
    assert "loss/rain_event" in logs_event


def test_apply_context_modality_dropout_radar_always_missing() -> None:
    context = torch.ones(2, 12, 4, 8, 8)
    dropped_context, availability = apply_context_modality_dropout(
        context=context,
        radar_channels=1,
        satellite_channels=10,
        rain_channels=1,
        drop_prob_radar=1.0,
        drop_prob_satellite=0.0,
        drop_prob_rain=0.0,
        min_available_modalities=1,
    )

    assert availability.shape == (2, 3)
    assert torch.all(availability[:, 0] == 0)
    assert torch.all(availability[:, 1] == 1)
    assert torch.all(availability[:, 2] == 1)
    assert torch.allclose(dropped_context[:, :1], torch.zeros_like(dropped_context[:, :1]))
    assert torch.allclose(dropped_context[:, 1:], torch.ones_like(dropped_context[:, 1:]))


def test_build_next_pred_batch_applies_missing_modality_when_enabled() -> None:
    trainer = _make_trainer_for_batch(target_mode="next_frame", block_size=1)
    trainer.train_cfg.next_pred.missing_modality = {
        "enabled": True,
        "drop_probs": {"radar": 1.0, "satellite": 0.0, "rain": 0.0},
        "min_available_modalities": 1,
    }
    batch = _make_batch(n_past=4, n_future=3)

    context, _target_seed, _target_gt, aux = trainer._build_next_pred_batch(batch, apply_missing_modality=True)
    availability = aux["context_modality_available"]

    assert isinstance(availability, torch.Tensor)
    assert availability.shape == (2, 3)
    assert torch.all(availability[:, 0] == 0)
    assert torch.all(availability[:, 1] == 1)
    assert torch.all(availability[:, 2] == 1)
    assert torch.allclose(context[:, :1], torch.zeros_like(context[:, :1]))


def test_build_next_pred_batch_outputs_time_for_autoregressive_seed() -> None:
    trainer = _make_trainer_for_batch(target_mode="next_frame", block_size=2)
    batch = _attach_time(_make_batch(n_past=4, n_future=3))

    _context, _target_seed, _target_gt, aux = trainer._build_next_pred_batch(batch, apply_missing_modality=False)
    assert torch.is_tensor(aux["context_time"])
    assert torch.is_tensor(aux["target_seed_time"])
    context_time = aux["context_time"]
    target_seed_time = aux["target_seed_time"]
    assert context_time.shape == (2, 4)
    assert target_seed_time.shape == (2, 1)


def _make_rollout_trainer(mode: str, block_size: int) -> RainTSNextFrameTrainer:
    trainer = _make_trainer_for_batch(target_mode="block", block_size=block_size)
    trainer.val_cfg = OmegaConf.create(
        {
            "rollout_mode": mode,
            "rollout_block_size": block_size,
            "rollout_history_detach": True,
            "rollout_use_gt_future_modalities": False,
            "after_roll_next": {"enabled": False, "roll_frames": block_size, "detach_history": True},
        }
    )
    trainer.accelerator = DummyAccelerator()
    trainer.model = DummyARModel()
    return trainer


def test_rollout_uses_autoregressive_predictions_frame_mode() -> None:
    trainer = _make_rollout_trainer(mode="frame", block_size=2)
    context = torch.zeros(1, 12, 2, 4, 4)

    pred = trainer._rollout_predict(context=context, total_future_frames=3)

    assert set(pred.keys()) == {"radar", "satellite", "rain"}
    assert pred["rain"].shape == (1, 1, 3, 4, 4)
    assert torch.allclose(pred["rain"][:, :, 0], torch.ones(1, 1, 4, 4) * 1.0)
    assert torch.allclose(pred["rain"][:, :, 1], torch.ones(1, 1, 4, 4) * 2.0)
    assert torch.allclose(pred["rain"][:, :, 2], torch.ones(1, 1, 4, 4) * 3.0)


def test_rollout_uses_autoregressive_predictions_block_mode() -> None:
    trainer = _make_rollout_trainer(mode="block", block_size=2)
    context = torch.zeros(1, 12, 2, 4, 4)

    pred = trainer._rollout_predict(context=context, total_future_frames=3)

    assert pred["radar"].shape == (1, 1, 3, 4, 4)
    assert torch.allclose(pred["radar"][:, :, 0], torch.ones(1, 1, 4, 4) * 1.0)
    assert torch.allclose(pred["radar"][:, :, 1], torch.ones(1, 1, 4, 4) * 2.0)
    assert torch.allclose(pred["radar"][:, :, 2], torch.ones(1, 1, 4, 4) * 3.0)


def test_rollout_with_time_passes_time_to_model() -> None:
    trainer = _make_rollout_trainer(mode="block", block_size=2)
    model = DummyARModelTimeProbe()
    trainer.model = model

    context = torch.zeros(1, 12, 2, 4, 4)
    context_time = torch.tensor([[0.1, 0.2]], dtype=torch.float32)
    future_time = torch.tensor([[0.3, 0.4, 0.5, 0.6]], dtype=torch.float32)

    pred = trainer._rollout_predict_with_settings(
        context=context,
        total_future_frames=4,
        mode="block",
        rollout_block_size=2,
        detach_history=True,
        context_time=context_time,
        future_time=future_time,
    )

    assert pred["rain"].shape == (1, 1, 4, 4, 4)
    assert model.time_call_count > 0


def test_rollout_can_use_gt_future_modalities_for_history_update() -> None:
    trainer = _make_rollout_trainer(mode="block", block_size=2)
    context = torch.zeros(1, 12, 2, 4, 4)
    future_modalities = {
        "radar": torch.ones(1, 1, 4, 4, 4) * 5.0,
        "satellite": torch.ones(1, 10, 4, 4, 4) * 7.0,
        "rain": torch.zeros(1, 1, 4, 4, 4),
    }

    pred = trainer._rollout_predict_with_settings(
        context=context,
        total_future_frames=4,
        mode="block",
        rollout_block_size=2,
        detach_history=True,
        future_modalities=future_modalities,
        use_gt_future_modalities=True,
    )

    assert pred["rain"].shape == (1, 1, 4, 4, 4)
    assert torch.allclose(pred["rain"][:, :, 0], torch.ones(1, 1, 4, 4) * 1.0)
    assert torch.allclose(pred["rain"][:, :, 1], torch.ones(1, 1, 4, 4) * 2.0)
    assert torch.allclose(pred["rain"][:, :, 2], torch.ones(1, 1, 4, 4) * 3.0)
    assert torch.allclose(pred["rain"][:, :, 3], torch.ones(1, 1, 4, 4) * 4.0)
    assert torch.allclose(pred["radar"][:, :, 0], torch.ones(1, 1, 4, 4) * 1.0)
    assert torch.allclose(pred["radar"][:, :, 1], torch.ones(1, 1, 4, 4) * 6.0)
    assert torch.allclose(pred["radar"][:, :, 2], torch.ones(1, 1, 4, 4) * 6.0)
    assert torch.allclose(pred["radar"][:, :, 3], torch.ones(1, 1, 4, 4) * 6.0)


def test_rollout_use_gt_future_modalities_requires_future_modalities() -> None:
    trainer = _make_rollout_trainer(mode="block", block_size=2)
    context = torch.zeros(1, 12, 2, 4, 4)

    try:
        trainer._rollout_predict_with_settings(
            context=context,
            total_future_frames=4,
            mode="block",
            rollout_block_size=2,
            detach_history=True,
            use_gt_future_modalities=True,
        )
    except ValueError as exc:
        assert "future_modalities should be provided" in str(exc)
        return
    raise AssertionError("Expected ValueError when use_gt_future_modalities=True without future_modalities.")


def test_val_step_and_inference_step_have_finite_losses() -> None:
    trainer = _make_rollout_trainer(mode="block", block_size=2)
    batch = _make_batch(batch=1, n_past=2, n_future=3)

    tf_logs = trainer.val_step(batch)
    pred_target, target, infer_loss, extra_logs = trainer._val_inference_step(batch)

    assert "loss" in tf_logs
    assert torch.isfinite(tf_logs["loss"])
    assert torch.isfinite(infer_loss)
    assert len(extra_logs) == 0
    assert pred_target["rain"].shape == target["rain"].shape


def test_val_inference_after_roll_next_loss_exists_when_enabled() -> None:
    trainer = _make_rollout_trainer(mode="block", block_size=2)
    trainer.val_cfg.after_roll_next.enabled = True
    trainer.val_cfg.after_roll_next.roll_frames = 2
    batch = _make_batch(batch=1, n_past=2, n_future=4)

    _pred_target, _target, infer_loss, extra_logs = trainer._val_inference_step(batch)

    assert torch.isfinite(infer_loss)
    assert "val/infer_after_roll_next_loss" in extra_logs
    assert torch.isfinite(extra_logs["val/infer_after_roll_next_loss"])


def test_train_step_with_gan_enabled_outputs_gan_logs() -> None:
    trainer = _make_trainer_for_batch(target_mode="block", block_size=2)
    trainer.accelerator = DummyAccelerator()
    trainer.model = DummyTrainModel()
    trainer.optim = torch.optim.Adam(trainer.model.parameters(), lr=1e-3)
    trainer.sched = torch.optim.lr_scheduler.LambdaLR(trainer.optim, lr_lambda=lambda _step: 1.0)
    trainer.discriminator = DummyTrainDiscriminator()
    trainer.disc_optim = torch.optim.Adam(trainer.discriminator.parameters(), lr=1e-3)
    trainer.disc_sched = torch.optim.lr_scheduler.LambdaLR(trainer.disc_optim, lr_lambda=lambda _step: 1.0)
    trainer.use_gan = True
    trainer.gan_cfg = OmegaConf.create(
        {
            "max_grad_norm": 1.0,
            "loss": {
                "type": "ns",
                "g_weight": 0.1,
                "d_weight": 1.0,
                "r1_weight": 0.0,
                "r2_weight": 0.0,
            },
        }
    )
    trainer.ema_model = None
    trainer.global_step = 0

    batch = _make_batch(batch=1, n_past=2, n_future=2)
    logs, did_step = trainer.train_step(batch)

    assert did_step
    assert "gan/d_loss" in logs
    assert "gan/g_loss" in logs
    assert float(logs["meta/gan_enabled"].item()) == 1.0
