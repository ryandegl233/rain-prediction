import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file

from src.trainer.rain_trainer_ts_next_frame_missing_modality import (
    RainCausalPatchTransformerMissingModality,
    RainTSNextFrameMissingModalityTrainer,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "src" / "config" / "ts_rain_train"
STAGE2_CONFIG_NAME = "rain_trainer_ts_next_frame_missing_modality_stage2_usegt"


def _compose_stage2_cfg() -> DictConfig:
    """Compose the standalone Route-B Stage-2 Hydra config."""
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = hydra.compose(config_name=STAGE2_CONFIG_NAME)
    OmegaConf.resolve(cfg)
    return cfg


class _RecordingFutureModalitiesModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frame_patch_size = 1
        self.use_modality_mask_token = False
        self.proj = nn.Conv3d(4, 4, kernel_size=1)
        self.seen_contexts: list[torch.Tensor] = []

    def forward_ar(
        self,
        context_x: torch.Tensor,
        target_x: torch.Tensor,
        predict_frames: int = 1,
        strict_target_isolation: bool = False,
        return_modality_dict: bool = True,
        context_time: torch.Tensor | None = None,
        target_time: torch.Tensor | None = None,
        context_modality_available: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del strict_target_isolation, return_modality_dict, context_time, target_time, context_modality_available
        self.seen_contexts.append(context_x.detach().clone())
        seq = torch.cat([context_x[:, :, :-1], target_x], dim=2)
        out = self.proj(seq[:, :, -predict_frames:])
        return {
            "radar": out[:, :1],
            "satellite": out[:, 1:3],
            "rain": out[:, 3:4],
        }


class _DummyAccelerator:
    @staticmethod
    def unwrap_model(model: nn.Module) -> nn.Module:
        return model

    @staticmethod
    def autocast():
        return nullcontext()


def test_stage2_usegt_config_overrides_and_inheritance() -> None:
    """Verify Route-B Stage-2 YAML resolves expected overrides and retains missing-modality settings."""
    cfg = _compose_stage2_cfg()

    assert int(cfg.dataset.time_interval) == 30
    assert int(cfg.dataset.train.batch_size) == 2
    assert int(cfg.train.max_steps) == 25000
    assert int(cfg.train.save_every) == 1000
    assert int(cfg.train.checkpoint.archive_every) == 5
    assert cfg.train.resume_path is None

    init_path = Path(str(cfg.train.init_model_path))
    assert init_path.name == "model.safetensors"
    assert init_path.parent.name == "step_00060000"
    assert init_path.exists(), f"Missing Stage-1 init checkpoint: {init_path}"

    assert float(cfg.train.optim.lr) == 5.0e-5
    assert int(cfg.train.scheduler.num_training_steps) == 25000
    assert float(cfg.train.scheduler.min_lr) == 1.0e-6
    assert str(cfg.train.log.run_comment) == "stage2_usegt_from_60k"

    rollout_cfg = cfg.train.next_pred.rollout_branch
    assert bool(rollout_cfg.enabled) is True
    assert bool(rollout_cfg.use_gt_future_modalities) is True
    assert bool(rollout_cfg.detach_rain_anchor) is True
    assert bool(rollout_cfg.detach_history) is True
    assert bool(cfg.train.next_pred.missing_modality.enabled) is True

    assert int(cfg.val.val_duration) == 1000
    assert bool(cfg.val.rollout_use_gt_future_modalities) is True


def test_stage2_checkpoint_weights_match_missing_modality_model() -> None:
    """Verify the 60k safetensors checkpoint loads strictly into RainCausalPatchTransformerMissingModality."""
    cfg = _compose_stage2_cfg()
    model_cfg = OmegaConf.to_container(cfg.rain_prediction_model, resolve=True)
    assert isinstance(model_cfg, dict)
    model_cfg.pop("_target_", None)

    model = RainCausalPatchTransformerMissingModality(**model_cfg)
    state_dict = load_file(str(cfg.train.init_model_path), device="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
    assert missing_keys == []
    assert unexpected_keys == []


def test_rollout_uses_gt_future_modalities_and_detaches_rain_anchor() -> None:
    """Verify use_gt_future_modalities feeds unmasked radar/satellite and detach_rain_anchor blocks step-to-step rain anchor grad."""
    trainer = RainTSNextFrameMissingModalityTrainer.__new__(RainTSNextFrameMissingModalityTrainer)
    model = _RecordingFutureModalitiesModel()
    trainer.model = model
    trainer.accelerator = _DummyAccelerator()
    trainer.device = torch.device("cpu")
    trainer.radar_c = 1
    trainer.satellite_c = 2
    trainer.rain_c = 1
    trainer.train_cfg = SimpleNamespace(strict_target_isolation=False)
    trainer._split_modalities = lambda x: {"radar": x[:, :1], "satellite": x[:, 1:3], "rain": x[:, 3:4]}
    trainer._merge_modalities = lambda r, s, rn: torch.cat([r, s, rn], dim=1)
    trainer._resolve_frame_patch_size = lambda: 1
    trainer._resolve_rain_residual_cfg = lambda: (True, False, 0.0)
    trainer._apply_rain_residual_output = lambda delta, ref: ref + delta

    # Use distinct constants in future radar/satellite so step 1 context confirms GT feeding.
    masked_context = torch.zeros((1, 4, 3, 4, 4), dtype=torch.float32)
    future_gt = torch.zeros((1, 4, 2, 4, 4), dtype=torch.float32)
    future_gt[:, :1, 0] = 0.77
    future_gt[:, 1:3, 0] = -0.55
    future_modalities = {"radar": future_gt[:, :1], "satellite": future_gt[:, 1:3]}

    def _run_rollout(initial_rain_anchor: torch.Tensor, detach_rain_anchor: bool) -> dict[str, torch.Tensor]:
        return trainer._rollout_predict_from_masked_context(
            masked_context=masked_context,
            initial_rain_anchor=initial_rain_anchor,
            total_future_frames=2,
            detach_history=True,
            detach_rain_anchor=detach_rain_anchor,
            history_window_frames=3,
            use_gt_future_modalities=True,
            future_modalities=future_modalities,
            rain_grad_only=True,
        )

    pred = _run_rollout(torch.zeros((1, 1, 1, 4, 4), dtype=torch.float32), detach_rain_anchor=True)
    assert pred["rain"].shape == (1, 1, 2, 4, 4)
    assert len(model.seen_contexts) == 2
    step1_last_frame = model.seen_contexts[1][:, :, -1]
    assert torch.allclose(step1_last_frame[:, :1], torch.full((1, 1, 4, 4), 0.77), atol=1e-6)
    assert torch.allclose(step1_last_frame[:, 1:3], torch.full((1, 2, 4, 4), -0.55), atol=1e-6)

    # Zero conv weight + bias ensures step 0 delta is 0 so step 1 output carries only anchor gradient when undetached.
    with torch.no_grad():
        model.proj.weight.zero_()
        model.proj.bias.zero_()

    for detach_rain_anchor in (True, False):
        rain_anchor = torch.ones((1, 1, 1, 4, 4), dtype=torch.float32, requires_grad=True)
        out = _run_rollout(rain_anchor, detach_rain_anchor=detach_rain_anchor)
        out["rain"][:, :, 1:].sum().backward()
        if detach_rain_anchor:
            # Slicing frame 1 from concatenated outputs yields zero gradient on frame 0 / anchor.
            assert rain_anchor.grad is None or torch.allclose(
                rain_anchor.grad,
                torch.zeros_like(rain_anchor.grad),
            )
        else:
            assert rain_anchor.grad is not None
            assert torch.allclose(rain_anchor.grad, torch.ones_like(rain_anchor.grad))


if __name__ == "__main__":
    test_stage2_usegt_config_overrides_and_inheritance()
    test_stage2_checkpoint_weights_match_missing_modality_model()
    test_rollout_uses_gt_future_modalities_and_detaches_rain_anchor()
    print("All Stage-2 use_gt tests passed.")
