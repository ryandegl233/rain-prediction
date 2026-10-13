from collections import Counter
from pathlib import Path
import os
import subprocess
import sys

import hydra
import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from src.tests.trainer.test_next_frame_transfer import make_config
from src.trainer.rain_trainer_ts_next_frame import RainTSNextFrameTrainer
from src.trainer.rain_trainer_ts_next_frame_transfer import (
    RainCausalPatchTransformerRainDecode,
    RainTSNextFrameTransferTrainer,
)


def make_real_config(tmp_path: Path) -> DictConfig:
    cfg = make_config(tmp_path)
    OmegaConf.set_struct(cfg, False)
    cfg.rain_prediction_model = {
        "_target_": "src.networks.time_series.causal_patch_transformer_next_frame.RainCausalPatchTransformerNextFrame",
        "input_size": 8,
        "patch_size": 4,
        "frame_patch_size": 1,
        "stem_channels": 16,
        "dim": 16,
        "depth": 1,
        "num_heads": 2,
        "decoder_base_channels": 8,
        "decoder_type": "3d",
        "decoder_k_size": 3,
        "dropout": 0.0,
        "drop_path": 0.0,
        "use_time_embedding": True,
        "activation_checkpoint": True,
    }
    return cfg


@pytest.mark.parametrize(
    ("detach_history", "residual", "with_time", "bf16"),
    [(True, True, True, False), (False, True, True, False), (True, False, False, False), (True, True, True, True)],
)
def test_rollout_preserves_rain_loss_gradients_and_gt_history(
    tmp_path: Path, detach_history: bool, residual: bool, with_time: bool, bf16: bool
) -> None:
    cfg = make_real_config(tmp_path / "baseline")
    cfg.train.loss.rain_residual.enabled = residual
    cfg.train.next_pred.rollout_branch = OmegaConf.merge(cfg.train.next_pred.rollout_branch, {
        "detach_history": detach_history,
        "delta_loss": {"enabled": True, "weight": 0.3},
        "weight_start": 0.1,
        "weight_end": 0.5,
        "ramp_steps": 20,
        "lead_weights": [1.0, 2.0, 3.0, 4.0],
        "lead_loss_weight": 0.2,
        "pod_floor": 0.9,
        "pod_floor_weight": 0.1,
    })
    baseline = RainTSNextFrameTrainer(cfg)
    cfg.train.proj_dir = str(tmp_path / "optimized")
    optimized = RainTSNextFrameTransferTrainer(cfg)
    assert isinstance(optimized.model, RainCausalPatchTransformerRainDecode)
    assert cfg.rain_prediction_model._target_.endswith(".RainCausalPatchTransformerNextFrame")
    optimized.model.load_state_dict(baseline.model.state_dict(), strict=True)
    baseline.global_step = optimized.global_step = 7
    batch = next(iter(baseline.train_dataloader))
    if not with_time:
        batch = {key: tensor for key, tensor in batch.items() if not key.startswith("time_")}

    inputs: dict[str, list[torch.Tensor]] = {"baseline": [], "optimized": []}
    rain_outputs: dict[str, list[torch.Tensor]] = {"baseline": [], "optimized": []}
    calls: Counter[str] = Counter()
    handles = []
    for label, trainer in (("baseline", baseline), ("optimized", optimized)):
        def record_input(module: torch.nn.Module, args: tuple[torch.Tensor, ...], label: str = label) -> None:
            inputs[label].append(args[0].detach().clone())

        def record_rain(
            module: torch.nn.Module, args: tuple[torch.Tensor, ...], output: torch.Tensor, label: str = label
        ) -> None:
            rain_outputs[label].append(output.detach().clone())

        handles.append(trainer.model.patch_embed.register_forward_pre_hook(record_input))
        handles.append(trainer.model.rain_decoder.register_forward_hook(record_rain))
    for name in ("radar", "satellite", "rain"):
        def record_decoder(module: torch.nn.Module, args: tuple[torch.Tensor, ...], name: str = name) -> None:
            calls[name] += 1

        handles.append(getattr(optimized.model, f"{name}_decoder").register_forward_pre_hook(record_decoder))
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
            expected_loss, expected_logs = baseline._train_rollout_branch_loss(batch)
            actual_loss, actual_logs = optimized._train_rollout_branch_loss(batch)
        assert expected_loss is not None and actual_loss is not None
        assert calls == Counter(rain=4)
        assert len(inputs["optimized"]) == len(rain_outputs["optimized"]) == 4
        for index, (expected, actual) in enumerate(zip(inputs["baseline"], inputs["optimized"], strict=True)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if index > 0:
                torch.testing.assert_close(actual[:, :1, 4:4 + index], batch["radar_future"][:, :, :index])
                torch.testing.assert_close(actual[:, 1:11, 4:4 + index], batch["satellite_future"][:, :, :index])
        for expected, actual in zip(rain_outputs["baseline"], rain_outputs["optimized"], strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        omitted_logs = {"rollout/loss/radar", "rollout/loss/satellite"}
        assert set(actual_logs) == set(expected_logs) - omitted_logs
        torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
        for name, actual in actual_logs.items():
            torch.testing.assert_close(actual, expected_logs[name], rtol=0, atol=0)

        expected_loss.backward()
        actual_loss.backward()
        for name, parameter in optimized.model.named_parameters():
            expected_gradient = dict(baseline.model.named_parameters())[name].grad
            if name.startswith(("radar_decoder.", "satellite_decoder.")):
                assert parameter.grad is None
                assert expected_gradient is not None and torch.count_nonzero(expected_gradient) == 0
            elif expected_gradient is None:
                assert parameter.grad is None
            else:
                torch.testing.assert_close(parameter.grad, expected_gradient, rtol=1e-5, atol=1e-6)
    finally:
        for handle in handles:
            handle.remove()


def test_teacher_forcing_and_validation_keep_all_modalities(tmp_path: Path) -> None:
    trainer = RainTSNextFrameTransferTrainer(make_real_config(tmp_path))
    batch = next(iter(trainer.train_dataloader))
    context, seed, target, aux = trainer._build_next_pred_batch(batch)
    pred = trainer.model.forward_ar(
        context_x=context,
        target_x=seed,
        predict_frames=int(aux["target_frames"]),
        context_time=aux["context_time"],
        target_time=aux["target_seed_time"],
    )
    loss, logs = trainer._next_prediction_loss(pred, target, aux)
    loss.backward()
    for name in ("radar", "satellite", "rain"):
        gradient = getattr(trainer.model, f"{name}_decoder").out_head.weight.grad
        assert gradient is not None and torch.count_nonzero(gradient) > 0
        assert f"loss/{name}" in logs

    trainer.model.eval()
    context, target, context_time, future_time = trainer._prepare_val_inference_batch(batch)
    with torch.no_grad():
        pred = trainer._rollout_predict_with_settings(
            context, 4, "frame", 1, True, context_time, future_time, target, True
        )
    assert set(pred) == {"radar", "satellite", "rain"}


@pytest.mark.parametrize("setting", ["all", "no_gt", "block", "enhanced", "temporal_diff"])
def test_other_rollout_objectives_keep_original_path(tmp_path: Path, setting: str) -> None:
    cfg = make_real_config(tmp_path / "baseline")
    if setting == "all":
        cfg.train.next_pred.rollout_branch.loss_on = "all"
    elif setting == "no_gt":
        cfg.train.next_pred.rollout_branch.use_gt_future_modalities = False
    elif setting == "block":
        cfg.train.next_pred.rollout_branch.mode = "block"
        cfg.train.next_pred.rollout_branch.rollout_block_size = 2
    elif setting == "enhanced":
        cfg.train.loss.mode = "enhanced"
        cfg.train.loss.rain_event_aux.enabled = True
    else:
        cfg.train.loss.temporal_diff.enabled = True
    baseline = RainTSNextFrameTrainer(cfg)
    cfg.train.proj_dir = str(tmp_path / "optimized")
    optimized = RainTSNextFrameTransferTrainer(cfg)
    batch = next(iter(baseline.train_dataloader))
    expected_loss, expected_logs = baseline._train_rollout_branch_loss(batch)
    actual_loss, actual_logs = optimized._train_rollout_branch_loss(batch)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    assert actual_logs.keys() == expected_logs.keys()


@pytest.mark.parametrize(
    ("decoder_type", "condition", "output_mode"),
    [("2d", "none", "pixels"), ("2d", "film", "residual"), ("3d", "none", "pixels"), ("3d", "film", "residual")],
)
def test_rain_decode_preserves_decoder_options(
    tmp_path: Path, decoder_type: str, condition: str, output_mode: str
) -> None:
    cfg = make_real_config(tmp_path)
    model_cfg = OmegaConf.merge(cfg.rain_prediction_model, {
        "decoder_type": decoder_type,
        "decoder_base_channels": 32,
        "decoder_condition_mode": condition,
        "decoder_output_mode": output_mode,
    })
    baseline = hydra.utils.instantiate(model_cfg)
    model_cfg._target_ = "src.trainer.rain_trainer_ts_next_frame_transfer.RainCausalPatchTransformerRainDecode"
    optimized = hydra.utils.instantiate(model_cfg)
    optimized.load_state_dict(baseline.state_dict(), strict=True)
    context = torch.rand(1, 12, 4, 8, 8)
    seed = context[:, :, -1:]
    expected = baseline.forward_ar(context, seed, return_modality_dict=False)
    actual = optimized.forward_ar(context, seed, return_modality_dict=False)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("launch_mode", ["module", "file"])
def test_entrypoint_uses_rain_only_rollout(tmp_path: Path, launch_mode: str) -> None:
    cfg = make_real_config(tmp_path / "run")
    cfg.train.log.log_every = 1
    OmegaConf.save(cfg, tmp_path / "smoke.yaml")
    root = Path(__file__).resolve().parents[3]
    entrypoint = (
        ["-m", "src.trainer.rain_trainer_ts_next_frame_transfer"]
        if launch_mode == "module"
        else [str(root / "src/trainer/rain_trainer_ts_next_frame_transfer.py")]
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(pytest.__file__).resolve().parents[1])
    result = subprocess.run(
        [
            sys.executable, *entrypoint,
            "--config-path", str(tmp_path), "--config-name", "smoke",
            f"hydra.run.dir={tmp_path / 'run'}",
        ],
        cwd=root if launch_mode == "module" else tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    logs = (tmp_path / "run/log.log").read_text()
    assert "[Train][1/1]" in logs
    assert "loss/radar:" in logs and "loss/satellite:" in logs
    assert "rollout/loss/rain:" in logs
    assert "rollout/loss/radar:" not in logs and "rollout/loss/satellite:" not in logs
