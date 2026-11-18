import json
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from src.tests.inference_demo_transfer import (
    load_checkpoint_weights,
    resolve_checkpoint_files,
    select_inference_window,
)
from src.tests.inference_demo_transfer_visualization import (
    _plot_diff_frame,
    save_conditioned_timeline,
    save_conditioning_diff_analysis,
    save_modality_diff_analysis,
    save_rain_residual_diagnostic_analysis,
)


class AcceleratorStub:
    @staticmethod
    def unwrap_model(model: nn.Module) -> nn.Module:
        return model


class WindowDatasetStub:
    def __init__(self) -> None:
        start = datetime(2025, 8, 10, 21, 0)
        self.times_pairs = []
        self.indices_pairs = []
        for index in range(2):
            window_start = start + timedelta(hours=index)
            context_times = [window_start + timedelta(minutes=30 * offset) for offset in range(4)]
            future_times = [window_start + timedelta(minutes=30 * offset) for offset in range(4, 16)]
            self.times_pairs.append((context_times, future_times))
            self.indices_pairs.append((list(range(index, index + 4)), list(range(index + 4, index + 16))))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"window_id": torch.tensor(index)}


def _write_checkpoint_tree(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints" / "step_00000012"
    checkpoint_dir.mkdir(parents=True)
    config_file = run_dir / "config" / "config_total.yaml"
    config_file.parent.mkdir()
    config_file.write_text("train: {}\n", encoding="utf-8")
    (checkpoint_dir / "metadata.json").write_text(
        json.dumps({
            "status": "complete",
            "global_step": 12,
            "config": "../../config/config_total.yaml",
        }),
        encoding="utf-8",
    )
    (run_dir / "checkpoints" / "latest.json").write_text(
        json.dumps({"checkpoint": checkpoint_dir.name}), encoding="utf-8"
    )
    return run_dir, checkpoint_dir


def test_select_inference_window_uses_target_valid_time() -> None:
    dataset = WindowDatasetStub()

    batch, selection = select_inference_window(
        dataset,
        dataset_index=None,
        target_valid_time="2025-08-10 23:00:00",
    )

    assert batch["window_id"].tolist() == [0]
    assert selection["dataset_index"] == 0
    assert selection["target_valid_time"] == "2025-08-10 23:00:00"
    assert len(selection["context_times"]) == 4
    assert len(selection["future_times"]) == 12

    indexed_batch, indexed_selection = select_inference_window(
        dataset,
        dataset_index=1,
        target_valid_time=None,
    )
    assert indexed_batch["window_id"].tolist() == [1]
    assert indexed_selection["dataset_index"] == 1

    default_batch, default_selection = select_inference_window(
        dataset,
        dataset_index=None,
        target_valid_time=None,
    )
    assert default_batch["window_id"].tolist() == [0]
    assert default_selection["dataset_index"] == 0


def test_select_inference_window_rejects_ambiguous_or_out_of_range_selection() -> None:
    dataset = WindowDatasetStub()
    dataset.times_pairs[1] = dataset.times_pairs[0]

    with pytest.raises(ValueError, match="found 2"):
        select_inference_window(
            dataset,
            dataset_index=None,
            target_valid_time="2025-08-10 23:00:00",
        )
    with pytest.raises(IndexError, match="outside"):
        select_inference_window(dataset, dataset_index=2, target_valid_time=None)
    with pytest.raises(ValueError, match="mutually exclusive"):
        select_inference_window(
            dataset,
            dataset_index=0,
            target_valid_time="2025-08-10 23:00:00",
        )


@pytest.mark.parametrize("path_kind", ["run", "root", "latest", "step", "ema", "model"])
def test_resolve_checkpoint_files_supports_saved_layout(tmp_path: Path, path_kind: str) -> None:
    run_dir, checkpoint_dir = _write_checkpoint_tree(tmp_path)
    (checkpoint_dir / "ema.pt").touch()
    (checkpoint_dir / "model.safetensors").touch()
    paths = {
        "run": run_dir,
        "root": run_dir / "checkpoints",
        "latest": run_dir / "checkpoints" / "latest.json",
        "step": checkpoint_dir,
        "ema": checkpoint_dir / "ema.pt",
        "model": checkpoint_dir / "model.safetensors",
    }

    files = resolve_checkpoint_files(paths[path_kind])
    assert files.directory == checkpoint_dir
    assert files.config == run_dir / "config" / "config_total.yaml"
    assert files.metadata == checkpoint_dir / "metadata.json"


def test_resolve_checkpoint_rejects_incomplete_checkpoint(tmp_path: Path) -> None:
    _, checkpoint_dir = _write_checkpoint_tree(tmp_path)
    (checkpoint_dir / "ema.pt").touch()
    (checkpoint_dir / "metadata.json").write_text(
        json.dumps({"status": "writing", "config": "../../config/config_total.yaml"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="not marked complete"):
        resolve_checkpoint_files(checkpoint_dir)


@pytest.mark.parametrize("weight_source", ["ema", "model"])
def test_load_checkpoint_weights_loads_only_model_parameters(tmp_path: Path, weight_source: str) -> None:
    _, checkpoint_dir = _write_checkpoint_tree(tmp_path)
    model = nn.Linear(3, 2)
    expected_weight = torch.full_like(model.weight, 4.0)
    expected_bias = torch.full_like(model.bias, -2.0)
    if weight_source == "ema":
        torch.save(
            {
                "initted": torch.tensor(True),
                "step": torch.tensor(12),
                "ema_model.weight": expected_weight,
                "ema_model.bias": expected_bias,
                "online_model.weight": torch.zeros_like(expected_weight),
                "online_model.bias": torch.zeros_like(expected_bias),
            },
            checkpoint_dir / "ema.pt",
        )
    else:
        save_file({"weight": expected_weight, "bias": expected_bias}, checkpoint_dir / "model.safetensors")
    files = resolve_checkpoint_files(checkpoint_dir)
    trainer = SimpleNamespace(model=model, accelerator=AcceleratorStub(), global_step=0)

    selected, selected_file = load_checkpoint_weights(trainer, files, weight_source)
    assert selected == weight_source
    assert selected_file.name == ("ema.pt" if weight_source == "ema" else "model.safetensors")
    assert trainer.global_step == 12
    torch.testing.assert_close(model.weight, expected_weight)
    torch.testing.assert_close(model.bias, expected_bias)


def test_load_checkpoint_weights_rejects_uninitialized_ema(tmp_path: Path) -> None:
    _, checkpoint_dir = _write_checkpoint_tree(tmp_path)
    model = nn.Linear(3, 2)
    torch.save(
        {
            "initted": torch.tensor(False),
            "ema_model.weight": model.weight.detach(),
            "ema_model.bias": model.bias.detach(),
        },
        checkpoint_dir / "ema.pt",
    )
    files = resolve_checkpoint_files(checkpoint_dir)
    trainer = SimpleNamespace(model=model, accelerator=AcceleratorStub(), global_step=0)

    with pytest.raises(ValueError, match="not initialized"):
        load_checkpoint_weights(trainer, files, "ema")


def test_direct_legacy_ema_uses_embedded_step(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    ema_dir = run_dir / "ema"
    ema_dir.mkdir(parents=True)
    config_file = run_dir / "config" / "config_total.yaml"
    config_file.parent.mkdir()
    config_file.write_text("train: {}\n", encoding="utf-8")
    model = nn.Linear(3, 2)
    torch.save(
        {
            "initted": torch.tensor(True),
            "step": torch.tensor(37),
            "ema_model.weight": model.weight.detach(),
            "ema_model.bias": model.bias.detach(),
        },
        ema_dir / "ema.pt",
    )
    files = resolve_checkpoint_files(ema_dir / "ema.pt")
    trainer = SimpleNamespace(model=model, accelerator=AcceleratorStub(), global_step=0)

    selected, _ = load_checkpoint_weights(trainer, files, "auto")
    assert selected == "ema"
    assert trainer.global_step == 37


def test_transfer_visualization_uses_current_matplotlib_colormap(tmp_path: Path) -> None:
    diff_image = _plot_diff_frame(torch.tensor([[[-1.0, 0.0], [0.5, 1.0]]]), vmax=1.0)
    assert diff_image.shape == (2, 2, 3)
    assert diff_image.dtype == np.uint8

    trainer = SimpleNamespace(
        proj_dir=tmp_path,
        global_step=12,
        modality_zero_centering=False,
    )
    context = torch.linspace(0.0, 0.4, 128).reshape(1, 1, 2, 8, 8)
    target = torch.linspace(0.2, 0.8, 128).reshape(1, 1, 2, 8, 8)
    prediction = target * 0.9
    context_modalities = {
        "radar": context,
        "satellite": context.repeat(1, 10, 1, 1, 1),
        "rain": context,
    }
    target_modalities = {
        "radar": target,
        "satellite": target.repeat(1, 10, 1, 1, 1),
        "rain": target,
    }
    save_conditioned_timeline(
        trainer,
        context_modalities,
        prediction,
        target_modalities,
        future_start=0,
        future_end=2,
        sample_idx=0,
        output_prefix="test_future_0_2",
    )
    save_modality_diff_analysis(
        trainer,
        context,
        prediction,
        target,
        modality_name="rain",
        future_start=0,
        future_end=2,
        sample_idx=0,
        output_prefix="test",
        diff_vmax=None,
    )
    save_conditioning_diff_analysis(
        trainer,
        context,
        target,
        modality_name="radar",
        future_start=0,
        future_end=2,
        sample_idx=0,
        output_prefix="test",
        diff_vmax=None,
    )

    rain_ref = torch.cat([context[:, :, -1:], prediction[:, :, :-1]], dim=2)
    save_rain_residual_diagnostic_analysis(
        trainer,
        {
            "rain_ref": rain_ref,
            "pred_delta": prediction - rain_ref,
            "gt_delta": target - rain_ref,
            "final_rain": prediction,
        },
        target,
        torch.zeros_like(target),
        torch.ones_like(target),
        future_start=0,
        future_end=2,
        sample_idx=0,
        output_prefix="test",
        delta_vmax=None,
        radar_diff_vmax=None,
    )

    viz_dir = tmp_path / "val_viz" / "step_00000012"
    assert (viz_dir / "test_future_0_2_sample0_timeline.jpg").is_file()
    assert (viz_dir / "test_rain_sample0_frames_0_2_diff_grid.png").is_file()
    assert (viz_dir / "test_rain_sample0_frames_0_2_diff_stats.txt").is_file()
    assert (viz_dir / "test_radar_sample0_frames_0_2_diff_grid.png").is_file()
    assert (viz_dir / "test_radar_sample0_frames_0_2_diff_stats.txt").is_file()
    assert (viz_dir / "test_rain_residual_sample0_frames_0_2_grid.png").is_file()
    assert (viz_dir / "test_rain_residual_sample0_frames_0_2_stats.txt").is_file()
