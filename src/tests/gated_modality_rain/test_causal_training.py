import subprocess
import sys
from pathlib import Path

import hydra
import pytest
import torch
from accelerate import Accelerator
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.dataset.rain_ts_litdata import WindowIndexIterWrapper
from src.gated_modality_rain.io_trainer import GatedModalityRainIOTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_NAME = "gated_modality_rain_causal"


def small_config() -> DictConfig:
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "src/config/ts_rain_train"), version_base=None):
        cfg = compose(config_name=CONFIG_NAME)
    cfg.rain_prediction_model.update(
        input_size=32,
        patch_size=4,
        stem_channels=16,
        dim=32,
        depth=1,
        num_heads=4,
        decoder_base_channels=32,
        activation_checkpoint=False,
    )
    return cfg


def sample_batch(device: str = "cpu") -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(77)
    batch = {
        f"{modality}_{period}": torch.rand(1, channels, 4, 32, 32, generator=generator).to(device)
        for modality, channels in (("radar", 1), ("satellite", 10), ("rain", 1))
        for period in ("past", "future")
    }
    batch["time_past"] = torch.tensor([[0.1, 0.2, 0.3, 0.4]], device=device)
    batch["time_future"] = torch.tensor([[0.5, 0.6, 0.7, 0.8]], device=device)
    return batch


def small_trainer(device: str = "cpu") -> GatedModalityRainIOTrainer:
    torch.set_num_threads(2)
    torch.manual_seed(2025)
    cfg = small_config()
    trainer = GatedModalityRainIOTrainer.__new__(GatedModalityRainIOTrainer)
    trainer.model = hydra.utils.instantiate(cfg.rain_prediction_model).to(device).eval()
    trainer.train_cfg = cfg.train
    trainer.val_cfg = cfg.val
    trainer.device = torch.device(device)
    trainer.accelerator = Accelerator(cpu=device == "cpu", mixed_precision="no")
    trainer.radar_c, trainer.satellite_c, trainer.rain_c = 1, 10, 1
    trainer.global_step = 8000
    trainer.modality_zero_centering = False
    return trainer


def test_future_seed_frames_cannot_change_earlier_predictions() -> None:
    trainer = small_trainer()
    context = torch.rand(1, 12, 1, 32, 32)
    seed = torch.rand(1, 12, 7, 32, 32)
    changed = seed.clone()
    changed[:, :, -2:] = torch.randn_like(changed[:, :, -2:]) * 3
    with torch.no_grad():
        expected = trainer.model.forward_ar(context, seed)
        actual = trainer.model.forward_ar(context, changed)
    for modality in expected:
        torch.testing.assert_close(actual[modality][:, :, :4], expected[modality][:, :, :4], rtol=0, atol=1e-6)


def test_decoding_a_frame_is_independent_of_other_decoded_frames() -> None:
    model = small_trainer().model.double()
    tokens = torch.randn(1, 7, 64, 32, dtype=torch.float64)
    arguments = dict(hp=8, wp=8, h=32, w=32, batch_size=1)
    with torch.no_grad():
        together = model._decode_target_tokens(tokens, predict_frames=7, **arguments)
        separately = [
            model._decode_target_tokens(tokens[:, index : index + 1], predict_frames=1, **arguments)
            for index in range(7)
        ]
    for modality in together:
        expected = torch.cat([output[modality] for output in separately], dim=2)
        torch.testing.assert_close(together[modality], expected, rtol=1e-10, atol=1e-10)


def test_teacher_forced_step_uses_the_same_context_and_seed_as_rollout() -> None:
    trainer = small_trainer()
    batch = sample_batch()
    context, seed, target, auxiliary = trainer._build_next_pred_batch(batch, apply_missing_modality=False)
    expected_context = torch.cat([batch[f"{name}_past"] for name in ("radar", "satellite", "rain")], dim=1)
    torch.testing.assert_close(context, expected_context)
    torch.testing.assert_close(seed, expected_context[:, :, -1:])
    assert auxiliary["target_frames"] == 1
    for modality in target:
        torch.testing.assert_close(target[modality], batch[f"{modality}_future"][:, :, :1])
    with torch.no_grad():
        direct = trainer.model.forward_ar(
            context, seed, context_time=auxiliary["context_time"], target_time=auxiliary["target_seed_time"]
        )
        direct["rain"] = trainer._apply_rain_residual_output(direct["rain"], auxiliary["rain_residual_ref"])
        rollout, _, _, _ = trainer._val_inference_step(batch)
    for modality in direct:
        torch.testing.assert_close(direct[modality], rollout[modality][:, :, :1], rtol=1e-5, atol=2e-6)


def test_rollout_loss_trains_every_decoder_and_the_spatial_gate() -> None:
    trainer = small_trainer()
    loss, _ = trainer._train_rollout_branch_loss(sample_batch())
    loss.backward()
    assert torch.isfinite(loss)
    for module in (
        trainer.model.radar_decoder,
        trainer.model.satellite_decoder,
        trainer.model.rain_decoder,
        trainer.model.spatial_modality_gate,
    ):
        gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(gradient.abs().sum().item() for gradient in gradients) > 0


def test_fixed_origin_predictions_do_not_read_future_observations() -> None:
    trainer = small_trainer()
    original_batch = sample_batch()
    changed_batch = {name: tensor.clone() for name, tensor in original_batch.items()}
    for modality in ("radar", "satellite", "rain"):
        changed_batch[f"{modality}_future"] = torch.randn_like(changed_batch[f"{modality}_future"]) * 5
    with torch.no_grad():
        original, _, _, _ = trainer._val_inference_step(original_batch)
        changed, _, _, _ = trainer._val_inference_step(changed_batch)
    for modality in original:
        torch.testing.assert_close(changed[modality], original[modality], rtol=0, atol=0)


def test_validation_window_order_is_stable_across_rounds() -> None:
    cfg = small_config()
    wrapper = WindowIndexIterWrapper(mode=cfg.dataset.val.iter_index_mode, seed=2025)
    first = wrapper.build_indices(total=415, epoch=0)
    later = wrapper.build_indices(total=415, epoch=3)
    assert first == later
    assert sorted(first) == list(range(415))


def tiny_loader() -> tuple[list[dict[str, torch.Tensor]], DataLoader]:
    sample = {name: tensor[0] for name, tensor in sample_batch().items()}
    dataset = [sample, sample]
    return dataset, DataLoader(dataset, batch_size=1)


def run_training_case(directory: Path, use_cuda: bool = False) -> None:
    torch.set_num_threads(2)
    cfg = small_config()
    OmegaConf.update(cfg, "accelerator.cpu", not use_cuda, force_add=True)
    cfg.accelerator.mixed_precision = "bf16" if use_cuda else "no"
    cfg.accelerator.log_with = None
    cfg.ema.beta = 0
    cfg.train.proj_dir = str(directory)
    cfg.train.debug = True
    cfg.train.log.log_with_time = False
    cfg.train.log.run_comment = ""
    cfg.train.max_steps = 2
    cfg.train.save_every = 1
    cfg.val.val_duration = 2
    cfg.val.max_val_iters = 1
    cfg.val.save_visuals = False
    cfg.rain_prediction_model.activation_checkpoint = True
    factory = "src.tests.gated_modality_rain.test_causal_training.tiny_loader"
    cfg.dataset.train = {"_target_": factory}
    cfg.dataset.val = {"_target_": factory}
    trainer = GatedModalityRainIOTrainer(cfg)
    model = trainer.accelerator.unwrap_model(trainer.model)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    trainer.run()
    assert trainer.global_step == 2
    for prefix in ("radar_decoder.", "satellite_decoder.", "rain_decoder.", "spatial_modality_gate."):
        assert any(
            not torch.equal(before[name], parameter)
            for name, parameter in model.named_parameters()
            if name.startswith(prefix)
        )
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    assert (directory / "checkpoint-00000001").is_dir()
    assert (directory / "checkpoint-00000002").is_dir()


@pytest.mark.parametrize("use_cuda", [False, True], ids=["cpu", "cuda_bf16"])
def test_training_updates_modules_and_saves_checkpoints(tmp_path: Path, use_cuda: bool) -> None:
    if use_cuda and not torch.cuda.is_available():
        pytest.skip("CUDA is required for the BF16 training check")
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            "-m",
            "src.tests.gated_modality_rain.test_causal_training",
            str(tmp_path),
            str(use_cuda),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_training_cli_resolves_the_independent_configuration() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            "-m",
            "src.trainer.gated_modality_rain_io_trainer",
            "--config-name",
            CONFIG_NAME,
            "--cfg",
            "job",
            "--resolve",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    cfg = OmegaConf.create(result.stdout)
    assert cfg.train.resume_path is None and cfg.train.init_model_path is None
    assert cfg.dataset.val.batch_size == 1
    assert not cfg.dataset.val.shuffle and not cfg.dataset.val.drop_last
    assert cfg.val.max_val_iters >= 415
    assert not cfg.accelerator.project_config.automatic_checkpoint_naming
    assert cfg.accelerator.project_config.total_limit is None


if __name__ == "__main__":
    run_training_case(Path(sys.argv[1]), use_cuda=sys.argv[2] == "True")
