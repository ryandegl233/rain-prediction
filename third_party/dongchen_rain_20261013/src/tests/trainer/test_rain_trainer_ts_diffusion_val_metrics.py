import torch
from omegaconf import OmegaConf

from src.networks.time_series.causal_forcing.scheduler import FlowMatchScheduler
from src.networks.time_series.diffusion.fm_solvers import FlowDPMSolverMultistepScheduler
from src.networks.time_series.diffusion.fm_solvers_unipc import FlowUniPCMultistepScheduler
from src.networks.time_series.diffusion.gaussian_scheduler import GaussianDiffusionScheduler
from src.trainer.rain_trainer_ts_diffusion import RainTSDiffusionTrainer


def test_scheduler_set_inference_timesteps_ddim_descending_unique() -> None:
    scheduler = GaussianDiffusionScheduler(num_train_timesteps=10)

    timesteps = scheduler.set_inference_timesteps(
        sampler="ddim",
        num_inference_steps=7,
        min_timestep=0,
        max_timestep=9,
        device=torch.device("cpu"),
    )

    assert timesteps.ndim == 1
    assert timesteps[0].item() == 9
    assert timesteps[-1].item() == 0
    assert bool(torch.all(timesteps[:-1] >= timesteps[1:]))
    assert timesteps.unique().numel() == timesteps.numel()


def test_scheduler_set_inference_timesteps_ddpm_full_chain() -> None:
    scheduler = GaussianDiffusionScheduler(num_train_timesteps=10)

    timesteps = scheduler.set_inference_timesteps(
        sampler="ddpm",
        min_timestep=0,
        max_timestep=9,
        device=torch.device("cpu"),
    )

    assert timesteps.tolist() == [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]


def test_scheduler_denoise_api_runs() -> None:
    scheduler = GaussianDiffusionScheduler(num_train_timesteps=10)
    latents = torch.randn(2, 3, 2, 8, 8)

    def predict_fn(x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        assert timestep.shape == (2, 2)
        return torch.zeros_like(x)

    out = scheduler.denoise(
        latents=latents,
        predict_fn=predict_fn,
        prediction_type="epsilon",
        sampler="ddim",
        num_inference_steps=5,
        min_timestep=0,
        max_timestep=9,
        clip_x0=False,
    )
    assert out.shape == latents.shape
    assert torch.isfinite(out).all()


def test_psnr_ssim_sums_return_valid_counts() -> None:
    pred = torch.rand(2, 3, 2, 8, 8)
    target = torch.rand(2, 3, 2, 8, 8)

    psnr_sum, ssim_sum, count = RainTSDiffusionTrainer._psnr_ssim_sums(pred, target, data_range=1.0)

    assert count.item() == 4.0
    assert torch.isfinite(psnr_sum)
    assert torch.isfinite(ssim_sum)


def test_expand_timestep_to_bt_shapes() -> None:
    t_b = torch.tensor([3, 5], dtype=torch.long)
    out_b = RainTSDiffusionTrainer._expand_timestep_to_bt(
        timestep=t_b,
        batch=2,
        frames=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert out_b.shape == (6,)
    assert torch.allclose(out_b, torch.tensor([3, 3, 3, 5, 5, 5], dtype=torch.float32))

    t_bt = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    out_bt = RainTSDiffusionTrainer._expand_timestep_to_bt(
        timestep=t_bt,
        batch=2,
        frames=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert out_bt.shape == (6,)
    assert torch.allclose(out_bt, torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float32))


def test_scheduler_add_noise_fm_supports_bcthw() -> None:
    trainer = object.__new__(RainTSDiffusionTrainer)
    trainer.diffusion_mode = "fm"
    trainer.noise_schedule = FlowMatchScheduler(num_train_timesteps=1000, num_inference_steps=1000)
    trainer.noise_schedule.set_timesteps(num_inference_steps=1000, training=True)

    clean = torch.randn(2, 3, 4, 8, 8)
    noise = torch.randn_like(clean)
    timestep = torch.randint(0, 1000, (2, 4), dtype=torch.long)
    noisy = trainer._scheduler_add_noise(clean=clean, noise=noise, timestep=timestep)
    assert noisy.shape == clean.shape
    assert torch.isfinite(noisy).all()


def test_build_fm_inference_scheduler_types() -> None:
    trainer = object.__new__(RainTSDiffusionTrainer)
    trainer.train_cfg = OmegaConf.create(
        {
            "diffusion": {
                "num_train_timesteps": 1000,
                "fm_shift": 3.0,
                "fm_sigma_max": 1.0,
                "fm_sigma_min": 0.0029940119760479044,
                "fm_inverse_timesteps": False,
                "fm_extra_one_step": False,
                "fm_reverse_sigmas": False,
            }
        }
    )
    trainer.val_cfg = OmegaConf.create({"num_inference_steps": 8})

    scheduler_euler = trainer._build_fm_inference_scheduler("fm_euler")
    scheduler_dpmpp = trainer._build_fm_inference_scheduler("fm_dpmpp")
    scheduler_unipc = trainer._build_fm_inference_scheduler("fm_unipc")
    assert isinstance(scheduler_euler, FlowMatchScheduler)
    assert isinstance(scheduler_dpmpp, FlowDPMSolverMultistepScheduler)
    assert isinstance(scheduler_unipc, FlowUniPCMultistepScheduler)


def test_denormalize_rain_for_metrics_uses_dataset_stats() -> None:
    trainer = object.__new__(RainTSDiffusionTrainer)
    trainer.modality_zero_centering = True
    trainer.rain_norm_mean = 0.5
    trainer.rain_norm_std = 2.0

    rain_norm = torch.tensor([[-1.0, 0.0, 1.0]], dtype=torch.float32)
    rain_raw = trainer._denormalize_rain_for_metrics(rain_norm)

    expected = torch.tensor([[-1.5, 0.5, 2.5]], dtype=torch.float32)
    assert torch.allclose(rain_raw, expected)


def test_denormalize_rain_for_metrics_noop_when_disabled() -> None:
    trainer = object.__new__(RainTSDiffusionTrainer)
    trainer.modality_zero_centering = False
    trainer.rain_norm_mean = None
    trainer.rain_norm_std = None

    rain = torch.tensor([[0.1, 0.2]], dtype=torch.float32)
    out = trainer._denormalize_rain_for_metrics(rain)
    assert out is rain
