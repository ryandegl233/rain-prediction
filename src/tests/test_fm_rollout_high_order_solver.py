import torch

from src.networks.time_series.diffusion.fm_solvers import FlowDPMSolverMultistepScheduler
from src.networks.time_series.diffusion.fm_solvers_unipc import FlowUniPCMultistepScheduler
from src.networks.time_series.fm_rollout import fm_denoise_target_with_kv_cache


class DummyKVCacheModel:
    def __init__(self, radar_channels: int = 1, satellite_channels: int = 10, rain_channels: int = 1):
        self.radar_channels = radar_channels
        self.satellite_channels = satellite_channels
        self.rain_channels = rain_channels

    def forward_with_context_cache(
        self,
        target_x: torch.Tensor,
        target_timestep: torch.Tensor,
        predict_frames: int,
        strict_target_isolation: bool,
        return_modality_dict: bool,
    ) -> dict[str, torch.Tensor]:
        _ = target_timestep, predict_frames, strict_target_isolation, return_modality_dict
        r = self.radar_channels
        s = self.satellite_channels
        return {
            "radar": target_x[:, :r],
            "satellite": target_x[:, r : r + s],
            "rain": target_x[:, r + s :],
        }


def _run_one_scheduler(scheduler: FlowDPMSolverMultistepScheduler | FlowUniPCMultistepScheduler) -> None:
    torch.manual_seed(0)
    model = DummyKVCacheModel()
    target_noisy = torch.randn(2, 12, 2, 8, 8)

    out = fm_denoise_target_with_kv_cache(
        model=model,  # type: ignore[arg-type]
        scheduler=scheduler,
        target_noisy=target_noisy,
        target_frames=2,
        num_inference_steps=4,
        strict_target_isolation=True,
    )

    assert set(out.keys()) == {"radar", "satellite", "rain"}
    assert out["radar"].shape == (2, 1, 2, 8, 8)
    assert out["satellite"].shape == (2, 10, 2, 8, 8)
    assert out["rain"].shape == (2, 1, 2, 8, 8)
    assert torch.isfinite(out["radar"]).all()
    assert torch.isfinite(out["satellite"]).all()
    assert torch.isfinite(out["rain"]).all()


def test_fm_rollout_with_flow_dpm_solver() -> None:
    scheduler = FlowDPMSolverMultistepScheduler(num_train_timesteps=1000, solver_order=2)
    _run_one_scheduler(scheduler)


def test_fm_rollout_with_flow_unipc_solver() -> None:
    scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, solver_order=2)
    _run_one_scheduler(scheduler)
