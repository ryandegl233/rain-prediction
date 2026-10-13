from types import SimpleNamespace

import torch

from src.networks.time_series.causal_forcing.diffusion import CausalDiffusion
from src.networks.time_series.causal_forcing.scheduler import FlowMatchScheduler


class DummyRainGenerator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_channels = 12
        self.radar_out_channels = 1
        self.satellite_out_channels = 10
        self.rain_out_channels = 1

    def forward(
        self,
        x: torch.Tensor,
        diffusion_timestep: torch.Tensor,
        predict_frames: int,
        strict_target_isolation: bool,
        return_modality_dict: bool,
    ) -> dict[str, torch.Tensor]:
        _ = diffusion_timestep, strict_target_isolation, return_modality_dict
        target = x[:, :, -predict_frames:]
        return {
            "radar": target[:, :1],
            "satellite": target[:, 1:11],
            "rain": target[:, 11:12],
        }


def test_flow_match_scheduler_conversions_are_consistent() -> None:
    scheduler = FlowMatchScheduler(num_inference_steps=32, num_train_timesteps=1000, shift=5.0, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(1000, training=True)

    x0 = torch.randn(4, 3, 8, 8)
    noise = torch.randn_like(x0)
    index = torch.randint(0, scheduler.timesteps.numel(), (4,))
    timestep = scheduler.timesteps[index]

    xt = scheduler.add_noise(x0, noise, timestep)
    flow = scheduler.training_target(x0, noise, timestep)

    x0_from_flow = scheduler.flow_pred_to_x0(flow_pred=flow, xt=xt, timestep=timestep)
    flow_from_x0 = scheduler.x0_to_flow_pred(x0=x0, xt=xt, timestep=timestep)
    noise_from_x0 = scheduler.convert_x0_to_noise(x0=x0, xt=xt, timestep=timestep)
    x0_from_noise = scheduler.convert_noise_to_x0(noise=noise_from_x0, xt=xt, timestep=timestep)

    torch.testing.assert_close(x0_from_flow, x0, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(flow_from_x0, flow, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(x0_from_noise, x0, rtol=1e-4, atol=2e-5)

    weight = scheduler.training_weight(timestep)
    assert weight.shape == (4,)
    assert torch.isfinite(weight).all()


def test_causal_diffusion_defaults_to_flow_match_scheduler() -> None:
    args = SimpleNamespace(
        generator=DummyRainGenerator(),
        num_train_timestep=1000,
        num_frame_per_block=1,
        independent_first_frame=False,
        mixed_precision=False,
        timestep_shift=5.0,
        teacher_forcing=True,
        noise_augmentation_max_timestep=0,
        denoising_loss_type="flow",
    )

    model = CausalDiffusion(args=args, device=torch.device("cpu"))

    assert isinstance(model.scheduler, FlowMatchScheduler)
    clean_latent = torch.randn(2, 4, 12, 8, 8)
    loss, log_dict = model.generator_loss(
        image_or_video_shape=list(clean_latent.shape),
        conditional_dict={},
        unconditional_dict={},
        clean_latent=clean_latent,
        initial_latent=None,
    )

    assert torch.isfinite(loss)
    assert "x0" in log_dict
    assert "x0_pred" in log_dict
    assert log_dict["x0"].shape == clean_latent[:, :1].shape
    assert log_dict["x0_pred"].shape == clean_latent[:, :1].shape
