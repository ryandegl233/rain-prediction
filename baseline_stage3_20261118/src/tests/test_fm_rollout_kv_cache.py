import torch

from src.networks.time_series.causal_patch_transformer_diffusion import (
    RainCausalPatchTransformerDiffusion,
)
from src.networks.time_series.fm_rollout import (
    ModalitySpec,
    fm_denoise_target_with_kv_cache,
    rollout_with_kv_cache,
)
from src.networks.time_series.diffusion.fm_scheduler import FlowMatchScheduler


def test_fm_denoise_and_rollout_with_kv_cache():
    # Keep this test deterministic and lightweight across environments.
    device = "cpu"
    torch.manual_seed(0)

    spec = ModalitySpec()
    model = RainCausalPatchTransformerDiffusion(
        in_channels=spec.total_channels,
        radar_out_channels=spec.radar_channels,
        satellite_out_channels=spec.satellite_channels,
        rain_out_channels=spec.rain_channels,
        input_size=64,
        patch_size=4,
        stem_channels=64,
        dim=128,
        depth=2,
        num_heads=8,
        dropout=0.0,
        drop_path=0.0,
        max_frames=16,
        decoder_base_channels=64,
    ).to(device).eval()
    scheduler = FlowMatchScheduler(
        num_inference_steps=4,
        num_train_timesteps=1000,
        shift=5.0,
        sigma_max=1.0,
        sigma_min=0.003 / 1.002,
    )

    b, h, w = 2, 64, 64
    context_t = 4
    target_t = 2
    context_clean = torch.randn(b, spec.total_channels, context_t, h, w, device=device)
    target_noisy = torch.randn(b, spec.total_channels, target_t, h, w, device=device)

    # build context cache
    model.clear_context_cache()
    model.build_context_cache(
        context_x=context_clean,
        context_timestep=torch.zeros((b, context_t), device=device, dtype=torch.float32),
    )

    # denoise one target block
    out = fm_denoise_target_with_kv_cache(
        model=model,
        scheduler=scheduler,
        target_noisy=target_noisy,
        target_frames=target_t,
        num_inference_steps=4,
        strict_target_isolation=True,
    )
    assert set(out.keys()) == {"radar", "satellite", "rain"}
    assert out["radar"].shape == (b, spec.radar_channels, target_t, h, w)
    assert out["satellite"].shape == (b, spec.satellite_channels, target_t, h, w)
    assert out["rain"].shape == (b, spec.rain_channels, target_t, h, w)
    assert torch.isfinite(out["radar"]).all()
    assert torch.isfinite(out["satellite"]).all()
    assert torch.isfinite(out["rain"]).all()

    # rollout with cache append
    rollout = rollout_with_kv_cache(
        model=model,
        scheduler=scheduler,
        context_clean=context_clean,
        horizon_blocks=3,
        block_frames=1,
        num_inference_steps=4,
        strict_target_isolation=True,
        seed=123,
    )
    assert rollout["radar"].shape == (b, spec.radar_channels, 3, h, w)
    assert rollout["satellite"].shape == (b, spec.satellite_channels, 3, h, w)
    assert rollout["rain"].shape == (b, spec.rain_channels, 3, h, w)
    assert torch.isfinite(rollout["radar"]).all()
    assert torch.isfinite(rollout["satellite"]).all()
    assert torch.isfinite(rollout["rain"]).all()
