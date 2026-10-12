import torch

from src.residual_diffusion_stage.robust_occurrence_corrector import (
    RobustOccurrenceCorrector,
    robust_occurrence_correction_loss,
)
from src.residual_diffusion_stage.train import _load_corrector_checkpoint


def _config() -> dict:
    return {
        "history_channels": 12,
        "corrector_base_channels": 8,
        "initial_gate_logit": -2.2,
        "corrector_use_occurrence_head": True,
        "initial_occurrence_logit": -2.2,
        "robust_occurrence": {
            "minimum_keep": 0.1,
            "occurrence_power": 1.5,
            "protection_threshold": 0.05,
            "protection_temperature": 0.01,
        },
    }


def test_robust_occurrence_corrector_suppresses_weak_rain_and_preserves_strong_rain() -> None:
    model = RobustOccurrenceCorrector(_config())
    coarse = torch.full((1, 1, 1, 16, 16), 0.02)
    coarse[..., 4:8, 4:8] = 0.2
    history = torch.zeros(1, 12, 3, 16, 16)
    outputs = model(coarse, history)

    assert outputs["center"][..., 0, 0].item() < 0.5 * outputs["center_raw"][..., 0, 0].item()
    torch.testing.assert_close(
        outputs["center"][..., 5, 5],
        outputs["center_raw"][..., 5, 5],
        rtol=1.0e-5,
        atol=1.0e-5,
    )
    assert 0.0 <= outputs["suppression"].min().item()
    assert outputs["suppression"].max().item() <= 1.0


def test_robust_occurrence_loss_penalizes_weak_dry_mass_without_hiding_light_rain() -> None:
    target = torch.zeros(1, 1, 1, 8, 8)
    target[..., 1, 1] = 0.03
    center = torch.full_like(target, 0.02)
    raw_center = center.clone()
    coarse = torch.zeros_like(target)
    gate = torch.ones_like(target)
    occurrence_logits = torch.zeros_like(target)
    suppression = torch.full_like(target, 0.5)
    loss, terms = robust_occurrence_correction_loss(
        center=center,
        raw_center=raw_center,
        target=target,
        coarse=coarse,
        mu=center,
        gate=gate,
        occurrence_logits=occurrence_logits,
        suppression=suppression,
        cfg={
            "active_threshold": 0.01,
            "soft_csi_thresholds": [0.01],
            "weak_rain_cap": 0.05,
            "weak_dry_mass_weight": 2.0,
            "light_rain_upper": 0.1,
            "light_rain_l1_weight": 1.0,
            "strong_rain_retention_weight": 1.0,
            "occurrence_brier_weight": 0.25,
        },
    )

    assert torch.isfinite(loss)
    assert terms["weak_dry_mass"].item() > 0.0
    assert terms["light_rain_l1"].item() > 0.0
    assert terms["strong_rain_retention"].item() == 0.0


def test_loader_restores_robust_occurrence_variant(tmp_path) -> None:
    model = RobustOccurrenceCorrector(_config())
    checkpoint = tmp_path / "robust.pt"
    torch.save(
        {
            "corrector_variant": "robust_occurrence",
            "stage_config": {"model": _config()},
            "model": model.state_dict(),
        },
        checkpoint,
    )
    loaded = _load_corrector_checkpoint(checkpoint, _config(), torch.device("cpu"))
    assert isinstance(loaded, RobustOccurrenceCorrector)
