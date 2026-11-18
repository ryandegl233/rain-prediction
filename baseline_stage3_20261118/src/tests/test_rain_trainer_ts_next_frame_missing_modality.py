from contextlib import nullcontext
import sys
from pathlib import Path
from types import SimpleNamespace

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn

from src.trainer.rain_trainer_ts_next_frame_missing_modality import (
    NONEMPTY_MODALITY_COMBINATIONS,
    RainCausalPatchTransformerMissingModality,
    RainTSNextFrameMissingModalityTrainer,
    compute_carry_forward_rain_ref,
    sample_per_frame_modality_mask,
)


class _DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frame_patch_size = 1
        self.use_modality_mask_token = True
        self.radar_mask_token = nn.Parameter(torch.full((1, 1, 1, 1, 1), 0.11))
        self.satellite_mask_token = nn.Parameter(torch.full((1, 2, 1, 1, 1), -0.22))
        self.rain_mask_token = nn.Parameter(torch.full((1, 1, 1, 1, 1), 0.33))
        self.proj = nn.Conv3d(4, 4, kernel_size=1)

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


def test_sample_per_frame_modality_mask_distribution_and_first_frame_anchor() -> None:
    torch.manual_seed(2025)
    mask = sample_per_frame_modality_mask(
        batch_size=4000,
        num_frames=5,
        device=torch.device("cpu"),
        full_modality_prob=0.6,
        anchor_first_frame=True,
    )
    assert mask.shape == (4000, 3, 5)
    # Frame 0 must be 100% (True, True, True)
    assert bool(mask[:, :, 0].all().item())

    # Frames 1..4 should have ~60% full modality and ~40%/6 per incomplete combination
    subsequent = mask[:, :, 1:].permute(0, 2, 1).reshape(-1, 3)  # [16000, 3]
    is_full = subsequent.all(dim=-1).float().mean().item()
    assert 0.58 <= is_full <= 0.62, f"Expected ~0.60 full modality rate, got {is_full}"

    for combo in NONEMPTY_MODALITY_COMBINATIONS[1:]:
        combo_tensor = torch.tensor(combo, dtype=torch.bool)
        rate = (subsequent == combo_tensor).all(dim=-1).float().mean().item()
        expected = 0.4 / 6.0
        assert abs(rate - expected) < 0.015, f"Combo {combo} rate {rate:.4f} far from {expected:.4f}"


def test_compute_carry_forward_rain_ref_preserves_latest_available_rain() -> None:
    # Shape: [B=2, C=1, T=4, H=2, W=2]
    rain = torch.zeros((2, 1, 4, 2, 2), dtype=torch.float32)
    for t_idx, val in enumerate((1.0, 2.0, 3.0, 4.0)):
        rain[:, :, t_idx] = val

    # Sample 0: [True, False, False, True] -> refs should be [1, 1, 1, 4]
    # Sample 1: [True, True, False, False] -> refs should be [1, 2, 2, 2]
    rain_available = torch.tensor(
        [
            [True, False, False, True],
            [True, True, False, False],
        ],
        dtype=torch.bool,
    )
    refs = compute_carry_forward_rain_ref(rain, rain_available)
    assert refs.shape == rain.shape

    expected_0 = [1.0, 1.0, 1.0, 4.0]
    expected_1 = [1.0, 2.0, 2.0, 2.0]
    for t_idx in range(4):
        assert torch.allclose(refs[0, :, t_idx], torch.full((1, 2, 2), expected_0[t_idx]))
        assert torch.allclose(refs[1, :, t_idx], torch.full((1, 2, 2), expected_1[t_idx]))


def test_mask_token_replacement_and_rollout_gradient_flow() -> None:
    trainer = RainTSNextFrameMissingModalityTrainer.__new__(RainTSNextFrameMissingModalityTrainer)
    trainer.model = _DummyModel()
    trainer.accelerator = _DummyAccelerator()
    trainer.device = torch.device("cpu")
    trainer.radar_c = 1
    trainer.satellite_c = 2
    trainer.rain_c = 1
    trainer.train_cfg = SimpleNamespace(
        strict_target_isolation=False,
        next_pred={
            "missing_modality": {
                "enabled": True,
                "full_modality_prob": 0.6,
                "anchor_first_frame": True,
                "mask_future_seed_in_sequence_loss": True,
            }
        },
    )
    trainer._split_modalities = lambda x: {"radar": x[:, :1], "satellite": x[:, 1:3], "rain": x[:, 3:4]}
    trainer._merge_modalities = lambda r, s, rn: torch.cat([r, s, rn], dim=1)
    trainer._resolve_frame_patch_size = lambda: 1
    trainer._resolve_rain_residual_cfg = lambda: (True, False, 0.0)
    trainer._apply_rain_residual_output = lambda delta, ref: ref + delta

    raw_context = torch.ones((2, 4, 4, 4, 4), dtype=torch.float32)
    raw_context[:, 3:4, 0] = 0.5  # Frame 0 rain = 0.5
    raw_context[:, 3:4, 1:] = 9.0  # Subsequent rain = 9.0 (will be masked)

    # Frame 0 all True, Frame 1 misses radar, Frame 2 misses sat, Frame 3 misses rain
    mask = torch.tensor(
        [
            [
                [True, False, True, True],   # radar
                [True, True, False, True],   # satellite
                [True, True, True, False],   # rain
            ],
            [
                [True, True, True, False],   # radar
                [True, False, True, False],  # satellite
                [True, False, False, False], # rain (all 1..3 missing -> falls back to Frame 0 = 0.5)
            ],
        ],
        dtype=torch.bool,
    )

    masked_context = trainer._apply_per_frame_mask_tokens(raw_context, mask)
    rain_ref = compute_carry_forward_rain_ref(raw_context[:, 3:4], mask[:, 2, :])
    initial_rain_anchor = rain_ref[:, :, -1:]

    # Sample 1 had rain masked on frames 1, 2, 3 -> initial_rain_anchor must equal Frame 0's rain (0.5), NOT 9.0 or 0.33
    assert torch.allclose(initial_rain_anchor[1], torch.full((1, 1, 4, 4), 0.5))

    out = trainer._rollout_predict_from_masked_context(
        masked_context=masked_context,
        initial_rain_anchor=initial_rain_anchor,
        total_future_frames=4,
        detach_history=False,
        use_gt_future_modalities=False,
        return_rain_delta=True,
    )
    loss = out["rain"].mean() + out["radar"].mean() + out["satellite"].mean()
    loss.backward()

    assert trainer.model.radar_mask_token.grad is not None
    assert trainer.model.satellite_mask_token.grad is not None
    assert trainer.model.rain_mask_token.grad is not None
    assert float(trainer.model.radar_mask_token.grad.abs().sum().item()) > 0.0
    assert float(trainer.model.satellite_mask_token.grad.abs().sum().item()) > 0.0
    assert float(trainer.model.rain_mask_token.grad.abs().sum().item()) > 0.0


def test_rollout_aux_no_grad_skips_aux_decoder_autograd() -> None:
    model = RainCausalPatchTransformerMissingModality(
        in_channels=4,
        out_channels=1,
        radar_out_channels=1,
        satellite_out_channels=2,
        rain_out_channels=1,
        input_size=16,
        patch_size=8,
        frame_patch_size=1,
        dim=32,
        depth=1,
        num_heads=4,
        decoder_base_channels=16,
        decoder_type="3d",
        use_modality_mask_token=True,
        activation_checkpoint=False,
    )
    model.train()
    model.rollout_aux_no_grad = True

    tokens = torch.randn((2, 1, 4, 32), requires_grad=True)
    out = model._decode_target_tokens(
        target_tokens=tokens,
        hp=2,
        wp=2,
        h=16,
        w=16,
        batch_size=2,
        predict_frames=1,
        return_modality_dict=True,
    )
    assert isinstance(out, dict)
    assert not out["radar"].requires_grad
    assert not out["satellite"].requires_grad
    assert out["rain"].requires_grad

    out["rain"].mean().backward()
    assert tokens.grad is not None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.rain_decoder.parameters())
    assert all(p.grad is None for p in model.radar_decoder.parameters())
    assert all(p.grad is None for p in model.satellite_decoder.parameters())


if __name__ == "__main__":
    test_sample_per_frame_modality_mask_distribution_and_first_frame_anchor()
    test_compute_carry_forward_rain_ref_preserves_latest_available_rain()
    test_mask_token_replacement_and_rollout_gradient_flow()
    test_rollout_aux_no_grad_skips_aux_decoder_autograd()
    print("All tests passed.")
