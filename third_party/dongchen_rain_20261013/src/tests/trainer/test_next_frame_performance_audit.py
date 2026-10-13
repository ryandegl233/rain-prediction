from collections import Counter
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.networks.time_series.causal_patch_transformer_next_frame import RainCausalPatchTransformerNextFrame
from src.tests.trainer.test_rain_trainer_ts_next_frame import DummyAccelerator
from src.trainer.rain_trainer_ts_next_frame import RainTSNextFrameTrainer


def test_current_rollout_computes_zero_weight_decoder_gradients() -> None:
    torch.manual_seed(2025)
    root = Path(__file__).resolve().parents[3]
    cfg = OmegaConf.load(root / "src/config/ts_rain_train/rain_trainer_ts_next_frame.yaml")
    trainer = object.__new__(RainTSNextFrameTrainer)
    trainer.device = torch.device("cpu")
    trainer.train_cfg = cfg.train
    trainer.radar_c, trainer.satellite_c, trainer.rain_c = 1, 10, 1
    trainer.accelerator = DummyAccelerator()
    trainer.global_step = 0
    trainer.model = RainCausalPatchTransformerNextFrame(
        in_channels=12,
        input_size=16,
        patch_size=8,
        frame_patch_size=1,
        stem_channels=32,
        dim=32,
        depth=1,
        num_heads=4,
        decoder_base_channels=16,
        decoder_type="3d",
        decoder_k_size=7,
        encoder_conv_style="wan_factorized",
        activation_checkpoint=True,
    )
    batch = {
        f"{modality}_{period}": torch.rand(1, channels, 4, 16, 16)
        for modality, channels in (("radar", 1), ("satellite", 10), ("rain", 1))
        for period in ("past", "future")
    }
    context, seed, _, aux = trainer._build_next_pred_batch(batch)
    assert (context.shape[2], seed.shape[2], aux["target_frames"]) == (1, 7, 7)

    input_frames: list[int] = []
    decoder_calls: Counter[str] = Counter()

    def record_frames(module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        input_frames.append(int(args[0].shape[1]))

    def record_decoder(module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        decoder_calls[str(id(module))] += 1

    handles = [trainer.model.blocks[0].register_forward_pre_hook(record_frames)]
    decoders = [trainer.model.radar_decoder, trainer.model.satellite_decoder, trainer.model.rain_decoder]
    for decoder in decoders:
        handles.append(decoder.up_layers[-1]._checkpoint_wrapped_module[0].register_forward_pre_hook(record_decoder))
    try:
        loss, _ = trainer._train_rollout_branch_loss(batch)
        assert loss is not None
        assert input_frames == [5, 6, 7, 8]
        assert sorted(decoder_calls.values()) == [4, 4, 4]
        loss.backward()
        assert sorted(decoder_calls.values()) == [8, 8, 8]
        for decoder in decoders[:2]:
            gradient = decoder.out_head.weight.grad
            assert gradient is not None
            assert torch.count_nonzero(gradient).item() == 0
        rain_gradient = trainer.model.rain_decoder.out_head.weight.grad
        assert rain_gradient is not None
        assert torch.count_nonzero(rain_gradient).item() > 0
        print("rollout input frames: [5, 6, 7, 8]; all three decoders: 4 forwards + 4 checkpoint recomputations")
        print("radar/satellite gradients: allocated, all zero; rain gradients: nonzero")
    finally:
        for handle in handles:
            handle.remove()


def test_rain_decoder_alone_preserves_output_and_shared_feature_gradients() -> None:
    torch.manual_seed(2025)
    model = RainCausalPatchTransformerNextFrame(
        input_size=16,
        patch_size=8,
        stem_channels=32,
        dim=32,
        depth=1,
        num_heads=4,
        decoder_base_channels=16,
        decoder_k_size=7,
        activation_checkpoint=True,
        dropout=0.0,
        drop_path=0.0,
    )
    tokens = torch.randn(1, 1, 4, 32, requires_grad=True)
    full = model._decode_target_tokens(
        tokens, hp=2, wp=2, h=16, w=16, batch_size=1, predict_frames=1
    )
    full_loss = full["rain"].square().mean() + 0.0 * (
        full["radar"].square().mean() + full["satellite"].square().mean()
    )
    differentiated = (tokens, *model.rain_decoder.parameters())
    expected_gradients = torch.autograd.grad(full_loss, differentiated)

    features = tokens.reshape(1, 1, 2, 2, 32).permute(0, 4, 1, 2, 3)
    rain_only = model.rain_decoder(features, target_thw=(1, 16, 16))
    actual_gradients = torch.autograd.grad(rain_only.square().mean(), differentiated)
    torch.testing.assert_close(rain_only, full["rain"], rtol=0, atol=0)
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
