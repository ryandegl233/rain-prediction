import pytest
import torch

from src.networks.time_series.causal_patch_transformer_next_frame import RainCausalPatchTransformerNextFrame


def _build_model() -> RainCausalPatchTransformerNextFrame:
    return RainCausalPatchTransformerNextFrame(
        in_channels=12,
        out_channels=1,
        radar_out_channels=1,
        satellite_out_channels=10,
        rain_out_channels=1,
        input_size=32,
        patch_size=4,
        stem_channels=32,
        dim=64,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        drop_path=0.0,
        max_frames=8,
        decoder_base_channels=32,
        activation_checkpoint=False,
    )


def test_forward_shapes_and_return_modes() -> None:
    model = _build_model()
    x = torch.randn(2, 12, 5, 32, 32)

    out_dict = model(x=x, predict_frames=1, strict_target_isolation=True, return_modality_dict=True)
    assert set(out_dict.keys()) == {"radar", "satellite", "rain"}
    assert out_dict["radar"].shape == (2, 1, 1, 32, 32)
    assert out_dict["satellite"].shape == (2, 10, 1, 32, 32)
    assert out_dict["rain"].shape == (2, 1, 1, 32, 32)

    out_rain = model(x=x, predict_frames=2, strict_target_isolation=True, return_modality_dict=False)
    assert out_rain.shape == (2, 1, 2, 32, 32)


def test_forward_ar_block_shapes() -> None:
    model = _build_model()
    context = torch.randn(2, 12, 4, 32, 32)
    target_seed = torch.randn(2, 12, 3, 32, 32)

    out = model.forward_ar(
        context_x=context,
        target_x=target_seed,
        predict_frames=3,
        strict_target_isolation=True,
        return_modality_dict=True,
    )
    assert out["radar"].shape == (2, 1, 3, 32, 32)
    assert out["satellite"].shape == (2, 10, 3, 32, 32)
    assert out["rain"].shape == (2, 1, 3, 32, 32)


def test_forward_ar_with_time_embedding_runs() -> None:
    model = RainCausalPatchTransformerNextFrame(
        in_channels=12,
        out_channels=1,
        radar_out_channels=1,
        satellite_out_channels=10,
        rain_out_channels=1,
        input_size=32,
        patch_size=4,
        stem_channels=32,
        dim=64,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        drop_path=0.0,
        max_frames=8,
        decoder_base_channels=32,
        use_time_embedding=True,
        activation_checkpoint=False,
    )
    context = torch.randn(2, 12, 4, 32, 32)
    target_seed = torch.randn(2, 12, 3, 32, 32)
    context_time = torch.rand(2, 4)
    target_time = torch.rand(2, 3)

    out = model.forward_ar(
        context_x=context,
        target_x=target_seed,
        predict_frames=3,
        strict_target_isolation=True,
        return_modality_dict=True,
        context_time=context_time,
        target_time=target_time,
    )
    assert out["radar"].shape == (2, 1, 3, 32, 32)
    assert out["satellite"].shape == (2, 10, 3, 32, 32)
    assert out["rain"].shape == (2, 1, 3, 32, 32)


def test_forward_ar_frame_patch_size_upsample() -> None:
    model = RainCausalPatchTransformerNextFrame(
        in_channels=12,
        out_channels=1,
        radar_out_channels=1,
        satellite_out_channels=10,
        rain_out_channels=1,
        input_size=32,
        patch_size=4,
        frame_patch_size=2,
        stem_channels=32,
        dim=64,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        drop_path=0.0,
        max_frames=8,
        decoder_base_channels=32,
        activation_checkpoint=False,
    )
    context = torch.randn(2, 12, 4, 32, 32)
    target_seed = torch.randn(2, 12, 2, 32, 32)

    out = model.forward_ar(
        context_x=context,
        target_x=target_seed,
        predict_frames=2,
        strict_target_isolation=True,
        return_modality_dict=True,
    )
    assert out["radar"].shape == (2, 1, 2, 32, 32)
    assert out["satellite"].shape == (2, 10, 2, 32, 32)
    assert out["rain"].shape == (2, 1, 2, 32, 32)


def test_forward_frame_patch_size_requires_divisible_predict_frames() -> None:
    model = RainCausalPatchTransformerNextFrame(
        in_channels=12,
        out_channels=1,
        radar_out_channels=1,
        satellite_out_channels=10,
        rain_out_channels=1,
        input_size=32,
        patch_size=4,
        frame_patch_size=2,
        stem_channels=32,
        dim=64,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        drop_path=0.0,
        max_frames=8,
        decoder_base_channels=32,
        activation_checkpoint=False,
    )
    x = torch.randn(2, 12, 6, 32, 32)

    with pytest.raises(ValueError, match="predict_frames must be divisible"):
        _ = model(x=x, predict_frames=1, strict_target_isolation=True, return_modality_dict=True)


def test_forward_with_pixelshuffle_and_replicate_padding() -> None:
    model = RainCausalPatchTransformerNextFrame(
        in_channels=12,
        out_channels=1,
        radar_out_channels=1,
        satellite_out_channels=10,
        rain_out_channels=1,
        input_size=32,
        patch_size=4,
        stem_channels=32,
        dim=64,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        drop_path=0.0,
        max_frames=8,
        decoder_base_channels=32,
        stem_pad_mode="replicate",
        decoder_pad_mode="replicate",
        decoder_upsample_mode="pixelshuffle",
        activation_checkpoint=False,
    )
    x = torch.randn(2, 12, 5, 32, 32)

    out = model(x=x, predict_frames=2, strict_target_isolation=True, return_modality_dict=True)
    assert out["radar"].shape == (2, 1, 2, 32, 32)
    assert out["satellite"].shape == (2, 10, 2, 32, 32)
    assert out["rain"].shape == (2, 1, 2, 32, 32)


def test_forward_with_decoder_kernel_size_five_runs() -> None:
    model = RainCausalPatchTransformerNextFrame(
        in_channels=12,
        out_channels=1,
        radar_out_channels=1,
        satellite_out_channels=10,
        rain_out_channels=1,
        input_size=32,
        patch_size=4,
        stem_channels=32,
        dim=64,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        drop_path=0.0,
        max_frames=8,
        decoder_base_channels=32,
        decoder_k_size=5,
        activation_checkpoint=False,
    )
    x = torch.randn(2, 12, 5, 32, 32)

    out = model(x=x, predict_frames=2, strict_target_isolation=True, return_modality_dict=True)
    assert out["radar"].shape == (2, 1, 2, 32, 32)
    assert out["satellite"].shape == (2, 10, 2, 32, 32)
    assert out["rain"].shape == (2, 1, 2, 32, 32)


def test_decoder_kernel_size_requires_odd_value() -> None:
    with pytest.raises(ValueError, match="decoder_k_size must be odd"):
        _ = RainCausalPatchTransformerNextFrame(
            in_channels=12,
            out_channels=1,
            radar_out_channels=1,
            satellite_out_channels=10,
            rain_out_channels=1,
            input_size=32,
            patch_size=4,
            stem_channels=32,
            dim=64,
            depth=2,
            num_heads=4,
            mlp_ratio=2.0,
            dropout=0.0,
            drop_path=0.0,
            max_frames=8,
            decoder_base_channels=32,
            decoder_k_size=4,
            activation_checkpoint=False,
        )


def test_forward_residual_output_uses_last_context_anchor() -> None:
    model = RainCausalPatchTransformerNextFrame(
        in_channels=12,
        out_channels=1,
        radar_out_channels=1,
        satellite_out_channels=10,
        rain_out_channels=1,
        input_size=32,
        patch_size=4,
        stem_channels=32,
        dim=64,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        drop_path=0.0,
        max_frames=8,
        decoder_base_channels=32,
        decoder_output_mode="residual",
        activation_checkpoint=False,
    )
    for decoder in (model.radar_decoder, model.satellite_decoder, model.rain_decoder):
        for param in decoder.parameters():
            param.detach().zero_()

    x = torch.randn(2, 12, 5, 32, 32)
    out = model(x=x, predict_frames=2, strict_target_isolation=True, return_modality_dict=True)

    context_last = x[:, :, 2:3]
    radar_anchor = context_last[:, :1].expand(-1, -1, 2, -1, -1)
    satellite_anchor = context_last[:, 1:11].expand(-1, -1, 2, -1, -1)
    rain_anchor = context_last[:, 11:12].expand(-1, -1, 2, -1, -1)

    assert torch.allclose(out["radar"], radar_anchor, atol=1e-6)
    assert torch.allclose(out["satellite"], satellite_anchor, atol=1e-6)
    assert torch.allclose(out["rain"], rain_anchor, atol=1e-6)


def test_forward_with_film_condition_runs() -> None:
    model = RainCausalPatchTransformerNextFrame(
        in_channels=12,
        out_channels=1,
        radar_out_channels=1,
        satellite_out_channels=10,
        rain_out_channels=1,
        input_size=32,
        patch_size=4,
        stem_channels=32,
        dim=64,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        drop_path=0.0,
        max_frames=8,
        decoder_base_channels=32,
        decoder_condition_mode="film",
        decoder_output_mode="residual",
        activation_checkpoint=False,
    )
    x = torch.randn(2, 12, 5, 32, 32)

    out = model(x=x, predict_frames=2, strict_target_isolation=True, return_modality_dict=True)
    assert out["radar"].shape == (2, 1, 2, 32, 32)
    assert out["satellite"].shape == (2, 10, 2, 32, 32)
    assert out["rain"].shape == (2, 1, 2, 32, 32)


def test_forward_resnet_encoder_shapes_and_multiple_predict_frames() -> None:
    model = RainCausalPatchTransformerNextFrame(
        in_channels=12,
        out_channels=1,
        radar_out_channels=1,
        satellite_out_channels=10,
        rain_out_channels=1,
        input_size=32,
        patch_size=4,
        frame_patch_size=2,
        encoder_type="resnet",
        encoder_spatial_downsample_stages=2,
        encoder_temporal_downsample_stages=1,
        encoder_causal=True,
        stem_channels=32,
        dim=64,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        drop_path=0.0,
        max_frames=8,
        decoder_base_channels=32,
        activation_checkpoint=False,
    )
    x = torch.randn(2, 12, 6, 32, 32)

    for predict_frames in (2, 4):
        out = model(x=x, predict_frames=predict_frames, strict_target_isolation=True, return_modality_dict=True)
        assert out["radar"].shape == (2, 1, predict_frames, 32, 32)
        assert out["satellite"].shape == (2, 10, predict_frames, 32, 32)
        assert out["rain"].shape == (2, 1, predict_frames, 32, 32)


def test_resnet_encoder_downsample_token_shape_matches_patch_scales() -> None:
    model = RainCausalPatchTransformerNextFrame(
        in_channels=12,
        out_channels=1,
        radar_out_channels=1,
        satellite_out_channels=10,
        rain_out_channels=1,
        input_size=64,
        patch_size=8,
        frame_patch_size=2,
        encoder_type="resnet",
        encoder_spatial_downsample_stages=3,
        encoder_temporal_downsample_stages=1,
        encoder_causal=True,
        stem_channels=32,
        dim=64,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        drop_path=0.0,
        max_frames=8,
        decoder_base_channels=32,
        activation_checkpoint=False,
    )
    x = torch.randn(1, 12, 6, 64, 64)

    tokens, hp, wp, _, _, _ = model._encode_tokens(x=x, frame_offset=0)
    assert hp == 8
    assert wp == 8
    assert tokens.shape == (1, 3, 64, 64)


def test_resnet_encoder_requires_patch_size_consistency() -> None:
    with pytest.raises(ValueError, match="resnet encoder requires patch_size"):
        _ = RainCausalPatchTransformerNextFrame(
            in_channels=12,
            out_channels=1,
            radar_out_channels=1,
            satellite_out_channels=10,
            rain_out_channels=1,
            input_size=32,
            patch_size=4,
            frame_patch_size=1,
            encoder_type="resnet",
            encoder_spatial_downsample_stages=3,
            encoder_temporal_downsample_stages=0,
            stem_channels=32,
            dim=64,
            depth=2,
            num_heads=4,
            mlp_ratio=2.0,
            dropout=0.0,
            drop_path=0.0,
            max_frames=8,
            decoder_base_channels=32,
            activation_checkpoint=False,
        )


def test_resnet_encoder_requires_frame_patch_size_consistency() -> None:
    with pytest.raises(ValueError, match="resnet encoder requires frame_patch_size"):
        _ = RainCausalPatchTransformerNextFrame(
            in_channels=12,
            out_channels=1,
            radar_out_channels=1,
            satellite_out_channels=10,
            rain_out_channels=1,
            input_size=32,
            patch_size=4,
            frame_patch_size=2,
            encoder_type="resnet",
            encoder_spatial_downsample_stages=2,
            encoder_temporal_downsample_stages=0,
            stem_channels=32,
            dim=64,
            depth=2,
            num_heads=4,
            mlp_ratio=2.0,
            dropout=0.0,
            drop_path=0.0,
            max_frames=8,
            decoder_base_channels=32,
            activation_checkpoint=False,
        )


def test_context_modality_mask_token_replaces_missing_modalities() -> None:
    model = _build_model()
    context = torch.randn(2, 12, 4, 16, 16)
    availability = torch.tensor(
        [
            [0, 1, 1],
            [1, 0, 1],
        ],
        dtype=torch.bool,
    )

    masked = model._apply_context_modality_mask_token(
        context_x=context,
        context_modality_available=availability,
    )

    radar_token = model.radar_mask_token.expand(2, -1, 4, 16, 16)
    satellite_token = model.satellite_mask_token.expand(2, -1, 4, 16, 16)

    assert torch.allclose(masked[0:1, :1], radar_token[0:1], atol=1e-6)
    assert torch.allclose(masked[1:2, 1:11], satellite_token[1:2], atol=1e-6)
    assert torch.allclose(masked[0:1, 1:11], context[0:1, 1:11], atol=1e-6)
    assert torch.allclose(masked[1:2, :1], context[1:2, :1], atol=1e-6)
