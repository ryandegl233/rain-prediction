import torch

from src.dataset.rain_ts_augmentation import RainTimeSeriesAugmentor


def _build_inputs() -> dict[str, torch.Tensor]:
    radar_past = torch.arange(1 * 3 * 8 * 8, dtype=torch.float32).reshape(1, 3, 8, 8)
    radar_future = torch.arange(1 * 2 * 8 * 8, dtype=torch.float32).reshape(1, 2, 8, 8)
    satellite_past = torch.arange(10 * 3 * 8 * 8, dtype=torch.float32).reshape(10, 3, 8, 8)
    satellite_future = torch.arange(10 * 2 * 8 * 8, dtype=torch.float32).reshape(10, 2, 8, 8)
    rain_past = torch.arange(1 * 3 * 8 * 8, dtype=torch.float32).reshape(1, 3, 8, 8)
    rain_future = torch.arange(1 * 2 * 8 * 8, dtype=torch.float32).reshape(1, 2, 8, 8)
    time_past = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    time_future = torch.tensor([0.4, 0.5], dtype=torch.float32)
    return {
        "radar_past": radar_past,
        "radar_future": radar_future,
        "satellite_past": satellite_past,
        "satellite_future": satellite_future,
        "rain_past": rain_past,
        "rain_future": rain_future,
        "time_past": time_past,
        "time_future": time_future,
    }


def test_random_crop_outputs_bbox_and_keeps_shape_when_configured() -> None:
    torch.manual_seed(0)
    augmentor = RainTimeSeriesAugmentor(
        enabled=True,
        random_crop_prob=1.0,
        random_crop_min_scale=0.5,
        random_crop_max_scale=0.5,
        random_crop_keep_size=True,
        temporal_reverse_prob=0.0,
    )

    output = augmentor(**_build_inputs())

    assert output["radar_past"].shape == (1, 3, 8, 8)
    assert output["radar_future"].shape == (1, 2, 8, 8)
    assert output["satellite_past"].shape == (10, 3, 8, 8)
    assert output["satellite_future"].shape == (10, 2, 8, 8)

    bbox_abs = output["aug_crop_box_xyxy"]
    bbox_norm = output["aug_crop_box_norm_xyxy"]
    width = float(bbox_abs[2] - bbox_abs[0])
    height = float(bbox_abs[3] - bbox_abs[1])
    assert width == 4.0
    assert height == 4.0
    assert torch.all((bbox_norm >= 0.0) & (bbox_norm <= 1.0))
    assert int(output["aug_time_reversed"].item()) == 0


def test_temporal_reverse_flips_full_window_and_resplits() -> None:
    augmentor = RainTimeSeriesAugmentor(
        enabled=True,
        random_crop_prob=0.0,
        random_crop_min_scale=1.0,
        random_crop_max_scale=1.0,
        temporal_reverse_prob=1.0,
    )

    radar_past = torch.tensor([[[[10.0]], [[11.0]]]])
    radar_future = torch.tensor([[[[20.0]], [[21.0]], [[22.0]]]])
    satellite_past = radar_past.repeat(10, 1, 1, 1)
    satellite_future = radar_future.repeat(10, 1, 1, 1)
    rain_past = radar_past.clone()
    rain_future = radar_future.clone()
    time_past = torch.tensor([0.1, 0.2], dtype=torch.float32)
    time_future = torch.tensor([0.3, 0.4, 0.5], dtype=torch.float32)

    output = augmentor(
        radar_past=radar_past,
        radar_future=radar_future,
        satellite_past=satellite_past,
        satellite_future=satellite_future,
        rain_past=rain_past,
        rain_future=rain_future,
        time_past=time_past,
        time_future=time_future,
    )

    assert torch.allclose(output["radar_past"].flatten(), torch.tensor([22.0, 21.0]))
    assert torch.allclose(output["radar_future"].flatten(), torch.tensor([20.0, 11.0, 10.0]))
    assert torch.allclose(output["time_past"], torch.tensor([0.5, 0.4], dtype=torch.float32))
    assert torch.allclose(output["time_future"], torch.tensor([0.3, 0.2, 0.1], dtype=torch.float32))
    assert int(output["aug_time_reversed"].item()) == 1


def test_random_crop_uses_same_box_for_all_modalities() -> None:
    torch.manual_seed(0)
    augmentor = RainTimeSeriesAugmentor(
        enabled=True,
        random_crop_prob=1.0,
        random_crop_min_scale=0.5,
        random_crop_max_scale=0.5,
        random_crop_keep_size=False,
        temporal_reverse_prob=0.0,
    )

    base_past = torch.arange(1 * 3 * 8 * 8, dtype=torch.float32).reshape(1, 3, 8, 8)
    base_future = torch.arange(1 * 2 * 8 * 8, dtype=torch.float32).reshape(1, 2, 8, 8)
    satellite_past = base_past.repeat(10, 1, 1, 1)
    satellite_future = base_future.repeat(10, 1, 1, 1)
    time_past = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    time_future = torch.tensor([0.4, 0.5], dtype=torch.float32)

    output = augmentor(
        radar_past=base_past,
        radar_future=base_future,
        satellite_past=satellite_past,
        satellite_future=satellite_future,
        rain_past=base_past.clone(),
        rain_future=base_future.clone(),
        time_past=time_past,
        time_future=time_future,
    )

    assert output["radar_past"].shape[-2:] == (4, 4)
    assert output["rain_past"].shape[-2:] == (4, 4)
    assert output["satellite_past"].shape[-2:] == (4, 4)
    assert torch.allclose(output["radar_past"], output["rain_past"])
    assert torch.allclose(output["radar_past"], output["satellite_past"][:1])
    assert torch.allclose(output["radar_future"], output["rain_future"])
    assert torch.allclose(output["radar_future"], output["satellite_future"][:1])
