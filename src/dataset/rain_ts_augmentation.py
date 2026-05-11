import torch
import torch.nn.functional as F


def _validate_probability(value: float, *, field_name: str) -> float:
    prob = float(value)
    if prob < 0.0 or prob > 1.0:
        raise ValueError(f"{field_name} must be in [0, 1], got {prob}")
    return prob


def _validate_scale(min_scale: float, max_scale: float) -> tuple[float, float]:
    min_value = float(min_scale)
    max_value = float(max_scale)
    if min_value <= 0.0 or max_value <= 0.0:
        raise ValueError(f"crop scales must be > 0, got min={min_value}, max={max_value}")
    if min_value > max_value:
        raise ValueError(f"crop min scale must be <= max scale, got min={min_value}, max={max_value}")
    return min_value, max_value


def _resize_sequence(sequence: torch.Tensor, *, target_hw: tuple[int, int]) -> torch.Tensor:
    c, t, _, _ = sequence.shape
    resized = F.interpolate(
        sequence.reshape(1, c * t, sequence.shape[-2], sequence.shape[-1]),
        size=target_hw,
        mode="bilinear",
        align_corners=False,
    )
    return resized.reshape(c, t, target_hw[0], target_hw[1])


def _crop_sequence(
    sequence: torch.Tensor,
    *,
    top: int,
    left: int,
    crop_h: int,
    crop_w: int,
    keep_size: bool,
) -> torch.Tensor:
    cropped = sequence[..., top : top + crop_h, left : left + crop_w]
    if not keep_size:
        return cropped
    return _resize_sequence(cropped, target_hw=(int(sequence.shape[-2]), int(sequence.shape[-1])))


def _reverse_past_future(
    past: torch.Tensor,
    future: torch.Tensor,
    *,
    n_past: int,
    n_futures: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    full = torch.cat([past, future], dim=1)
    full = torch.flip(full, dims=[1])
    return full[:, :n_past], full[:, n_past : n_past + n_futures]


class RainTimeSeriesAugmentor:
    def __init__(
        self,
        *,
        enabled: bool = False,
        random_crop_prob: float = 0.0,
        random_crop_min_scale: float = 1.0,
        random_crop_max_scale: float = 1.0,
        random_crop_keep_size: bool = True,
        temporal_reverse_prob: float = 0.0,
    ) -> None:
        self.enabled = bool(enabled)
        self.random_crop_prob = _validate_probability(random_crop_prob, field_name="random_crop_prob")
        self.random_crop_min_scale, self.random_crop_max_scale = _validate_scale(
            random_crop_min_scale, random_crop_max_scale
        )
        self.random_crop_keep_size = bool(random_crop_keep_size)
        self.temporal_reverse_prob = _validate_probability(temporal_reverse_prob, field_name="temporal_reverse_prob")

    def _should_apply(self, prob: float) -> bool:
        if prob <= 0.0:
            return False
        return bool(torch.rand((), dtype=torch.float32).item() < prob)

    def __call__(
        self,
        *,
        radar_past: torch.Tensor,
        radar_future: torch.Tensor,
        satellite_past: torch.Tensor,
        satellite_future: torch.Tensor,
        rain_past: torch.Tensor,
        rain_future: torch.Tensor,
        time_past: torch.Tensor,
        time_future: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if not self.enabled:
            return {
                "radar_past": radar_past,
                "radar_future": radar_future,
                "satellite_past": satellite_past,
                "satellite_future": satellite_future,
                "rain_past": rain_past,
                "rain_future": rain_future,
                "time_past": time_past,
                "time_future": time_future,
                "aug_crop_box_xyxy": torch.tensor(
                    [0.0, 0.0, float(radar_past.shape[-1]), float(radar_past.shape[-2])], dtype=torch.float32
                ),
                "aug_crop_box_norm_xyxy": torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32),
                "aug_time_reversed": torch.tensor(0, dtype=torch.int64),
            }

        h = int(radar_past.shape[-2])
        w = int(radar_past.shape[-1])
        top = 0
        left = 0
        crop_h = h
        crop_w = w

        if self._should_apply(self.random_crop_prob) and (h > 1 and w > 1):
            scale = float(
                torch.empty((), dtype=torch.float32).uniform_(self.random_crop_min_scale, self.random_crop_max_scale).item()
            )
            crop_h = max(1, min(h, int(round(h * scale))))
            crop_w = max(1, min(w, int(round(w * scale))))
            max_top = max(0, h - crop_h)
            max_left = max(0, w - crop_w)
            top = int(torch.randint(low=0, high=max_top + 1, size=(1,)).item()) if max_top > 0 else 0
            left = int(torch.randint(low=0, high=max_left + 1, size=(1,)).item()) if max_left > 0 else 0

            radar_past = _crop_sequence(
                radar_past,
                top=top,
                left=left,
                crop_h=crop_h,
                crop_w=crop_w,
                keep_size=self.random_crop_keep_size,
            )
            radar_future = _crop_sequence(
                radar_future,
                top=top,
                left=left,
                crop_h=crop_h,
                crop_w=crop_w,
                keep_size=self.random_crop_keep_size,
            )
            satellite_past = _crop_sequence(
                satellite_past,
                top=top,
                left=left,
                crop_h=crop_h,
                crop_w=crop_w,
                keep_size=self.random_crop_keep_size,
            )
            satellite_future = _crop_sequence(
                satellite_future,
                top=top,
                left=left,
                crop_h=crop_h,
                crop_w=crop_w,
                keep_size=self.random_crop_keep_size,
            )
            rain_past = _crop_sequence(
                rain_past,
                top=top,
                left=left,
                crop_h=crop_h,
                crop_w=crop_w,
                keep_size=self.random_crop_keep_size,
            )
            rain_future = _crop_sequence(
                rain_future,
                top=top,
                left=left,
                crop_h=crop_h,
                crop_w=crop_w,
                keep_size=self.random_crop_keep_size,
            )

        is_time_reversed = False
        if self._should_apply(self.temporal_reverse_prob):
            n_past = int(radar_past.shape[1])
            n_futures = int(radar_future.shape[1])
            radar_past, radar_future = _reverse_past_future(radar_past, radar_future, n_past=n_past, n_futures=n_futures)
            satellite_past, satellite_future = _reverse_past_future(
                satellite_past, satellite_future, n_past=n_past, n_futures=n_futures
            )
            rain_past, rain_future = _reverse_past_future(rain_past, rain_future, n_past=n_past, n_futures=n_futures)
            time_full = torch.cat([time_past, time_future], dim=0)
            time_full = torch.flip(time_full, dims=[0])
            time_past = time_full[:n_past]
            time_future = time_full[n_past : n_past + n_futures]
            is_time_reversed = True

        x0 = float(left)
        y0 = float(top)
        x1 = float(left + crop_w)
        y1 = float(top + crop_h)

        return {
            "radar_past": radar_past,
            "radar_future": radar_future,
            "satellite_past": satellite_past,
            "satellite_future": satellite_future,
            "rain_past": rain_past,
            "rain_future": rain_future,
            "time_past": time_past,
            "time_future": time_future,
            "aug_crop_box_xyxy": torch.tensor([x0, y0, x1, y1], dtype=torch.float32),
            "aug_crop_box_norm_xyxy": torch.tensor(
                [x0 / float(w), y0 / float(h), x1 / float(w), y1 / float(h)],
                dtype=torch.float32,
            ),
            "aug_time_reversed": torch.tensor(1 if is_time_reversed else 0, dtype=torch.int64),
        }
