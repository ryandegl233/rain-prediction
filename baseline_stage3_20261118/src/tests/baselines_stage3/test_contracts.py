from datetime import datetime, timedelta, timezone

import pytest
import torch

from src.baselines_stage3 import contracts
from src.baselines_stage3.contracts import make_inputs, sample_masks
from src.baselines_stage3.models import LucasKanade, Persistence


def batch(size: int = 16, count: int = 3) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(12)
    result = {
        f"{name}_{period}": torch.rand(count, channels, 4, size, size, generator=generator)
        for name, channels in (("radar", 1), ("satellite", 10), ("rain", 1))
        for period in ("past", "future")
    }
    stamps = 1_700_000_000 + torch.arange(count)[:, None] * 36000 + torch.arange(8)[None] * 1800
    result.update(time_past_timestamp=stamps[:, :4].double(), time_future_timestamp=stamps[:, 4:].double())
    result.update(
        time_past=torch.arange(4).repeat(count, 1).float() / 48,
        time_future=torch.arange(4, 8).repeat(count, 1).float() / 48,
        aug_time_reversed=torch.zeros(count),
    )
    return result


def test_masks_are_anchored_deterministic_and_do_not_change_global_rng() -> None:
    stamps = torch.arange(2000)[:, None] * 1800 + torch.arange(4)[None] * 1800
    before = torch.get_rng_state().clone()
    masks = sample_masks(stamps, "sampled", 2025, 0.6)
    assert torch.equal(before, torch.get_rng_state())
    assert masks[:, :, 0].all() and masks.any(dim=1).all()
    assert abs(float(masks[:, :, 1:].all(dim=1).float().mean()) - 0.6) < 0.03
    assert torch.equal(masks, torch.cat([sample_masks(x, "sampled", 2025, 0.6) for x in stamps.split(41)]))


def test_missing_rain_and_future_truth_are_not_forecast_inputs() -> None:
    data = batch()
    mask = torch.ones(3, 3, 4, dtype=torch.bool)
    mask[:, 2, 1:] = False
    inputs = make_inputs(data, mask, True)
    expected = data["rain_past"][:, :, :1].expand(-1, -1, 4, -1, -1)
    torch.testing.assert_close(Persistence()(inputs), expected)
    data["rain_past"][:, :, 1:] = float("nan")
    data["rain_future"].fill_(float("nan"))
    changed = make_inputs(data, mask, True)
    torch.testing.assert_close(changed.history, inputs.history)
    torch.testing.assert_close(LucasKanade()(changed), expected)


def test_same_dataset_calendar_windows_have_same_masks_across_host_timezones(monkeypatch: pytest.MonkeyPatch) -> None:
    records = []
    for zone in (timezone.utc, timezone(timedelta(hours=8))):

        class HostDateTime(datetime):
            @classmethod
            def fromtimestamp(cls, timestamp: float) -> datetime:
                return datetime.fromtimestamp(timestamp, zone).replace(tzinfo=None)

        monkeypatch.setattr(contracts, "datetime", HostDateTime)
        start = datetime(2025, 7, 1, tzinfo=zone).timestamp()
        stamps = start + torch.arange(200, dtype=torch.float64)[:, None] * 3600 + torch.arange(4)[None] * 1800
        records.append((contracts.window_times(stamps), sample_masks(stamps, "sampled", 2025, 0.6)))
    assert records[0][0] == records[1][0]
    assert torch.equal(records[0][1], records[1][1])


def test_lk_respects_missing_observation_time_gaps() -> None:
    data = batch(size=80, count=1)
    data["rain_past"].zero_()
    texture = 0.2 + torch.rand(16, 16, generator=torch.Generator().manual_seed(8))
    for step in range(4):
        data["rain_past"][0, 0, step, 28:44, 18 + 2 * step : 34 + 2 * step] = texture
    mask = torch.ones(1, 3, 4, dtype=torch.bool)
    mask[:, 2, [1, 3]] = False
    inputs = make_inputs(data, mask, True)
    output = LucasKanade(size_opening=0)(inputs)
    expected = torch.zeros_like(output)
    for step in range(4):
        expected[0, 0, step, 28:44, 26 + 2 * step : 42 + 2 * step] = texture
    assert (output - expected).square().mean() < (Persistence()(inputs) - expected).square().mean() * 0.3
