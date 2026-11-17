import torch
import pytest

from src.baselines.classical import LucasKanade, Persistence
from src.baselines.contracts import prepare_history


def batch(size: int = 48, count: int = 2) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(42)
    values = {
        f"{name}_{period}": torch.rand(count, channels, 4, size, size, generator=generator)
        for name, channels in (("radar", 1), ("satellite", 10), ("rain", 1))
        for period in ("past", "future")
    }
    stamps = 1_700_000_000 + torch.arange(8, dtype=torch.float64) * 1800
    values["time_past_timestamp"] = stamps[:4].repeat(count, 1)
    values["time_future_timestamp"] = stamps[4:].repeat(count, 1)
    values["time_past"] = torch.arange(4).repeat(count, 1).float() / 48
    return values


def test_persistence_uses_last_history_frame_without_mutating_input() -> None:
    data = batch()
    original = data["rain_past"].clone()
    history = prepare_history(data, ("radar", "satellite", "rain"))
    prediction = Persistence()(history)
    torch.testing.assert_close(prediction, original[:, :, -1:].expand(-1, -1, 4, -1, -1))
    prediction.zero_()
    torch.testing.assert_close(data["rain_past"], original)
    assert sum(value.numel() for value in Persistence().parameters()) == 0


def test_future_values_are_not_in_history_contract() -> None:
    data = batch()
    before = prepare_history(data, ("rain",))
    for name in ("radar_future", "satellite_future", "rain_future", "time_future_timestamp"):
        data[name].fill_(float("nan"))
    after = prepare_history(data, ("rain",))
    torch.testing.assert_close(before.values, after.values)
    assert not before.values[:, :11].any()
    torch.testing.assert_close(Persistence()(before), Persistence()(after))


def test_missing_rain_is_rejected() -> None:
    history = prepare_history(batch(), ("satellite",))
    with pytest.raises(ValueError, match="rain"):
        Persistence()(history)


@pytest.mark.parametrize("bad_modalities", [(), ("unknown",), ("rain", "rain")])
def test_invalid_modality_selection(bad_modalities: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        prepare_history(batch(), bad_modalities)


def test_lk_dry_and_stationary_fields() -> None:
    data = batch()
    data["rain_past"].zero_()
    data["rain_past"][1, :, :, 12:30, 12:30] = 1
    output = LucasKanade()(prepare_history(data, ("rain",)))
    torch.testing.assert_close(output, data["rain_past"][:, :, -1:].expand_as(output), atol=1e-4, rtol=1e-4)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("start", [18, 58])
def test_real_lk_translates_rain_to_the_right(start: int) -> None:
    data = batch(size=80, count=1)
    data["rain_past"].zero_()
    texture = 0.2 + torch.rand(16, 16, generator=torch.Generator().manual_seed(8))
    for step in range(4):
        left = start + 2 * step
        data["rain_past"][0, 0, step, 28:44, left : left + 16] = texture
    output = LucasKanade(size_opening=0)(prepare_history(data, ("rain",)))
    expected = torch.zeros_like(output)
    for step in range(4):
        left = start + 8 + 2 * step
        right = min(left + 16, 80)
        expected[0, 0, step, 28:44, left:right] = texture[:, : right - left]
    persistence = Persistence()(prepare_history(data, ("rain",)))
    assert (output - expected).square().mean() < (persistence - expected).square().mean() * 0.25
    assert torch.isfinite(output).all()


def test_nonfinite_observed_rain_rejected() -> None:
    data = batch()
    data["rain_past"][0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        prepare_history(data, ("rain",))
