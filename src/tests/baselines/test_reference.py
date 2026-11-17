import torch
from torch import nn

from src.baselines.contracts import prepare_history
from src.baselines.reference import NextFrameForecaster
from src.tests.baselines.fixtures import make_loader


class IncrementNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.increment = nn.Parameter(torch.tensor(0.1))
        self.frame_patch_size = 1
        self.use_time_embedding = True
        self.seen: list[torch.Tensor] = []

    def _apply_context_modality_mask_token(
        self, context_x: torch.Tensor, context_modality_available: torch.Tensor
    ) -> torch.Tensor:
        available = context_modality_available.repeat_interleave(torch.tensor([1, 10, 1]), dim=1)
        return torch.where(available[:, :, None, None, None], context_x, 0)

    def forward_ar(self, context_x: torch.Tensor, target_x: torch.Tensor, **kwargs: object) -> dict[str, torch.Tensor]:
        if kwargs.get("context_modality_available") is not None:
            context_x = self._apply_context_modality_mask_token(context_x, kwargs["context_modality_available"])
        self.seen.append(context_x.detach().clone())
        return {
            "radar": target_x[:, :1],
            "satellite": target_x[:, 1:11],
            "rain": torch.ones_like(target_x[:, 11:12]) * self.increment,
        }


def test_reference_rolls_its_own_predictions_and_has_gradients() -> None:
    _, loader = make_loader()
    data = next(iter(loader))
    network = IncrementNetwork()
    model = NextFrameForecaster(network)
    history = prepare_history(data, ("radar", "satellite", "rain"))
    output = model(history)
    expected = data["rain_past"][:, :, -1:] + torch.arange(1, 5).view(1, 1, 4, 1, 1) * 0.1
    torch.testing.assert_close(output, expected)
    output.mean().backward()
    assert network.increment.grad > 0
    assert [value.shape[2] for value in network.seen] == [4, 5, 6, 7]


def test_reference_cannot_access_future_or_missing_rain_anchor() -> None:
    _, loader = make_loader()
    data = next(iter(loader))
    model = NextFrameForecaster(IncrementNetwork())
    original = model(prepare_history(data, ("satellite",)))
    for name in ("rain_past", "radar_past", "radar_future", "satellite_future", "rain_future"):
        data[name].fill_(999)
    changed = model(prepare_history(data, ("satellite",)))
    torch.testing.assert_close(original, changed)
    expected = torch.arange(1, 5).view(1, 1, 4, 1, 1).expand_as(changed) * 0.1
    torch.testing.assert_close(changed, expected)


def test_missing_modalities_retain_generated_history() -> None:
    _, loader = make_loader()
    network = IncrementNetwork()
    model = NextFrameForecaster(network)
    model(prepare_history(next(iter(loader)), ("satellite",)))
    generated_rain = network.seen[2][:, 11:12, 4:]
    expected = torch.tensor([0.1, 0.2]).view(1, 1, 2, 1, 1).expand_as(generated_rain)
    torch.testing.assert_close(generated_rain, expected)
