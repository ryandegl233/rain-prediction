import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

import torch

COMBINATIONS = {
    "full": (True, True, True),
    "radar+satellite": (True, True, False),
    "radar+rain": (True, False, True),
    "satellite+rain": (False, True, True),
    "radar": (True, False, False),
    "satellite": (False, True, False),
    "rain": (False, False, True),
}


@dataclass(frozen=True)
class ForecastInput:
    history: torch.Tensor
    available: torch.Tensor
    time: torch.Tensor | None
    future_aux: torch.Tensor | None
    timestamps: torch.Tensor | None = None


def window_times(stamps: torch.Tensor) -> list[list[str]]:
    return [
        [datetime.fromtimestamp(stamp).isoformat() for stamp in window] for window in stamps.detach().cpu().tolist()
    ]


def sample_masks(stamps: torch.Tensor, scenario: str, seed: int, full_probability: float) -> torch.Tensor:
    if stamps.ndim != 2 or stamps.shape[1] != 4 or not torch.isfinite(stamps).all():
        raise ValueError("Historical timestamps must be finite [B,4].")
    if not 0 <= full_probability <= 1:
        raise ValueError("full_probability must be in [0,1].")
    if scenario == "satellite_only":
        return torch.tensor((False, True, False))[:, None].expand(stamps.shape[0], 3, 4).clone()
    if scenario != "sampled" and scenario not in COMBINATIONS:
        raise ValueError(f"Unknown scenario: {scenario}")
    combos = torch.tensor(list(COMBINATIONS.values()), dtype=torch.bool)
    probability = torch.tensor([full_probability] + [(1 - full_probability) / 6] * 6)
    masks = []
    for window in window_times(stamps):
        if scenario == "sampled":
            key = json.dumps([seed, window], separators=(",", ":")).encode()
            generator = torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))
            indices = torch.multinomial(probability, 4, replacement=True, generator=generator)
            mask = combos[indices].T.contiguous()
        else:
            mask = torch.tensor(COMBINATIONS[scenario])[:, None].repeat(1, 4)
        mask[:, 0] = True
        masks.append(mask)
    return torch.stack(masks)


def make_inputs(batch: Mapping[str, torch.Tensor], available: torch.Tensor, use_gt: bool) -> ForecastInput:
    available = available.detach().cpu()
    if available.dtype != torch.bool or available.ndim != 3 or available.shape[1:] != (3, 4):
        raise ValueError("Availability must be boolean [B,3,4].")
    satellite_only = not available[:, (0, 2)].any() and available[:, 1].all()
    if not satellite_only and (not available[:, :, 0].all() or not available.any(dim=1).all()):
        raise ValueError("The first frame must be fully observed and every frame must retain a modality.")
    history = []
    shape = None
    for index, (name, channels) in enumerate((("radar", 1), ("satellite", 10), ("rain", 1))):
        if satellite_only and name != "satellite":
            value = torch.zeros_like(batch["satellite_past"][:, :1], device="cpu", dtype=torch.float32)
        else:
            value = batch[f"{name}_past"].detach().float().cpu()
        if value.ndim != 5 or value.shape[:3] != (available.shape[0], channels, 4):
            raise ValueError(f"{name}_past must be [B,{channels},4,H,W].")
        spatial = value.shape[-2:]
        if min(spatial) < 1 or (shape is not None and shape != spatial):
            raise ValueError("Modalities must share nonempty spatial dimensions.")
        shape = spatial
        value = torch.where(available[:, index, None, :, None, None], value, 0)
        if not torch.isfinite(value).all():
            raise ValueError(f"Observed {name} history is nonfinite.")
        history.append(value)
    time = batch.get("time_past")
    if time is not None:
        time = time.detach().float().cpu()
        if time.shape != (available.shape[0], 4) or not torch.isfinite(time).all():
            raise ValueError("time_past must be finite [B,4].")
    future_aux = None
    if use_gt and satellite_only:
        raise ValueError("Satellite-only NPM cannot consume future observations.")
    if use_gt:
        auxiliary = []
        for name, channels in (("radar", 1), ("satellite", 10)):
            value = batch[f"{name}_future"].detach().float().cpu()
            if value.shape != (available.shape[0], channels, 4, *shape):
                raise ValueError(f"Invalid future {name} shape.")
            value = value[:, :, :3]
            if not torch.isfinite(value).all():
                raise ValueError(f"Future {name} observations are nonfinite.")
            auxiliary.append(value)
        future_aux = torch.cat(auxiliary, dim=1)
    stamps = batch.get("time_past_timestamp")
    return ForecastInput(
        torch.cat(history, dim=1), available, time, future_aux, stamps.detach().cpu() if stamps is not None else None
    )


def latest_rain(inputs: ForecastInput) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.arange(4).expand_as(inputs.available[:, 2])
    last = torch.where(inputs.available[:, 2], indices, -1).amax(dim=1)
    if (last < 0).any():
        raise ValueError("Rain-dependent methods require at least one historical rain observation.")
    rain = inputs.history[:, 11:12]
    selected = rain.gather(2, last[:, None, None, None, None].expand(-1, 1, 1, *rain.shape[-2:]))
    return selected, last
