from collections.abc import Mapping
from dataclasses import dataclass

import torch

MODALITIES = {"radar": slice(0, 1), "satellite": slice(1, 11), "rain": slice(11, 12)}


@dataclass(frozen=True)
class HistoryBatch:
    values: torch.Tensor
    available: torch.Tensor
    time: torch.Tensor | None = None


def prepare_history(
    batch: Mapping[str, torch.Tensor],
    modalities: tuple[str, ...],
) -> HistoryBatch:
    if not modalities or len(set(modalities)) != len(modalities) or set(modalities) - MODALITIES.keys():
        raise ValueError(f"Invalid modalities: {modalities}")
    fields = []
    shape = None
    for name, channel_slice in MODALITIES.items():
        value = batch[f"{name}_past"].detach().float()
        channels = channel_slice.stop - channel_slice.start
        if value.ndim != 5 or value.shape[1:3] != (channels, 4) or value.shape[0] < 1:
            raise ValueError(f"{name}_past must be [B,{channels},4,H,W], got {tuple(value.shape)}")
        current = (value.shape[0], value.shape[3], value.shape[4])
        if min(current) < 1 or (shape is not None and shape != current):
            raise ValueError("History fields must share nonempty batch and spatial dimensions.")
        shape = current
        if name not in modalities:
            value = torch.zeros_like(value)
        elif not torch.isfinite(value).all():
            raise ValueError(f"Observed {name} history must be finite.")
        fields.append(value)

    values = torch.cat(fields, dim=1)
    available = torch.tensor([name in modalities for name in MODALITIES], device=values.device)
    available = available.unsqueeze(0).expand(values.shape[0], -1)
    time = batch.get("time_past")
    if time is not None:
        time = time.detach().to(device=values.device, dtype=torch.float32)
        if time.shape != (values.shape[0], 4) or not torch.isfinite(time).all():
            raise ValueError("time_past must be finite [B,4].")
    return HistoryBatch(values, available, time)
