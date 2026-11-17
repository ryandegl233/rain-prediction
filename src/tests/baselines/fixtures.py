import torch
from torch.utils.data import DataLoader


def make_loader(batch_size: int = 2, count: int = 3, **kwargs: object) -> tuple[list[dict], DataLoader]:
    generator = torch.Generator().manual_seed(12)
    samples = []
    for index in range(count):
        sample = {
            f"{name}_{period}": torch.rand(channels, 4, 16, 16, generator=generator)
            for name, channels in (("radar", 1), ("satellite", 10), ("rain", 1))
            for period in ("past", "future")
        }
        stamps = 1_700_000_000 + index * 36000 + torch.arange(8, dtype=torch.float64) * 1800
        sample.update(
            time_past_timestamp=stamps[:4],
            time_future_timestamp=stamps[4:],
            time_past=torch.arange(4).float() / 48,
            time_future=torch.arange(4, 8).float() / 48,
            aug_time_reversed=torch.tensor(0),
        )
        samples.append(sample)
    return samples, DataLoader(samples, batch_size=batch_size, shuffle=False, drop_last=False)
