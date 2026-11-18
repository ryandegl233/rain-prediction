import torch
from torch.utils.data import DataLoader

from src.tests.baselines_stage3.test_contracts import batch


def make_loader(batch_size: int = 2, count: int = 3, **kwargs: object) -> tuple[list[dict], DataLoader]:
    data = batch(count=count)
    records = [{name: value[i] for name, value in data.items()} for i in range(count)]
    return records, DataLoader(records, batch_size=batch_size, shuffle=False, drop_last=False)
