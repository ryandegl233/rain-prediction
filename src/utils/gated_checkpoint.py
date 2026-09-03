import json
from pathlib import Path

import accelerate
import torch
from accelerate.utils import load_state_dict
from accelerate.utils.constants import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from torch import nn


def load_gated_model_initialization(model: nn.Module, checkpoint: Path) -> None:
    selected_path = checkpoint
    if checkpoint.is_dir():
        if (checkpoint / WEIGHTS_NAME).is_file():
            selected_path = checkpoint / WEIGHTS_NAME
        elif (checkpoint / SAFE_WEIGHTS_NAME).is_file():
            selected_path = checkpoint / SAFE_WEIGHTS_NAME
        else:
            indexes = sorted(checkpoint.glob("*.index.json"))
            if len(indexes) != 1:
                raise ValueError(f"Checkpoint directory requires one weight file or one index: {checkpoint}")
            selected_path = indexes[0]
    if not selected_path.is_file():
        raise FileNotFoundError(f"Checkpoint file does not exist: {selected_path}")

    checkpoint_files = [selected_path]
    if selected_path.suffix == ".json":
        with selected_path.open(encoding="utf-8") as stream:
            index = json.load(stream)
        weight_map = index.get("weight_map", index)
        checkpoint_files = [selected_path.parent / name for name in sorted(set(weight_map.values()))]

    expected = model.state_dict()
    expected_keys = set(expected)
    gate_keys = {key for key in expected_keys if key.startswith("spatial_modality_gate.")}
    actual_keys: set[str] = set()
    invalid_shapes: list[str] = []
    for checkpoint_file in checkpoint_files:
        state = load_state_dict(str(checkpoint_file))
        duplicate_keys = actual_keys.intersection(state)
        if duplicate_keys:
            raise ValueError(f"Checkpoint contains duplicate keys across shards: {sorted(duplicate_keys)}")
        actual_keys.update(state)
        for key, value in state.items():
            if key in expected and (not isinstance(value, torch.Tensor) or value.shape != expected[key].shape):
                invalid_shapes.append(key)
        del state

    missing_keys = expected_keys - actual_keys
    unexpected_keys = actual_keys - expected_keys
    if unexpected_keys or invalid_shapes or (missing_keys and missing_keys != gate_keys):
        raise ValueError(
            f"Invalid gated initialization checkpoint {checkpoint}: "
            f"missing_keys={sorted(missing_keys)}, unexpected_keys={sorted(unexpected_keys)}, "
            f"invalid_shapes={sorted(invalid_shapes)}"
        )
    accelerate.load_checkpoint_in_model(model, str(checkpoint), strict=False)
