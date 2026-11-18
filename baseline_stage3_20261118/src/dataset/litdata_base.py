import random
from collections.abc import Iterable, Iterator
from contextlib import suppress
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import PIL.Image as Image
import torch
from easydict import EasyDict as edict
from kornia.augmentation import AugmentationSequential, RandomResizedCrop, Resize
from litdata import (
    CombinedStreamingDataset,
    ParallelStreamingDataset,
    StreamingDataLoader,
    StreamingDataset,
)
import litdata as ld
from loguru import logger
from torch import Tensor
from torchvision.transforms.functional import to_tensor  # ty: ignore[unresolved-import]
from typing_extensions import Any, Optional, Union, cast


def _get_index_file_state() -> dict[str, str]:
    return {
        "constants": ld.constants._INDEX_FILENAME,
        "dataset_utilities": ld.utilities.dataset_utilities._INDEX_FILENAME,
        "streaming_dataset": ld.streaming.dataset._INDEX_FILENAME,
        "streaming_cache": ld.streaming.cache._INDEX_FILENAME,
        "streaming_reader": ld.streaming.reader._INDEX_FILENAME,
        "streaming_config": ld.streaming.config._INDEX_FILENAME,
    }


def _apply_index_file_state(state: dict[str, str]) -> None:
    targets = {
        "constants": ld.constants,
        "dataset_utilities": ld.utilities.dataset_utilities,
        "streaming_dataset": ld.streaming.dataset,
        "streaming_cache": ld.streaming.cache,
        "streaming_reader": ld.streaming.reader,
        "streaming_config": ld.streaming.config,
    }
    for key, module in targets.items():
        setattr(module, "_INDEX_FILENAME", state[key])


def _set_index_file(file_name: str = "index.json") -> None:
    _apply_index_file_state(
        {
            "constants": file_name,
            "dataset_utilities": file_name,
            "streaming_dataset": file_name,
            "streaming_cache": file_name,
            "streaming_reader": file_name,
            "streaming_config": file_name,
        }
    )


def _reset_index_file():
    ld.constants._INDEX_FILENAME = "index.json"
    ld.utilities.dataset_utilities._INDEX_FILENAME = "index.json"
    ld.streaming.dataset._INDEX_FILENAME = "index.json"
    ld.streaming.cache._INDEX_FILENAME = "index.json"
    ld.streaming.reader._INDEX_FILENAME = "index.json"
    ld.streaming.config._INDEX_FILENAME = "index.json"


def _as_index_filename(index_file_name: str | None) -> str | None:
    if index_file_name is None:
        return None
    if index_file_name.endswith(".json"):
        return index_file_name
    return f"{index_file_name}.json"


class _IndexFileOverride:
    def __init__(self, index_file_name: str | None) -> None:
        self.index_file_name = _as_index_filename(index_file_name)
        self._prev: dict[str, str] | None = None

    def __enter__(self) -> "_IndexFileOverride":
        if self.index_file_name is None:
            return self
        self._prev = _get_index_file_state()
        _set_index_file(self.index_file_name)
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        if self._prev is not None:
            _apply_index_file_state(self._prev)


def to_tensor_img(
    img: Image.Image | np.ndarray | Tensor,
    is_permuted: bool = True,
    repeat_gray_n: int = 3,
    force_to_rgb=False,
) -> torch.Tensor | None:
    if img is None:
        return None

    is_arr = False
    if torch.is_tensor(img) or (is_arr := isinstance(img, np.ndarray)):
        if is_arr:
            img = torch.from_numpy(img)

        img = cast(torch.Tensor, img)
        if img.ndim == 2:
            img = img.unsqueeze(0)

        if not is_permuted:
            # img = img.permute(2, 0, 1)  # hwc -> chw
            # in serilizer, img is chw orignally, need to permute to back
            # chw (dataset) -> wch (serializer) -> permute(1, -1, 0)
            img = img.permute(1, -1, 0)

        if img.shape[0] == 1 and repeat_gray_n > 1:
            # use expand instead of repeat
            img = img.expand(repeat_gray_n, -1, -1)
        if img.shape[0] == 4 and force_to_rgb:
            img = img[:3]

        return img
    else:
        if isinstance(img, Image.Image):
            img = img.convert("RGB")  # gray to rgb

        img = to_tensor(img)  # hwc -> chw
        if is_permuted:
            # img is chw orignally, need to permute to back
            # chw -> whc in to_tensor
            img = img.permute(2, 1, 0)
        return img


class IndexedCombinedStreamingDataset(CombinedStreamingDataset):
    """A CombinedStreamingDataset that also returns the index of each sample."""

    def __init__(self, combined_is_cycled=False, *args, **kwargs) -> None:
        kwargs.setdefault("seed", 2025)
        super().__init__(*args, **kwargs)
        self.combined_is_cycled = combined_is_cycled

        if not combined_is_cycled:
            self.__check_can_be_indexed()
            self.accum_lens = np.cumsum([0] + [len(ds) for ds in self._datasets])

        # set the epoch to 0
        self.current_epoch = 0

    def __check_can_be_indexed(self) -> None:
        """Check if the dataset can be indexed."""
        if not self._iterate_over_all:
            raise ValueError("IndexedCombinedStreamingDataset only supports iterate_over_all=True")

    def _check_datasets(self, datasets: list[StreamingDataset]) -> None:
        # if any(not isinstance(d, StreamingDataset) for d in datasets):
        #     raise RuntimeError("The provided datasets should be instances of the StreamingDataset.")
        return

    def set_weights(self, weights: list[float]):
        self._weights = weights
        logger.debug(f"Set weights to {weights}", not_rank0_print=True)

    def __len__(self) -> int | float:
        """Return the total number of samples across all datasets."""
        if self.combined_is_cycled:
            return float("inf")
        return self.accum_lens[-1]

    def __getitem__(self, idx: int) -> dict:
        if self.combined_is_cycled:
            logger.error(
                "The combined dataset is cycled, not supported for indexing. Return using __iter__ method to sample."
            )
            raise IndexError(f"Index {idx} not supported for cycled combined dataset.")

        # Check bounds
        total_length = self.accum_lens[-1]
        if idx < 0 or idx >= total_length:
            raise IndexError(f"Index {idx} out of range for dataset of length {total_length}")

        # find the dataset containing this index
        ds_idx = int(np.searchsorted(self.accum_lens, idx, side="right") - 1)
        ds_idx = int(max(0, min(int(ds_idx), len(self._datasets) - 1)))
        sample_idx = idx - int(self.accum_lens[ds_idx])

        # Ensure sample_idx is within bounds
        if sample_idx < 0 or sample_idx >= len(self._datasets[ds_idx]):
            raise IndexError(f"Index {idx} resulted in invalid sample_idx {sample_idx} for dataset {ds_idx}")

        return self._datasets[ds_idx][sample_idx]

    def _split_num_samples_yielded(self, total: int) -> list[int]:
        if total <= 0 or len(self._datasets) == 0:
            return [0 for _ in range(len(self._datasets))]

        weights = list(self._weights) if self._weights is not None else []
        clean_weights = [float(w) if w is not None and float(w) > 0 else 0.0 for w in weights]

        if len(clean_weights) != len(self._datasets):
            clean_weights = [0.0 for _ in range(len(self._datasets))]

        weight_sum = sum(clean_weights)
        if weight_sum <= 0:
            lens = [max(int(get_dataset_len(ds)), 0) for ds in self._datasets]
            len_sum = sum(lens)
            if len_sum <= 0:
                per = total // max(1, len(self._datasets))
                out = [per for _ in range(len(self._datasets))]
                out[0] += total - sum(out)
                return out
            clean_weights = [l / len_sum for l in lens]
        else:
            clean_weights = [w / weight_sum for w in clean_weights]

        raw = [total * w for w in clean_weights]
        counts = [int(v) for v in raw]
        remainder = total - sum(counts)
        if remainder > 0:
            frac = [rv - int(rv) for rv in raw]
            for idx in sorted(range(len(frac)), key=lambda i: frac[i], reverse=True)[:remainder]:
                counts[idx] += 1
        return counts

    def state_dict(
        self,
        num_workers: int,
        batch_size: int,
        num_samples_yielded: int | list[int] | None = None,
    ) -> dict[str, Any]:
        if isinstance(num_samples_yielded, int):
            num_samples_yielded = self._split_num_samples_yielded(num_samples_yielded)
        return super().state_dict(num_workers, batch_size, num_samples_yielded)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not state_dict:
            return
        if "dataset" in state_dict:
            super().load_state_dict(state_dict)
            return

        for dataset_idx, dataset in enumerate(self._datasets):
            key = str(dataset_idx)
            if key in state_dict:
                dataset.load_state_dict(state_dict[key])


class SingleCycleStreamingDataset(ParallelStreamingDataset):
    def __init__(
        self,
        dataset: Any,  # can be a wrapper dataset as well
        *,
        length: int | float | None = float("inf"),
        seed: int = 2025,
        resume: bool = True,
        reset_rngs: bool = False,
        force_override_state_dict: bool = False,
    ) -> None:
        super().__init__(
            cast(list[StreamingDataset], [dataset]),
            length=length,
            force_override_state_dict=force_override_state_dict,
            transform=cast(Any, self._transform),
            seed=seed,
            resume=resume,
            reset_rngs=reset_rngs,
        )

    def _check_datasets(self, datasets: list[StreamingDataset]) -> None:
        # do nothing check
        return

    def _transform(self, samples: tuple[Any, ...], rngs: Any = None) -> Any:
        (sample_d,) = samples
        return sample_d

    def __getitem__(self, index: int) -> Any:
        return self._datasets[0][index]

    def __iter__(self) -> Iterator[Any]:
        while True:
            iterator = super().__iter__()
            yielded_any = False
            while True:
                try:
                    item = next(iterator)
                except StopIteration:
                    break
                yielded_any = True
                yield item

            if not yielded_any:
                yield None

    def state_dict(
        self,
        num_workers: int,
        batch_size: int,
        num_samples_yielded: int | list[int] | None = None,
    ) -> dict[str, Any]:
        inner = self._datasets[0]
        if isinstance(num_samples_yielded, list):
            total_yielded = int(num_samples_yielded[0]) if len(num_samples_yielded) > 0 else 0
        elif isinstance(num_samples_yielded, int):
            total_yielded = int(num_samples_yielded)
        else:
            total_yielded = 0

        if hasattr(inner, "state_dict"):
            cycle_length = int(len(inner)) if hasattr(inner, "__len__") else 0
            in_cycle = 0 if cycle_length <= 0 else (total_yielded % cycle_length)
            return {
                "0": inner.state_dict(
                    num_samples_yielded=in_cycle,
                    num_workers=num_workers,
                    batch_size=batch_size,
                )
            }

        return {
            "0": {"num_samples_yielded": total_yielded, "num_workers": int(num_workers), "batch_size": int(batch_size)}
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        inner = self._datasets[0]
        if not state_dict:
            return

        if "dataset" in state_dict:
            super().load_state_dict(state_dict)
            return

        if "0" in state_dict and hasattr(inner, "load_state_dict"):
            inner.load_state_dict(state_dict["0"])
            return

        if hasattr(inner, "load_state_dict"):
            inner.load_state_dict(state_dict)


class _BaseStreamingDataset(StreamingDataset):
    """
    Fixed index file name support.
    """

    def __init__(self, *, input_dir: str, **kwargs: Any) -> None:
        input_dir, index_file_name = self._change_litdata_index_file(input_dir, kwargs)
        kwargs.setdefault("seed", 2025)
        super().__init__(input_dir=str(input_dir), **kwargs)

        self.index_file_name = index_file_name

        # change back
        # Preload via base StreamingDataset getter to avoid calling subclass __getitem__
        # before subclass attributes are initialized.
        if self.__len__() > 0:
            try:
                _ = super().__getitem__(0)
            except Exception as e:
                logger.warning(f"Failed to pre-load the first item for {input_dir}: {e}")
            finally:
                self._reset_preloaded_item_loader_state()

        if index_file_name is not None:
            _reset_index_file()

    def _reset_preloaded_item_loader_state(self) -> None:
        if self.cache is None:
            return

        reader = getattr(self.cache, "_reader", None)
        if reader is None:
            return

        item_loader = getattr(reader, "_item_loader", None)
        if item_loader is None:
            return

        open_handle = getattr(item_loader, "_open_handle", None)
        if open_handle is not None:
            with suppress(Exception):
                open_handle.close()

        item_loader._open_handle = None
        item_loader._chunk_filepath = None

    def _change_litdata_index_file(self, path: str | Path, kwargs: dict[str, Any]):
        """
        Change the index file name of litdata.
        """
        path = Path(path)
        if not path.is_absolute():
            repo_root = Path(__file__).resolve().parents[2]
            if not path.exists() and (repo_root / path).exists():
                # Resolve relative paths against the repo root for Hydra runs.
                path = repo_root / path
        index_file_name = None
        if path.is_dir():
            pass
        elif path.is_file() and path.suffix == ".json":
            index_file_name = path.stem
            path = path.parent
        else:
            raise ValueError(f"input_dir must be a directory or a index json file, got: {path}")

        index_file_name = kwargs.pop("index_file_name", None) or index_file_name
        if index_file_name is not None:
            index_filename = _as_index_filename(index_file_name)
            _set_index_file(cast(str, index_filename))
            assert (path / cast(str, index_filename)).exists(), (
                f"Index file not found: {path / cast(str, index_filename)}"
            )
            logger.debug(f"Set litdata index file name to: {index_filename}")

        return path, index_file_name

    def _check_datasets(self, datasets: list[StreamingDataset]) -> None:
        """override the original method
        do nothing check.
        """

        # if any(not isinstance(d, StreamingDaaset) for d in datasets):
        #     raise RuntimeError("The provided datasets should be instances of the StreamingDataset.")
        return

    def _create_cache(self, worker_env: Any):
        with _IndexFileOverride(self.index_file_name):
            return super()._create_cache(worker_env=worker_env)

    @classmethod
    def create_dataset(
        cls,
        input_dir: str | list[str],
        other_ds: StreamingDataLoader | list[StreamingDataLoader] | None = None,
        combined_kwargs: dict = {"batching_method": "stratified"},
        is_cycled: bool = False,
        **kwargs: Any,
    ):
        """
        combined_kwargs: dict
            weights: Optional[Sequence[float]] = None,
            iterate_over_all: bool = True,
            batching_method: BatchingMethodType = "stratified",
            force_override_state_dict: bool = False,
        """
        if isinstance(input_dir, str):
            ds = cls(input_dir=input_dir, **kwargs)
        elif isinstance(input_dir, list) and len(input_dir) == 1:
            ds = cls(input_dir=input_dir[0], **kwargs)
        else:
            streams = [cls(input_dir=d, **kwargs) for d in input_dir]
            ds = IndexedCombinedStreamingDataset(datasets=streams, **combined_kwargs)

        if other_ds is not None:
            ds = IndexedCombinedStreamingDataset(
                datasets=[ds] + ([other_ds] if not isinstance(other_ds, list) else other_ds),
                **combined_kwargs,
            )

        if is_cycled:
            ds = SingleCycleStreamingDataset(dataset=ds)
            # logger.debug(f"Create a cycled dataset for input dir: {input_dir}")

        return ds

    @classmethod
    def create_dataloader(
        cls,
        input_dir: str | list[str],
        stream_ds_kwargs: dict = {},
        combined_kwargs: dict = {"batching_method": "per_stream"},
        loader_kwargs: dict = {},
    ):
        ds = cls.create_dataset(input_dir=input_dir, combined_kwargs=combined_kwargs, **stream_ds_kwargs)
        dl = StreamingDataLoader(ds, **loader_kwargs)
        return ds, dl
