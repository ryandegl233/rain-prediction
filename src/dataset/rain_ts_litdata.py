import io
import json
import os
import random
from collections import defaultdict
from contextlib import suppress
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterator

import jsonlines as jsonl
import numpy as np
import pandas as pd
import PIL.Image as Image
import torch
import torch.distributed as dist
from easydict import EasyDict as edict
from kornia.augmentation import Resize
from litdata import (
    CombinedStreamingDataset,
    ParallelStreamingDataset,
    StreamingDataLoader,
    StreamingDataset,
)
from litdata.streaming import serializers
from litdata.utilities import dataset_utilities as litdata_utils
from loguru import logger
from natsort import natsorted
from numpy.typing import ArrayLike, NDArray
from torch import Tensor
from torch.utils.data import Dataset as MappedDataset
from torch.utils.data import get_worker_info
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm
from typing_extensions import Any, Optional, Union, cast

from src.dataset.litdata_base import (
    IndexedCombinedStreamingDataset,
    SingleCycleStreamingDataset,
    _BaseStreamingDataset,
    to_tensor_img,
)
from src.dataset.rain_ts_augmentation import RainTimeSeriesAugmentor


class JsonlSerializer(serializers.Serializer):
    """Serializer for JSONL (JSON Lines) format."""

    def serialize(self, obj: dict) -> tuple[bytes, str]:
        """Serialize a dictionary to JSONL bytes format."""
        buf = io.BytesIO()
        writer = jsonl.Writer(buf)
        writer.write(obj)
        writer.close()
        return buf.getvalue(), "jsonl"

    def deserialize(self, byte_data: bytes) -> dict | list[dict]:
        """Deserialize JSONL bytes back to a dictionary or list of dictionaries."""
        text_data = byte_data.decode("utf-8").strip()

        if not text_data:
            return {}

        lines = text_data.split("\n")
        objects = [json.loads(line) for line in lines if line.strip()]

        if len(objects) == 1:
            return objects[0]
        else:
            return objects

    def can_serialize(self, data: Any) -> bool:
        """Check if the data can be serialized as JSONL."""
        return isinstance(data, dict)


class TiffFileSerializer(serializers.TIFFSerializer):
    def deserialize(self, data: bytes) -> torch.Tensor:
        """Deserialize bytes into an object."""
        arr = super().deserialize(data)
        # additional transport like in JPEGSerializer
        if arr.ndim == 3:
            arr = arr.transpose([-1, 0, 1])
        return torch.from_numpy(arr)


class JPEGGeneralSerializer(serializers.JPEGSerializer):
    def _force_to_rgb(self, arr: np.ndarray | torch.Tensor):
        if arr.shape[0] == 4:
            # png with alpha channel
            logger.debug(f"Warning: 4 channels, shape {arr.shape} -> change to 3 RGB channels.")
            arr = arr[:3]
        return arr

    def deserialize(self, data: bytes) -> torch.Tensor | None:
        """Deserialize bytes into an object."""
        # filtering and libpng C lib warnings
        saved_stdout = os.dup(1)
        saved_stderr = os.dup(2)

        try:
            devnull = os.open("/dev/null", os.O_WRONLY)
            os.dup2(devnull, 1)  # stdout
            os.dup2(devnull, 2)  # stderr
            os.close(devnull)

            with suppress(RuntimeError):
                # torchvision decode failed
                arr = super().deserialize(data)
                arr = self._force_to_rgb(arr)
                return arr

            # Use general PIL decoder
            arr = img_decode_io(data)

            if arr is None:
                return None
            elif arr.ndim == 3:
                arr = arr.transpose([-1, 0, 1])

            arr = self._force_to_rgb(arr)

            return torch.from_numpy(arr)
        finally:
            # Restore stdout and stderr
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)


serializers._SERIALIZERS["jsonl"] = JsonlSerializer()
serializers._SERIALIZERS["tifffile"] = TiffFileSerializer()
serializers._SERIALIZERS["jpeg"] = JPEGGeneralSerializer()
logger.debug("Registered JsonlSerializer for litdata")
logger.debug("Modified TiffFileSerializer for litdata")
logger.debug("Modified JPEGGeneralSerializer for litdata")

# ---------- Rain dataset ------------- #


def find_consecutive_time(
    time_str_list: list[str], time_format="%Y-%m-%d %H:%M:%S", time_interval: int = 30
) -> list[tuple[list[datetime], list[int]]]:
    interval = timedelta(minutes=time_interval)
    # sort times with original indices
    time_list_with_indices = [(datetime.strptime(t, time_format), i) for i, t in enumerate(time_str_list)]
    time_list_with_indices.sort(key=lambda x: x[0].timestamp())

    consecutive_times = []
    current_consecutive = []
    current_indices = []

    for i in range(len(time_list_with_indices)):
        time, original_index = time_list_with_indices[i]
        if i == 0:
            current_consecutive.append(time)
            current_indices.append(original_index)
        elif time - time_list_with_indices[i - 1][0] <= interval:
            current_consecutive.append(time)
            current_indices.append(original_index)
        else:
            consecutive_times.append((current_consecutive, current_indices))
            current_consecutive = [time]
            current_indices = [original_index]

    consecutive_times.append((current_consecutive, current_indices))
    return consecutive_times


def normalize_rain_linear(rain: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    if std <= 0:
        raise ValueError(f"rain std must be > 0, got {std}")
    return (rain - mean) / std


def denormalize_rain_linear(rain_norm: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    if std <= 0:
        raise ValueError(f"rain std must be > 0, got {std}")
    return rain_norm * std + mean


class WindowIndexIterWrapper:
    """Build dataset iteration order for each epoch."""

    def __init__(self, mode: str = "sequential", seed: int = 2025) -> None:
        normalized_mode = str(mode).lower()
        if normalized_mode not in {"sequential", "shuffle_each_epoch"}:
            raise ValueError(f"iter_index_mode must be 'sequential' or 'shuffle_each_epoch', got {mode}")
        self.mode = normalized_mode
        self.seed = int(seed)

    def build_indices(self, total: int, epoch: int) -> list[int]:
        if total <= 0:
            return []
        if self.mode == "sequential":
            return list(range(total))

        generator = torch.Generator()
        generator.manual_seed(self.seed + int(epoch))
        return torch.randperm(total, generator=generator).tolist()


class RainTimeSeriesDataset(IndexedCombinedStreamingDataset):
    def __init__(
        self,
        inp_dirs: list[str],
        time_interval: int = 30,
        n_past=2,
        n_futures=2,
        img_resize: int = 384,
        target_img_resize: int | None = None,
        stack_data=True,
        is_cycled=False,
        ##### dataset configs #####
        index_file_name: str | None = None,
        modality_zero_centering: bool = False,
        rain_norm_mean: float | None = None,
        rain_norm_std: float | None = None,
        clip_values: bool = True,
        radar_clip_min: float | None = 0.0,
        radar_clip_max: float | None = 60.0,
        satellite_clip_min: float | None = 0.0,
        satellite_clip_max: float | None = 300.0,
        rain_clip_min: float | None = 0.0,
        rain_clip_max: float | None = None,
        iter_index_mode: str = "sequential",
        iter_index_seed: int = 2025,
        rain_ratio_filter_enabled: bool = False,
        rain_ratio_filter_file_name: str = "metadata_rain_ratio.parquet",
        rain_ratio_filter_column: str | None = None,
        rain_ratio_filter_min_value: float = 0.0,
        rain_ratio_filter_mode: str = "future_any",
        aug_enabled: bool = False,
        aug_random_crop_prob: float = 0.0,
        aug_random_crop_min_scale: float = 1.0,
        aug_random_crop_max_scale: float = 1.0,
        aug_random_crop_keep_size: bool = True,
        aug_temporal_reverse_prob: float = 0.0,
        *args,
        **kwargs,
    ):
        """
        Metadata: get raw information: radar/satellite file paths, time, station id, lat, lon,
                and collected rain-level station numbers. Can be used to index out the mild/heavy rain if needs.
        """
        repo_root = Path(__file__).resolve().parents[2]
        resolved_inp_dirs: list[str] = []
        for inp_dir in inp_dirs:
            path = Path(inp_dir)
            if not path.is_absolute() and not path.exists():
                candidate = repo_root / path
                if candidate.exists():
                    path = candidate
            resolved_inp_dirs.append(str(path))

        # Create base datasets
        datasets = []
        for inp_dir in resolved_inp_dirs:
            ds = _BaseStreamingDataset.create_dataset(
                input_dir=os.path.join(inp_dir, "pairs"),
                other_ds=None,
                index_file_name=index_file_name,
                is_cycled=is_cycled,
            )
            datasets.append(ds)

        metadata = pd.concat(
            [pd.read_parquet(Path(inp_dir) / "metadata.parquet") for inp_dir in resolved_inp_dirs],
            ignore_index=True,
        )
        self.metadata = metadata
        self.rain_ratio_filter_enabled = bool(rain_ratio_filter_enabled)
        self.rain_ratio_filter_file_name = str(rain_ratio_filter_file_name)
        self.rain_ratio_filter_column = rain_ratio_filter_column
        self.rain_ratio_filter_min_value = float(rain_ratio_filter_min_value)
        self.rain_ratio_filter_mode = str(rain_ratio_filter_mode).lower()
        self.rain_ratio_values: pd.Series | None = self._load_rain_ratio_values(inp_dirs=resolved_inp_dirs)

        super().__init__(combined_is_cycled=is_cycled, datasets=datasets, *args, **kwargs)
        total_base_rows = sum(int(len(ds)) for ds in datasets)
        if total_base_rows != int(len(self.metadata)):
            raise ValueError(
                "metadata row count mismatch with base streaming samples: "
                f"metadata_rows={len(self.metadata)}, base_rows={total_base_rows}"
            )

        self.n_past = n_past
        self.n_futures = n_futures
        self.time_interval = time_interval
        self.stack_data = stack_data  # see docstring
        if bool(aug_enabled) and (not bool(self.stack_data)):
            raise ValueError("dataset augmentation requires stack_data=True.")
        if target_img_resize is not None and bool(aug_enabled):
            raise ValueError("target_img_resize currently requires aug_enabled=False to keep LR/HR targets aligned.")

        self.augmentor = RainTimeSeriesAugmentor(
            enabled=bool(aug_enabled),
            random_crop_prob=float(aug_random_crop_prob),
            random_crop_min_scale=float(aug_random_crop_min_scale),
            random_crop_max_scale=float(aug_random_crop_max_scale),
            random_crop_keep_size=bool(aug_random_crop_keep_size),
            temporal_reverse_prob=float(aug_temporal_reverse_prob),
        )
        self.modality_zero_centering = bool(modality_zero_centering)
        self.rain_norm_mean = rain_norm_mean
        self.rain_norm_std = rain_norm_std
        self.use_rain_linear_norm = self.rain_norm_mean is not None and self.rain_norm_std is not None
        if self.modality_zero_centering and not self.use_rain_linear_norm:
            raise ValueError(
                "modality_zero_centering=True requires rain_norm_mean and rain_norm_std for rain zero-centering."
            )
        if self.use_rain_linear_norm and float(self.rain_norm_std) <= 0:
            raise ValueError(f"rain_norm_std must be > 0 when rain norm is enabled, got {self.rain_norm_std}")

        self.clip_values = bool(clip_values)
        self.radar_clip_min = radar_clip_min
        self.radar_clip_max = radar_clip_max
        self.satellite_clip_min = satellite_clip_min
        self.satellite_clip_max = satellite_clip_max
        self.rain_clip_min = rain_clip_min
        self.rain_clip_max = rain_clip_max
        self.iter_index_mode = str(iter_index_mode).lower()
        self.iter_index_seed = int(iter_index_seed)
        self._iter_epoch = 0
        self.iter_wrapper = WindowIndexIterWrapper(mode=self.iter_index_mode, seed=self.iter_index_seed)

        # Initialize resizers
        self.img_resize = int(img_resize)
        self.target_img_resize = None if target_img_resize is None else int(target_img_resize)
        self._warned_low_resolution_hr_target = False
        self.resizer = Resize((img_resize, img_resize), align_corners=False, keepdim=True)
        self.target_resizer = (
            Resize((int(target_img_resize), int(target_img_resize)), align_corners=False, keepdim=True)
            if target_img_resize is not None
            else None
        )

        # contruct consecutive time groups
        self.times_pairs = []
        self.indices_pairs = []
        self._construct_group_pairs()

    def __len__(self) -> int:
        return len(self.times_pairs)

    def _construct_group_pairs(self):
        consecutive_times_indices = find_consecutive_time(
            self.metadata["time"].tolist(),
            time_interval=self.time_interval,
        )
        consecutive_times = [t[0] for t in consecutive_times_indices]
        consecutive_indices = [t[1] for t in consecutive_times_indices]

        logger.info(f"Found {len(consecutive_times)} consecutive time intervals for the whole dataset")
        filtered_times: list[list[datetime]] = []
        filtered_indices: list[list[int]] = []
        for i, (times, indices) in enumerate(zip(consecutive_times, consecutive_indices)):
            logger.info(
                f"Interval group {i}: Times: {times[0].strftime('%Y%m%d_%H%M%S')}-{times[-1].strftime('%Y%m%d_%H%M%S')}"
            )
            if len(times) < self.n_past + self.n_futures:
                logger.info(f"Skipping interval group {i} due to insufficient data points: {len(times)}")
                continue
            filtered_times.append(times)
            filtered_indices.append(indices)

        # partition the data into overlapping windows
        times_pairs, indices_pairs = self._window_sliding_partition(filtered_times, filtered_indices)
        if len(times_pairs) != len(indices_pairs):
            raise RuntimeError(
                "times_pairs and indices_pairs should have the same length, "
                f"got {len(times_pairs)} vs {len(indices_pairs)}"
            )
        if bool(getattr(self, "rain_ratio_filter_enabled", False)):
            times_pairs, indices_pairs = self._apply_rain_ratio_window_filter(times_pairs=times_pairs, indices_pairs=indices_pairs)

        self.times_pairs.extend(times_pairs)
        self.indices_pairs.extend(indices_pairs)

    def _load_rain_ratio_values(self, inp_dirs: list[str]) -> pd.Series | None:
        if not self.rain_ratio_filter_enabled:
            return None

        if self.rain_ratio_filter_min_value < 0:
            raise ValueError(
                f"rain_ratio_filter_min_value must be >= 0, got {self.rain_ratio_filter_min_value}"
            )
        if self.rain_ratio_filter_mode not in {"future_any", "future_all", "window_any", "window_all"}:
            raise ValueError(
                "rain_ratio_filter_mode must be one of "
                f"{['future_any', 'future_all', 'window_any', 'window_all']}, got {self.rain_ratio_filter_mode}"
            )

        ratio_metadata = pd.concat(
            [pd.read_parquet(Path(inp_dir) / self.rain_ratio_filter_file_name) for inp_dir in inp_dirs],
            ignore_index=True,
        )
        if int(len(ratio_metadata)) != int(len(self.metadata)):
            raise ValueError(
                "metadata_rain_ratio row count mismatch with metadata.parquet: "
                f"ratio_rows={len(ratio_metadata)}, metadata_rows={len(self.metadata)}"
            )

        ratio_column = self.rain_ratio_filter_column
        if ratio_column is None:
            ratio_candidates = [col for col in ratio_metadata.columns if str(col).startswith("rain_ratio_gt_")]
            if len(ratio_candidates) <= 0:
                raise ValueError(
                    f"No rain_ratio_gt_* column found in {self.rain_ratio_filter_file_name}. "
                    "Please set rain_ratio_filter_column explicitly."
                )
            ratio_column = str(ratio_candidates[0])
        if ratio_column not in ratio_metadata.columns:
            raise ValueError(
                f"rain_ratio_filter_column={ratio_column} not found in {self.rain_ratio_filter_file_name}."
            )

        ratio_values = pd.to_numeric(ratio_metadata[ratio_column], errors="coerce")
        self.rain_ratio_filter_column = ratio_column
        valid_count = int(np.isfinite(ratio_values.to_numpy(dtype=np.float32, na_value=np.nan)).sum())
        logger.info(
            "[RainRatioFilter] enabled=True | "
            f"column={self.rain_ratio_filter_column} | "
            f"threshold={self.rain_ratio_filter_min_value} | "
            f"mode={self.rain_ratio_filter_mode} | "
            f"valid_rows={valid_count}/{len(ratio_values)}"
        )
        return ratio_values

    def _apply_rain_ratio_window_filter(
        self,
        *,
        times_pairs: list[tuple[list[datetime], list[datetime]]],
        indices_pairs: list[tuple[list[int], list[int]]],
    ) -> tuple[list[tuple[list[datetime], list[datetime]]], list[tuple[list[int], list[int]]]]:
        if self.rain_ratio_values is None:
            raise RuntimeError("rain_ratio_filter_enabled=True but rain_ratio_values is None.")

        mode = self.rain_ratio_filter_mode
        threshold = self.rain_ratio_filter_min_value
        ratio_values = self.rain_ratio_values.to_numpy(dtype=np.float32, na_value=np.nan)

        kept_times: list[tuple[list[datetime], list[datetime]]] = []
        kept_indices: list[tuple[list[int], list[int]]] = []
        total = len(indices_pairs)
        for tp, ip in zip(times_pairs, indices_pairs):
            past_indices, future_indices = ip
            if mode.startswith("future_"):
                target_indices = future_indices
            else:
                target_indices = [*past_indices, *future_indices]
            if len(target_indices) <= 0:
                continue

            target_ratio = ratio_values[target_indices]
            valid_mask = np.isfinite(target_ratio)
            if not bool(valid_mask.any()):
                continue

            if mode.endswith("_any"):
                keep = bool((target_ratio[valid_mask] >= threshold).any())
            else:
                keep = bool(valid_mask.all() and (target_ratio >= threshold).all())

            if keep:
                kept_times.append(tp)
                kept_indices.append(ip)

        logger.info(
            "[RainRatioFilter] Window filter applied: "
            f"{len(kept_indices)}/{total} windows kept | "
            f"column={self.rain_ratio_filter_column} | "
            f"threshold={threshold} | mode={mode}"
        )
        return kept_times, kept_indices

    def set_num_workers(self, num_workers: int) -> None:
        """
        Keep outer dataloader num_workers, but keep inner streaming datasets in global-index mode.
        """
        self.num_workers = int(num_workers)
        for dataset in self._datasets:
            dataset.set_num_workers(1)

    def set_batch_size(self, batch_size: int | list[int]) -> None:
        """
        Keep outer dataloader batch_size, but keep inner streaming datasets in global-index mode.
        """
        self.batch_size = batch_size
        for dataset in self._datasets:
            dataset.set_batch_size(1)

    def _window_sliding_partition(self, consecutive_times, consecutive_indices):
        """
        Partition the input data into overlapping windows of a specified size.
        """
        w = self.n_past + self.n_futures

        times_pairs = []
        indices_pairs = []

        for i, (ct, ci) in enumerate(zip(consecutive_times, consecutive_indices)):
            if len(ct) < w:
                logger.info(f"Skipping group with insufficient data: {len(ct)} < {w}")
                continue

            for i in tqdm(range(len(ct) - w + 1), desc=f"Sliding window group {i}", leave=False):
                ct_window = ct[i : i + w]
                ic_window = ci[i : i + w]

                ct_w_past = ct_window[: self.n_past]
                ct_w_future = ct_window[self.n_past :]

                ic_w_past = ic_window[: self.n_past]
                ic_w_future = ic_window[self.n_past :]

                times_pairs.append((ct_w_past, ct_w_future))
                indices_pairs.append((ic_w_past, ic_w_future))

        return times_pairs, indices_pairs

    def _date_time_to_float(self, time: datetime) -> float:
        h = time.hour * 60
        m = time.minute
        return (h + m) / 24 / 60  # 0 ~ 1 time

    def _to_float_tensor(self, value: Any, *, field_name: str, index: int) -> torch.Tensor:
        if isinstance(value, (bytes, bytearray, memoryview)):
            try:
                value = serializers._SERIALIZERS["tifffile"].deserialize(bytes(value))
            except Exception as exc:
                raise TypeError(
                    f"Failed to decode bytes field '{field_name}' at sample index={index} "
                    f"as TIFF bytes. bytes_len={len(value)}"
                ) from exc

        try:
            return torch.as_tensor(value, dtype=torch.float32)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise TypeError(
                f"Failed to convert field '{field_name}' at sample index={index} into float32 tensor. "
                f"Got type={type(value).__name__}"
            ) from exc

    def _ensure_chw(self, value: torch.Tensor, *, field_name: str, index: int) -> torch.Tensor:
        if value.ndim == 2:
            return value.unsqueeze(0)
        if value.ndim == 3:
            return value
        raise ValueError(
            f"Expected field '{field_name}' at sample index={index} to be HW/CHW tensor, got shape={tuple(value.shape)}"
        )

    def _prepare_sample_resolutions(
        self, sample: dict, *, index: int
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None]:
        radar = self._to_float_tensor(sample["radar"], field_name="radar", index=index)
        sat = self._to_float_tensor(sample["satellite"], field_name="satellite", index=index)
        rain_int = self._to_float_tensor(sample["rain_interpolated"], field_name="rain_interpolated", index=index)
        if (
            self.target_img_resize is not None
            and not self._warned_low_resolution_hr_target
            and min(int(rain_int.shape[-2]), int(rain_int.shape[-1])) <= self.img_resize
        ):
            logger.warning(
                "target_img_resize is enabled, but raw rain_interpolated is not larger than img_resize; "
                "HR target may not contain more spatial information than the LR input."
            )
            self._warned_low_resolution_hr_target = True

        radar_hr = self.target_resizer(radar) if self.target_resizer is not None else None
        sat_hr = self.target_resizer(sat) if self.target_resizer is not None else None
        rain_int_hr = self.target_resizer(rain_int) if self.target_resizer is not None else None

        radar = self.resizer(radar)
        sat = self.resizer(sat)
        rain_int = self.resizer(rain_int)

        radar = self._ensure_chw(radar, field_name="radar", index=index)
        sat = self._ensure_chw(sat, field_name="satellite", index=index)
        rain_int = self._ensure_chw(rain_int, field_name="rain_interpolated", index=index)
        if radar_hr is not None and sat_hr is not None and rain_int_hr is not None:
            radar_hr = self._ensure_chw(radar_hr, field_name="radar_hr", index=index)
            sat_hr = self._ensure_chw(sat_hr, field_name="satellite_hr", index=index)
            rain_int_hr = self._ensure_chw(rain_int_hr, field_name="rain_interpolated_hr", index=index)

        radar = self._sanitize_and_clip(
            radar,
            min_value=self.radar_clip_min,
            max_value=self.radar_clip_max,
            fill_value=0.0,
        )
        sat = self._sanitize_and_clip(
            sat,
            min_value=self.satellite_clip_min,
            max_value=self.satellite_clip_max,
            fill_value=0.0,
        )
        rain_int = self._sanitize_and_clip(
            rain_int,
            min_value=self.rain_clip_min,
            max_value=self.rain_clip_max,
            fill_value=0.0,
        )
        if radar_hr is not None and sat_hr is not None and rain_int_hr is not None:
            radar_hr = self._sanitize_and_clip(
                radar_hr,
                min_value=self.radar_clip_min,
                max_value=self.radar_clip_max,
                fill_value=0.0,
            )
            sat_hr = self._sanitize_and_clip(
                sat_hr,
                min_value=self.satellite_clip_min,
                max_value=self.satellite_clip_max,
                fill_value=0.0,
            )
            rain_int_hr = self._sanitize_and_clip(
                rain_int_hr,
                min_value=self.rain_clip_min,
                max_value=self.rain_clip_max,
                fill_value=0.0,
            )

        # norm
        _sat_max_value = 300
        _radar_max_value = 60
        sat = sat / _sat_max_value
        radar = radar / _radar_max_value
        if radar_hr is not None and sat_hr is not None:
            radar_hr = radar_hr / _radar_max_value
            sat_hr = sat_hr / _sat_max_value
        if self.modality_zero_centering:
            sat = sat * 2.0 - 1.0
            radar = radar * 2.0 - 1.0
            rain_int = normalize_rain_linear(
                rain_int,
                mean=float(self.rain_norm_mean),
                std=float(self.rain_norm_std),
            )
            if radar_hr is not None and sat_hr is not None and rain_int_hr is not None:
                sat_hr = sat_hr * 2.0 - 1.0
                radar_hr = radar_hr * 2.0 - 1.0
                rain_int_hr = normalize_rain_linear(
                    rain_int_hr,
                    mean=float(self.rain_norm_mean),
                    std=float(self.rain_norm_std),
                )

        low = (radar, sat, rain_int)
        high = (radar_hr, sat_hr, rain_int_hr) if radar_hr is not None and sat_hr is not None and rain_int_hr is not None else None
        return low, high

    def _get_sample(self, index):
        sample = super().__getitem__(index)
        low, _high = self._prepare_sample_resolutions(sample, index=index)
        return low

    def _get_sample_with_target(self, index):
        sample = super().__getitem__(index)
        return self._prepare_sample_resolutions(sample, index=index)

    def _sanitize_and_clip(
        self,
        value: torch.Tensor,
        *,
        min_value: float | None,
        max_value: float | None,
        fill_value: float,
    ) -> torch.Tensor:
        posinf_fill = float(max_value) if max_value is not None else float(fill_value)
        neginf_fill = float(min_value) if min_value is not None else float(fill_value)
        out = torch.nan_to_num(value, nan=float(fill_value), posinf=posinf_fill, neginf=neginf_fill)
        if not self.clip_values:
            return out
        if min_value is None and max_value is None:
            return out
        return torch.clamp(out, min=min_value, max=max_value)

    def _export_metadata_with_rain_ratio(
        self,
        output_parquet_path: str,
        *,
        ratio_threshold: float = 0.1,
        crop_slices: tuple[slice, slice] | None = None,
        overwrite: bool = False,
    ) -> str:
        def format_ratio_threshold_key(threshold: float) -> str:
            token = f"{float(threshold):.6f}".rstrip("0").rstrip(".")
            if token == "":
                token = "0"
            return token.replace("-", "m").replace(".", "p")

        if ratio_threshold < 0:
            raise ValueError(f"ratio_threshold must be >= 0, got {ratio_threshold}")

        output_path = Path(output_parquet_path)
        if output_path.suffix != ".parquet":
            raise ValueError(f"output_parquet_path must end with .parquet, got {output_parquet_path}")
        if output_path.exists() and (not overwrite):
            raise FileExistsError(f"Output parquet already exists: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total = int(len(self.metadata))
        ratios: list[float] = []
        row_slice: slice | None = None
        col_slice: slice | None = None
        if crop_slices is not None:
            row_slice, col_slice = crop_slices

        for sample_index in tqdm(range(total), desc="Compute rain ratio", leave=False):
            ratio_value = float("nan")
            try:
                raw = IndexedCombinedStreamingDataset.__getitem__(self, sample_index)
                rain_raw = raw.get("rain_interpolated", None) if isinstance(raw, dict) else None
                rain = self._to_float_tensor(rain_raw, field_name="rain_interpolated", index=sample_index)
                rain = self.resizer(rain)
                rain = self._sanitize_and_clip(
                    rain,
                    min_value=self.rain_clip_min,
                    max_value=self.rain_clip_max,
                    fill_value=0.0,
                )

                if row_slice is not None and col_slice is not None and rain.ndim >= 2:
                    rain = rain[..., row_slice, col_slice]

                ratio_value = float((rain > float(ratio_threshold)).to(torch.float32).mean().item())
            except Exception as exc:
                logger.warning(
                    f"[RainRatio] sample_index={sample_index} decode/compute failed: {exc}. Fill ratio with NaN."
                )

            ratios.append(ratio_value)

        ratio_col = f"rain_ratio_gt_{format_ratio_threshold_key(ratio_threshold)}"
        metadata_out = self.metadata.copy()
        if output_path.exists():
            existing_out = pd.read_parquet(output_path)
            if int(len(existing_out)) != int(len(metadata_out)):
                raise ValueError(
                    "existing metadata_rain_ratio row count mismatch with metadata.parquet: "
                    f"existing_rows={len(existing_out)}, metadata_rows={len(metadata_out)}"
                )
            existing_ratio_cols = [
                col for col in existing_out.columns if str(col).startswith("rain_ratio_gt_") and col != ratio_col
            ]
            for col in existing_ratio_cols:
                metadata_out[col] = pd.to_numeric(existing_out[col], errors="coerce")
        metadata_out[ratio_col] = ratios
        metadata_out.to_parquet(output_path, index=False)

        valid_count = int(np.isfinite(np.asarray(ratios, dtype=np.float32)).sum())
        logger.info(
            f"[RainRatio] Saved parquet: {output_path} | total={total} | valid_ratio_rows={valid_count} | "
            f"column={ratio_col}"
        )
        return str(output_path)

    def __getitem__(self, index) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        ind_past, ind_future = self.indices_pairs[index]
        times = self.times_pairs[index]

        # data past
        radar_past = []
        sat_past = []
        rain_int_past = []
        radar_past_hr = []
        sat_past_hr = []
        rain_int_past_hr = []
        for _ip in ind_past:
            low, high = self._get_sample_with_target(_ip)
            radar, sat, rain_int = low
            radar_past.append(radar)
            sat_past.append(sat)
            rain_int_past.append(rain_int)
            if high is not None:
                radar_hr, sat_hr, rain_int_hr = high
                radar_past_hr.append(radar_hr)
                sat_past_hr.append(sat_hr)
                rain_int_past_hr.append(rain_int_hr)

        # data future
        radar_future = []
        sat_future = []
        rain_int_future = []
        radar_future_hr = []
        sat_future_hr = []
        rain_int_future_hr = []
        for _if in ind_future:
            low, high = self._get_sample_with_target(_if)
            radar, sat, rain_int = low
            radar_future.append(radar)
            sat_future.append(sat)
            rain_int_future.append(rain_int)
            if high is not None:
                radar_hr, sat_hr, rain_int_hr = high
                radar_future_hr.append(radar_hr)
                sat_future_hr.append(sat_hr)
                rain_int_future_hr.append(rain_int_hr)

        # stack
        if self.stack_data:
            radar_past = torch.stack(radar_past, dim=1)
            sat_past = torch.stack(sat_past, dim=1)
            rain_int_past = torch.stack(rain_int_past, dim=1)
            if len(radar_past_hr) > 0:
                radar_past_hr = torch.stack(radar_past_hr, dim=1)
                sat_past_hr = torch.stack(sat_past_hr, dim=1)
                rain_int_past_hr = torch.stack(rain_int_past_hr, dim=1)

            radar_future = torch.stack(radar_future, dim=1)
            sat_future = torch.stack(sat_future, dim=1)
            rain_int_future = torch.stack(rain_int_future, dim=1)
            if len(radar_future_hr) > 0:
                radar_future_hr = torch.stack(radar_future_hr, dim=1)
                sat_future_hr = torch.stack(sat_future_hr, dim=1)
                rain_int_future_hr = torch.stack(rain_int_future_hr, dim=1)

        # time
        time_past = torch.tensor([self._date_time_to_float(t) for t in times[0]])
        time_future = torch.tensor([self._date_time_to_float(t) for t in times[1]])

        if self.stack_data:
            augmented = self.augmentor(
                radar_past=radar_past,
                radar_future=radar_future,
                satellite_past=sat_past,
                satellite_future=sat_future,
                rain_past=rain_int_past,
                rain_future=rain_int_future,
                time_past=time_past,
                time_future=time_future,
            )
            radar_past = augmented["radar_past"]
            radar_future = augmented["radar_future"]
            sat_past = augmented["satellite_past"]
            sat_future = augmented["satellite_future"]
            rain_int_past = augmented["rain_past"]
            rain_int_future = augmented["rain_future"]
            time_past = augmented["time_past"]
            time_future = augmented["time_future"]
            aug_crop_box_xyxy = augmented["aug_crop_box_xyxy"]
            aug_crop_box_norm_xyxy = augmented["aug_crop_box_norm_xyxy"]
            aug_time_reversed = augmented["aug_time_reversed"]
        else:
            aug_crop_box_xyxy = torch.tensor(
                [0.0, 0.0, float(radar_past[0].shape[-1]), float(radar_past[0].shape[-2])], dtype=torch.float32
            )
            aug_crop_box_norm_xyxy = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)
            aug_time_reversed = torch.tensor(0, dtype=torch.int64)

        out = edict(
            {
                "radar_past": radar_past,  # (bs, c, n_past, h, w)
                "radar_future": radar_future,  # (bs, c, n_future, h, w)
                "satellite_past": sat_past,
                "satellite_future": sat_future,
                "rain_past": rain_int_past,
                "rain_future": rain_int_future,
                "time_past": time_past,
                "time_future": time_future,
                "aug_crop_box_xyxy": aug_crop_box_xyxy,
                "aug_crop_box_norm_xyxy": aug_crop_box_norm_xyxy,
                "aug_time_reversed": aug_time_reversed,
            }
        )
        if len(radar_past_hr) > 0:
            out["radar_past_hr"] = radar_past_hr
            out["satellite_past_hr"] = sat_past_hr
            out["rain_past_hr"] = rain_int_past_hr
            out["radar_future_hr"] = radar_future_hr
            out["satellite_future_hr"] = sat_future_hr
            out["rain_future_hr"] = rain_int_future_hr
        return out

    def __iter__(self):
        """
        Litdata dataloader will cast the dataset into a iter dataset, not a mapped dataset.
        It means the interator will use __iter__ method not __getitem__ method to get a sample to collate function.

        TODO: find a way to get a sample using __getitem__ method in the iter method, especially when want to
        get heavy rain samples only.
        """
        worker_info = get_worker_info()
        worker_id = int(worker_info.id) if worker_info is not None else 0
        num_workers = int(worker_info.num_workers) if worker_info is not None else 1

        rank = int(dist.get_rank()) if dist.is_available() and dist.is_initialized() else 0
        world_size = int(dist.get_world_size()) if dist.is_available() and dist.is_initialized() else 1

        global_worker_id = rank * num_workers + worker_id
        global_num_workers = world_size * num_workers

        total = len(self.times_pairs)
        indices = self.iter_wrapper.build_indices(total=total, epoch=self._iter_epoch)
        self._iter_epoch += 1
        for offset in range(global_worker_id, total, global_num_workers):
            yield self.__getitem__(indices[offset])


def get_litdata_rain_ts_dataloader(
    inp_dirs: list[str],
    *,
    time_interval: int = 30,
    n_past: int = 5,
    n_futures: int = 5,
    img_resize: int = 256,
    target_img_resize: int | None = None,
    stack_data: bool = True,
    index_file_name: str | None = None,
    modality_zero_centering: bool = False,
    rain_norm_mean: float | None = None,
    rain_norm_std: float | None = None,
    clip_values: bool = True,
    radar_clip_min: float | None = 0.0,
    radar_clip_max: float | None = 60.0,
    satellite_clip_min: float | None = 0.0,
    satellite_clip_max: float | None = 300.0,
    rain_clip_min: float | None = 0.0,
    rain_clip_max: float | None = None,
    batching_method: str = "per_stream",
    iterate_over_all: bool = True,
    is_cycled: bool = True,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 8,
    drop_last: bool = True,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int | None = 2,
    use_streaming_loader: bool = True,
    iter_index_mode: str = "sequential",
    iter_index_seed: int = 2025,
    rain_ratio_filter_enabled: bool = False,
    rain_ratio_filter_file_name: str = "metadata_rain_ratio.parquet",
    rain_ratio_filter_column: str | None = None,
    rain_ratio_filter_min_value: float = 0.0,
    rain_ratio_filter_mode: str = "future_any",
    aug_enabled: bool = False,
    aug_random_crop_prob: float = 0.0,
    aug_random_crop_min_scale: float = 1.0,
    aug_random_crop_max_scale: float = 1.0,
    aug_random_crop_keep_size: bool = True,
    aug_temporal_reverse_prob: float = 0.0,
):
    """
    Build RainTimeSeriesDataset + dataloader for training.

    Returns:
        tuple[RainTimeSeriesDataset, StreamingDataLoader]
    """
    ds = RainTimeSeriesDataset(
        inp_dirs=inp_dirs,
        time_interval=time_interval,
        n_past=n_past,
        n_futures=n_futures,
        img_resize=img_resize,
        target_img_resize=target_img_resize,
        stack_data=stack_data,
        is_cycled=is_cycled,
        index_file_name=index_file_name,
        modality_zero_centering=modality_zero_centering,
        rain_norm_mean=rain_norm_mean,
        rain_norm_std=rain_norm_std,
        clip_values=clip_values,
        radar_clip_min=radar_clip_min,
        radar_clip_max=radar_clip_max,
        satellite_clip_min=satellite_clip_min,
        satellite_clip_max=satellite_clip_max,
        rain_clip_min=rain_clip_min,
        rain_clip_max=rain_clip_max,
        iter_index_mode=iter_index_mode,
        iter_index_seed=iter_index_seed,
        rain_ratio_filter_enabled=rain_ratio_filter_enabled,
        rain_ratio_filter_file_name=rain_ratio_filter_file_name,
        rain_ratio_filter_column=rain_ratio_filter_column,
        rain_ratio_filter_min_value=rain_ratio_filter_min_value,
        rain_ratio_filter_mode=rain_ratio_filter_mode,
        aug_enabled=aug_enabled,
        aug_random_crop_prob=aug_random_crop_prob,
        aug_random_crop_min_scale=aug_random_crop_min_scale,
        aug_random_crop_max_scale=aug_random_crop_max_scale,
        aug_random_crop_keep_size=aug_random_crop_keep_size,
        aug_temporal_reverse_prob=aug_temporal_reverse_prob,
        batching_method=batching_method,
        iterate_over_all=iterate_over_all,
    )

    # Keep arg for backward compatibility, but this project now always uses StreamingDataLoader.
    if not use_streaming_loader:
        logger.warning("use_streaming_loader=False is deprecated; forcing StreamingDataLoader.")

    stream_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        stream_kwargs["persistent_workers"] = persistent_workers
    if prefetch_factor is not None and num_workers > 0:
        stream_kwargs["prefetch_factor"] = prefetch_factor
    dl = StreamingDataLoader(ds, **stream_kwargs)

    return ds, dl


def export_rain_filter_ratio(
    inp_dirs: list[str],
    *,
    ratio_threshold: float = 0.1,
    img_resize: int = 256,
    crop_slices: tuple[slice, slice] | None = None,
    overwrite: bool = False,
    index_file_name: str | None = None,
    time_interval: int = 30,
    rain_clip_min: float | None = 0.0,
    rain_clip_max: float | None = None,
) -> list[str]:
    if len(inp_dirs) == 0:
        raise ValueError("inp_dirs must not be empty.")

    def resolve_existing_input_dir(inp_dir: str) -> Path:
        path = Path(inp_dir)
        if path.is_dir():
            return path

        fallback = Path(str(inp_dir).replace("litdata_train_2025", "litdata_train"))
        if fallback.is_dir():
            logger.warning(
                f"[RainRatio] Input dir not found: {inp_dir}. Auto fallback to existing dir: {fallback}"
            )
            return fallback

        raise ValueError(
            f"Input data dir does not exist: {inp_dir}. "
            "Please check your dataset path (e.g., litdata_train vs litdata_train_2025)."
        )

    exported_paths: list[str] = []
    for inp_dir in inp_dirs:
        resolved_dir = resolve_existing_input_dir(inp_dir)
        dataset = RainTimeSeriesDataset(
            inp_dirs=[str(resolved_dir)],
            time_interval=time_interval,
            n_past=1,
            n_futures=1,
            img_resize=img_resize,
            stack_data=True,
            is_cycled=False,
            index_file_name=index_file_name,
            modality_zero_centering=False,
            clip_values=True,
            rain_clip_min=rain_clip_min,
            rain_clip_max=rain_clip_max,
            batching_method="per_stream",
            iterate_over_all=True,
        )
        output_path = resolved_dir / "metadata_rain_ratio.parquet"
        exported_path = dataset._export_metadata_with_rain_ratio(
            output_parquet_path=str(output_path),
            ratio_threshold=ratio_threshold,
            crop_slices=crop_slices,
            overwrite=overwrite,
        )
        exported_paths.append(exported_path)
        logger.info(f"[RainRatio] Exported {exported_path}")

    return exported_paths


if __name__ == "__main__":
    paths = [
        "data2/litdata_train/litdata_interval_30/202305",
        "data2/litdata_train/litdata_interval_30/202306",
        "data2/litdata_train/litdata_interval_30/202307",
        "data2/litdata_train/litdata_interval_30/202308",
        "data2/litdata_train/litdata_interval_30/202309",
        "data2/litdata_train/litdata_interval_30/202506",
        "data2/litdata_train/litdata_interval_30/202507",
        "data2/litdata_train/litdata_interval_30/202508",
        "data2/litdata_train/litdata_interval_30/202509",
    ]
    exported_paths = export_rain_filter_ratio(
        inp_dirs=paths,
        ratio_threshold=0.1,
        overwrite=False,
    )
    for exported_path in exported_paths:
        print(exported_path)
