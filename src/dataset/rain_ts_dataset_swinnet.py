"""
Rain prediction time series dataset module.
This module provides functionality to handle time series datasets for rain prediction tasks.
It includes methods to find consecutive time intervals and to create datasets from given data.
Two types of datasets are implemented:
1. Stochastic loading of hierarchical data in directories
2. Wids dataset format for fast loading

Author: Zihan Cao
Date: 2025-08-10
"""

import io
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import smogn
import random
import os
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import tifffile
import torch as th
import ImbalancedLearningRegression as iblr
from imblearn.over_sampling._smote import SMOTE, SMOTEN, SMOTENC, SVMSMOTE, BorderlineSMOTE, KMeansSMOTE
import webdataset
import wids
from kornia.augmentation import Resize
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
import torchvision.transforms.functional as TF

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.dataset.utils import gaussian_data
from src.utils.logging import log_print
from src.dataset.crop_utils import get_crop_coords
# Add this function near the top of rain_ts_dataset.py
def identity(x):
    return x

# * --- times utilities --- #

#BOUNDS = [0, 0.01, 0.1, 0.3, 0.6, 1]
#BOUNDS = [0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1]
BOUNDS = [0, 0.01, 0.1, 0.2,0.5, 1]

def find_consecutive_time(
    time_str_list: list[str], time_format="%Y-%m-%d %H:%M:%S", time_interval: int = 30
) -> list[tuple[list[datetime], list[int]]]:
    interval = timedelta(minutes=time_interval)
    # sort times with original indices
    time_list_with_indices = [
        (datetime.strptime(t, time_format), i) for i, t in enumerate(time_str_list)
    ]
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


# * --- Dataset module --- #


class RainTimeSeriesDataset(th.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        data_parts: list[str] | None = None,
        time_interval: int = 30,
        n_past=2,
        n_futures=2,
        stack_data=True,
        img_resize: int = 384,
        expand_rain: bool = True,
        return_radar_satellite_futures: bool = False,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.time_interval = time_interval
        self.n_past = n_past
        self.n_futures = n_futures
        self.stack_data = stack_data
        self.return_radar_satellite_futures = return_radar_satellite_futures
        self.expand_rain = expand_rain
        self._expand_rain_n = 2

        self.radar_files = []
        self.satellite_files = []
        self.rain_files = []

        self.times_pairs = []
        self.indices_pairs = []

        for sub_dir in Path(data_dir).iterdir():
            if data_parts is not None and sub_dir.name not in data_parts:
                log_print(f"skip the sub dir {sub_dir.name}")
                continue
            self._construct_group_pairs(sub_dir)

        self.resizer = Resize(
            (img_resize, img_resize), align_corners=False, keepdim=True
        )

    def _construct_group_pairs(self, data_dir: str | Path):
        data_dir = Path(data_dir)
        metadata_file = data_dir / "metadata.parquet"
        assert metadata_file.exists(), f"Metadata file not found: {metadata_file}"
        metadata = pd.read_parquet(metadata_file)

        consecutive_times_indices = find_consecutive_time(
            metadata["time"].tolist(),
            time_interval=self.time_interval,
        )
        consecutive_times = [t[0] for t in consecutive_times_indices]
        consecutive_indices = [t[1] for t in consecutive_times_indices]

        radar_files = [data_dir / f for f in metadata["radar_file"].tolist()]
        satellite_files = [data_dir / f for f in metadata["satellite_file"].tolist()]
        rain_files = [data_dir / f for f in metadata["rain_file"].tolist()]

        self.radar_files.extend(radar_files)
        self.satellite_files.extend(satellite_files)
        self.rain_files.extend(rain_files)

        log_print(
            f"Found {len(consecutive_times)} consecutive time intervals at subdir {data_dir}"
        )
        for i, (times, indices) in enumerate(
            zip(consecutive_times, consecutive_indices)
        ):
            log_print(
                f"Interval group {i}: Times: {times[0].strftime('%Y%m%d_%H%M%S')}-{times[-1].strftime('%Y%m%d_%H%M%S')}"
            )
            if len(times) < self.n_past + self.n_futures:
                log_print(
                    f"Skipping interval group {i} due to insufficient data points: {len(times)}"
                )
                consecutive_times.pop(i)
                consecutive_indices.pop(i)
                continue

        # partition the data into overlapping windows
        times_pairs, indices_pairs = self._window_sliding_partition(
            consecutive_times, consecutive_indices
        )

        self.times_pairs.extend(times_pairs)
        self.indices_pairs.extend(indices_pairs)

    def _window_sliding_partition(self, consecutive_times, consecutive_indices):
        """
        Partition the input data into overlapping windows of a specified size.
        """
        w = self.n_past + self.n_futures

        times_pairs = []
        indices_pairs = []

        for i, (ct, ci) in enumerate(zip(consecutive_times, consecutive_indices)):
            if len(ct) < w:
                log_print(f"Skipping group with insufficient data: {len(ct)} < {w}")
                continue

            for i in tqdm(
                range(len(ct) - w + 1), desc=f"Sliding window group {i}", leave=False
            ):
                ct_window = ct[i : i + w]
                ic_window = ci[i : i + w]

                ct_w_past = ct_window[: self.n_past]
                ct_w_future = ct_window[self.n_past :]

                ic_w_past = ic_window[: self.n_past]
                ic_w_future = ic_window[self.n_past :]

                times_pairs.append((ct_w_past, ct_w_future))
                indices_pairs.append((ic_w_past, ic_w_future))

        return times_pairs, indices_pairs

    def __len__(self):
        # return sum([len(times) for times in self.consecutive_times])
        return len(self.times_pairs)

    def _read_data(self, time_past: list[int], time_future: list[int]):
        def _read_tiff(file: str, is_rain: bool = False):
            # import time

            # t0 = time.time()
            img = tifffile.imread(
                file, maxworkers=8, mode="r", buffersize=16 * 1024 * 1024
            ).astype(np.float32)
            # print(f"Loading time: {time.time() - t0:.4f}s")

            img = th.tensor(img)
            img = self.resizer(img)  # Resize the image

            if is_rain and self.expand_rain:
                img = gaussian_data(
                    img,
                    sigma=1.5,
                    unchanged_amp=True,
                    n_times=self._expand_rain_n,
                    use_cuda=False,
                )
                assert isinstance(img, th.Tensor), "Image should be a tensor"

            if img.ndim == 3:
                img = img.permute([-1, 0, 1])  # Ensure the image is in (C, H, W) format
            elif img.ndim == 2:
                img = img[None]
            else:
                raise ValueError(
                    f"Image {file} has unexpected dimensions: {img.ndim}. Expected 2D or 3D with channels first."
                )

            return img

        if self.stack_data:
            # data shape: [batch_size, channels, times, height, width]
            read_data_past = lambda file, is_rain: th.stack(
                [_read_tiff(file[i], is_rain) for i in time_past], dim=-3
            )
            read_data_future = lambda file, is_rain: th.stack(
                [_read_tiff(file[i], is_rain) for i in time_future], dim=-3
            )
        else:
            read_data_past = lambda file, is_rain: [
                _read_tiff(file[i], is_rain) for i in time_past
            ]
            read_data_future = lambda file, is_rain: [
                _read_tiff(file[i], is_rain) for i in time_future
            ]

        radar_past = read_data_past(self.radar_files, False)
        satellite_past = read_data_past(self.satellite_files, False)
        rain_past = read_data_past(self.rain_files, True)

        rain_future = read_data_future(self.rain_files, True)

        if self.return_radar_satellite_futures:
            radar_future = read_data_future(self.radar_files, False)
            satellite_future = read_data_future(self.satellite_files, False)

            return (
                radar_past,
                radar_future,
                satellite_past,
                satellite_future,
                rain_past,
                rain_future,
            )
        else:
            return (radar_past, satellite_past, rain_past, rain_future)

    def _norm_img(self, img, value):
        if th.is_tensor(img):
            img = img / value
        elif isinstance(img, list):
            img = [im / value for im in img]

        return img

    def _date_time_to_float(self, time: datetime) -> float:
        h = time.hour * 60
        m = time.minute
        return (h + m) / 24 / 60

    def __getitem__(self, index):
        _sat_max_value = 300
        _radar_max_value = 60

        tp = self.indices_pairs[index]
        times = self.times_pairs[index]
        time_past, time_future = tp

        out = self._read_data(time_past, time_future)
        if self.return_radar_satellite_futures:
            rp, rf, sp, sf, ra_p, ra_f = out

            sp = self._norm_img(sp, _sat_max_value)
            sf = self._norm_img(sf, _sat_max_value)

            rp = self._norm_img(rp, _radar_max_value)
            rf = self._norm_img(rf, _radar_max_value)

            ret = {
                # pasts
                "radar_past": rp,
                "satellite_past": sp,
                "rain_past": ra_p,
                # futures
                "radar_future": rf,
                "satellite_future": sf,
                "rain_future": ra_f,
                # times
                "time_past": th.tensor([self._date_time_to_float(t) for t in times[0]]),
                "time_future": th.tensor(
                    [self._date_time_to_float(t) for t in times[1]]
                ),
            }

        else:
            rp, sp, ra_p, ra_f = out

            sp = self._norm_img(sp, _sat_max_value)
            rp = self._norm_img(rp, _radar_max_value)

            ret = {
                # pasts
                "radar_past": rp,
                "satellite_past": sp,
                "rain_past": ra_p,
                # futures
                "rain_future": ra_f,
                # times
                "time_past": th.tensor([self._date_time_to_float(t) for t in times[0]]),
                "time_future": th.tensor(
                    [self._date_time_to_float(t) for t in times[1]]
                ),
            }

        return ret


class RainTimeSeriesWidsDataset(th.utils.data.Dataset):
    def __init__(
        self,
        index_file: str | Path,
        time_interval: int = 30,
        n_past=2,
        n_futures=2,
        stack_data=True,
        img_resize: int = 384,
    ) -> None:
        super().__init__()
        self.index_file = Path(index_file)
        self.n_past = n_past
        self.n_futures = n_futures
        self.stack_data = stack_data
        self.time_interval = time_interval

        self.dataset = wids.ShardListDataset(
            shards=index_file,
            localname=lambda name: local_name_fn(name, None),
            transformations=[  # type: ignore
                wids_transform,
                remove_undecoded_keys,
                wids_remove_none_keys,
            ],
        )
        metadata_path = self.index_file.parent / "metadata.parquet"
        self.metadata = pd.read_parquet(metadata_path)

        assert self.metadata.shape[0] == len(self.dataset), (
            f"Metadata length does not match dataset length, but got {self.metadata.shape[0]} and {len(self.dataset)}"
        )

        # Initialize resizer
        self.resizer = Resize(
            (img_resize, img_resize), align_corners=False, keepdim=True
        )

        # contruct consecutive time groups
        self.times_pairs = []
        self.indices_pairs = []
        self._construct_group_pairs_wids()

    def _construct_group_pairs_wids(self):
        consecutive_times_indices = find_consecutive_time(
            self.metadata["time"].tolist(),
            time_interval=self.time_interval,
        )
        consecutive_times = [t[0] for t in consecutive_times_indices]
        consecutive_indices = [t[1] for t in consecutive_times_indices]

        log_print(
            f"Found {len(consecutive_times)} consecutive time intervals for the whole dataset"
        )
        for i, (times, indices) in enumerate(
            zip(consecutive_times, consecutive_indices)
        ):
            log_print(
                f"Interval group {i}: Times: {times[0].strftime('%Y%m%d_%H%M%S')}-{times[-1].strftime('%Y%m%d_%H%M%S')}"
            )
            if len(times) < self.n_past + self.n_futures:
                log_print(
                    f"Skipping interval group {i} due to insufficient data points: {len(times)}"
                )
                consecutive_times.pop(i)
                consecutive_indices.pop(i)
                continue

        # partition the data into overlapping windows
        times_pairs, indices_pairs = self._window_sliding_partition(
            consecutive_times, consecutive_indices
        )

        self.times_pairs.extend(times_pairs)
        self.indices_pairs.extend(indices_pairs)

    def _window_sliding_partition(self, consecutive_times, consecutive_indices):
        """
        Partition the input data into overlapping windows of a specified size.
        """
        w = self.n_past + self.n_futures

        times_pairs = []
        indices_pairs = []

        for i, (ct, ci) in enumerate(zip(consecutive_times, consecutive_indices)):
            if len(ct) < w:
                log_print(f"Skipping group with insufficient data: {len(ct)} < {w}")
                continue

            for i in tqdm(
                range(len(ct) - w + 1), desc=f"Sliding window group {i}", leave=False
            ):
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

    def _get_sample_from_wids(self, index):
        sample = self.dataset[index]  # single process here
        radar, sat, rain_int = (
            sample["radar"],
            sample["satellite"],
            sample["rain_interpolated"],
        )
        to_tensor = lambda x: th.as_tensor(x, dtype=th.float32)
        radar, sat, rain_int = map(to_tensor, [radar, sat, rain_int])

        # Apply resizing
        radar = self.resizer(radar)
        sat = self.resizer(sat)
        rain_int = self.resizer(rain_int)

        # norm
        _sat_max_value = 300
        _radar_max_value = 60
        sat = sat / _sat_max_value
        radar = radar / _radar_max_value

        return radar, sat, rain_int
    
    def _hash_cfg(self, cfg: dict) -> str:
        """根据数据集配置生成唯一哈希，用于区分不同缓存"""
        s = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
        return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]

    def compute_sample_rain_intensity(
        self,
        cache_dir: str | os.PathLike | None = None,
        *,
        force_recompute: bool = False,
        log_every: int = 500,
    ):
        """
        预计算每个样本未来帧的降雨强度（基于 90% 分位数的稳健指标），并缓存。
        """

        # 构建缓存路径
        cache_path = None
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            sig = {
                "num_samples": len(self.indices_pairs),
                "time_interval": getattr(self, "time_interval", None),
                "n_past": getattr(self, "n_past", None),
                "n_futures": getattr(self, "n_futures", None),
                "img_resize": getattr(self, "img_resize", None),
            }
            if hasattr(self, "index_file") and self.index_file is not None:
                sig["index_file"] = str(self.index_file)
            if hasattr(self, "data_dir") and self.data_dir is not None:
                sig["data_dir"] = str(self.data_dir)

            cache_name = f"rain_strengths_{self._hash_cfg(sig)}.npy"
            cache_path = os.path.join(cache_dir, cache_name)

        # 尝试读取缓存
        if (cache_path is not None) and (not force_recompute) and os.path.isfile(cache_path):
            self.sample_rain_strengths = np.load(cache_path).astype(np.float16, copy=False)
            log_print(f"[cache] loaded from {cache_path}")
            return

        # 计算
        N = len(self.indices_pairs)
        rain_strengths = np.zeros((N,), dtype=np.float16)

        for i, (_, ind_future) in enumerate(self.indices_pairs):
            vals = []
            for if_ in ind_future:
                _, _, rain = self._get_sample_from_wids(if_)
                vals.append(float(rain.max().item()))  # 每帧最大降雨量

            if len(vals) > 0:
                score = np.percentile(vals, 90)

                # 极弱雨(<0.01)归为无雨
                if score < 0.01:
                    score = 0.0

                rain_strengths[i] = score
            else:
                rain_strengths[i] = 0.0

            if (i + 1) % log_every == 0:
                log_print(f"[compute] processed {i+1}/{N}")

        self.sample_rain_strengths = rain_strengths

        # 保存缓存
        if cache_path is not None:
            np.save(cache_path, rain_strengths)
            log_print(f"[cache] saved to {cache_path}")
    

    def __len__(self):
        return len(self.indices_pairs)

    def __getitem__(self, index) -> dict[str, th.Tensor | list[th.Tensor]]:
        ind_past, ind_future = self.indices_pairs[index]
        times = self.times_pairs[index]

        # data past
        radar_past = []
        sat_past = []
        rain_int_past = []
        for ip in ind_past:
            radar, sat, rain_int = self._get_sample_from_wids(ip)
            radar_past.append(radar)
            sat_past.append(sat)
            rain_int_past.append(rain_int)

        # data future
        radar_future = []
        sat_future = []
        rain_int_future = []
        for if_ in ind_future:
            radar, sat, rain_int = self._get_sample_from_wids(if_)
            radar_future.append(radar)
            sat_future.append(sat)
            rain_int_future.append(rain_int)

        # stack
        if self.stack_data:
            radar_past = th.stack(radar_past, dim=-3)
            sat_past = th.stack(sat_past, dim=-3)
            rain_int_past = th.stack(rain_int_past, dim=-3)

            radar_future = th.stack(radar_future, dim=-3)
            sat_future = th.stack(sat_future, dim=-3)
            rain_int_future = th.stack(rain_int_future, dim=-3)

        # time
        time_past = th.tensor([self._date_time_to_float(t) for t in times[0]])
        time_future = th.tensor([self._date_time_to_float(t) for t in times[1]])

        return {
            "radar_past": radar_past,  # (bs, c, n_past, h, w)
            "radar_future": radar_future,  # (bs, c, n_future, h, w)
            "satellite_past": sat_past,
            "satellite_future": sat_future,
            "rain_past": rain_int_past,
            "rain_future": rain_int_future,
            "time_past": time_past,
            "time_future": time_future,
        }


class RainTimeSeriesWidsDataset_CLS(RainTimeSeriesWidsDataset):
    def __getitem__(self, index) -> dict[str, th.Tensor | list[th.Tensor]]:
        ind_past, ind_future = self.indices_pairs[index]
        times = self.times_pairs[index]

        radar_past, sat_past, rain_int_past = [], [], []
        for ip in ind_past:
            radar, sat, rain = self._get_sample_from_wids(ip)
            radar_past.append(radar)
            sat_past.append(sat)
            rain_int_past.append(rain)

        radar_future, sat_future, rain_int_future = [], [], []
        for if_ in ind_future:
            radar, sat, rain = self._get_sample_from_wids(if_)
            radar_future.append(radar)
            sat_future.append(sat)
            rain_int_future.append(rain)

        if self.stack_data:
            radar_past = th.stack(radar_past, dim=-3)
            sat_past = th.stack(sat_past, dim=-3)
            rain_int_past = th.stack(rain_int_past, dim=-3)
            radar_future = th.stack(radar_future, dim=-3)
            sat_future = th.stack(sat_future, dim=-3)
            rain_int_future = th.stack(rain_int_future, dim=-3)  # [1, n_future, H, W]

        # 将未来雨量离散化
        bounds = np.array(BOUNDS, dtype=np.float32)
        rain_np = rain_int_future.numpy()  # float
        rain_cls = _bin_by_bounds(rain_np, bounds)  # int
        # rain_cls = np.squeeze(rain_cls, axis=0)
        rain_cls = th.as_tensor(rain_cls, dtype=th.long)

        time_past = th.tensor([self._date_time_to_float(t) for t in times[0]])
        time_future = th.tensor([self._date_time_to_float(t) for t in times[1]])

        return {
            "radar_past": radar_past,       
            "satellite_past": sat_past,
            "rain_past": rain_int_past,      
            "radar_future": radar_future,
            "satellite_future": sat_future,
            "rain_future": rain_int_future,   
            "rain_future_cls": rain_cls, 
            "time_past": time_past,
            "time_future": time_future,
        }


class RainTimeSeriesWidsDataset_CLS_Crop(RainTimeSeriesWidsDataset_CLS):
    """
    训练集：
        - 只对 strong & no-rain 进行裁剪
        - strong sample 使用 threshold-guided crop
        - no-rain sample 使用随机 crop
        - 使训练数据分布更平衡

    验证/测试：
        - 不裁剪，保持完整图像
    """

    def __init__(
        self,
        *args,
        strong_threshold: float = 0.3,
        is_train: bool = False,
        patch_size: int = 128,
        cache_dir: str = '__cache__',
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.strong_threshold = strong_threshold
        self.patch_size = patch_size
        self.is_train = is_train

        self.strong_cache_path = os.path.join(
            cache_dir, f"strong_indices_thr{strong_threshold}_{patch_size}.npy"
        )
        self.bg_cache_path = os.path.join(
            cache_dir, f"norain_indices_thr{strong_threshold}_{patch_size}.npy"
        )

        if not self.is_train:
            self.strong_indices = None
            self.norain_indices = None
            return


        if os.path.exists(self.strong_cache_path):
            log_print(f"Loading strong indices from cache: {self.strong_cache_path}")
            self.strong_indices = np.load(self.strong_cache_path).tolist()
        else:
            log_print("Scanning strong rain samples (first time)...")
            self.strong_indices = []

            for idx in tqdm(range(len(self.indices_pairs)), desc="Scanning strong"):
                _, ind_future = self.indices_pairs[idx]
                has_strong = False

                for if_ in ind_future:
                    _, _, rain = self._get_sample_from_wids(if_)
                    if rain.max() >= strong_threshold:
                        has_strong = True
                        break

                if has_strong:
                    self.strong_indices.append(idx)

            np.save(self.strong_cache_path, np.array(self.strong_indices))
            log_print(f"Saved strong cache: {self.strong_cache_path}")


        if os.path.exists(self.bg_cache_path):
            log_print(f"Loading no-rain indices from cache: {self.bg_cache_path}")
            self.norain_indices = np.load(self.bg_cache_path).tolist()
        else:
            log_print("Scanning no-rain samples (first time)...")
            self.norain_indices = []

            for idx in tqdm(range(len(self.indices_pairs)), desc="Scanning no-rain"):
                _, ind_future = self.indices_pairs[idx]
                has_strong = False

                for if_ in ind_future:
                    _, _, rain = self._get_sample_from_wids(if_)
                    if rain.max() >= strong_threshold:
                        has_strong = True
                        break

                if not has_strong:
                    self.norain_indices.append(idx)

            np.save(self.bg_cache_path, np.array(self.norain_indices))
            log_print(f"Saved bg cache: {self.bg_cache_path}")

        log_print(
            f"Strong samples: {len(self.strong_indices)} / {len(self.indices_pairs)} | "
            f"NoRain samples: {len(self.norain_indices)} / {len(self.indices_pairs)}"
        )


    def __len__(self):
        if self.is_train:
            return len(self.strong_indices)
        return super().__len__()

 
    def _crop_item(self, item, top, left):
        P = self.patch_size

        def c(x):
            return x[..., top:top+P, left:left+P]

        for key in [
            "radar_past",
            "satellite_past",
            "rain_past",
            "radar_future",
            "satellite_future",
            "rain_future",
            "rain_future_cls",
        ]:
            if key in item and isinstance(item[key], th.Tensor):
                item[key] = c(item[key])

        return item
    
    
    def apply_rain_augmentation(self, item):
        """对雷达/卫星/雨图像做一致增强"""
        keys = ["radar_past", "satellite_past", "rain_past",
                "radar_future", "satellite_future", "rain_future", "rain_future_cls"]

        # 水平翻转
        if random.random() < 0.5:
            for k in keys:
                if k in item and isinstance(item[k], th.Tensor):
                    item[k] = th.flip(item[k], dims=[-1])   # W维

        # 垂直翻转
        if random.random() < 0.5:
            for k in keys:
                if k in item and isinstance(item[k], th.Tensor):
                    item[k] = th.flip(item[k], dims=[-2])   # H维

        # 随机旋转
        if random.random() < 0.5:
            k90 = random.choice([1, 2, 3])  # 90,180,270°
            for k in keys:
                if k in item and isinstance(item[k], th.Tensor):
                    item[k] = th.rot90(item[k], k=k90, dims=[-2, -1])

        return item


    def __getitem__(self, index):
        if not self.is_train:
            return super().__getitem__(index)

        # # 20% 采样 NO-RAIN 样本
        # if random.random() < 0.20 and len(self.norain_indices) > 0:
        #     true_index = random.choice(self.norain_indices)
        #     item = super().__getitem__(true_index)

        #     # 随机裁剪一个 patch
        #     _, _, H, W = item["rain_future"].shape
        #     top = random.randint(0, max(0, H - self.patch_size))
        #     left = random.randint(0, max(0, W - self.patch_size))
        #     item = self._crop_item(item, top, left)

        #     # 不增强
        #     return item

        # 80% 采样 STRONG 样本
        true_index = self.strong_indices[index]
        item = super().__getitem__(true_index)
        rain_future = item["rain_future"]

        # threshold crop
        coords = get_crop_coords(
            rain_future,
            patch_size=self.patch_size,
            threshold=self.strong_threshold
        )

        if coords is None:
            _, _, H, W = rain_future.shape
            top = (H - self.patch_size) // 2
            left = (W - self.patch_size) // 2
        else:
            top, left = coords

        item = self._crop_item(item, top, left)

        # 增强
        max_rain = item["rain_future"].max().item()

        if max_rain >= self.strong_threshold:
            # 强降雨，一定要增强
            item = self.apply_rain_augmentation(item)

        elif max_rain >= 0.01:
            # 小雨，30% 概率增强
            if random.random() < 0.3:
                item = self.apply_rain_augmentation(item)

        return item


class RainTimeSeriesWidsDataset_CLS_filter(RainTimeSeriesWidsDataset_CLS):
    """
    train:
        -按照min_rain和min_ratio去筛选数据
        -min_rain:降雨阈值判断
        -min_ratio:阈值占比
    val:
        -不筛选,正常val
    """
    def __init__(
        self,
        *args,
        is_train: bool = False,
        min_rain: float = 0.1,
        min_ratio: float = 0.05,
        cache_dir: str = '__cache__',
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.is_train = is_train
        self.min_rain = min_rain
        self.min_ratio = min_ratio

        self.filter_cache_path = os.path.join(
            cache_dir,
            f"filter_indices_rain{min_rain}_ratio{min_ratio}.npy"
        )

        if not self.is_train:
            self.filtered_indices = None
            log_print(f"[Filter] 非训练模式，使用完整数据集: {len(self.indices_pairs)} 样本")
            return

        # 优先读取缓存
        if os.path.exists(self.filter_cache_path):
            log_print(f"[Filter] 从缓存加载过滤索引: {self.filter_cache_path}")
            self.filtered_indices = np.load(self.filter_cache_path).tolist()
        else:
            log_print(f"[Filter] 扫描并过滤降雨样本...")
            self.filtered_indices = []
            
            for idx in tqdm(range(len(self.indices_pairs)), desc="过滤扫描"):
                _, ind_future = self.indices_pairs[idx]
                valid = False
                
                for if_ in ind_future:
                    try:
                        _, _, rain = self._get_sample_from_wids(if_)
                        rain_np = rain.numpy() if hasattr(rain, "numpy") else np.array(rain)
                        max_rain = rain_np.max()
                        ratio = (rain_np > self.min_rain).sum() / rain_np.size
                        
                        if max_rain > self.min_rain and ratio >= self.min_ratio:
                            valid = True
                            break
                    except Exception as e:
                        log_print(f"[Filter] 加载样本 {if_} 时出错: {e}")
                        continue
                
                if valid:
                    self.filtered_indices.append(idx)
            
            # 保存缓存
            os.makedirs(cache_dir, exist_ok=True)
            np.save(self.filter_cache_path, np.array(self.filtered_indices))
            log_print(f"[Filter] 保存过滤缓存: {self.filter_cache_path}")

        log_print(
            f"[Filter] 过滤条件: min_rain={min_rain}, min_ratio={min_ratio}"
            f" | 原始: {len(self.indices_pairs)} → 过滤后: {len(self.filtered_indices)}"
        )

        # 详细统计信息
        #if self.filtered_indices:
            # 分析过滤后样本的降雨强度分布
            #rain_strengths = []
            #for idx in self.filtered_indices[:min(50, len(self.filtered_indices))]:  # 采样前50个
               # _, ind_future = self.indices_pairs[idx]
                #for if_ in ind_future:
                    #try:
                        #_, _, rain = self._get_sample_from_wids(if_)
                        #rain_strengths.append(float(rain.max().item()))
                        #break
                    #except:
                    #    continue
            
            #if rain_strengths:
            #    log_print(f"[Filter] 样本降雨强度统计 - 平均: {np.mean(rain_strengths):.3f}, "
            #             f"最大: {np.max(rain_strengths):.3f}, 最小: {np.min(rain_strengths):.3f}")

    def __len__(self):
        if self.is_train and self.filtered_indices is not None:
            return len(self.filtered_indices)
        return super().__len__()

    def __getitem__(self, index):
        if not self.is_train or self.filtered_indices is None:
            return super().__getitem__(index)
        
        if index >= len(self.filtered_indices):
            raise IndexError(f"索引 {index} 超出过滤后数据集范围 {len(self.filtered_indices)}")
            
        true_index = self.filtered_indices[index]
        try:
            item = super().__getitem__(true_index)
            return item
        except Exception as e:
            log_print(f"[Filter] 获取样本 {true_index} 时出错: {e}")
            # 返回一个有效的替代样本
            return super().__getitem__(0)


class RainTimeSeriesWidsDataset_CLS_filter_721(RainTimeSeriesWidsDataset_CLS):
    """
    Train:
        - 混合采样策略 (70% Strong, 20% Medium, 10% Light/Others)
        - Strong: rain > min_rain & ratio >= min_ratio
        - Medium: rain > min_rain & 0.005 <= ratio < min_ratio (作为过渡)
        - Light:  Others (包括无雨)
    Val:
        - 不筛选，正常返回所有数据
    """
    def __init__(
        self,
        *args,
        is_train: bool = False,
        min_rain: float = 0.1,
        min_ratio: float = 0.02, 
        cache_dir: str = '__cache__',
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.is_train = is_train
        self.min_rain = min_rain
        self.min_ratio = min_ratio
        
        # 定义三类数据的缓存路径
        self.cache_strong = os.path.join(cache_dir, f"indices_strong_r{min_rain}_rat{min_ratio}.npy")
        self.cache_medium = os.path.join(cache_dir, f"indices_medium_r{min_rain}_rat{min_ratio}.npy")
        self.cache_light = os.path.join(cache_dir, f"indices_light_r{min_rain}_rat{min_ratio}.npy")

        if not self.is_train:
            self.indices_pool = None
            log_print(f"[Filter] 非训练模式，使用完整数据集: {len(self.indices_pairs)} 样本")
            return

        # --- 扫描或加载缓存 ---
        if os.path.exists(self.cache_strong) and os.path.exists(self.cache_medium) and os.path.exists(self.cache_light):
            log_print(f"[Filter] 从缓存加载三类索引...")
            self.strong_indices = np.load(self.cache_strong).tolist()
            self.medium_indices = np.load(self.cache_medium).tolist()
            self.light_indices = np.load(self.cache_light).tolist()
        else:
            log_print(f"[Filter] 开始扫描数据集并分类 (Strong/Medium/Light)...")
            self.strong_indices = []
            self.medium_indices = []
            self.light_indices = []
            
            # Medium 的下界
            medium_ratio_lower = 0.005 
            
            for idx in tqdm(range(len(self.indices_pairs)), desc="扫描样本分类"):
                _, ind_future = self.indices_pairs[idx]
                
                # 获取该样本未来帧的最大雨量特征
                max_r_val = 0.0
                max_ratio_val = 0.0
                
                # 取未来几帧里最强的那个作为代表
                for if_ in ind_future:
                    try:
                        _, _, rain = self._get_sample_from_wids(if_)
                        rain_np = rain.numpy() if hasattr(rain, "numpy") else np.array(rain)
                        
                        curr_max = rain_np.max()
                        curr_ratio = (rain_np > self.min_rain).sum() / rain_np.size
                        
                        if curr_max > max_r_val:
                            max_r_val = curr_max
                            max_ratio_val = curr_ratio
                    except:
                        continue
                
                # --- 分类逻辑 ---
                if max_r_val > self.min_rain and max_ratio_val >= self.min_ratio:
                    self.strong_indices.append(idx)
                elif max_r_val > self.min_rain and max_ratio_val >= medium_ratio_lower:
                    self.medium_indices.append(idx)
                else:
                    self.light_indices.append(idx)
            
            # 保存缓存
            os.makedirs(cache_dir, exist_ok=True)
            np.save(self.cache_strong, np.array(self.strong_indices))
            np.save(self.cache_medium, np.array(self.medium_indices))
            np.save(self.cache_light, np.array(self.light_indices))

        log_print(
            f"[Filter] 分类统计:\n"
            f"  - Strong (70%): {len(self.strong_indices)} (rain>{min_rain}, ratio>={min_ratio})\n"
            f"  - Medium (20%): {len(self.medium_indices)} (rain>{min_rain}, 0.005<=ratio<{min_ratio})\n"
            f"  - Light  (10%): {len(self.light_indices)} (Others)"
        )
        
        # 兜底防止某一类为空
        if len(self.strong_indices) == 0: self.strong_indices = list(range(len(self.indices_pairs)))
        if len(self.medium_indices) == 0: self.medium_indices = self.light_indices 
        if len(self.light_indices) == 0: self.light_indices = self.strong_indices # 极端情况

    def __len__(self):
        # 训练时，长度定义为 Strong 样本数量除以 0.7，保证每个 epoch 能大概把 Strong 样本跑一遍
        if self.is_train and hasattr(self, 'strong_indices'):
            return int(len(self.strong_indices) / 0.7)
        return super().__len__()

    def __getitem__(self, index):
        if not self.is_train:
            return super().__getitem__(index)
        
        # --- 概率采样逻辑 ---
        r = random.random()
        
        if r < 0.7:
            # 70% 概率取 Strong
            pool = self.strong_indices
        elif r < 0.9:
            # 20% 概率取 Medium (0.7 + 0.2)
            pool = self.medium_indices
        else:
            # 10% 概率取 Light
            pool = self.light_indices
            
        # 随机从选定的池子里取一个索引
        rand_idx = random.choice(pool)
        
        try:
            return super().__getitem__(rand_idx)
        except Exception as e:
            log_print(f"[Filter] Error loading sample {rand_idx}: {e}")
            return super().__getitem__(0)
def _bin_by_bounds(values: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """将连续雨量值按分级边界划分为类别索引"""
    cls = np.digitize(values, bounds, right=False) - 1
    K = len(bounds) - 1
    return np.clip(cls, 0, K - 1).astype(np.int64)


def extract_smote_features(dataset, cache_dir="__cache__"):
    """
    从 RainTimeSeriesWidsDataset_CLS 提取用于 SMOTE 的时空特征。
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True, parents=True)
    cache_path = cache_dir / "rain_features_with_pastrain.parquet"

    if cache_path.exists():
        print(f"[ILR] 已检测到缓存特征文件：{cache_path}")
        df_features = pd.read_parquet(cache_path)
        print(f"[ILR] 已从缓存加载特征，共 {len(df_features)} 条样本")
        if "idx" not in df_features.columns:
            df_features["idx"] = np.arange(len(df_features))
            print(f"[ILR] 自动补充 idx 列，共 {len(df_features)} 条样本。")
        return df_features

    print("[ILR] 未检测到缓存特征文件，开始计算时空特征...")
    mean_radar, std_radar, max_radar, var_radar = [], [], [], []
    mean_sat, std_sat, max_sat, var_sat = [], [], [], []
    mean_rain, std_rain, max_rain, var_rain = [], [], [], []
    delta_mean_radar = []

    for i in tqdm(range(len(dataset.indices_pairs)), desc="[ILR] 提取特征中"):
        ind_past, _ = dataset.indices_pairs[i]
        radar_seq, sat_seq, rain_seq = [], [], []

        # 遍历过去帧
        for ip in ind_past:
            radar, sat, rain = dataset._get_sample_from_wids(ip)
            radar = radar.numpy() if hasattr(radar, "numpy") else np.array(radar)
            sat = sat.numpy() if hasattr(sat, "numpy") else np.array(sat)
            rain = rain.numpy() if hasattr(rain, "numpy") else np.array(rain)
            radar_seq.append(radar)
            sat_seq.append(sat)
            rain_seq.append(rain)

        radar_seq = np.stack(radar_seq, axis=1)  # (C, T, H, W)
        sat_seq = np.stack(sat_seq, axis=1)
        rain_seq = np.stack(rain_seq, axis=1)

        # === 基本统计 ===
        mean_radar.append(radar_seq.mean())
        std_radar.append(radar_seq.std())
        max_radar.append(radar_seq.max())

        mean_sat.append(sat_seq.mean())
        std_sat.append(sat_seq.std())
        max_sat.append(sat_seq.max())

        mean_rain.append(rain_seq.mean())
        std_rain.append(rain_seq.std())
        max_rain.append(rain_seq.max())

        # === 雷达帧间变化 ===
        delta = np.diff(radar_seq, axis=1)
        delta_mean_radar.append(delta.mean())

    df_features = pd.DataFrame({
        "mean_radar": mean_radar,
        "std_radar": std_radar,
        "max_radar": max_radar,
        "mean_sat": mean_sat,
        "std_sat": std_sat,
        "max_sat": max_sat,
        "mean_rain": mean_rain,
        "std_rain": std_rain,
        "max_rain": max_rain,
        "delta_mean_radar": delta_mean_radar,
    })

    df_features.to_parquet(cache_path)
    print(f"[ILR] 特征计算完成，共 {len(df_features)} 条样本，已缓存到：{cache_path}")
    return df_features


def get_dataloader_weight(
    data_dir: str | Path | None = None,
    data_parts: list[str] | None = None,
    index_file: str | Path | None = None,
    *,
    time_interval: int = 30,
    n_past: int = 2,
    n_futures: int = 2,
    stack_data: bool = True,
    img_resize: int = 384,
    expand_rain: bool = True,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    persistent_workers: bool = False,
    pin_memory: bool = False,
    use_wids: bool = False,
    use_weight_sampler: bool = False,
    cache_dir: str | Path = "__cache__"  # 对应 compute_sample_rain_intensity()
) -> tuple:
    """
    创建 RainTimeSeriesDataset 或 RainTimeSeriesWidsDataset 及其对应的 DataLoader。

    Args:
        ...
        use_weight_sampler (bool): 是否启用类别均衡采样（WeightedRandomSampler）

    Returns:
        (dataset, dataloader)
    """

    if use_wids:
        if index_file is None:
            raise ValueError("index_file must be provided when use_wids is True")
        dataset = RainTimeSeriesWidsDataset_CLS_Crop(
            index_file=index_file,
            time_interval=time_interval,
            n_past=n_past,
            n_futures=n_futures,
            stack_data=stack_data,
            img_resize=img_resize,
        )

    else:
        assert data_dir is not None, "data_dir must be provided when use_wids is False"

        dataset = RainTimeSeriesDataset(
            data_dir=data_dir,
            data_parts=data_parts,
            time_interval=time_interval,
            n_past=n_past,
            n_futures=n_futures,
            stack_data=stack_data,
            img_resize=img_resize,
            expand_rain=expand_rain,
        )

    if use_weight_sampler:
            if not hasattr(dataset, "sample_rain_strengths"):
                if hasattr(dataset, "compute_sample_rain_intensity"):
                    dataset.compute_sample_rain_intensity(cache_dir=cache_dir)
                else:
                    raise AttributeError(
                        "Dataset 缺少 sample_rain_strengths 属性且无 compute_sample_rain_intensity() 方法。"
                    )

    sampler = None
    if use_weight_sampler:
        rain_strengths = dataset.sample_rain_strengths
        classes = _bin_by_bounds(rain_strengths, BOUNDS)
        K = len(BOUNDS) - 1

        # 类别统计
        class_counts = np.bincount(classes, minlength=K)
        log_print(
            "类别统计: "
            + ", ".join(f"[{BOUNDS[i]:.1f},{BOUNDS[i+1]:.1f})={class_counts[i]}" for i in range(K))
        )

        target_ratio = np.array([0.05, 0.35, 0.35, 0.25])
        
        weights_per_class = target_ratio / np.maximum(class_counts, 1)
        weights_per_class /= weights_per_class.mean()

        sample_weights = weights_per_class[classes]
        weights_tensor = th.tensor(sample_weights, dtype=th.double)

        sampler = th.utils.data.WeightedRandomSampler(
            weights=weights_tensor,
            num_samples=len(sample_weights),
            replacement=True,
        )

        # 打印采样分布验证
        sampled_indices = list(sampler)
        sampled_cls = classes[sampled_indices]
        sampled_counts = np.bincount(sampled_cls, minlength=K)
        log_print(
            "[Sampled distribution]: "
            + ", ".join(f"[{BOUNDS[i]},{BOUNDS[i+1]})={sampled_counts[i]}" for i in range(K))
        )


    dataloader = th.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and not use_weight_sampler),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=6 if num_workers > 0 else None,
    )

    return dataset, dataloader

def compute_lmu_lsigma(y):
    """
    计算 l_mu, l_sigma
    """
    y = np.array(y)
    y = y[y > 0]  # 只取有雨样本
    log_y = np.log(y)
    l_mu = log_y.mean()
    l_sigma = log_y.std()
    return l_mu, l_sigma


def get_dataloader_ilr(
    data_dir: str | Path | None = None,
    data_parts: list[str] | None = None,
    index_file: str | Path | None = None,
    *,
    time_interval: int = 30,
    n_past: int = 2,
    n_futures: int = 2,
    stack_data: bool = True,
    img_resize: int = 384,
    expand_rain: bool = True,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    persistent_workers: bool = False,
    pin_memory: bool = False,
    use_wids: bool = False,
    use_ilr_oversample: bool = False,
    ilr_cfg: dict | None = None,
    cache_dir: str | Path = "__cache__",
    is_class: bool = False,
    is_train: bool = False,
    # patch_size: int = 128,
) -> tuple:
    """
    创建使用 ILR 过采样的 DataLoader（回归不平衡增强）：
    """

    if use_wids:
        assert index_file is not None, "index_file must be provided when use_wids is True"
        
        # === [修复点 1] 无论 is_train 是 True 还是 False，都必须初始化 dataset ===
        # 移除了外层的 if is_train: 判断，直接初始化
        dataset = RainTimeSeriesWidsDataset_CLS(
            index_file=index_file,
            time_interval=time_interval,
            n_past=n_past,
            n_futures=n_futures,
            stack_data=stack_data,
            img_resize=img_resize,
            # strong_threshold=0.3, # 如果需要这些参数，请取消注释并确保类定义支持它们
            # is_train=is_train,
            # patch_size=patch_size,
        )
        # ======================================================================

    else:
        assert data_dir is not None, "data_dir must be provided when use_wids is False"
        dataset = RainTimeSeriesDataset(
            data_dir=data_dir,
            data_parts=data_parts,
            time_interval=time_interval,
            n_past=n_past,
            n_futures=n_futures,
            stack_data=stack_data,
            img_resize=img_resize,
            expand_rain=expand_rain,
        )

    # 只有 dataset 成功初始化后，才进行后续操作
    if not hasattr(dataset, "sample_rain_strengths"):
        # 确保 compute_sample_rain_intensity 存在再调用
        if hasattr(dataset, "compute_sample_rain_intensity"):
            dataset.compute_sample_rain_intensity(cache_dir=cache_dir)

    # === [修复点 2] 只有在启用过采样且是训练模式时，才执行 ILR ===
    # 通常测试时不应该进行过采样
    if use_ilr_oversample and is_train:
        y_cont = dataset.sample_rain_strengths.astype(float)
        idx = np.arange(len(y_cont))

        if is_class:
            df_features = extract_smote_features(dataset, cache_dir)
            bounds = np.array(BOUNDS, dtype=np.float32)
            y_cls = np.digitize(y_cont, bins=bounds) - 1
            df = pd.DataFrame({
                "idx": idx,
                "rain_class": y_cls,
            })
        else:
            df = pd.DataFrame({
                "idx": idx,
                "rain_strength": y_cont,
                "x_dummy": idx
            })

        # ... (省略中间的统计打印代码，保持原样) ...
        # 为了简洁，这里省略了打印统计信息的代码，请保留你原有的打印逻辑

        ilr_cfg = ilr_cfg or {}
        method = ilr_cfg.pop("method", "smote" if is_class else "ro")

        #  分类任务：imblearn.SMOTE 
        if is_class:
            log_print(f"[ILR] 分类任务：使用 imblearn.BorderlineSMOTE 进行过采样")

            # ... (保持原有的 SMOTE 配置代码) ...
            # 为了简洁，这里直接使用最核心的逻辑
            smote = BorderlineSMOTE(
                sampling_strategy={3:2000, 4:2200, 5:2400},
                random_state=42,
                k_neighbors=3,
                m_neighbors=10,
                kind="borderline-1",
            )
            selected_features = ["mean_rain", "max_rain", "std_rain"]

            y = df["rain_class"].to_numpy()
            X = df_features[selected_features].to_numpy()
            idx = df_features["idx"].to_numpy()
            X_with_idx = np.column_stack([idx, X])
            
            # 执行重采样
            try:
                X_res, y_res = smote.fit_resample(X_with_idx, y)
                df_res = pd.DataFrame({
                    "idx": X_res[:, 0].astype(int),
                    "rain_class": y_res
                })
            except Exception as e:
                log_print(f"[ILR Error] SMOTE 失败: {e}", level="ERROR")
                df_res = df # 回退

        #  回归任务
        else:
            if not hasattr(iblr, method):
                raise ValueError(f"[ILR] 无效方法名: {method}")

            func = getattr(iblr, method)
            log_print(f"[ILR] 回归任务：使用 {method}() 方法过采样")
            df_res = func(data=df, y="rain_strength", **ilr_cfg)

        log_print(f"[ILR] 原样本数: {len(df)} → 过采样后样本数: {len(df_res)}")

        # 更新 Dataset 索引
        new_indices = df_res["idx"].astype(int).tolist()
        # 必须做越界检查
        max_idx = len(dataset.indices_pairs)
        dataset.indices_pairs = [dataset.indices_pairs[i] for i in new_indices if i < max_idx]
        dataset.times_pairs   = [dataset.times_pairs[i]   for i in new_indices if i < max_idx]

    # 创建 DataLoader
    dataloader = th.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        # 如果启用了 ILR 过采样，通常不再 shuffle，或者由 Sampler 控制
        # 这里保持你的逻辑：如果用了 ILR 就不 shuffle (因为已经是重采样过的了)
        shuffle=(shuffle and not use_ilr_oversample),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=6 if num_workers > 0 else None,
    )

    return dataset, dataloader


def get_dataloader_smogn(
    data_dir: str | Path | None = None,
    data_parts: list[str] | None = None,
    index_file: str | Path | None = None,
    *,
    time_interval: int = 30,
    n_past: int = 2,
    n_futures: int = 2,
    stack_data: bool = True,
    img_resize: int = 384,
    expand_rain: bool = True,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    persistent_workers: bool = False,
    pin_memory: bool = False,
    cache_dir: str | Path = "__cache__",
    smogn_cfg: dict | None = None,
    use_wids: bool = False,
):
    """
    使用 SMOGN 对高雨量样本进行过采样的 DataLoader 生成函数。
    """

    if use_wids:
        assert index_file is not None, "index_file must be provided when use_wids is True"
        dataset = RainTimeSeriesWidsDataset_CLS(
            index_file=index_file,
            time_interval=time_interval,
            n_past=n_past,
            n_futures=n_futures,
            stack_data=stack_data,
            img_resize=img_resize,
        )
    else:
        assert data_dir is not None, "data_dir must be provided when use_wids is False"
        dataset = RainTimeSeriesDataset(
            data_dir=data_dir,
            data_parts=data_parts,
            time_interval=time_interval,
            n_past=n_past,
            n_futures=n_futures,
            stack_data=stack_data,
            img_resize=img_resize,
            expand_rain=expand_rain,
        )

    if not hasattr(dataset, "sample_rain_strengths"):
        dataset.compute_sample_rain_intensity(cache_dir=cache_dir)

    y = dataset.sample_rain_strengths.astype(float)
    df = pd.DataFrame({"idx": np.arange(len(y)), "rain_strength": y})

    smogn_default = dict(
        y="rain_strength",
        samp_method="extreme", 
        rel_method="auto",
        rel_thres=0.8,
        k=5,
        rel_xtrm_type="high",
    )
    if smogn_cfg:
        smogn_default.update(smogn_cfg)

    log_print(f"[SMOGN] 原样本数: {len(df)}")

    df_res = smogn.smoter(data=df, **smogn_default)
    log_print(f"[SMOGN] 过采样后样本数: {len(df_res)}")

    new_y = df_res["rain_strength"].to_numpy()

    class_counts, _ = np.histogram(new_y, bins=BOUNDS)
    log_print(
        "[SMOGN] 类别统计: "
        + ", ".join(
            f"[{BOUNDS[i]:.1f},{BOUNDS[i+1]:.1f})={class_counts[i]}"
            for i in range(len(class_counts))
        )
    )

    new_indices = df_res["idx"].astype(int).tolist()
    dataset.indices_pairs = [dataset.indices_pairs[i] for i in new_indices if i < len(dataset.indices_pairs)]
    dataset.times_pairs = [dataset.times_pairs[i] for i in new_indices if i < len(dataset.times_pairs)]

    dataloader = th.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=6 if num_workers > 0 else None,
    )

    return dataset, dataloader


def get_dataloader_filter(
    data_dir: str | Path | None = None,
    data_parts: list[str] | None = None,
    index_file: str | Path | None = None,
    *,
    time_interval: int = 30,
    n_past: int = 2,
    n_futures: int = 2,
    stack_data: bool = True,
    img_resize: int = 384,
    expand_rain: bool = True,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    persistent_workers: bool = False,
    pin_memory: bool = False,
    use_wids: bool = False,
    is_train: bool = False,
    min_rain: float = 0.1,
    min_ratio: float = 0.05,
    cache_dir: str = "__cache__",
):
    """
    Create dataset and dataloader with rain intensity filtering.
    """
    if use_wids:
        if index_file is None:
            raise ValueError("index_file must be provided when use_wids is True")

        dataset = RainTimeSeriesWidsDataset_CLS_filter(
            index_file=index_file,
            time_interval=time_interval,
            n_past=n_past,
            n_futures=n_futures,
            stack_data=stack_data,
            img_resize=img_resize,
            is_train=is_train,
            min_rain=min_rain,
            min_ratio=min_ratio,
            cache_dir=cache_dir,
        )

    else:
        # 对于非Wids数据集，目前不支持过滤功能
        assert data_dir is not None, "data_dir must be provided when use_wids is False"
        dataset = RainTimeSeriesDataset(
            data_dir=data_dir,
            data_parts=data_parts,
            time_interval=time_interval,
            n_past=n_past,
            n_futures=n_futures,
            stack_data=stack_data,
            img_resize=img_resize,
            expand_rain=expand_rain,
        )
        log_print("Warning: Filtering is only supported for WIDS datasets. Using original dataset without filtering.")

    # 关键修复：检查数据集长度
    dataset_length = len(dataset)
    log_print(f"数据集长度: {dataset_length}, batch_size: {batch_size}")
    
    if dataset_length == 0:
        raise ValueError(f"数据集为空！过滤条件可能太严格: min_rain={min_rain}, min_ratio={min_ratio}")
    
    if dataset_length < batch_size:
        log_print(f"警告: 数据集长度({dataset_length})小于batch_size({batch_size})，调整batch_size")
        batch_size = max(1, dataset_length // 2)  # 调整为数据集长度的一半或1

    dataloader = th.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,  # 减少prefetch factor
    )

    # 验证数据加载器
    try:
        # 尝试获取第一个batch来验证数据加载器工作正常
        first_batch = next(iter(dataloader))
        log_print(f"数据加载器验证成功，第一个batch包含 {len(first_batch)} 个键")
    except StopIteration:
        raise ValueError("数据加载器为空，无法获取任何batch！")
    except Exception as e:
        log_print(f"数据加载器验证时出现警告: {e}")

    return dataset, dataloader


def get_dataloader_filter_721(
    data_dir: str | Path | None = None,
    data_parts: list[str] | None = None,
    index_file: str | Path | None = None,
    *,
    time_interval: int = 30,
    n_past: int = 2,
    n_futures: int = 2,
    stack_data: bool = True,
    img_resize: int = 384,
    expand_rain: bool = True,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    persistent_workers: bool = False,
    pin_memory: bool = False,
    use_wids: bool = False,
    is_train: bool = False,
    min_rain: float = 0.1,
    min_ratio: float = 0.05,
    cache_dir: str = "__cache__",
):
    """
    Create dataset and dataloader with rain intensity filtering.
    """
    if use_wids:
        if index_file is None:
            raise ValueError("index_file must be provided when use_wids is True")

        dataset = RainTimeSeriesWidsDataset_CLS_filter_721(
            index_file=index_file,
            time_interval=time_interval,
            n_past=n_past,
            n_futures=n_futures,
            stack_data=stack_data,
            img_resize=img_resize,
            is_train=is_train,
            min_rain=min_rain,
            min_ratio=min_ratio,
            cache_dir=cache_dir,
        )

    else:
        # 对于非Wids数据集，目前不支持过滤功能
        assert data_dir is not None, "data_dir must be provided when use_wids is False"
        dataset = RainTimeSeriesDataset(
            data_dir=data_dir,
            data_parts=data_parts,
            time_interval=time_interval,
            n_past=n_past,
            n_futures=n_futures,
            stack_data=stack_data,
            img_resize=img_resize,
            expand_rain=expand_rain,
        )
        log_print("Warning: Filtering is only supported for WIDS datasets. Using original dataset without filtering.")

    # 关键修复：检查数据集长度
    dataset_length = len(dataset)
    log_print(f"数据集长度: {dataset_length}, batch_size: {batch_size}")
    
    if dataset_length == 0:
        raise ValueError(f"数据集为空！过滤条件可能太严格: min_rain={min_rain}, min_ratio={min_ratio}")
    
    if dataset_length < batch_size:
        log_print(f"警告: 数据集长度({dataset_length})小于batch_size({batch_size})，调整batch_size")
        batch_size = max(1, dataset_length // 2)  # 调整为数据集长度的一半或1

    dataloader = th.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,  # 减少prefetch factor
    )

    # 验证数据加载器
    try:
        # 尝试获取第一个batch来验证数据加载器工作正常
        first_batch = next(iter(dataloader))
        log_print(f"数据加载器验证成功，第一个batch包含 {len(first_batch)} 个键")
    except StopIteration:
        raise ValueError("数据加载器为空，无法获取任何batch！")
    except Exception as e:
        log_print(f"数据加载器验证时出现警告: {e}")

    return dataset, dataloader


def get_dataloader_crop(
    data_dir: str | Path | None = None,
    data_parts: list[str] | None = None,
    index_file: str | Path | None = None,
    *,
    time_interval: int = 30,
    n_past: int = 2,
    n_futures: int = 2,
    stack_data: bool = True,
    img_resize: int = 384,
    expand_rain: bool = True,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    persistent_workers: bool = False,
    pin_memory: bool = False,
    use_wids: bool = False,
    # is_class: bool = False,
    is_train: bool = False,
    patch_size: int = 128,
    strong_threshold: float = 0.3,
    cache_dir: str = "__cache__",
):
    """
    Create dataset and dataloader for both cropped and normal training.
    """


    if use_wids:
        if index_file is None:
            raise ValueError("index_file must be provided when use_wids is True")

        dataset = RainTimeSeriesWidsDataset_CLS_Crop(
            index_file=index_file,
            time_interval=time_interval,
            n_past=n_past,
            n_futures=n_futures,
            stack_data=stack_data,
            img_resize=img_resize,
            strong_threshold=strong_threshold,
            is_train=is_train,
            patch_size=patch_size,
            cache_dir=cache_dir,
        )


    else:
        assert data_dir is not None, "data_dir must be provided when use_wids is False"

        dataset = RainTimeSeriesDataset(
            data_dir=data_dir,
            data_parts=data_parts,
            time_interval=time_interval,
            n_past=n_past,
            n_futures=n_futures,
            stack_data=stack_data,
            img_resize=img_resize,
            expand_rain=expand_rain,
        )


    dataloader = th.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=6 if num_workers > 0 else None,
    )

    return dataset, dataloader



def get_dataloader(
    data_dir: str | Path | None = None,
    data_parts: list[str] | None = None,
    index_file: str | Path | None = None,
    *,
    time_interval: int = 30,
    n_past: int = 2,
    n_futures: int = 2,
    stack_data: bool = True,
    img_resize: int = 384,
    expand_rain: bool = True,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    persistent_workers: bool = False,
    pin_memory: bool = False,
    use_wids: bool = False,
):
    """
    Create a RainTimeSeriesDataset or RainTimeSeriesWidsDataset and corresponding DataLoader.

    Args:
        data_dir (str | Path): Path to the data directory.
        data_parts (list[str] | None): List of data parts to include (for RainTimeSeriesDataset).
        time_interval (int): Time interval in minutes between consecutive data points. Default is 30.
        n_past (int): Number of past time steps to use as input. Default is 2.
        n_futures (int): Number of future time steps to predict. Default is 2.
        stack_data (bool): Whether to stack data along the time dimension. Default is True.
        img_resize (int): Size to resize images to. Default is 384.
        expand_rain (bool): Whether to expand rain data using Gaussian expansion. Default is True.
        batch_size (int): Batch size for the DataLoader. Default is 1.
        shuffle (bool): Whether to shuffle the data. Default is False.
        num_workers (int): Number of worker processes for data loading. Default is 0.
        persistent_workers (bool): Whether to keep workers alive after dataset has been consumed once.
        use_wids (bool): Whether to use RainTimeSeriesWidsDataset instead of RainTimeSeriesDataset.
        index_file (str | Path | None): Path to the index file for Wids dataset.

    Returns:
        tuple: A tuple containing (dataset, dataloader).
    """
    if use_wids:
        if index_file is None:
            raise ValueError("index_file must be provided when use_wids is True")

        dataset = RainTimeSeriesWidsDataset(
            index_file=index_file,
            time_interval=time_interval,
            n_past=n_past,
            n_futures=n_futures,
            stack_data=stack_data,
            img_resize=img_resize,
        )

    else:
        assert data_dir is not None, "data_dir must be provided when use_wids is False"

        dataset = RainTimeSeriesDataset(
            data_dir=data_dir,
            data_parts=data_parts,
            time_interval=time_interval,
            n_past=n_past,
            n_futures=n_futures,
            stack_data=stack_data,
            img_resize=img_resize,
            expand_rain=expand_rain,
        )

    dataloader = th.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=6 if num_workers > 0 else None,
    )

    return dataset, dataloader


# * --- utilities --- #


def tiff_decode_io(
    tiff_bytes: bytes,
    use_out_param: bool = True,  # Flag to control using 'out' parameter
    backend: str = "tifffile",  # Backend to use for TIFF decoding
) -> np.ndarray:
    """
    Decodes TIFF formatted bytes into a NumPy array.
    Optionally reads metadata first to pre-allocate memory for the 'out' parameter.
    """

    if backend == "tifffile":
        __predefined_buf_size = 16 * 1024 * 1024
        __max_workers = 8

        if use_out_param:
            try:
                # Step 1: Read metadata using TiffFile to get shape and dtype from the first series
                expected_shape: tuple[int, ...]
                expected_dtype: np.dtype
                with io.BytesIO(tiff_bytes) as metadata_buffer:
                    with tifffile.TiffFile(metadata_buffer) as tif:
                        if not tif.series or not tif.series[0]:
                            raise ValueError(
                                "TIFF file contains no series or series[0] is invalid."
                            )
                        current_series: tifffile.TiffPageSeries = tif.series[0]
                        expected_shape = current_series.shape
                        expected_dtype = np.dtype(
                            current_series.dtype
                        )  # Ensure it's a numpy.dtype

                # Step 2: Pre-allocate the output array
                output_array: np.ndarray = np.empty(
                    expected_shape, dtype=expected_dtype
                )

                # Step 3: Read image data into the pre-allocated array
                # Use a new BytesIO object for imread to ensure the stream is at the beginning
                with io.BytesIO(tiff_bytes) as data_buffer:
                    tifffile.imread(
                        data_buffer,
                        out=output_array,
                        buffersize=__predefined_buf_size,  # Example buffer size
                        maxworkers=__max_workers,  # Example max workers
                    )
                return output_array
            except Exception as e:
                # Fallback to the simpler method if any error occurs during the 'out' optimization path
                # You might want to log the error 'e' for debugging purposes
                # print(f"Warning: Failed to use 'out' parameter optimization: {e}. Falling back.")
                with io.BytesIO(tiff_bytes) as buffer:
                    img: np.ndarray = tifffile.imread(
                        buffer,
                        buffersize=__predefined_buf_size,
                        maxworkers=__max_workers,
                    )
                return img
        else:
            # Original behavior if use_out_param is False
            with io.BytesIO(tiff_bytes) as buffer:
                img: np.ndarray = tifffile.imread(
                    buffer,
                    buffersize=__predefined_buf_size,
                    maxworkers=__max_workers,
                )
            return img
    else:
        raise NotImplementedError(
            f"TIFF decoding with backend '{backend}' is not implemented. "
            "Please use 'tifffile' backend."
        )


def wids_transform(sample):
    modalities = [
        ".radar.tiff",
        ".satellite.tiff",
        # ".rain.tiff",
        ".rain_interpolated.tiff",
    ]
    for m in modalities:
        name = m.rsplit(".", 1)[0][1:]
        img = sample.pop(m)
        img = tiff_decode_io(img.getvalue())
        if img.ndim == 2:
            img = img[..., np.newaxis]  # Add channel dimension if missing
        elif img.ndim == 3:
            pass
        else:
            raise ValueError(f"Unexpected image dimensions: {img.ndim} for {m}")

        img = img.transpose([-1, 0, 1])  # [h, w, c]
        sample[name] = img

    sample.pop("__dataset__")
    return sample


def remove_undecoded_keys(sample):
    # check all keys
    for k, v in sample.items():
        if not k.startswith("__") and isinstance(v, bytes):
            log_print(f"{k} is undecoded", "warning")
            return None
    return sample


def wids_remove_none_keys(sample: dict[str, Any]) -> dict[str, Any]:
    """Remove keys with None values from the sample dictionary."""
    # remove None key/values
    _key_to_del = []
    for k, v in sample.items():
        if v is None:
            _key_to_del.append(k)
    for k in _key_to_del:
        del sample[k]

    return sample


def local_name_fn(name, prefix: str | None = None):
    """Helper function to create local names with a prefix."""

    if prefix is None:
        return name
    else:
        return f"{prefix}/{name}"


# * --- test --- #


def test_wids():
    ds = wids.ShardListDataset(
        "data_original/zihan_processed/interval_30/202305/satellite_shardindex.json",
        transformations=[wids_transform],
        localname=lambda name: local_name_fn(name),
    )
    dl = webdataset.WebLoader(
        ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )

    import time

    total_time = 0
    t0 = time.time()

    for i, sample in enumerate(dl):
        # 处理 batch 数据
        print(f"Batch {i}, img shape: {sample['img'].shape}")

        # 累积时间
        batch_time = time.time() - t0
        total_time += batch_time
        print(f"Batch {i} time taken: {batch_time:.4f} seconds")

        t0 = time.time()


def test_wids_loading(check_shape=False):
    index_file = "/home/JieYiZhu/Dataset/wds_interval_30/shardindex.json"
    metadata_file = "/home/JieYiZhu/Dataset/wds_interval_30/metadata.parquet"

    # Test with get_dataloader function
    dataset, dl = get_dataloader(
        index_file=index_file,
        use_wids=True,
        time_interval=30,
        n_past=2,
        n_futures=2,
        stack_data=True,
        img_resize=384,
        batch_size=16,
        shuffle=True,
        num_workers=12,
    )

    import time

    t0 = time.time()
    for i, sample in enumerate(dl):
        t1 = time.time()
        print(f"load using {t1 - t0} s")

        # Print sample keys
        if i == 0:
            print("Sample keys:", list(sample.keys()))

        if check_shape:
            # Check shapes for all modalities
            # Expected format: (batch_size, channels, time, height, width)
            for key, value in sample.items():
                if isinstance(value, th.Tensor) and ("past" in key or "future" in key):
                    print(f"{key} shape: {value.shape}")

                    # Verify shape format is (bs, c, t, h, w)
                    if value.dim() != 5:
                        print(
                            f"WARNING: {key} does not have expected 5 dimensions (bs, c, t, h, w)"
                        )
                    else:
                        bs, c, t, h, w = value.shape
                        print(f"  - Batch size: {bs}")
                        print(f"  - Channels: {c}")
                        print(f"  - Time steps: {t}")
                        print(f"  - Height: {h}")
                        print(f"  - Width: {w}")

                        # Check if height and width match the resize parameter
                        if h != 384 or w != 384:
                            print(
                                f"WARNING: {key} height/width {h}x{w} does not match expected 384x384"
                            )

        # Test a few specific modalities
        if i == 0:  # Just test the first batch
            # Check specific modalities
            modalities = [
                "radar_past",
                "radar_future",
                "satellite_past",
                "satellite_future",
                "rain_past",
                "rain_future",
            ]

            for modality in modalities:
                if modality in sample:
                    shape = sample[modality].shape
                    print(f"{modality} shape: {shape}")

                    # Validate shape format
                    if len(shape) == 5:
                        bs, c, t, h, w = shape
                        print(
                            f"  Valid 5D format - bs:{bs}, c:{c}, t:{t}, h:{h}, w:{w}"
                        )
                    else:
                        print(f"  ERROR: Expected 5D tensor, got {len(shape)}D")

        t0 = time.time()

        # Just test one batch for shape verification
        if i >= 0 and check_shape:
            break


def test_loading():
    ds = RainTimeSeriesDataset(
        data_dir="data_original/zihan_processed/interval_30",
        time_interval=30,
        n_past=1,
        n_futures=1,
        expand_rain=True,
    )

    dl = th.utils.data.DataLoader(
        ds,
        batch_size=2,
        shuffle=False,
        num_workers=6,
        # persistent_workers=True,
    )

    for sample in tqdm(dl, total=len(ds) // 12):
        ...

    log_print("Testing loading finished.")


def test_imgs(use_wids=True):
    from src.utils.visualization.plot import plot_any_modality

    if use_wids:
        ds, dl = get_dataloader_ilr(
            index_file="/home/rainpred/RainPrediction/DATA/wds_interval_30/train/shardindex.json",
            use_wids=True,
            time_interval=30,
            n_past=10,
            n_futures=1,
            stack_data=True,
            img_resize=256,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            use_ilr_oversample=True,
            ilr_cfg=None,
            cache_dir="__cache__",
            is_class=True,
            is_train=True,
            patch_size=128,
        )
    else:
        ds = RainTimeSeriesDataset(
            data_dir="data_original/zihan_processed/interval_30/202308",
            time_interval=30,
            n_past=2,
            n_futures=1,
            expand_rain=False,
        )

        dl = th.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    import time

    # t1 = time.time()
    for sample in dl:
        # print(sample["time_past"])
        print(sample["rain_past"])

        rain_past = sample["rain_past"][0:1][:, :, 0]
        rain_future = sample["rain_future"][0:1][:, :, 0]

        radar_past = sample["radar_past"][0:1][:, :, 0]
        radar_future = sample["radar_future"][0:1][:, :, 0]

        sate_past = sample["satellite_past"][0:1][:, :, 0]
        sate_future = sample["satellite_future"][0:1][:, :, 0]

        # plot
        rain_past_p = plot_any_modality(rain_past, "rain")
        rain_future_p = plot_any_modality(rain_future, "rain")

        sat_past_p = plot_any_modality(sate_past, "satellite")
        sat_future_p = plot_any_modality(sate_future, "satellite")

        radar_past_p = plot_any_modality(radar_past, "radar")
        radar_future_p = plot_any_modality(radar_future, "radar")
        pass

        # print(time.time() - t1)
        # t1 = time.time()


def save_image(arr, save_path):
    from PIL import Image
    arr = np.array(arr)

    # squeeze掉多余维度
    arr = np.squeeze(arr)  # (H, W, 1) → (H, W)

    if arr.ndim != 2:
        raise ValueError(f"save_image expects 2D array, got shape={arr.shape}")

    # normalize to 0–255
    arr_norm = arr - arr.min()
    if arr_norm.max() > 0:
        arr_norm = arr_norm / arr_norm.max()
    arr_norm = (arr_norm * 255).astype(np.uint8)

    img = Image.fromarray(arr_norm)
    img.save(save_path)



def analyze_patches(
    use_wids=True,
    save_dir="threshold_check",
    strong_threshold=0.3,
    patch_size=128
):
    os.makedirs(save_dir, exist_ok=True)

    # -----------------------------
    # 1. Load dataloader
    # -----------------------------
    if use_wids:
        ds, dl = get_dataloader_crop(
            index_file="/home/rainpred/RainPrediction/DATA/wds_interval_30/train/shardindex.json",
            use_wids=True,
            time_interval=30,
            n_past=10,
            n_futures=1,
            stack_data=True,
            img_resize=256,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            is_train=True,
            patch_size=patch_size,
            strong_threshold=strong_threshold,
        )
    else:
        ds = RainTimeSeriesDataset(
            data_dir="data_original/zihan_processed/interval_30/202308",
            time_interval=30,
            n_past=2,
            n_futures=1,
            expand_rain=False,
        )
        dl = th.utils.data.DataLoader(ds, batch_size=1)

    print("Loaded dataset:", len(ds))

    # -----------------------------
    # 2. Statistics accumulator
    # -----------------------------
    total_bg_ratio = []
    total_small_ratio = []
    total_strong_ratio = []

    sample_id = 0
    max_samples = 200

    # -----------------------------
    # 3. Iterate over samples
    # -----------------------------
    for sample_id, sample in enumerate(dl):
        # sample["rain_past"] shape: [1, 1, 10, 128, 128]

        rain_past = sample["rain_past"][0, 0, -1]   # 取最后一帧
        rain_future = sample["rain_future"][0, 0, 0]   # 取唯一未来帧

        # 转 numpy
        rain_past_np = rain_past.numpy()
        rain_future_np = rain_future.numpy()

        # 统计比例
        strong_threshold = 0.3
        future = rain_future_np

        bg_ratio = np.mean(future < 0.1)
        small_ratio = np.mean((future >= 0.1) & (future < strong_threshold))
        strong_ratio = np.mean(future >= strong_threshold)

        print(f"[{sample_id}] bg={bg_ratio:.4f}, small={small_ratio:.4f}, strong={strong_ratio:.4f}")

        # 保存图片
        save_image(rain_past_np, f"{save_dir}/sample_{sample_id}_rain_past.png")
        save_image(rain_future_np, f"{save_dir}/sample_{sample_id}_rain_future.png")

        if sample_id >= max_samples:
            break


    # -----------------------------
    # 4. Final stats summary
    # -----------------------------
    print("\n====== FINAL STATISTICS ======")
    print(f"Avg background ratio (<0.1): {np.mean(total_bg_ratio):.4f}")
    print(f"Avg small rain ratio (0.1–{strong_threshold}): {np.mean(total_small_ratio):.4f}")
    print(f"Avg strong ratio (>{strong_threshold}): {np.mean(total_strong_ratio):.4f}")
    print("Saved images to:", save_dir)



if __name__ == "__main__":
    th.cuda.set_device("cuda:0")
    # test_loading()
    test_imgs()
    # test_wids()
    # test_wids_loading()

    # analyze_patches(
    #     use_wids=True,
    #     save_dir="threshold_check_03",
    #     strong_threshold=0.3,
    #     patch_size=128,
    # )
