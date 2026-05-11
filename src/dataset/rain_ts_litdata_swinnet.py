import os
import random
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm
from litdata import StreamingDataLoader

from src.dataset.litdata_base import IndexedCombinedStreamingDataset
from src.dataset.rain_ts_litdata import RainTimeSeriesDataset

# === 1. 定义降雨分类边界 (来自 SwinNet) ===
# 0-0.01: 无雨/极小雨 (Class 0)
# 0.01-0.1: 小雨 (Class 1)
# 0.1-0.3: 中雨 (Class 2)
# 0.3-10.0: 大雨/暴雨 (Class 3)
BOUNDS = [0, 0.01, 0.1, 0.2, 0.5, 10]


class RainLitData_CLS_Filter(RainTimeSeriesDataset):
    """
    1. LitData 底层读取
    2. 经纬度定点裁剪 (LatLon Crop)
    3. min_rain/min_ratio(Filter)
    4. 降雨分类标签输出 (CLS)
    """

    def __init__(
        self,
        *args,
        center_latlon: tuple[float, float],
        patch_size: int = 128,
        img_resize: int = 256,
        max_resample_tries: int = 64,
        # 过滤参数
        is_train: bool = False,
        min_rain: float = 0.1,
        min_ratio: float = 0.03,
        cache_dir: str = "__cache__",
        validate_cache: bool = True,
        cache_validate_sample_size: int = 128,
        cache_validate_min_ok_ratio: float = 0.2,
        # 四川地理范围
        lon_min: float = 97.0,
        lon_max: float = 109.0,
        lat_min: float = 26.0,
        lat_max: float = 35.0,
        **kwargs,
    ):
        super().__init__(*args, img_resize=img_resize, **kwargs)

        self.center_latlon = center_latlon
        self.patch_size = patch_size
        self.img_size = img_resize
        self.geo_bounds = (lon_min, lon_max, lat_min, lat_max)

        self.BOUNDS = BOUNDS

        self.is_train = is_train
        self.min_rain = min_rain
        self.min_ratio = min_ratio
        self.max_resample_tries = max_resample_tries
        self.validate_cache = bool(validate_cache)
        self.cache_validate_sample_size = max(1, int(cache_validate_sample_size))
        self.cache_validate_min_ok_ratio = float(cache_validate_min_ok_ratio)

        # 预计算裁剪切片
        self.crop_slices = self._precompute_fixed_crop_slices()

        # 预转换 BOUNDS 为 Tensor
        self.bounds_tensor = torch.tensor(BOUNDS, dtype=torch.float32)

        # 执行过滤
        if self.is_train:
            self._apply_swinnet_filter(cache_dir)
        else:
            self.valid_indices = list(range(len(self.indices_pairs)))

    def _precompute_fixed_crop_slices(self):
        """计算固定裁剪区域"""
        lat, lon = self.center_latlon
        lon_min, lon_max, lat_min, lat_max = self.geo_bounds
        S = self.img_size
        P = self.patch_size

        x = (lon - lon_min) / (lon_max - lon_min) * (S - 1)
        y = (lat_max - lat) / (lat_max - lat_min) * (S - 1)

        cy, cx = int(round(y)), int(round(x))
        top = max(0, min(cy - P // 2, S - P))
        left = max(0, min(cx - P // 2, S - P))

        return (slice(top, top + P), slice(left, left + P))

    def _extract_fixed_area(self, item: dict) -> dict:
        """提取区域"""
        row_slice, col_slice = self.crop_slices
        for k, v in item.items():
            if isinstance(v, torch.Tensor) and v.ndim >= 2:
                item[k] = v[..., row_slice, col_slice]
        return item

    def _compute_cls(self, rain_tensor: torch.Tensor) -> torch.Tensor:
        """
        将连续雨量转换为分类标签
        """
        if not rain_tensor.is_contiguous():
            rain_tensor = rain_tensor.contiguous()

        # 确保 bounds 在同一个 device 上
        if self.bounds_tensor.device != rain_tensor.device:
            self.bounds_tensor = self.bounds_tensor.to(rain_tensor.device)

        # torch.bucketize 类似于 np.digitize
        # right=False: buckets are [bounds[i-1], bounds[i])
        cls_idx = torch.bucketize(rain_tensor, self.bounds_tensor, right=False) - 1

        # 限制范围在 [0, K-1]
        K = len(BOUNDS) - 1
        cls_idx = torch.clamp(cls_idx, 0, K - 1)

        return cls_idx.long()

    def _item_has_bad_value(self, item: dict) -> bool:
        for _, v in item.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                if torch.isnan(v).any() or torch.isinf(v).any():
                    return True
                if v.abs().max() > 1e5:
                    return True
        return False

    def _validate_cached_indices(self, sample_size: int | None = None, min_ok_ratio: float | None = None) -> bool:
        if sample_size is None:
            sample_size = self.cache_validate_sample_size
        if min_ok_ratio is None:
            min_ok_ratio = self.cache_validate_min_ok_ratio

        if len(self.valid_indices) == 0:
            return False

        n = min(sample_size, len(self.valid_indices))
        sampled = random.sample(self.valid_indices, n) if len(self.valid_indices) > n else list(self.valid_indices)

        ok = 0
        for real_index in sampled:
            try:
                rain_future = self._get_rain_future_only(int(real_index))
                if rain_future is None:
                    continue
                if not self._item_has_bad_value({"rain_future": rain_future}):
                    ok += 1
            except Exception:
                continue

        ok_ratio = ok / max(1, len(sampled))
        logger.info(f"[Filter-CLS] Cache validation: ok={ok}/{len(sampled)} ({ok_ratio:.2%})")
        return ok_ratio >= min_ok_ratio

    def _build_cache_name(self) -> str:
        return (
            f"cls_filter_interval{self.time_interval}_"
            f"cls_filter_lat{self.center_latlon}_"
            f"rain{self.min_rain}_ratio{self.min_ratio}"
            ".npy"
        )

    def _resolve_metadata_ratio_col(self) -> str | None:
        if not hasattr(self, "metadata") or "rain_range_max" not in self.metadata.columns:
            return None

        token = f"{float(self.min_rain):.6f}".rstrip("0").rstrip(".")
        if token == "":
            token = "0"
        token = token.replace("-", "m").replace(".", "p")
        ratio_col = f"rain_ratio_gt_{token}"
        if ratio_col in self.metadata.columns:
            return ratio_col
        return None

    def _metadata_filter_indices(self, ratio_col: str | None = None) -> list[int]:
        if not hasattr(self, "metadata") or "rain_range_max" not in self.metadata.columns:
            return []

        use_ratio = ratio_col is not None and ratio_col in self.metadata.columns
        valid: list[int] = []
        for idx, (_, ind_future) in enumerate(
            tqdm(
                self.indices_pairs,
                desc="Prefilter",
                dynamic_ncols=True,
                mininterval=1.0,
            )
        ):
            future_rows = self.metadata.iloc[list(ind_future)]
            max_vals = future_rows["rain_range_max"].to_numpy()
            max_val = float(np.nanmax(max_vals))
            if max_val <= float(self.min_rain):
                continue

            if use_ratio:
                ratio_vals = future_rows[ratio_col].to_numpy(dtype=np.float32)
                ratio_val = float(np.nanmean(ratio_vals))
                if (not np.isfinite(ratio_val)) or ratio_val < float(self.min_ratio):
                    continue

            valid.append(idx)
        return valid

    def _apply_swinnet_filter(self, cache_dir):
        """筛选逻辑"""
        cache_dir = os.path.abspath(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        cache_name = self._build_cache_name()
        cache_path = os.path.join(cache_dir, cache_name)
        logger.info(f"[Filter-CLS] cache_path={cache_path}")

        if os.path.exists(cache_path):
            self.valid_indices = np.load(cache_path).tolist()
            if len(self.valid_indices) > 0 and ((not self.validate_cache) or self._validate_cached_indices()):
                logger.info(f"[Filter-CLS] Loaded {len(self.valid_indices)} indices from cache.")
                return
            logger.warning(
                "[Filter-CLS] Cached indices are empty or mostly unreadable. Will rebuild cache by rescanning dataset."
            )

        ratio_col = self._resolve_metadata_ratio_col()
        if ratio_col is not None:
            metadata_valid_indices = self._metadata_filter_indices(ratio_col=ratio_col)
            if len(metadata_valid_indices) > 0:
                self.valid_indices = metadata_valid_indices
                if (not self.validate_cache) or self._validate_cached_indices():
                    np.save(cache_path, np.array(self.valid_indices))
                    logger.info(
                        f"[Filter-CLS] Metadata ratio filter hit column={ratio_col}, "
                        f"valid={len(self.valid_indices)}. Skip scanning."
                    )
                    return
                logger.warning(
                    "[Filter-CLS] Metadata ratio filtered indices are mostly unreadable. "
                    "Will fallback to scanning."
                )

        logger.info(f"[Filter-CLS] Scanning dataset...")
        valid_indices = []
        readable_indices = []
        candidate_indices = list(range(len(self.indices_pairs)))

        # Metadata prefilter to reduce decoding workload.
        try:
            prefiltered = self._metadata_filter_indices(ratio_col=ratio_col)
            if len(prefiltered) > 0:
                logger.info(f"[Filter-CLS] Metadata prefilter: {len(self.indices_pairs)} -> {len(prefiltered)}")
                candidate_indices = prefiltered
        except Exception as e:
            logger.warning(f"[Filter-CLS] Metadata prefilter skipped due to error: {e}")

        for idx in tqdm(
            tqdm(
                candidate_indices,
                desc="Filtering",
                dynamic_ncols=True,
                mininterval=1.0,
            )
        ):
            try:
                rain_future = self._get_rain_future_only(idx)
                if rain_future is None:
                    continue
                readable_indices.append(idx)
                rain_np = rain_future.detach().cpu().numpy()

                # 筛选逻辑
                max_val = rain_np.max()
                ratio_val = (rain_np > self.min_rain).sum() / rain_np.size

                if max_val > self.min_rain and ratio_val >= self.min_ratio:
                    valid_indices.append(idx)
            except:
                continue

        self.valid_indices = valid_indices
        if len(self.valid_indices) == 0:
            # 阈值过严时，退化为“仅保留可读窗口”，避免把坏窗口放回训练集
            if len(readable_indices) > 0:
                self.valid_indices = readable_indices
                logger.warning(
                    "[Filter-CLS] Valid samples is 0 after threshold filtering. "
                    "Fallback to readable samples only. "
                    "Please lower min_rain/min_ratio."
                )
            else:
                raise RuntimeError(
                    "[Filter-CLS] No readable samples found in dataset. "
                    "Please check data integrity/index and preprocessing outputs "
                    "(possible serialization mismatch or corrupted chunks)."
                )
        np.save(cache_path, np.array(self.valid_indices))
        logger.info(f"[Filter-CLS] Valid samples: {len(self.valid_indices)}")

    def _get_rain_future_only(self, window_index: int) -> torch.Tensor | None:
        """Fast path for filtering: decode only future rain frames."""
        _, ind_future = self.indices_pairs[window_index]
        rain_future = []
        for fidx in ind_future:
            raw = IndexedCombinedStreamingDataset.__getitem__(self, int(fidx))
            rain_raw = raw.get("rain_interpolated", None) if isinstance(raw, dict) else None
            rain = self._to_float_tensor(rain_raw, field_name="rain_interpolated", index=int(fidx))
            rain = self.resizer(rain)
            rain = self._sanitize_and_clip(
                rain,
                min_value=self.rain_clip_min,
                max_value=self.rain_clip_max,
                fill_value=0.0,
            )
            rain_future.append(rain)

        if len(rain_future) == 0:
            return None

        rain_future = torch.stack(rain_future, dim=-3)
        if rain_future.dim() == 3:
            rain_future = rain_future.unsqueeze(0)

        item = {"rain_future": rain_future}
        item = self._extract_fixed_area(item)
        return item["rain_future"]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index):
        if len(self.valid_indices) == 0:
            raise RuntimeError("Dataset has 0 valid indices after filtering.")

        last_err: Exception | None = None
        attempts = max(1, int(self.max_resample_tries))
        mapped_index = int(index) % len(self.valid_indices)

        for attempt in range(attempts):
            real_index = self.valid_indices[mapped_index]
            try:
                # 1. 尝试读取数据
                item = super().__getitem__(real_index)
                # 2. 裁剪区域
                item = self._extract_fixed_area(item)

                has_bad_value = False
                if self._item_has_bad_value(item):
                    has_bad_value = True
                    logger.warning(f"[Bad Data Skip] Index: {real_index} has NaN/Inf/huge values. Resampling...")

                if has_bad_value:
                    mapped_index = np.random.randint(0, len(self.valid_indices))
                    continue

                rain_future = item["rain_future"]
                rain_cls = self._compute_cls(rain_future)
                item["rain_future_cls"] = rain_cls
                return item

            except Exception as e:
                last_err = e
                logger.warning(
                    f"[Load Error] Index: {real_index}, attempt={attempt + 1}/{attempts}, error={e}. Resampling..."
                )
                mapped_index = np.random.randint(0, len(self.valid_indices))
                continue

        raise RuntimeError(f"Failed to load a valid sample after {attempts} attempts. Last error: {last_err}")


# === DataLoader 构造函数 ===
def get_dataloader(
    data_dirs: list[str],
    center_latlon: tuple[float, float],
    batch_size: int = 4,
    time_interval: int = 30,
    n_past: int = 2,
    n_futures: int = 1,
    img_resize: int = 256,
    patch_size: int = 256,
    min_rain: float = 0.1,
    min_ratio: float = 0.03,
    is_train: bool = True,
    num_workers: int = 4,
    shuffle: bool = False,
    prefetch_factor: int = 1,
    persistent_workers: bool = True,
    **kwargs,
):
    dataset = RainLitData_CLS_Filter(
        inp_dirs=data_dirs,
        center_latlon=center_latlon,
        time_interval=time_interval,
        n_past=n_past,
        n_futures=n_futures,
        img_resize=img_resize,
        patch_size=patch_size,
        min_rain=min_rain,
        min_ratio=min_ratio,
        is_train=is_train,
        stack_data=True,
        **kwargs,
    )

    loader = StreamingDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
        prefetch_factor=prefetch_factor if num_workers and num_workers > 0 else None,
        persistent_workers=bool(persistent_workers and num_workers and num_workers > 0),
    )

    return dataset, loader
