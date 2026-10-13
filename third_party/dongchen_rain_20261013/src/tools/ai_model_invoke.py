#!/usr/bin/env python3
from datetime import datetime
from pathlib import Path
from typing import Any

import accelerate
import h5py
import torch
from kornia.augmentation import Resize
from omegaconf import OmegaConf

from src.dataset.rain_ts_litdata import RainTimeSeriesDataset, normalize_rain_linear
from src.networks.SwinNet import SwinNet


EMA_DIR = Path(
    "/home/rainpred/RainPrediction/runs/swinnet_cls_10min_AR/"
    "2026-05-09_23-55-33_rain_train_pasts_n=5_future_n=5/ema"
)
CFG_PATH = Path(
    "/home/rainpred/RainPrediction/runs/swinnet_cls_10min_AR/"
    "2026-05-09_23-55-33_rain_train_pasts_n=5_future_n=5/config/config_total.yaml"
)

SAT_KEYS: list[str] = [f"tbb_{i:02d}" for i in range(7, 17)]
RADAR_KEY_CANDIDATES: list[str] = ["radar", "var", "qref"]
RAIN_KEY_CANDIDATES: list[str] = ["rain", "rain_interpolated", "var"]


class DatasetPreprocessor:
    def __init__(self, cfg: Any, img_size: int) -> None:
        dataset_cfg = cfg.get("dataset", {})
        train_cfg = dataset_cfg.get("train", {})

        self.clip_values = bool(train_cfg.get("clip_values", True))
        self.radar_clip_min = train_cfg.get("radar_clip_min", 0.0)
        self.radar_clip_max = train_cfg.get("radar_clip_max", 60.0)
        self.satellite_clip_min = train_cfg.get("satellite_clip_min", 0.0)
        self.satellite_clip_max = train_cfg.get("satellite_clip_max", 300.0)
        self.rain_clip_min = train_cfg.get("rain_clip_min", 0.0)
        self.rain_clip_max = train_cfg.get("rain_clip_max", None)

        self.modality_zero_centering = bool(train_cfg.get("modality_zero_centering", False))
        self.rain_norm_mean = float(train_cfg.get("rain_norm_mean", 0.0))
        self.rain_norm_std = float(train_cfg.get("rain_norm_std", 1.0))

        self.resizer = Resize((img_size, img_size), align_corners=False, keepdim=True)

    def process_frame(
        self,
        radar_raw: torch.Tensor,
        sat_raw: torch.Tensor,
        rain_raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        radar = RainTimeSeriesDataset._to_float_tensor(self, radar_raw, field_name="radar", index=0)
        sat = RainTimeSeriesDataset._to_float_tensor(self, sat_raw, field_name="satellite", index=0)
        rain = RainTimeSeriesDataset._to_float_tensor(self, rain_raw, field_name="rain_interpolated", index=0)

        radar = self.resizer(radar)
        sat = self.resizer(sat)
        rain = self.resizer(rain)

        radar = RainTimeSeriesDataset._ensure_chw(self, radar, field_name="radar", index=0)
        sat = RainTimeSeriesDataset._ensure_chw(self, sat, field_name="satellite", index=0)
        rain = RainTimeSeriesDataset._ensure_chw(self, rain, field_name="rain_interpolated", index=0)

        radar = RainTimeSeriesDataset._sanitize_and_clip(
            self,
            radar,
            min_value=self.radar_clip_min,
            max_value=self.radar_clip_max,
            fill_value=0.0,
        )
        sat = RainTimeSeriesDataset._sanitize_and_clip(
            self,
            sat,
            min_value=self.satellite_clip_min,
            max_value=self.satellite_clip_max,
            fill_value=0.0,
        )
        rain = RainTimeSeriesDataset._sanitize_and_clip(
            self,
            rain,
            min_value=self.rain_clip_min,
            max_value=self.rain_clip_max,
            fill_value=0.0,
        )

        sat = sat / 300.0
        radar = radar / 60.0

        if self.modality_zero_centering:
            sat = sat * 2.0 - 1.0
            radar = radar * 2.0 - 1.0
            rain = normalize_rain_linear(rain, mean=self.rain_norm_mean, std=self.rain_norm_std)

        return radar, sat, rain


class _Runtime:
    _instance: "_Runtime | None" = None

    def __init__(self) -> None:
        cfg = OmegaConf.load(CFG_PATH)
        model_cfg = cfg.get("rain_prediction_model", {})

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_past = int(model_cfg.get("n_past", 5))
        self.output_frames = int(model_cfg.get("output_frames", 5))
        self.num_classes = int(model_cfg.get("num_classes", 5))
        self.img_size = int(model_cfg.get("input_resolution", [256, 256])[0])

        self.preprocessor = DatasetPreprocessor(cfg=cfg, img_size=self.img_size)

        self.model = SwinNet(
            input_channel=int(model_cfg.get("input_channel", 12)),
            hidden_dim=int(model_cfg.get("hidden_dim", 64)),
            downscaling_factors=tuple(model_cfg.get("downscaling_factors", [4, 2, 1, 1])),
            layers=tuple(model_cfg.get("layers", [2, 2, 2, 2])),
            heads=tuple(model_cfg.get("heads", [4, 4, 4, 4])),
            head_dim=int(model_cfg.get("head_dim", 64)),
            window_size=int(model_cfg.get("window_size", 8)),
            input_resolution=tuple(model_cfg.get("input_resolution", [256, 256])),
            num_classes=self.num_classes,
            n_past=self.n_past,
            lstm_layers=int(model_cfg.get("lstm_layers", 2)),
            output_frames=self.output_frames,
        ).to(self.device)

        try:
            accelerate.load_checkpoint_in_model(self.model, EMA_DIR / "rain_model", strict=True)
        except TypeError:
            accelerate.load_checkpoint_in_model(self.model, EMA_DIR / "rain_model")
        self.model.eval()

    @classmethod
    def get(cls) -> "_Runtime":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


def _parse_time_from_name(path: Path) -> datetime:
    return datetime.strptime(path.stem, "%Y%m%d_%H%M")


def _list_nc_by_time(folder: str) -> dict[datetime, Path]:
    files = sorted(Path(folder).glob("*.nc"))
    return {_parse_time_from_name(file): file for file in files}


def _read_2d_first_match(path: Path, keys: list[str]) -> torch.Tensor:
    with h5py.File(path, "r") as file:
        for key in keys:
            if key in file:
                arr = torch.as_tensor(file[key][()], dtype=torch.float32)
                if arr.ndim != 2:
                    raise ValueError(f"{path}::{key} is not 2D, shape={tuple(arr.shape)}")
                return arr
    raise KeyError(f"{path} missing key from {keys}")


def _read_sat_10ch(path: Path) -> torch.Tensor:
    with h5py.File(path, "r") as file:
        channels: list[torch.Tensor] = []
        for key in SAT_KEYS:
            if key not in file:
                raise KeyError(f"{path} missing channel {key}")
            arr = torch.as_tensor(file[key][()], dtype=torch.float32)
            if arr.ndim != 2:
                raise ValueError(f"{path}::{key} is not 2D, shape={tuple(arr.shape)}")
            channels.append(arr)
    return torch.stack(channels, dim=0)


def _read_rain_aligned(path: Path, img_size: int) -> torch.Tensor:
    with h5py.File(path, "r") as file:
        rain = None
        for key in RAIN_KEY_CANDIDATES:
            if key in file:
                rain = file[key][()]
                break
        if rain is None:
            raise KeyError(f"{path} missing key from {RAIN_KEY_CANDIDATES}")

        lat_key = "latitude" if "latitude" in file else ("lat" if "lat" in file else None)
        if lat_key is not None:
            lat = file[lat_key][()]
            if getattr(lat, "ndim", 1) == 1 and len(lat) >= 2 and bool(lat[0] > lat[-1]):
                rain = rain[::-1, :].copy()

    rain_t = torch.as_tensor(rain, dtype=torch.float32)[None, None]
    rain_t = torch.nn.functional.interpolate(rain_t, size=(img_size, img_size), mode="bilinear", align_corners=False)
    return rain_t[0, 0]


def build_input_tensors(
    processed_data: dict[str, Any],
    n_past: int,
    preprocessor: DatasetPreprocessor,
    img_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    radar_map = _list_nc_by_time(processed_data.get("radar", {}).get("data", ""))
    sat_map = _list_nc_by_time(processed_data.get("satellite", {}).get("data", ""))
    rain_map = _list_nc_by_time(processed_data.get("rain", {}).get("data", ""))

    common_times = sorted(set(radar_map) & set(sat_map) & set(rain_map))
    if len(common_times) < n_past:
        raise ValueError(f"aligned timestamps not enough: {len(common_times)} < n_past={n_past}")

    times = common_times[-n_past:]
    radar_frames: list[torch.Tensor] = []
    sat_frames: list[torch.Tensor] = []
    rain_frames: list[torch.Tensor] = []

    for ts in times:
        radar_raw = _read_2d_first_match(radar_map[ts], RADAR_KEY_CANDIDATES).unsqueeze(0)
        sat_raw = _read_sat_10ch(sat_map[ts])
        rain_raw = _read_rain_aligned(rain_map[ts], img_size=img_size).unsqueeze(0)
        radar, sat, rain = preprocessor.process_frame(radar_raw=radar_raw, sat_raw=sat_raw, rain_raw=rain_raw)
        radar_frames.append(radar)
        sat_frames.append(sat)
        rain_frames.append(rain)

    radar_past = torch.stack(radar_frames, dim=1).unsqueeze(0)
    sat_past = torch.stack(sat_frames, dim=1).unsqueeze(0)
    rain_past = torch.stack(rain_frames, dim=1).unsqueeze(0)
    used_times = [t.strftime("%Y%m%d_%H%M") for t in times]
    return radar_past, sat_past, rain_past, used_times


def process_ai_predict_rainfall(
    event_id: str,
    file_time: int,
    model_type: str,
    processed_data: dict[str, Any],
) -> dict[str, Any]:
    runtime = _Runtime.get()

    radar_past, sat_past, rain_past, used_times = build_input_tensors(
        processed_data=processed_data,
        n_past=runtime.n_past,
        preprocessor=runtime.preprocessor,
        img_size=runtime.img_size,
    )

    with torch.no_grad():
        pred = runtime.model(
            radar_past.to(runtime.device),
            sat_past.to(runtime.device),
            rain_past.to(runtime.device),
        )
        pred_cls = torch.argmax(pred, dim=1)

    hist = torch.bincount(pred_cls.reshape(-1).cpu(), minlength=runtime.num_classes)

    return {
        "event_id": event_id,
        "file_time": file_time,
        "model_type": model_type,
        "success": True,
        "used_times": used_times,
        "pred_shape": list(pred.shape),
        "pred_distribution": hist.tolist(),
        "input_stats": {
            "radar_min": float(radar_past.min().item()),
            "radar_max": float(radar_past.max().item()),
            "sat_min": float(sat_past.min().item()),
            "sat_max": float(sat_past.max().item()),
            "rain_min": float(rain_past.min().item()),
            "rain_max": float(rain_past.max().item()),
        },
    }
