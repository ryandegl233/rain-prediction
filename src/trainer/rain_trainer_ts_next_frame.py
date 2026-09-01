"""
Time-series next-frame/block trainer (Accelerate) for rain prediction.

Objective:
- autoregressive teacher-forcing next-token style training
- no diffusion/noise scheduler dependencies
"""

import json
import os
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

# Ensure the project root is on sys.path so `src.*` imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import accelerate
import hydra
import torch
import torch.nn.functional as F
import torch.nn as nn
from accelerate import Accelerator
from accelerate.state import PartialState
from ema_pytorch import EMA
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw, ImageFont
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
from fvcore.nn import parameter_count_table
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.dataset.rain_ts_litdata import denormalize_rain_linear
from src.networks.losses.gan import gan_critic_total_loss, gan_generator_loss
from src.utils.visualization.plot import plot_any_modality

try:
    import colored_traceback

    colored_traceback.add_hook()
except Exception:
    colored_traceback = None

try:
    from accelerate.utils import DummyOptim, DummyScheduler
except Exception:

    class DummyOptim:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            self.param_groups = [{"lr": 0.0}]

        def step(self):
            return None

        def zero_grad(self, *args, **kwargs):
            return None

    class DummyScheduler:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            pass

        def step(self):
            return None


def apply_context_modality_dropout(
    context: torch.Tensor,
    *,
    radar_channels: int,
    satellite_channels: int,
    rain_channels: int,
    drop_prob_radar: float,
    drop_prob_satellite: float,
    drop_prob_rain: float,
    min_available_modalities: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if context.ndim != 5:
        raise ValueError(f"context must be [B,C,T,H,W], got shape={tuple(context.shape)}")
    if radar_channels <= 0 or satellite_channels <= 0 or rain_channels <= 0:
        raise ValueError(
            "radar/satellite/rain channels must be > 0, "
            f"got radar={radar_channels}, satellite={satellite_channels}, rain={rain_channels}"
        )
    total_channels = radar_channels + satellite_channels + rain_channels
    if int(context.shape[1]) != total_channels:
        raise ValueError(
            f"context channel mismatch: expected {total_channels}, got {int(context.shape[1])}. "
            "Please check modality channels."
        )

    probs = [float(drop_prob_radar), float(drop_prob_satellite), float(drop_prob_rain)]
    for prob in probs:
        if prob < 0.0 or prob > 1.0:
            raise ValueError(f"drop probability must be in [0, 1], got {prob}")
    if min_available_modalities < 1 or min_available_modalities > 3:
        raise ValueError(f"min_available_modalities must be in [1, 3], got {min_available_modalities}")

    b = int(context.shape[0])
    drop_prob = torch.tensor(probs, device=context.device, dtype=torch.float32)
    missing = torch.rand((b, 3), device=context.device) < drop_prob[None, :]
    available = ~missing

    for batch_idx in range(b):
        available_count = int(available[batch_idx].sum().item())
        if available_count >= min_available_modalities:
            continue
        need_to_recover = min_available_modalities - available_count
        missing_idx = torch.nonzero(missing[batch_idx], as_tuple=False).squeeze(1)
        if int(missing_idx.numel()) <= 0:
            continue
        chosen = missing_idx[torch.randperm(int(missing_idx.numel()), device=context.device)[:need_to_recover]]
        available[batch_idx, chosen] = True
        missing[batch_idx, chosen] = False

    r = radar_channels
    s = satellite_channels
    modality_chunks = [
        context[:, :r],
        context[:, r : r + s],
        context[:, r + s : r + s + rain_channels],
    ]
    output_chunks: list[torch.Tensor] = []
    for modality_idx, modality_tensor in enumerate(modality_chunks):
        modality_missing = missing[:, modality_idx].view(b, 1, 1, 1, 1)
        output_chunks.append(torch.where(modality_missing, torch.zeros_like(modality_tensor), modality_tensor))

    dropped_context = torch.cat(output_chunks, dim=1)
    return dropped_context, available


class RainTSNextFrameTrainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.train_cfg = cfg.train
        self.val_cfg = cfg.val
        self.dataset_cfg = cfg.dataset
        self.ema_cfg = cfg.ema

        self.accelerator: Accelerator = hydra.utils.instantiate(cfg.accelerator)
        seed = int(getattr(self.train_cfg, "seed", 2025))
        accelerate.utils.set_seed(seed)

        self.tensorboard_writer: SummaryWriter | None = None
        self.log_file = self._configure_logger()
        self.device = self.accelerator.device

        self.log_msg(f"Log file: {self.log_file}")
        self.log_msg(f"Project dir: {self.proj_dir}")

        self.train_dataset, self.train_dataloader = hydra.utils.instantiate(self.dataset_cfg.train)
        self.val_dataset, self.val_dataloader = hydra.utils.instantiate(self.dataset_cfg.val)
        self._init_rain_norm_params()

        self.model = hydra.utils.instantiate(cfg.rain_prediction_model)
        self.radar_c = int(getattr(self.model, "radar_out_channels", 1))
        self.satellite_c = int(getattr(self.model, "satellite_out_channels", 10))
        self.rain_c = int(getattr(self.model, "rain_out_channels", 1))

        init_model_path = self.train_cfg.get("init_model_path")
        if init_model_path:
            init_path = Path(str(init_model_path))
            if not init_path.exists():
                raise FileNotFoundError(f"init_model_path does not exist: {init_path}")
            accelerate.load_checkpoint_in_model(self.model, str(init_path), strict=False)
            self.log_msg(f"Loaded model initialization weights from {init_path}. Optimizer and scheduler start fresh.")

        self.gan_cfg = self.train_cfg.get("gan", {})
        self.use_gan = bool(self.gan_cfg.get("enabled", False))
        self.discriminator: nn.Module | None = None
        self.disc_optim = None
        self.disc_sched = None

        self._validate_data_model_contract()
        self.optim, self.sched = self._build_optim_sched()
        if self.use_gan:
            self.discriminator = hydra.utils.instantiate(self.gan_cfg.discriminator)
            self.disc_optim, self.disc_sched = self._build_gan_optim_sched()
        self.log_msg(f"Model parameters:\n{parameter_count_table(self.model)}")

        if self.use_gan:
            if self.discriminator is None or self.disc_optim is None or self.disc_sched is None:
                raise ValueError("GAN is enabled but discriminator/optimizer/scheduler is not initialized.")
            (
                self.model,
                self.discriminator,
                self.optim,
                self.disc_optim,
                self.train_dataloader,
                self.val_dataloader,
                self.sched,
                self.disc_sched,
            ) = self.accelerator.prepare(
                self.model,
                self.discriminator,
                self.optim,
                self.disc_optim,
                self.train_dataloader,
                self.val_dataloader,
                self.sched,
                self.disc_sched,
            )
            self.train_dataloadaer = self.train_dataloader
        else:
            self.model, self.optim, self.train_dataloadaer, self.val_dataloader, self.sched = self.accelerator.prepare(
                self.model,
                self.optim,
                self.train_dataloader,
                self.val_dataloader,
                self.sched,
            )

        self.ema_model: EMA | None = None
        if float(self.ema_cfg.beta) > 0:
            self.ema_model = EMA(
                self.accelerator.unwrap_model(self.model),
                beta=float(self.ema_cfg.beta),
                update_after_step=int(self.ema_cfg.update_after_step),
                update_every=int(self.ema_cfg.update_every),
            ).to(self.device)
            self.log_msg("EMA enabled")

        self.global_step = 0
        self.log_msg(f"Model channels: radar={self.radar_c}, satellite={self.satellite_c}, rain={self.rain_c}")
        self.log_msg("Objective: autoregressive next-token style prediction")
        if self.use_gan:
            self.log_msg("GAN training enabled for next-frame trainer.")

    def _init_rain_norm_params(self) -> None:
        train_mzc = bool(self.dataset_cfg.train.get("modality_zero_centering", False))
        val_mzc = bool(self.dataset_cfg.val.get("modality_zero_centering", False))
        if train_mzc != val_mzc:
            raise ValueError("dataset.train and dataset.val modality_zero_centering should be identical.")
        self.modality_zero_centering = train_mzc

        train_mean = self.dataset_cfg.train.get("rain_norm_mean")
        train_std = self.dataset_cfg.train.get("rain_norm_std")
        val_mean = self.dataset_cfg.val.get("rain_norm_mean")
        val_std = self.dataset_cfg.val.get("rain_norm_std")

        use_train = train_mean is not None and train_std is not None
        use_val = val_mean is not None and val_std is not None
        if use_train != use_val:
            raise ValueError(
                "dataset.train and dataset.val should both set rain_norm_mean/rain_norm_std or both unset them."
            )

        self.rain_norm_mean: float | None = None
        self.rain_norm_std: float | None = None
        if not self.modality_zero_centering:
            return

        if not use_train:
            raise ValueError(
                "modality_zero_centering=True requires rain_norm_mean and rain_norm_std in train/val dataset config."
            )

        train_mean_f = float(train_mean)
        train_std_f = float(train_std)
        val_mean_f = float(val_mean)
        val_std_f = float(val_std)
        if train_std_f <= 0 or val_std_f <= 0:
            raise ValueError(f"rain_norm_std must be > 0, got train={train_std_f}, val={val_std_f}")
        if train_mean_f != val_mean_f or train_std_f != val_std_f:
            raise ValueError(
                "dataset.train and dataset.val rain_norm_mean/rain_norm_std must be identical "
                f"for stable evaluation. got train=({train_mean_f}, {train_std_f}), val=({val_mean_f}, {val_std_f})"
            )

        self.rain_norm_mean = train_mean_f
        self.rain_norm_std = train_std_f
        self.log_msg(
            f"Modality zero-centering enabled. rain mean={self.rain_norm_mean:.12f}, std={self.rain_norm_std:.12f}"
        )

    def _validate_data_model_contract(self) -> None:
        n_past = int(self.dataset_cfg.n_past)
        n_futures = int(self.dataset_cfg.n_futures)
        if n_past <= 0:
            raise ValueError(f"dataset.n_past must be > 0, got {n_past}")
        if n_futures <= 0:
            raise ValueError(f"dataset.n_futures must be > 0, got {n_futures}")

        max_frames = int(getattr(self.model, "max_frames", n_past + n_futures))
        total_frames = n_past + n_futures
        if total_frames > max_frames:
            raise ValueError(
                f"dataset.n_past + dataset.n_futures = {total_frames} exceeds model.max_frames={max_frames}. "
                "Increase model.max_frames or reduce dataset temporal window."
            )

        train_stack = bool(getattr(self.dataset_cfg.train, "stack_data", True))
        val_stack = bool(getattr(self.dataset_cfg.val, "stack_data", True))
        if not train_stack or not val_stack:
            raise ValueError("dataset.train/val.stack_data must be true for this trainer (expects tensor batches).")

    def _resolve_project_dir(self) -> Path:
        try:
            hydra_cfg = HydraConfig.get()
        except Exception:
            hydra_cfg = None

        if hydra_cfg is not None:
            hydra_output_dir = getattr(hydra_cfg.runtime, "output_dir", None)
            if hydra_output_dir not in (None, ""):
                return Path(str(hydra_output_dir))

        configured_proj_dir = self.train_cfg.get("proj_dir")
        if configured_proj_dir not in (None, ""):
            log_root = Path(str(configured_proj_dir))
            if bool(self.train_cfg.log.get("log_with_time", True)):
                stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
                log_root = log_root / stamp
            run_comment = self.train_cfg.log.get("run_comment", "")
            if run_comment:
                log_root = Path(f"{log_root.as_posix()}_{run_comment}")
            return log_root

        raise ValueError("Unable to resolve project directory: Hydra runtime output_dir and train.proj_dir are both empty.")

    def _configure_logger(self) -> Path:
        logger.remove()
        logger.add(
            sys.stdout,
            format="{time:HH:mm:ss} - {level.icon} <level>[{level}:{file.name}:{line}]</level> - <level>{message}</level>",
            level="DEBUG",
            colorize=True,
        )

        log_root = self._resolve_project_dir()
        log_file = log_root / "log.log"

        if self.accelerator.use_distributed:
            if self.accelerator.is_main_process:
                input_lst = [log_file] * self.accelerator.num_processes
            else:
                input_lst = [None] * self.accelerator.num_processes
            output_lst = [None]
            torch.distributed.scatter_object_list(output_lst, input_lst, src=0)
            log_file = output_lst[0]
            assert isinstance(log_file, Path)

        self.proj_dir = log_file.parent
        if self.accelerator.is_main_process:
            self.proj_dir.mkdir(parents=True, exist_ok=True)
            if not self.train_cfg.debug:
                logger.add(
                    log_file,
                    format="<green>[{time:MM-DD HH:mm:ss}]</green> - <level>[{level}]</level> - <cyan>{file}:{line}</cyan> - <level>{message}</level>",
                    level="INFO",
                    rotation="20 MB",
                    enqueue=False,
                    colorize=False,
                )
            cfg_dump = self.proj_dir / "config" / "config_total.yaml"
            cfg_dump.parent.mkdir(parents=True, exist_ok=True)
            cfg_dump.write_text(OmegaConf.to_yaml(self.cfg, resolve=True))

        tensorboard_root = self.proj_dir / "tensorboard"
        self.accelerator.project_configuration.project_dir = str(self.proj_dir)
        self.accelerator.project_configuration.logging_dir = str(tensorboard_root)
        if self.accelerator.is_main_process:
            tensorboard_root.mkdir(parents=True, exist_ok=True)
            if not self.train_cfg.debug:
                self.tensorboard_writer = SummaryWriter(log_dir=str(tensorboard_root / "rain_ts_next_frame"))
        return log_file

    def log_msg(self, msg: str, level: str = "info", only_rank_zero: bool = True) -> None:
        def append_fallback_log(line: str) -> None:
            if not hasattr(self, "log_file"):
                return
            now = datetime.now().strftime("%m-%d %H:%M:%S")
            fallback_line = f"[{now}] - [{level.upper()}] - rain_trainer_ts_next_frame.py - {line}\n"
            with Path(self.log_file).open("a", encoding="utf-8") as f:
                f.write(fallback_line)

        fn = getattr(logger, level.lower())
        if only_rank_zero:
            if self.accelerator.is_main_process:
                append_fallback_log(msg)
                fn(msg)
        else:
            with self.accelerator.main_process_first():
                line = f"rank-{self.accelerator.process_index} | {msg}"
                append_fallback_log(line)
                fn(line)

    def _log_tensorboard_scalars(self, scalars: dict[str, float], step: int) -> None:
        if not self.accelerator.is_main_process or self.tensorboard_writer is None:
            return
        for tag, value in scalars.items():
            self.tensorboard_writer.add_scalar(tag, float(value), step)
        self.tensorboard_writer.flush()

    def _close_tensorboard_writer(self) -> None:
        if self.tensorboard_writer is None:
            return
        self.tensorboard_writer.flush()
        self.tensorboard_writer.close()
        self.tensorboard_writer = None

    def _build_optim_sched(self):
        ds_plugin = self.accelerator.state.deepspeed_plugin
        if ds_plugin is None or "optimizer" not in ds_plugin.deepspeed_config:
            need_named_params = "muon" in self.train_cfg.optim._target_
            opt = hydra.utils.instantiate(self.train_cfg.optim)(
                self.model.parameters() if not need_named_params else self.model.named_parameters()
            )
        else:
            opt = DummyOptim([{"params": list(self.model.parameters())}])

        if ds_plugin is None or "scheduler" not in ds_plugin.deepspeed_config:
            sched = hydra.utils.instantiate(self.train_cfg.scheduler)(optimizer=opt)
        else:
            sched = DummyScheduler(opt)
        return opt, sched

    def _build_gan_optim_sched(self):
        ds_plugin = self.accelerator.state.deepspeed_plugin
        if ds_plugin is not None:
            raise ValueError("train.gan.enabled=True currently does not support deepspeed plugin.")
        if self.discriminator is None:
            raise ValueError("Discriminator is not initialized.")

        optim_cfg = self.gan_cfg.get("optimizer")
        sched_cfg = self.gan_cfg.get("scheduler")
        if optim_cfg is None or sched_cfg is None:
            raise ValueError("train.gan.optimizer and train.gan.scheduler must be set when GAN is enabled.")

        need_named_params = "muon" in optim_cfg.get("_target_", "")
        if need_named_params:
            opt = hydra.utils.instantiate(optim_cfg)(self.discriminator.named_parameters())
        else:
            opt = hydra.utils.instantiate(optim_cfg)(self.discriminator.parameters())
        sched = hydra.utils.instantiate(sched_cfg)(optimizer=opt)
        return opt, sched

    @staticmethod
    def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
        for param in module.parameters():
            param.requires_grad_(enabled)

    def _split_modalities(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        r = self.radar_c
        s = self.satellite_c
        return {
            "radar": x[:, :r],
            "satellite": x[:, r : r + s],
            "rain": x[:, r + s :],
        }

    @staticmethod
    def _ensure_bcthw(x: torch.Tensor, expected_channels: int, name: str) -> torch.Tensor:
        if x.ndim == 4:
            x = x.unsqueeze(1)
        if x.ndim != 5:
            raise ValueError(f"{name} must be 4D/5D tensor, got shape={tuple(x.shape)}")

        if x.shape[1] == expected_channels:
            return x
        if x.shape[2] == expected_channels:
            return x.permute(0, 2, 1, 3, 4).contiguous()

        raise ValueError(f"{name} channel mismatch, expected C={expected_channels}, got shape={tuple(x.shape)}")

    def _merge_modalities(self, radar: torch.Tensor, satellite: torch.Tensor, rain: torch.Tensor) -> torch.Tensor:
        return torch.cat([radar, satellite, rain], dim=1)

    def _denormalize_rain_for_metrics(self, rain: torch.Tensor) -> torch.Tensor:
        if not getattr(self, "modality_zero_centering", False):
            return rain
        if self.rain_norm_mean is None or self.rain_norm_std is None:
            return rain
        return denormalize_rain_linear(rain, mean=self.rain_norm_mean, std=self.rain_norm_std)

    def _resolve_rain_residual_cfg(self) -> tuple[bool, float, bool, float]:
        loss_cfg = self.train_cfg.get("loss", {})
        residual_cfg = loss_cfg.get("rain_residual", {})
        enabled = bool(residual_cfg.get("enabled", False))
        delta_weight = float(residual_cfg.get("delta_weight", 1.0))
        clamp_output = bool(residual_cfg.get("clamp_output", False))
        min_value = float(residual_cfg.get("min_value", 0.0))
        if delta_weight < 0:
            raise ValueError(f"train.loss.rain_residual.delta_weight must be >= 0, got {delta_weight}")
        return enabled, delta_weight, clamp_output, min_value

    def _apply_rain_residual_output(self, pred_rain_delta: torch.Tensor, rain_ref: torch.Tensor) -> torch.Tensor:
        if pred_rain_delta.shape != rain_ref.shape:
            raise ValueError(
                "rain residual reference shape mismatch: "
                f"pred_delta={tuple(pred_rain_delta.shape)}, rain_ref={tuple(rain_ref.shape)}"
            )
        _, _, clamp_output, min_value = self._resolve_rain_residual_cfg()
        pred_rain = rain_ref.to(device=pred_rain_delta.device, dtype=pred_rain_delta.dtype) + pred_rain_delta
        if clamp_output:
            pred_rain = pred_rain.clamp_min(min_value)
        return pred_rain

    def _resolve_target_frames(self, n_future: int) -> tuple[str, int]:
        next_cfg = self.train_cfg.next_pred
        target_mode = str(next_cfg.get("target_mode", "next_frame")).lower()
        if target_mode == "next_frame":
            return target_mode, 1
        if target_mode == "block":
            configured_block = next_cfg.get("block_size", None)
            target_frames = n_future if configured_block is None else int(configured_block)
            if target_frames <= 0 or target_frames > n_future:
                raise ValueError(f"train.next_pred.block_size must be in [1, {n_future}], got {target_frames}.")
            return target_mode, target_frames
        raise ValueError(f"train.next_pred.target_mode must be 'next_frame' or 'block', got {target_mode}.")

    def _resolve_sequence_loss_weights(self) -> tuple[bool, float, float]:
        next_cfg = self.train_cfg.next_pred
        seq_cfg = next_cfg.get("sequence_loss", None)
        if seq_cfg is None:
            return False, 1.0, 1.0

        enabled = bool(seq_cfg.get("enabled", False))
        context_weight = float(seq_cfg.get("context_weight", 1.0))
        future_weight = float(seq_cfg.get("future_weight", 1.0))
        if context_weight < 0 or future_weight < 0:
            raise ValueError(
                "train.next_pred.sequence_loss context_weight/future_weight must be >= 0, "
                f"got context_weight={context_weight}, future_weight={future_weight}"
            )
        if enabled and context_weight <= 0 and future_weight <= 0:
            raise ValueError(
                "train.next_pred.sequence_loss enabled=True requires context_weight>0 or future_weight>0."
            )
        return enabled, context_weight, future_weight

    def _resolve_frame_patch_size(self) -> int:
        base_model = self.accelerator.unwrap_model(self.model)
        frame_patch_size = int(getattr(base_model, "frame_patch_size", 1))
        if frame_patch_size <= 0:
            raise ValueError(f"model.frame_patch_size must be > 0, got {frame_patch_size}")
        return frame_patch_size

    def _maybe_apply_missing_modality(
        self,
        context: torch.Tensor,
        *,
        enable: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        default_available = torch.ones((int(context.shape[0]), 3), device=context.device, dtype=torch.bool)
        if not enable:
            return context, default_available

        mm_cfg = self.train_cfg.next_pred.get("missing_modality", {})
        if not bool(mm_cfg.get("enabled", False)):
            return context, default_available

        drop_probs = mm_cfg.get("drop_probs", {})
        return apply_context_modality_dropout(
            context,
            radar_channels=self.radar_c,
            satellite_channels=self.satellite_c,
            rain_channels=self.rain_c,
            drop_prob_radar=float(drop_probs.get("radar", 0.0)),
            drop_prob_satellite=float(drop_probs.get("satellite", 0.0)),
            drop_prob_rain=float(drop_probs.get("rain", 0.0)),
            min_available_modalities=int(mm_cfg.get("min_available_modalities", 1)),
        )

    def _build_next_pred_batch(
        self,
        batch: dict[str, torch.Tensor],
        apply_missing_modality: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, float | int | str | torch.Tensor | dict[str, torch.Tensor]],
    ]:
        radar_past = self._ensure_bcthw(
            batch["radar_past"].to(self.device, dtype=torch.float32),
            expected_channels=self.radar_c,
            name="radar_past",
        )
        satellite_past = self._ensure_bcthw(
            batch["satellite_past"].to(self.device, dtype=torch.float32),
            expected_channels=self.satellite_c,
            name="satellite_past",
        )
        rain_past = self._ensure_bcthw(
            batch["rain_past"].to(self.device, dtype=torch.float32),
            expected_channels=self.rain_c,
            name="rain_past",
        )

        radar_future = self._ensure_bcthw(
            batch["radar_future"].to(self.device, dtype=torch.float32),
            expected_channels=self.radar_c,
            name="radar_future",
        )
        satellite_future = self._ensure_bcthw(
            batch["satellite_future"].to(self.device, dtype=torch.float32),
            expected_channels=self.satellite_c,
            name="satellite_future",
        )
        rain_future = self._ensure_bcthw(
            batch["rain_future"].to(self.device, dtype=torch.float32),
            expected_channels=self.rain_c,
            name="rain_future",
        )
        time_past: torch.Tensor | None = None
        time_future: torch.Tensor | None = None
        if "time_past" in batch and "time_future" in batch:
            time_past = batch["time_past"].to(self.device, dtype=torch.float32)
            time_future = batch["time_future"].to(self.device, dtype=torch.float32)
            if time_past.ndim == 1:
                time_past = time_past.unsqueeze(0)
            if time_future.ndim == 1:
                time_future = time_future.unsqueeze(0)
            if time_past.ndim != 2 or time_future.ndim != 2:
                raise ValueError(
                    "time_past/time_future must be [B,T]. "
                    f"got time_past={tuple(time_past.shape)}, time_future={tuple(time_future.shape)}"
                )
            if int(time_past.shape[0]) != int(radar_past.shape[0]) or int(time_future.shape[0]) != int(radar_future.shape[0]):
                raise ValueError(
                    "time batch mismatch with modality tensors: "
                    f"time_past_batch={int(time_past.shape[0])}, radar_past_batch={int(radar_past.shape[0])}, "
                    f"time_future_batch={int(time_future.shape[0])}, radar_future_batch={int(radar_future.shape[0])}"
                )
            if int(time_past.shape[1]) != int(radar_past.shape[2]) or int(time_future.shape[1]) != int(radar_future.shape[2]):
                raise ValueError(
                    "time length mismatch with frames: "
                    f"time_past={int(time_past.shape[1])}, past_frames={int(radar_past.shape[2])}, "
                    f"time_future={int(time_future.shape[1])}, future_frames={int(radar_future.shape[2])}"
                )

        n_future = int(radar_future.shape[2])
        if n_future <= 0:
            raise ValueError("n_future must be > 0.")

        target_mode, target_frames = self._resolve_target_frames(n_future=n_future)
        sequence_loss_enabled, context_loss_weight, future_loss_weight = self._resolve_sequence_loss_weights()

        if sequence_loss_enabled and bool(self.train_cfg.strict_target_isolation):
            raise ValueError(
                "train.next_pred.sequence_loss.enabled=True requires train.strict_target_isolation=false. "
                "Sequence right-shift supervision needs target-target causal links."
            )

        context_radar = radar_past
        context_satellite = satellite_past
        context_rain = rain_past
        context = self._merge_modalities(context_radar, context_satellite, context_rain)

        if context.shape[2] <= 0:
            raise ValueError("context frames must be > 0 for next-token style training.")

        if sequence_loss_enabled:
            full_radar = torch.cat([radar_past, radar_future], dim=2)
            full_satellite = torch.cat([satellite_past, satellite_future], dim=2)
            full_rain = torch.cat([rain_past, rain_future], dim=2)
            full_tensor = self._merge_modalities(full_radar, full_satellite, full_rain)
            frame_patch_size = self._resolve_frame_patch_size()
            total_frames = int(full_tensor.shape[2])
            if total_frames % frame_patch_size != 0:
                raise ValueError(
                    "train.next_pred.sequence_loss enabled=True requires (n_past + n_futures) divisible by "
                    f"frame_patch_size({frame_patch_size}), got total_frames={total_frames}."
                )
            if total_frames <= frame_patch_size:
                raise ValueError(
                    "sequence right-shift supervision requires n_past + n_futures > frame_patch_size, "
                    f"got total_frames={total_frames}, frame_patch_size={frame_patch_size}"
                )

            sequence_context = full_tensor[:, :, :frame_patch_size]
            target_seed_tensor = full_tensor[:, :, :-frame_patch_size]
            target_tensor = full_tensor[:, :, frame_patch_size:]
            target_gt = self._split_modalities(target_tensor)
            diff_anchor = self._split_modalities(full_tensor[:, :, frame_patch_size - 1 : frame_patch_size])
            sequence_context_time: torch.Tensor | None = None
            target_seed_time: torch.Tensor | None = None
            target_gt_time: torch.Tensor | None = None
            if time_past is not None and time_future is not None:
                full_time = torch.cat([time_past, time_future], dim=1)
                sequence_context_time = full_time[:, :frame_patch_size]
                target_seed_time = full_time[:, :-frame_patch_size]
                target_gt_time = full_time[:, frame_patch_size:]
            sequence_context, context_modality_available = self._maybe_apply_missing_modality(
                sequence_context,
                enable=apply_missing_modality,
            )
            context_target_frames = max(int(context.shape[2] - frame_patch_size), 0)
            future_target_frames = int(target_tensor.shape[2]) - context_target_frames
            aux = {
                "target_mode": target_mode,
                "target_frames": int(target_seed_tensor.shape[2]),
                "context_frames": int(sequence_context.shape[2]),
                "sequence_loss_enabled": 1,
                "sequence_context_frames": context_target_frames,
                "sequence_future_frames": future_target_frames,
                "sequence_context_weight": context_loss_weight,
                "sequence_future_weight": future_loss_weight,
                "context_modality_available": context_modality_available,
                "context_time": sequence_context_time,
                "target_seed_time": target_seed_time,
                "target_gt_time": target_gt_time,
                "temporal_diff_anchor": diff_anchor,
                "rain_residual_ref": self._split_modalities(target_seed_tensor)["rain"],
            }
            return sequence_context, target_seed_tensor, target_gt, aux

        target_gt = {
            "radar": radar_future[:, :, :target_frames],
            "satellite": satellite_future[:, :, :target_frames],
            "rain": rain_future[:, :, :target_frames],
        }
        target_gt_time: torch.Tensor | None = None
        if time_future is not None:
            target_gt_time = time_future[:, :target_frames]

        anchor = {
            "radar": context_radar[:, :, -1:],
            "satellite": context_satellite[:, :, -1:],
            "rain": context_rain[:, :, -1:],
        }
        if target_frames == 1:
            target_seed = anchor
            target_seed_time = time_past[:, -1:] if time_past is not None else None
        else:
            target_seed = {
                "radar": torch.cat([anchor["radar"], target_gt["radar"][:, :, :-1]], dim=2),
                "satellite": torch.cat([anchor["satellite"], target_gt["satellite"][:, :, :-1]], dim=2),
                "rain": torch.cat([anchor["rain"], target_gt["rain"][:, :, :-1]], dim=2),
            }
            if time_past is not None and target_gt_time is not None:
                target_seed_time = torch.cat([time_past[:, -1:], target_gt_time[:, :-1]], dim=1)
            else:
                target_seed_time = None

        target_seed_tensor = self._merge_modalities(
            target_seed["radar"],
            target_seed["satellite"],
            target_seed["rain"],
        )
        context, context_modality_available = self._maybe_apply_missing_modality(
            context,
            enable=apply_missing_modality,
        )
        aux = {
            "target_mode": target_mode,
            "target_frames": target_frames,
            "context_frames": int(context.shape[2]),
            "sequence_loss_enabled": 0,
            "sequence_context_frames": 0,
            "sequence_future_frames": int(target_frames),
            "sequence_context_weight": context_loss_weight,
            "sequence_future_weight": future_loss_weight,
            "context_modality_available": context_modality_available,
            "context_time": time_past,
            "target_seed_time": target_seed_time,
            "target_gt_time": target_gt_time,
            "temporal_diff_anchor": anchor,
            "rain_residual_ref": target_seed["rain"],
        }
        return context, target_seed_tensor, target_gt, aux

    def _next_prediction_loss(
        self,
        pred: dict[str, torch.Tensor],
        target_gt: dict[str, torch.Tensor],
        aux: dict[str, float | int | str | torch.Tensor | dict[str, torch.Tensor]] | None = None,
        loss_weight_override: dict[str, float] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        def resolve_temporal_diff_weight() -> tuple[bool, float, str]:
            loss_cfg = self.train_cfg.get("loss", {})
            diff_cfg = loss_cfg.get("temporal_diff", {})
            enabled = bool(diff_cfg.get("enabled", False))
            if not enabled:
                return False, 0.0, "rain"

            apply_on = str(diff_cfg.get("apply_on", "rain")).lower()
            if apply_on not in {"rain", "all"}:
                raise ValueError(f"train.loss.temporal_diff.apply_on must be 'rain' or 'all', got {apply_on}")

            weight_init = float(diff_cfg.get("weight_init", 0.0))
            weight_max = float(diff_cfg.get("weight_max", weight_init))
            warmup_steps = int(diff_cfg.get("warmup_steps", 0))
            if weight_init < 0:
                raise ValueError(f"train.loss.temporal_diff.weight_init must be >= 0, got {weight_init}")
            if weight_max < 0:
                raise ValueError(f"train.loss.temporal_diff.weight_max must be >= 0, got {weight_max}")
            if warmup_steps < 0:
                raise ValueError(f"train.loss.temporal_diff.warmup_steps must be >= 0, got {warmup_steps}")

            if warmup_steps <= 0:
                return True, weight_max, apply_on

            step = int(getattr(self, "global_step", 0))
            alpha = min(max(float(step) / float(warmup_steps), 0.0), 1.0)
            weight = weight_init + (weight_max - weight_init) * alpha
            return True, weight, apply_on

        def temporal_diff_mse_map(
            pred_tensor: torch.Tensor,
            target_tensor: torch.Tensor,
            anchor_tensor: torch.Tensor | None = None,
            loss_weight: torch.Tensor | None = None,
        ) -> torch.Tensor | None:
            if int(target_tensor.shape[2]) <= 0:
                return None

            if anchor_tensor is not None:
                if int(anchor_tensor.shape[2]) != 1:
                    raise ValueError(
                        "temporal diff anchor must contain exactly 1 frame, "
                        f"got anchor_shape={tuple(anchor_tensor.shape)}"
                    )
                pred_seq = torch.cat([anchor_tensor, pred_tensor], dim=2)
                target_seq = torch.cat([anchor_tensor, target_tensor], dim=2)
            else:
                pred_seq = pred_tensor
                target_seq = target_tensor

            if int(pred_seq.shape[2]) <= 1:
                return None

            pred_diff = pred_seq[:, :, 1:] - pred_seq[:, :, :-1]
            target_diff = target_seq[:, :, 1:] - target_seq[:, :, :-1]
            diff_map = F.mse_loss(pred_diff, target_diff, reduction="none")
            if loss_weight is not None:
                diff_map = diff_map * loss_weight.to(device=diff_map.device, dtype=diff_map.dtype)
            return diff_map

        def get_temporal_diff_anchor() -> dict[str, torch.Tensor] | None:
            if aux is None:
                return None

            anchor = aux.get("temporal_diff_anchor")
            if not isinstance(anchor, dict):
                return None

            if not all(torch.is_tensor(anchor.get(name)) for name in ("radar", "satellite", "rain")):
                return None
            return anchor

        def build_radar_guided_diff_weight(anchor: dict[str, torch.Tensor] | None) -> torch.Tensor | None:
            loss_cfg = self.train_cfg.get("loss", {})
            diff_cfg = loss_cfg.get("temporal_diff", {})
            guided_cfg = diff_cfg.get("radar_guided", {})
            if not bool(guided_cfg.get("enabled", False)) or anchor is None:
                return None

            alpha = float(guided_cfg.get("alpha", 0.0))
            echo_beta = float(guided_cfg.get("echo_beta", 0.0))
            eps = float(guided_cfg.get("eps", 1.0e-6))
            max_weight = float(guided_cfg.get("max_weight", 0.0))
            if alpha < 0:
                raise ValueError(f"train.loss.temporal_diff.radar_guided.alpha must be >= 0, got {alpha}")
            if echo_beta < 0:
                raise ValueError(
                    f"train.loss.temporal_diff.radar_guided.echo_beta must be >= 0, got {echo_beta}"
                )
            if eps <= 0:
                raise ValueError(f"train.loss.temporal_diff.radar_guided.eps must be > 0, got {eps}")
            if max_weight < 0:
                raise ValueError(
                    f"train.loss.temporal_diff.radar_guided.max_weight must be >= 0, got {max_weight}"
                )
            if alpha == 0 and echo_beta == 0:
                return None

            radar_seq = torch.cat([anchor["radar"], target_gt["radar"]], dim=2).detach()
            radar_delta = radar_seq[:, :, 1:] - radar_seq[:, :, :-1]
            radar_change = radar_delta.abs()
            change_scale = radar_change.mean(dim=(1, 2, 3, 4), keepdim=True).clamp_min(eps)
            change_term = radar_change / change_scale

            radar_echo = torch.maximum(radar_seq[:, :, 1:], radar_seq[:, :, :-1]).clamp_min(0.0)
            echo_scale = radar_echo.mean(dim=(1, 2, 3, 4), keepdim=True).clamp_min(eps)
            echo_term = radar_echo / echo_scale

            weight = 1.0 + alpha * change_term + echo_beta * echo_term
            if max_weight > 0:
                weight = weight.clamp(max=max_weight)
            weight_mean = weight.mean(dim=(1, 2, 3, 4), keepdim=True).clamp_min(eps)
            return weight / weight_mean

        def compute_weighted_segment_loss(
            loss_chunk: torch.Tensor,
            context_frames: int,
            future_frames: int,
            context_weight: float,
            future_weight: float,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            zero = torch.zeros((), device=loss_chunk.device, dtype=loss_chunk.dtype)
            weighted_sum = zero
            weight_sum = 0.0

            if context_frames > 0:
                context_loss = loss_chunk[:, :, :context_frames].mean()
                if context_weight > 0:
                    weighted_sum = weighted_sum + context_weight * context_loss
                    weight_sum += context_weight
            else:
                context_loss = zero

            if future_frames > 0:
                future_loss = loss_chunk[:, :, context_frames : context_frames + future_frames].mean()
                if future_weight > 0:
                    weighted_sum = weighted_sum + future_weight * future_loss
                    weight_sum += future_weight
            else:
                future_loss = zero

            if weight_sum <= 0:
                raise ValueError("At least one of sequence context/future weights must be > 0.")
            return weighted_sum / weight_sum, context_loss, future_loss

        def build_rain_regression_loss_map(
            pred_rain: torch.Tensor,
            target_rain: torch.Tensor,
        ) -> tuple[torch.Tensor, bool]:
            sq_error = F.mse_loss(pred_rain, target_rain, reduction="none")
            loss_cfg = self.train_cfg.get("loss", {})
            mode = str(loss_cfg.get("mode", "mse")).lower()
            if mode == "mse":
                return sq_error, False
            if mode != "enhanced":
                raise ValueError(f"train.loss.mode must be 'mse' or 'enhanced', got {mode}")

            rain_cfg = loss_cfg.get("rain_region_weight", {})
            if not bool(rain_cfg.get("enabled", False)):
                return sq_error, False

            alpha = float(rain_cfg.get("alpha", 0.0))
            r0 = float(rain_cfg.get("r0", 0.0))
            gamma = float(rain_cfg.get("gamma", 1.0))
            if alpha < 0:
                raise ValueError(f"train.loss.rain_region_weight.alpha must be >= 0, got {alpha}")
            if gamma <= 0:
                raise ValueError(f"train.loss.rain_region_weight.gamma must be > 0, got {gamma}")

            weight = 1.0 + alpha * torch.clamp(target_rain - r0, min=0.0).pow(gamma)
            return sq_error * weight, True

        def build_rain_event_loss_map(
            pred_rain: torch.Tensor,
            target_rain: torch.Tensor,
        ) -> tuple[torch.Tensor | None, float]:
            loss_cfg = self.train_cfg.get("loss", {})
            mode = str(loss_cfg.get("mode", "mse")).lower()
            if mode == "mse":
                return None, 0.0
            if mode != "enhanced":
                raise ValueError(f"train.loss.mode must be 'mse' or 'enhanced', got {mode}")

            event_cfg = loss_cfg.get("rain_event_aux", {})
            if not bool(event_cfg.get("enabled", False)):
                return None, 0.0

            event_weight = float(event_cfg.get("weight", 1.0))
            if event_weight < 0:
                raise ValueError(f"train.loss.rain_event_aux.weight must be >= 0, got {event_weight}")
            if event_weight == 0:
                return None, 0.0

            thresholds_cfg = event_cfg.get("thresholds", [0.1, 0.3, 0.5])
            if isinstance(thresholds_cfg, (int, float)):
                thresholds = [float(thresholds_cfg)]
            else:
                thresholds = [float(v) for v in thresholds_cfg]
            if len(thresholds) == 0:
                raise ValueError("train.loss.rain_event_aux.thresholds must contain at least 1 value.")

            event_type = str(event_cfg.get("type", "bce")).lower()
            if event_type not in {"bce", "focal"}:
                raise ValueError(f"train.loss.rain_event_aux.type must be 'bce' or 'focal', got {event_type}")

            logit_scale = float(event_cfg.get("logit_scale", 10.0))
            if logit_scale <= 0:
                raise ValueError(f"train.loss.rain_event_aux.logit_scale must be > 0, got {logit_scale}")

            focal_gamma = float(event_cfg.get("focal_gamma", 2.0))
            focal_alpha = float(event_cfg.get("focal_alpha", 0.25))
            if focal_gamma < 0:
                raise ValueError(f"train.loss.rain_event_aux.focal_gamma must be >= 0, got {focal_gamma}")
            if focal_alpha < 0 or focal_alpha > 1:
                raise ValueError(f"train.loss.rain_event_aux.focal_alpha must be in [0, 1], got {focal_alpha}")

            per_threshold_maps: list[torch.Tensor] = []
            for threshold in thresholds:
                threshold_tensor = torch.tensor(threshold, device=pred_rain.device, dtype=pred_rain.dtype)
                logits = (pred_rain - threshold_tensor) * logit_scale
                labels = (target_rain >= threshold_tensor).to(dtype=pred_rain.dtype)
                bce_map = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
                if event_type == "bce":
                    per_threshold_maps.append(bce_map)
                    continue

                prob = torch.sigmoid(logits)
                pt = prob * labels + (1.0 - prob) * (1.0 - labels)
                alpha_factor = focal_alpha * labels + (1.0 - focal_alpha) * (1.0 - labels)
                focal_map = alpha_factor * (1.0 - pt).pow(focal_gamma) * bce_map
                per_threshold_maps.append(focal_map)

            event_map = torch.stack(per_threshold_maps, dim=0).mean(dim=0)
            return event_map, event_weight

        rain_residual_enabled, rain_residual_delta_weight, _, _ = self._resolve_rain_residual_cfg()
        rain_residual_already_applied = bool(aux.get("rain_residual_already_applied", 0)) if aux is not None else False
        rain_residual_ref = aux.get("rain_residual_ref") if aux is not None else None
        pred_for_loss = dict(pred)
        rain_delta_loss_map: torch.Tensor | None = None
        if rain_residual_enabled and not rain_residual_already_applied:
            if not torch.is_tensor(rain_residual_ref):
                raise ValueError("train.loss.rain_residual.enabled=True requires aux['rain_residual_ref'].")
            rain_ref = rain_residual_ref.to(device=pred["rain"].device, dtype=pred["rain"].dtype)
            pred_for_loss["rain"] = self._apply_rain_residual_output(pred["rain"], rain_ref)
            target_delta = target_gt["rain"] - rain_ref
            rain_delta_loss_map = F.mse_loss(pred["rain"], target_delta, reduction="none")

        pred_tensor = self._merge_modalities(pred_for_loss["radar"], pred_for_loss["satellite"], pred_for_loss["rain"])
        target_tensor = self._merge_modalities(target_gt["radar"], target_gt["satellite"], target_gt["rain"])

        loss_map = F.mse_loss(pred_tensor, target_tensor, reduction="none")
        lw_cfg = self.train_cfg.loss_weights
        lw_radar = float(lw_cfg.radar) if loss_weight_override is None else float(loss_weight_override.get("radar", lw_cfg.radar))
        lw_satellite = float(lw_cfg.satellite) if loss_weight_override is None else float(loss_weight_override.get("satellite", lw_cfg.satellite))
        lw_rain = float(lw_cfg.rain) if loss_weight_override is None else float(loss_weight_override.get("rain", lw_cfg.rain))
        temporal_diff_enabled, temporal_diff_weight, temporal_diff_apply_on = resolve_temporal_diff_weight()
        temporal_diff_anchor = get_temporal_diff_anchor()
        radar_diff_anchor = temporal_diff_anchor["radar"] if temporal_diff_anchor is not None else None
        satellite_diff_anchor = temporal_diff_anchor["satellite"] if temporal_diff_anchor is not None else None
        rain_diff_anchor = temporal_diff_anchor["rain"] if temporal_diff_anchor is not None else None
        rain_diff_weight_map = (
            build_radar_guided_diff_weight(temporal_diff_anchor) if temporal_diff_enabled else None
        )

        sequence_loss_enabled = bool(aux.get("sequence_loss_enabled", 0)) if aux is not None else False
        if not sequence_loss_enabled:
            l_radar = loss_map[:, : self.radar_c].mean()
            l_satellite = loss_map[:, self.radar_c : self.radar_c + self.satellite_c].mean()
            rain_loss_map, rain_weighted_enabled = build_rain_regression_loss_map(
                pred_for_loss["rain"], target_gt["rain"]
            )
            l_rain = rain_loss_map.mean()
            event_map, event_weight = build_rain_event_loss_map(pred_for_loss["rain"], target_gt["rain"])
            if event_map is None:
                event_loss = torch.zeros((), device=loss_map.device, dtype=loss_map.dtype)
            else:
                event_loss = event_weight * event_map.mean()
            rain_delta_loss = (
                torch.zeros((), device=loss_map.device, dtype=loss_map.dtype)
                if rain_delta_loss_map is None
                else rain_delta_loss_map.mean()
            )
            rain_delta_loss_weighted = rain_residual_delta_weight * rain_delta_loss
            loss = (
                lw_radar * l_radar
                + lw_satellite * l_satellite
                + lw_rain * l_rain
                + event_loss
                + rain_delta_loss_weighted
            )

            zero = torch.zeros((), device=loss_map.device, dtype=loss_map.dtype)
            diff_radar = zero
            diff_satellite = zero
            diff_rain = zero
            diff_raw = zero
            diff_weighted = zero
            if temporal_diff_enabled and temporal_diff_weight > 0:
                if temporal_diff_apply_on == "all":
                    radar_diff_map = temporal_diff_mse_map(pred_for_loss["radar"], target_gt["radar"], radar_diff_anchor)
                    satellite_diff_map = temporal_diff_mse_map(
                        pred_for_loss["satellite"], target_gt["satellite"], satellite_diff_anchor
                    )
                    rain_diff_map = temporal_diff_mse_map(
                        pred_for_loss["rain"], target_gt["rain"], rain_diff_anchor, rain_diff_weight_map
                    )
                    if radar_diff_map is not None:
                        diff_radar = radar_diff_map.mean()
                    if satellite_diff_map is not None:
                        diff_satellite = satellite_diff_map.mean()
                    if rain_diff_map is not None:
                        diff_rain = rain_diff_map.mean()
                    diff_raw = (
                        lw_radar * diff_radar
                        + lw_satellite * diff_satellite
                        + lw_rain * diff_rain
                    )
                else:
                    rain_diff_map = temporal_diff_mse_map(
                        pred_for_loss["rain"], target_gt["rain"], rain_diff_anchor, rain_diff_weight_map
                    )
                    if rain_diff_map is not None:
                        diff_rain = rain_diff_map.mean()
                        diff_raw = diff_rain

                diff_weighted = temporal_diff_weight * diff_raw
                loss = loss + diff_weighted

            logs = {
                "loss": loss.detach(),
                "loss/radar": l_radar.detach(),
                "loss/satellite": l_satellite.detach(),
                "loss/rain": l_rain.detach(),
            }
            if temporal_diff_enabled:
                logs["loss/rain_diff"] = diff_rain.detach()
                if temporal_diff_apply_on == "all":
                    logs["loss/radar_diff"] = diff_radar.detach()
                    logs["loss/satellite_diff"] = diff_satellite.detach()
                logs["loss/temporal_diff_raw"] = diff_raw.detach()
                logs["loss/temporal_diff"] = diff_weighted.detach()
                logs["meta/temporal_diff_weight"] = torch.tensor(
                    temporal_diff_weight, device=loss_map.device, dtype=loss_map.dtype
                )
                if rain_diff_weight_map is not None:
                    logs["meta/radar_guided_diff_weight_mean"] = rain_diff_weight_map.mean().detach()
                    logs["meta/radar_guided_diff_weight_max"] = rain_diff_weight_map.max().detach()
            if rain_weighted_enabled:
                logs["loss/rain_weighted_reg"] = l_rain.detach()
            if event_map is not None:
                logs["loss/rain_event"] = event_loss.detach()
            if rain_residual_enabled:
                logs["loss/rain_residual_delta"] = rain_delta_loss.detach()
                logs["loss/rain_residual_delta_weighted"] = rain_delta_loss_weighted.detach()
                logs["meta/rain_residual_enabled"] = torch.tensor(1.0, device=loss_map.device, dtype=loss_map.dtype)
            return loss, logs

        context_frames = int(aux.get("sequence_context_frames", 0))
        future_frames = int(aux.get("sequence_future_frames", 0))
        context_weight = float(aux.get("sequence_context_weight", 1.0))
        future_weight = float(aux.get("sequence_future_weight", 1.0))
        total_frames = int(loss_map.shape[2])
        if context_frames + future_frames != total_frames:
            raise ValueError(
                "sequence loss frame partition mismatch: "
                f"context_frames({context_frames}) + future_frames({future_frames}) != total_frames({total_frames})"
            )

        radar_map = loss_map[:, : self.radar_c]
        satellite_map = loss_map[:, self.radar_c : self.radar_c + self.satellite_c]
        rain_map, rain_weighted_enabled = build_rain_regression_loss_map(pred_for_loss["rain"], target_gt["rain"])
        rain_event_map, event_weight = build_rain_event_loss_map(pred_for_loss["rain"], target_gt["rain"])

        l_radar, l_radar_context, l_radar_future = compute_weighted_segment_loss(
            radar_map, context_frames, future_frames, context_weight, future_weight
        )
        l_satellite, l_satellite_context, l_satellite_future = compute_weighted_segment_loss(
            satellite_map, context_frames, future_frames, context_weight, future_weight
        )
        l_rain, l_rain_context, l_rain_future = compute_weighted_segment_loss(
            rain_map, context_frames, future_frames, context_weight, future_weight
        )
        if rain_event_map is None:
            rain_event = torch.zeros((), device=loss_map.device, dtype=loss_map.dtype)
            rain_event_context = torch.zeros((), device=loss_map.device, dtype=loss_map.dtype)
            rain_event_future = torch.zeros((), device=loss_map.device, dtype=loss_map.dtype)
        else:
            rain_event, rain_event_context, rain_event_future = compute_weighted_segment_loss(
                rain_event_map, context_frames, future_frames, context_weight, future_weight
            )
            rain_event = event_weight * rain_event
            rain_event_context = event_weight * rain_event_context
            rain_event_future = event_weight * rain_event_future
        if rain_delta_loss_map is None:
            rain_delta_loss = torch.zeros((), device=loss_map.device, dtype=loss_map.dtype)
            rain_delta_loss_context = torch.zeros((), device=loss_map.device, dtype=loss_map.dtype)
            rain_delta_loss_future = torch.zeros((), device=loss_map.device, dtype=loss_map.dtype)
        else:
            rain_delta_loss, rain_delta_loss_context, rain_delta_loss_future = compute_weighted_segment_loss(
                rain_delta_loss_map, context_frames, future_frames, context_weight, future_weight
            )
        rain_delta_loss_weighted = rain_residual_delta_weight * rain_delta_loss
        loss = lw_radar * l_radar + lw_satellite * l_satellite + lw_rain * l_rain + rain_event + rain_delta_loss_weighted

        zero = torch.zeros((), device=loss_map.device, dtype=loss_map.dtype)
        diff_radar = zero
        diff_satellite = zero
        diff_rain = zero
        diff_raw = zero
        diff_weighted = zero
        if temporal_diff_enabled and temporal_diff_weight > 0:
            if temporal_diff_anchor is None:
                diff_context_frames = max(context_frames - 1, 0)
                diff_future_frames = max(future_frames if context_frames > 0 else future_frames - 1, 0)
            else:
                diff_context_frames = max(context_frames, 0)
                diff_future_frames = max(future_frames, 0)

            if temporal_diff_apply_on == "all":
                radar_diff_map = temporal_diff_mse_map(pred_for_loss["radar"], target_gt["radar"], radar_diff_anchor)
                satellite_diff_map = temporal_diff_mse_map(
                    pred_for_loss["satellite"], target_gt["satellite"], satellite_diff_anchor
                )
                rain_diff_map = temporal_diff_mse_map(
                    pred_for_loss["rain"], target_gt["rain"], rain_diff_anchor, rain_diff_weight_map
                )
                if radar_diff_map is not None and (diff_context_frames + diff_future_frames) > 0:
                    diff_radar, _, _ = compute_weighted_segment_loss(
                        radar_diff_map, diff_context_frames, diff_future_frames, context_weight, future_weight
                    )
                if satellite_diff_map is not None and (diff_context_frames + diff_future_frames) > 0:
                    diff_satellite, _, _ = compute_weighted_segment_loss(
                        satellite_diff_map, diff_context_frames, diff_future_frames, context_weight, future_weight
                    )
                if rain_diff_map is not None and (diff_context_frames + diff_future_frames) > 0:
                    diff_rain, _, _ = compute_weighted_segment_loss(
                        rain_diff_map, diff_context_frames, diff_future_frames, context_weight, future_weight
                    )
                diff_raw = (
                    lw_radar * diff_radar
                    + lw_satellite * diff_satellite
                    + lw_rain * diff_rain
                )
            else:
                rain_diff_map = temporal_diff_mse_map(
                    pred_for_loss["rain"], target_gt["rain"], rain_diff_anchor, rain_diff_weight_map
                )
                if rain_diff_map is not None and (diff_context_frames + diff_future_frames) > 0:
                    diff_rain, _, _ = compute_weighted_segment_loss(
                        rain_diff_map, diff_context_frames, diff_future_frames, context_weight, future_weight
                    )
                    diff_raw = diff_rain
            diff_weighted = temporal_diff_weight * diff_raw
            loss = loss + diff_weighted

        logs = {
            "loss": loss.detach(),
            "loss/radar": l_radar.detach(),
            "loss/satellite": l_satellite.detach(),
            "loss/rain": l_rain.detach(),
            "loss/radar_context": l_radar_context.detach(),
            "loss/radar_future": l_radar_future.detach(),
            "loss/satellite_context": l_satellite_context.detach(),
            "loss/satellite_future": l_satellite_future.detach(),
            "loss/rain_context": l_rain_context.detach(),
            "loss/rain_future": l_rain_future.detach(),
        }
        if temporal_diff_enabled:
            logs["loss/rain_diff"] = diff_rain.detach()
            if temporal_diff_apply_on == "all":
                logs["loss/radar_diff"] = diff_radar.detach()
                logs["loss/satellite_diff"] = diff_satellite.detach()
            logs["loss/temporal_diff_raw"] = diff_raw.detach()
            logs["loss/temporal_diff"] = diff_weighted.detach()
            logs["meta/temporal_diff_weight"] = torch.tensor(
                temporal_diff_weight, device=loss_map.device, dtype=loss_map.dtype
            )
            if rain_diff_weight_map is not None:
                logs["meta/radar_guided_diff_weight_mean"] = rain_diff_weight_map.mean().detach()
                logs["meta/radar_guided_diff_weight_max"] = rain_diff_weight_map.max().detach()
        if rain_weighted_enabled:
            logs["loss/rain_weighted_reg"] = l_rain.detach()
        if rain_event_map is not None:
            logs["loss/rain_event"] = rain_event.detach()
            logs["loss/rain_event_context"] = rain_event_context.detach()
            logs["loss/rain_event_future"] = rain_event_future.detach()
        if rain_residual_enabled:
            logs["loss/rain_residual_delta"] = rain_delta_loss.detach()
            logs["loss/rain_residual_delta_context"] = rain_delta_loss_context.detach()
            logs["loss/rain_residual_delta_future"] = rain_delta_loss_future.detach()
            logs["loss/rain_residual_delta_weighted"] = rain_delta_loss_weighted.detach()
            logs["meta/rain_residual_enabled"] = torch.tensor(1.0, device=loss_map.device, dtype=loss_map.dtype)
        return loss, logs

    def train_step(self, batch: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], bool]:
        context, target_seed, target_gt, aux = self._build_next_pred_batch(batch, apply_missing_modality=True)
        target_frames = int(aux["target_frames"])
        context_modality_available = aux.get("context_modality_available")
        context_time = aux.get("context_time")
        target_seed_time = aux.get("target_seed_time")

        with self.accelerator.accumulate(self.model):
            with self.accelerator.autocast():
                forward_kwargs = {
                    "context_x": context,
                    "target_x": target_seed,
                    "predict_frames": target_frames,
                    "strict_target_isolation": bool(self.train_cfg.strict_target_isolation),
                    "return_modality_dict": True,
                    "context_modality_available": context_modality_available,
                }
                if torch.is_tensor(context_time):
                    forward_kwargs["context_time"] = context_time
                if torch.is_tensor(target_seed_time):
                    forward_kwargs["target_time"] = target_seed_time
                pred = self.model.forward_ar(
                    **forward_kwargs,
                )
                rec_loss, logs = self._next_prediction_loss(pred=pred, target_gt=target_gt, aux=aux)
                rollout_branch_loss, rollout_logs = self._train_rollout_branch_loss(batch)

            total_g_loss = rec_loss
            if rollout_branch_loss is not None:
                total_g_loss = total_g_loss + rollout_branch_loss
                logs.update(rollout_logs)
            if self.use_gan:
                if self.discriminator is None or self.disc_optim is None or self.disc_sched is None:
                    raise ValueError("GAN is enabled but discriminator states are not initialized.")

                gan_loss_cfg = self.gan_cfg.get("loss", {})
                gan_loss_type = str(gan_loss_cfg.get("type", "ns"))
                d_weight = float(gan_loss_cfg.get("d_weight", 1.0))
                g_weight = float(gan_loss_cfg.get("g_weight", 1.0))
                r1_weight = float(gan_loss_cfg.get("r1_weight", 0.0))
                r2_weight = float(gan_loss_cfg.get("r2_weight", 0.0))

                target_tensor = self._merge_modalities(target_gt["radar"], target_gt["satellite"], target_gt["rain"])
                pred_tensor = self._merge_modalities(pred["radar"], pred["satellite"], pred["rain"])

                self._set_requires_grad(self.discriminator, True)
                disc_condition = target_seed.detach()
                real_disc_target = target_tensor.detach()
                fake_disc_target = pred_tensor.detach()
                if r1_weight > 0:
                    real_disc_target = real_disc_target.requires_grad_(True)
                if r2_weight > 0:
                    fake_disc_target = fake_disc_target.requires_grad_(True)

                with self.accelerator.autocast():
                    real_logits_d = self.discriminator(context=disc_condition, target=real_disc_target)
                    fake_logits_d = self.discriminator(context=disc_condition, target=fake_disc_target)
                    d_total_loss, d_logs = gan_critic_total_loss(
                        real_logits=real_logits_d,
                        fake_logits=fake_logits_d,
                        loss_type=gan_loss_type,
                        d_weight=d_weight,
                        real_input=real_disc_target if r1_weight > 0 else None,
                        fake_input=fake_disc_target if r2_weight > 0 else None,
                        r1_weight=r1_weight,
                        r2_weight=r2_weight,
                    )
                self.accelerator.backward(d_total_loss)
                if self.accelerator.sync_gradients:
                    default_gan_clip = float(self.train_cfg.get("max_grad_norm", 0.0))
                    gan_max_grad_norm = float(self.gan_cfg.get("max_grad_norm", default_gan_clip))
                    if gan_max_grad_norm > 0:
                        self.accelerator.clip_grad_norm_(self.discriminator.parameters(), gan_max_grad_norm)
                    self.disc_optim.step()
                    self.disc_sched.step()
                    self.disc_optim.zero_grad(set_to_none=True)

                self._set_requires_grad(self.discriminator, False)
                with self.accelerator.autocast():
                    fake_logits_g = self.discriminator(context=target_seed, target=pred_tensor)
                    real_logits_ref = None
                    if gan_loss_type == "rel_ns":
                        with torch.no_grad():
                            real_logits_ref = self.discriminator(context=target_seed, target=target_tensor)
                    g_adv_loss, g_logs = gan_generator_loss(
                        fake_logits=fake_logits_g,
                        real_logits=real_logits_ref,
                        loss_type=gan_loss_type,
                        weight=g_weight,
                    )
                total_g_loss = total_g_loss + g_adv_loss
                logs.update(d_logs)
                logs.update(g_logs)

            self.accelerator.backward(total_g_loss)
            max_grad_norm = float(self.train_cfg.get("max_grad_norm", 0.0))
            if self.accelerator.sync_gradients and max_grad_norm > 0:
                self.accelerator.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            if self.accelerator.sync_gradients:
                self.optim.step()
                self.sched.step()
                self.optim.zero_grad(set_to_none=True)

        did_step = bool(self.accelerator.sync_gradients)
        if did_step:
            if self.ema_model is not None:
                self.ema_model.update()
            self.global_step += 1

        logs["loss/rec_teacher_forced"] = rec_loss.detach()
        if "loss/rollout_branch" not in logs:
            logs["loss/rollout_branch"] = torch.zeros((), device=self.device)
            logs["loss/rollout_branch_raw"] = torch.zeros((), device=self.device)
            logs["meta/rollout_branch_weight"] = torch.zeros((), device=self.device)
        logs["loss/rec"] = (rec_loss.detach() + logs["loss/rollout_branch"])
        logs["loss"] = total_g_loss.detach()
        logs["meta/target_mode"] = torch.tensor(
            {"next_frame": 0.0, "block": 1.0}[str(aux["target_mode"])], device=self.device
        )
        logs["meta/target_frames"] = torch.tensor(float(aux["target_frames"]), device=self.device)
        logs["meta/context_frames"] = torch.tensor(float(aux["context_frames"]), device=self.device)
        logs["meta/sequence_loss"] = torch.tensor(float(aux["sequence_loss_enabled"]), device=self.device)
        logs["meta/sequence_context_frames"] = torch.tensor(float(aux["sequence_context_frames"]), device=self.device)
        logs["meta/sequence_future_frames"] = torch.tensor(float(aux["sequence_future_frames"]), device=self.device)
        logs["meta/rollout_branch_enabled"] = torch.tensor(
            1.0 if bool(self.train_cfg.next_pred.get("rollout_branch", {}).get("enabled", False)) else 0.0,
            device=self.device,
        )
        logs["meta/gan_enabled"] = torch.tensor(1.0 if self.use_gan else 0.0, device=self.device)
        return logs, did_step

    @torch.no_grad()
    def val_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        context, target_seed, target_gt, aux = self._build_next_pred_batch(batch, apply_missing_modality=False)
        target_frames = int(aux["target_frames"])
        context_time = aux.get("context_time")
        target_seed_time = aux.get("target_seed_time")
        with self.accelerator.autocast():
            forward_kwargs = {
                "context_x": context,
                "target_x": target_seed,
                "predict_frames": target_frames,
                "strict_target_isolation": bool(self.train_cfg.strict_target_isolation),
                "return_modality_dict": True,
            }
            if torch.is_tensor(context_time):
                forward_kwargs["context_time"] = context_time
            if torch.is_tensor(target_seed_time):
                forward_kwargs["target_time"] = target_seed_time
            pred = self.model.forward_ar(**forward_kwargs)
            _, logs = self._next_prediction_loss(pred=pred, target_gt=target_gt, aux=aux)
        return logs

    def _prepare_val_inference_batch(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor | None, torch.Tensor | None]:
        radar_past = self._ensure_bcthw(
            batch["radar_past"].to(self.device, dtype=torch.float32),
            expected_channels=self.radar_c,
            name="radar_past",
        )
        satellite_past = self._ensure_bcthw(
            batch["satellite_past"].to(self.device, dtype=torch.float32),
            expected_channels=self.satellite_c,
            name="satellite_past",
        )
        rain_past = self._ensure_bcthw(
            batch["rain_past"].to(self.device, dtype=torch.float32),
            expected_channels=self.rain_c,
            name="rain_past",
        )

        radar_future = self._ensure_bcthw(
            batch["radar_future"].to(self.device, dtype=torch.float32),
            expected_channels=self.radar_c,
            name="radar_future",
        )
        satellite_future = self._ensure_bcthw(
            batch["satellite_future"].to(self.device, dtype=torch.float32),
            expected_channels=self.satellite_c,
            name="satellite_future",
        )
        rain_future = self._ensure_bcthw(
            batch["rain_future"].to(self.device, dtype=torch.float32),
            expected_channels=self.rain_c,
            name="rain_future",
        )

        context = self._merge_modalities(radar_past, satellite_past, rain_past)
        target = {
            "radar": radar_future,
            "satellite": satellite_future,
            "rain": rain_future,
        }
        context_time: torch.Tensor | None = None
        target_time: torch.Tensor | None = None
        if "time_past" in batch and "time_future" in batch:
            context_time = batch["time_past"].to(self.device, dtype=torch.float32)
            target_time = batch["time_future"].to(self.device, dtype=torch.float32)
            if context_time.ndim == 1:
                context_time = context_time.unsqueeze(0)
            if target_time.ndim == 1:
                target_time = target_time.unsqueeze(0)
            if context_time.ndim != 2 or target_time.ndim != 2:
                raise ValueError(
                    "time_past/time_future must be [B,T] in val batch. "
                    f"got time_past={tuple(context_time.shape)}, time_future={tuple(target_time.shape)}"
                )
            if int(context_time.shape[0]) != int(context.shape[0]) or int(target_time.shape[0]) != int(context.shape[0]):
                raise ValueError(
                    "time batch mismatch in validation: "
                    f"context_batch={int(context.shape[0])}, "
                    f"time_past_batch={int(context_time.shape[0])}, time_future_batch={int(target_time.shape[0])}"
                )
            if int(context_time.shape[1]) != int(context.shape[2]) or int(target_time.shape[1]) != int(target["rain"].shape[2]):
                raise ValueError(
                    "time length mismatch in validation: "
                    f"context_time={int(context_time.shape[1])}, context_frames={int(context.shape[2])}, "
                    f"target_time={int(target_time.shape[1])}, target_frames={int(target['rain'].shape[2])}"
                )
        return context, target, context_time, target_time

    def _resolve_train_rollout_branch(self, total_future_frames: int) -> dict[str, float | int | bool | str] | None:
        rollout_cfg = self.train_cfg.next_pred.get("rollout_branch", {})
        if not bool(rollout_cfg.get("enabled", False)):
            return None
        if total_future_frames <= 0:
            raise ValueError(f"total_future_frames must be > 0, got {total_future_frames}")

        mode = str(rollout_cfg.get("mode", "block")).lower()
        if mode not in {"frame", "block"}:
            raise ValueError(f"train.next_pred.rollout_branch.mode must be 'frame' or 'block', got {mode}")

        weight = float(rollout_cfg.get("weight", 0.0))
        if weight < 0:
            raise ValueError(f"train.next_pred.rollout_branch.weight must be >= 0, got {weight}")
        if weight == 0:
            return None

        configured_frames = rollout_cfg.get("rollout_frames", None)
        rollout_frames = total_future_frames if configured_frames in (None, "") else int(configured_frames)
        if rollout_frames <= 0:
            raise ValueError(f"train.next_pred.rollout_branch.rollout_frames must be > 0, got {rollout_frames}")
        if rollout_frames > total_future_frames:
            raise ValueError(
                "train.next_pred.rollout_branch.rollout_frames cannot exceed available future frames, "
                f"got rollout_frames={rollout_frames}, total_future_frames={total_future_frames}"
            )

        default_block_size = int(self.train_cfg.next_pred.get("block_size", 1))
        rollout_block_size = int(rollout_cfg.get("rollout_block_size", default_block_size))
        if rollout_block_size <= 0:
            raise ValueError(
                f"train.next_pred.rollout_branch.rollout_block_size must be > 0, got {rollout_block_size}"
            )

        loss_on = str(rollout_cfg.get("loss_on", "rain")).lower()
        if loss_on not in {"rain", "all"}:
            raise ValueError(f"train.next_pred.rollout_branch.loss_on must be 'rain' or 'all', got {loss_on}")

        delta_loss_cfg = rollout_cfg.get("delta_loss", {})
        delta_loss_enabled = bool(delta_loss_cfg.get("enabled", False))
        delta_loss_weight = float(delta_loss_cfg.get("weight", 1.0))
        if delta_loss_weight < 0:
            raise ValueError(
                "train.next_pred.rollout_branch.delta_loss.weight must be >= 0, "
                f"got {delta_loss_weight}"
            )

        return {
            "mode": mode,
            "weight": weight,
            "rollout_frames": rollout_frames,
            "rollout_block_size": rollout_block_size,
            "detach_history": bool(rollout_cfg.get("detach_history", True)),
            "use_gt_future_modalities": bool(rollout_cfg.get("use_gt_future_modalities", True)),
            "loss_on": loss_on,
            "delta_loss_enabled": delta_loss_enabled,
            "delta_loss_weight": delta_loss_weight,
        }


    def _resolve_rollout_mode(self) -> tuple[str, int, bool, bool]:
        mode = str(self.val_cfg.get("rollout_mode", "block")).lower()
        if mode not in {"frame", "block"}:
            raise ValueError(f"val.rollout_mode must be 'frame' or 'block', got {mode}")

        default_block = int(self.train_cfg.next_pred.get("block_size", 1))
        rollout_block_size = int(self.val_cfg.get("rollout_block_size", default_block))
        rollout_block_size = max(1, rollout_block_size)
        rollout_history_detach = bool(self.val_cfg.get("rollout_history_detach", True))
        use_gt_future_modalities = bool(self.val_cfg.get("rollout_use_gt_future_modalities", False))
        return mode, rollout_block_size, rollout_history_detach, use_gt_future_modalities

    def _build_self_rolled_seed(
        self,
        context: torch.Tensor,
        seed_frames: int,
        detach_history: bool,
        context_time: torch.Tensor | None = None,
        future_time: torch.Tensor | None = None,
        future_modality_forcing: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if seed_frames <= 0:
            raise ValueError(f"seed_frames must be > 0, got {seed_frames}")
        frame_patch_size = self._resolve_frame_patch_size()
        context_frames = int(context.shape[2])
        if context_frames < frame_patch_size:
            raise ValueError(
                "context frames must be >= frame_patch_size for rollout seed, "
                f"got context_frames={context_frames}, frame_patch_size={frame_patch_size}"
            )
        if context_frames % frame_patch_size != 0:
            raise ValueError(
                "context frames must be divisible by frame_patch_size for rollout, "
                f"got context_frames={context_frames}, frame_patch_size={frame_patch_size}"
            )
        if seed_frames % frame_patch_size != 0:
            raise ValueError(
                "seed_frames must be divisible by frame_patch_size, "
                f"got seed_frames={seed_frames}, frame_patch_size={frame_patch_size}"
            )
        if future_modality_forcing is not None:
            if "radar" not in future_modality_forcing or "satellite" not in future_modality_forcing:
                raise ValueError("future_modality_forcing must contain 'radar' and 'satellite' when provided.")
            if int(future_modality_forcing["radar"].shape[2]) != seed_frames:
                raise ValueError(
                    "future_modality_forcing['radar'] length mismatch with seed_frames: "
                    f"{int(future_modality_forcing['radar'].shape[2])} vs {seed_frames}"
                )
            if int(future_modality_forcing["satellite"].shape[2]) != seed_frames:
                raise ValueError(
                    "future_modality_forcing['satellite'] length mismatch with seed_frames: "
                    f"{int(future_modality_forcing['satellite'].shape[2])} vs {seed_frames}"
                )

        seed_list: list[torch.Tensor] = [context[:, :, -frame_patch_size:, :, :]]
        context_cur = context
        context_time_cur = context_time
        generated = frame_patch_size

        while generated < seed_frames:
            with self.accelerator.autocast():
                forward_kwargs = {
                    "context_x": context_cur,
                    "target_x": seed_list[-1],
                    "predict_frames": frame_patch_size,
                    "strict_target_isolation": bool(self.train_cfg.strict_target_isolation),
                    "return_modality_dict": True,
                }
                if torch.is_tensor(context_time_cur):
                    forward_kwargs["context_time"] = context_time_cur
                    forward_kwargs["target_time"] = context_time_cur[:, -frame_patch_size:]
                pred_one = self.model.forward_ar(**forward_kwargs)
            if self._resolve_rain_residual_cfg()[0]:
                seed_modalities = self._split_modalities(seed_list[-1])
                pred_one = dict(pred_one)
                pred_one["rain"] = self._apply_rain_residual_output(pred_one["rain"], seed_modalities["rain"])
            pred_one_tensor = self._merge_modalities(pred_one["radar"], pred_one["satellite"], pred_one["rain"])
            pred_hist = pred_one_tensor.detach() if detach_history else pred_one_tensor
            if future_modality_forcing is not None:
                step_start = generated - frame_patch_size
                step_end = generated
                pred_modalities = self._split_modalities(pred_hist)
                pred_hist = self._merge_modalities(
                    future_modality_forcing["radar"][:, :, step_start:step_end],
                    future_modality_forcing["satellite"][:, :, step_start:step_end],
                    pred_modalities["rain"],
                )
            seed_list.append(pred_hist)
            context_cur = torch.cat([context_cur, pred_hist], dim=2)
            if torch.is_tensor(context_time_cur):
                if future_time is None:
                    raise ValueError("future_time should be provided when context_time is provided.")
                next_time = future_time[:, generated - frame_patch_size : generated]
                context_time_cur = torch.cat([context_time_cur, next_time], dim=1)
            generated += frame_patch_size

        return torch.cat(seed_list, dim=2)

    def _rollout_predict_with_settings(
        self,
        context: torch.Tensor,
        total_future_frames: int,
        mode: str,
        rollout_block_size: int,
        detach_history: bool,
        context_time: torch.Tensor | None = None,
        future_time: torch.Tensor | None = None,
        future_modalities: dict[str, torch.Tensor] | None = None,
        use_gt_future_modalities: bool = False,
        return_rain_delta: bool = False,
    ) -> dict[str, torch.Tensor]:
        if context.ndim != 5:
            raise ValueError(f"context must be [B,C,T,H,W], got {tuple(context.shape)}")
        if total_future_frames <= 0:
            raise ValueError(f"total_future_frames must be > 0, got {total_future_frames}")
        if mode not in {"frame", "block"}:
            raise ValueError(f"rollout mode must be 'frame' or 'block', got {mode}")
        if rollout_block_size <= 0:
            raise ValueError(f"rollout_block_size must be > 0, got {rollout_block_size}")
        frame_patch_size = self._resolve_frame_patch_size()
        context_frames = int(context.shape[2])
        if context_frames % frame_patch_size != 0:
            raise ValueError(
                "Validation context frames must be divisible by frame_patch_size, "
                f"got context_frames={context_frames}, frame_patch_size={frame_patch_size}"
            )
        if total_future_frames % frame_patch_size != 0:
            raise ValueError(
                "Validation total future frames must be divisible by frame_patch_size, "
                f"got total_future_frames={total_future_frames}, frame_patch_size={frame_patch_size}"
            )
        if context_time is not None or future_time is not None:
            if context_time is None or future_time is None:
                raise ValueError("context_time and future_time should be both set or both None in rollout.")
            if context_time.ndim != 2 or future_time.ndim != 2:
                raise ValueError(
                    "context_time/future_time must be [B,T] in rollout. "
                    f"got context_time={tuple(context_time.shape)}, future_time={tuple(future_time.shape)}"
                )
            if int(context_time.shape[0]) != int(context.shape[0]) or int(future_time.shape[0]) != int(context.shape[0]):
                raise ValueError(
                    "rollout time batch mismatch: "
                    f"context_batch={int(context.shape[0])}, "
                    f"context_time_batch={int(context_time.shape[0])}, future_time_batch={int(future_time.shape[0])}"
                )
            if int(context_time.shape[1]) != context_frames or int(future_time.shape[1]) != total_future_frames:
                raise ValueError(
                    "rollout time length mismatch: "
                    f"context_time={int(context_time.shape[1])}, context_frames={context_frames}, "
                    f"future_time={int(future_time.shape[1])}, total_future_frames={total_future_frames}"
                )
        if use_gt_future_modalities:
            if future_modalities is None:
                raise ValueError("future_modalities should be provided when use_gt_future_modalities=True.")
            if "radar" not in future_modalities or "satellite" not in future_modalities:
                raise ValueError("future_modalities must include 'radar' and 'satellite' when GT forcing is enabled.")
            radar_future = future_modalities["radar"]
            satellite_future = future_modalities["satellite"]
            if radar_future.ndim != 5 or satellite_future.ndim != 5:
                raise ValueError(
                    "future_modalities['radar'/'satellite'] must be [B,C,T,H,W], "
                    f"got radar={tuple(radar_future.shape)}, satellite={tuple(satellite_future.shape)}"
                )
            if int(radar_future.shape[0]) != int(context.shape[0]) or int(satellite_future.shape[0]) != int(context.shape[0]):
                raise ValueError(
                    "future_modalities batch mismatch with context: "
                    f"context_batch={int(context.shape[0])}, radar_batch={int(radar_future.shape[0])}, "
                    f"satellite_batch={int(satellite_future.shape[0])}"
                )
            if int(radar_future.shape[2]) != total_future_frames or int(satellite_future.shape[2]) != total_future_frames:
                raise ValueError(
                    "future_modalities time length mismatch with total_future_frames: "
                    f"radar_frames={int(radar_future.shape[2])}, satellite_frames={int(satellite_future.shape[2])}, "
                    f"total_future_frames={total_future_frames}"
                )

        remaining = total_future_frames
        context_cur = context
        context_time_cur = context_time
        produced = 0

        pred_radar: list[torch.Tensor] = []
        pred_satellite: list[torch.Tensor] = []
        pred_rain: list[torch.Tensor] = []
        pred_rain_delta: list[torch.Tensor] = []

        while remaining > 0:
            if mode == "frame":
                chunk = frame_patch_size
            else:
                raw_chunk = min(rollout_block_size, remaining)
                chunk = (raw_chunk // frame_patch_size) * frame_patch_size
                if chunk <= 0:
                    chunk = frame_patch_size
            if chunk > remaining:
                chunk = remaining
            chunk_future_time = None if future_time is None else future_time[:, produced : produced + chunk]
            chunk_future_modalities = None
            if future_modalities is not None:
                chunk_future_modalities = {
                    "radar": future_modalities["radar"][:, :, produced : produced + chunk],
                    "satellite": future_modalities["satellite"][:, :, produced : produced + chunk],
                }

            seed_block = self._build_self_rolled_seed(
                context=context_cur,
                seed_frames=chunk,
                detach_history=detach_history,
                context_time=context_time_cur,
                future_time=chunk_future_time,
                future_modality_forcing=chunk_future_modalities if use_gt_future_modalities else None,
            )
            with self.accelerator.autocast():
                forward_kwargs = {
                    "context_x": context_cur,
                    "target_x": seed_block,
                    "predict_frames": chunk,
                    "strict_target_isolation": bool(self.train_cfg.strict_target_isolation),
                    "return_modality_dict": True,
                }
                if torch.is_tensor(context_time_cur):
                    if chunk_future_time is None:
                        raise ValueError("chunk_future_time should be provided when context_time_cur is provided.")
                    target_seed_time = torch.cat(
                        [context_time_cur[:, -frame_patch_size:], chunk_future_time[:, :-frame_patch_size]],
                        dim=1,
                    )
                    forward_kwargs["context_time"] = context_time_cur
                    forward_kwargs["target_time"] = target_seed_time
                pred_block = self.model.forward_ar(**forward_kwargs)
            raw_rain_delta = pred_block["rain"]
            if self._resolve_rain_residual_cfg()[0]:
                seed_modalities = self._split_modalities(seed_block)
                pred_block = dict(pred_block)
                pred_block["rain"] = self._apply_rain_residual_output(pred_block["rain"], seed_modalities["rain"])
            pred_radar.append(pred_block["radar"])
            pred_satellite.append(pred_block["satellite"])
            pred_rain.append(pred_block["rain"])
            pred_rain_delta.append(raw_rain_delta)

            pred_block_tensor = self._merge_modalities(pred_block["radar"], pred_block["satellite"], pred_block["rain"])
            pred_hist = pred_block_tensor.detach() if detach_history else pred_block_tensor
            if use_gt_future_modalities:
                if chunk_future_modalities is None:
                    raise ValueError("chunk_future_modalities should be available when GT forcing is enabled.")
                pred_modalities = self._split_modalities(pred_hist)
                pred_hist = self._merge_modalities(
                    chunk_future_modalities["radar"],
                    chunk_future_modalities["satellite"],
                    pred_modalities["rain"],
                )
            context_cur = torch.cat([context_cur, pred_hist], dim=2)
            if torch.is_tensor(context_time_cur):
                if chunk_future_time is None:
                    raise ValueError("chunk_future_time should be provided when context_time_cur is provided.")
                context_time_cur = torch.cat([context_time_cur, chunk_future_time], dim=1)
            produced += chunk
            remaining -= chunk

        output = {
            "radar": torch.cat(pred_radar, dim=2),
            "satellite": torch.cat(pred_satellite, dim=2),
            "rain": torch.cat(pred_rain, dim=2),
        }
        if return_rain_delta:
            output["rain_delta"] = torch.cat(pred_rain_delta, dim=2)
        return output

    def _rollout_predict(
        self,
        context: torch.Tensor,
        total_future_frames: int,
        context_time: torch.Tensor | None = None,
        future_time: torch.Tensor | None = None,
        future_modalities: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        mode, rollout_block_size, detach_history, use_gt_future_modalities = self._resolve_rollout_mode()
        return self._rollout_predict_with_settings(
            context=context,
            total_future_frames=total_future_frames,
            mode=mode,
            rollout_block_size=rollout_block_size,
            detach_history=detach_history,
            context_time=context_time,
            future_time=future_time,
            future_modalities=future_modalities,
            use_gt_future_modalities=use_gt_future_modalities,
            return_rain_delta=False,
        )

    def _train_rollout_branch_loss(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
        context, target, context_time, target_time = self._prepare_val_inference_batch(batch)
        total_future_frames = int(target["rain"].shape[2])
        rollout_branch_cfg = self._resolve_train_rollout_branch(total_future_frames=total_future_frames)
        if rollout_branch_cfg is None:
            return None, {}

        rollout_frames = int(rollout_branch_cfg["rollout_frames"])
        target_slice = {
            "radar": target["radar"][:, :, :rollout_frames],
            "satellite": target["satellite"][:, :, :rollout_frames],
            "rain": target["rain"][:, :, :rollout_frames],
        }
        target_time_slice = None if target_time is None else target_time[:, :rollout_frames]
        pred_rollout = self._rollout_predict_with_settings(
            context=context,
            total_future_frames=rollout_frames,
            mode=str(rollout_branch_cfg["mode"]),
            rollout_block_size=int(rollout_branch_cfg["rollout_block_size"]),
            detach_history=bool(rollout_branch_cfg["detach_history"]),
            context_time=context_time,
            future_time=target_time_slice,
            future_modalities=target_slice,
            use_gt_future_modalities=bool(rollout_branch_cfg["use_gt_future_modalities"]),
            return_rain_delta=True,
        )

        loss_weight_override = None
        if str(rollout_branch_cfg["loss_on"]) == "rain":
            loss_weight_override = {"radar": 0.0, "satellite": 0.0, "rain": 1.0}

        aux = {
            "target_mode": "rollout_branch",
            "target_frames": rollout_frames,
            "context_frames": int(context.shape[2]),
            "sequence_loss_enabled": 0,
            "sequence_context_frames": 0,
            "sequence_future_frames": rollout_frames,
            "sequence_context_weight": 1.0,
            "sequence_future_weight": 1.0,
            "temporal_diff_anchor": self._split_modalities(context[:, :, -1:]),
            "rain_residual_already_applied": 1,
        }
        rollout_raw_loss, rollout_inner_logs = self._next_prediction_loss(
            pred=pred_rollout,
            target_gt=target_slice,
            aux=aux,
            loss_weight_override=loss_weight_override,
        )
        delta_loss_enabled = bool(rollout_branch_cfg["delta_loss_enabled"])
        delta_loss_weight = float(rollout_branch_cfg["delta_loss_weight"])
        rollout_delta_loss = torch.zeros((), device=rollout_raw_loss.device, dtype=rollout_raw_loss.dtype)
        rollout_delta_loss_weighted = rollout_delta_loss
        delta_mean_ratio = torch.zeros((), device=rollout_raw_loss.device, dtype=rollout_raw_loss.dtype)
        pred_delta_abs_mean = torch.zeros((), device=rollout_raw_loss.device, dtype=rollout_raw_loss.dtype)
        gt_delta_abs_mean = torch.zeros((), device=rollout_raw_loss.device, dtype=rollout_raw_loss.dtype)
        if delta_loss_enabled and delta_loss_weight > 0:
            if "rain_delta" not in pred_rollout:
                raise ValueError("rollout_branch.delta_loss.enabled=True requires rollout prediction to include rain_delta.")
            gt_delta_ref = torch.cat([context[:, -self.rain_c :, -1:], target_slice["rain"][:, :, :-1]], dim=2)
            gt_delta = target_slice["rain"] - gt_delta_ref
            pred_delta = pred_rollout["rain_delta"].to(device=gt_delta.device, dtype=gt_delta.dtype)
            rollout_delta_loss = F.mse_loss(pred_delta, gt_delta)
            rollout_delta_loss_weighted = delta_loss_weight * rollout_delta_loss
            rollout_raw_loss = rollout_raw_loss + rollout_delta_loss_weighted
            pred_delta_abs_mean = pred_delta.abs().mean()
            gt_delta_abs_mean = gt_delta.abs().mean()
            delta_mean_ratio = pred_delta_abs_mean / gt_delta_abs_mean.clamp_min(1.0e-8)

        rollout_weight = float(rollout_branch_cfg["weight"])
        rollout_scaled_loss = rollout_raw_loss * rollout_weight

        logs = {
            "loss/rollout_branch_raw": rollout_raw_loss.detach(),
            "loss/rollout_branch": rollout_scaled_loss.detach(),
            "loss/rollout_branch_delta": rollout_delta_loss.detach(),
            "loss/rollout_branch_delta_weighted_inner": rollout_delta_loss_weighted.detach(),
            "meta/rollout_branch_weight": torch.tensor(
                rollout_weight, device=rollout_raw_loss.device, dtype=rollout_raw_loss.dtype
            ),
            "meta/rollout_branch_delta_loss_weight": torch.tensor(
                delta_loss_weight, device=rollout_raw_loss.device, dtype=rollout_raw_loss.dtype
            ),
            "meta/rollout_branch_pred_delta_abs_mean": pred_delta_abs_mean.detach(),
            "meta/rollout_branch_gt_delta_abs_mean": gt_delta_abs_mean.detach(),
            "meta/rollout_branch_delta_mean_ratio": delta_mean_ratio.detach(),
        }
        for key, value in rollout_inner_logs.items():
            if key == "loss":
                continue
            logs[f"rollout/{key}"] = value.detach()
        return rollout_scaled_loss, logs

    @torch.no_grad()
    def _predict_next_after_roll_block(
        self,
        context: torch.Tensor,
        target: dict[str, torch.Tensor],
        context_time: torch.Tensor | None = None,
        target_time: torch.Tensor | None = None,
    ) -> tuple[
        dict[str, torch.Tensor] | None, dict[str, torch.Tensor] | None, torch.Tensor | None, torch.Tensor | None
    ]:
        after_cfg = self.val_cfg.get("after_roll_next", {})
        if not bool(after_cfg.get("enabled", False)):
            return None, None, None, None

        frame_patch_size = self._resolve_frame_patch_size()
        roll_frames = int(after_cfg.get("roll_frames", frame_patch_size))
        if roll_frames <= 0:
            raise ValueError(f"val.after_roll_next.roll_frames must be > 0, got {roll_frames}")
        if roll_frames % frame_patch_size != 0:
            raise ValueError(
                "val.after_roll_next.roll_frames must be divisible by frame_patch_size, "
                f"got roll_frames={roll_frames}, frame_patch_size={frame_patch_size}"
            )

        total_target_frames = int(target["rain"].shape[2])
        if total_target_frames - roll_frames < frame_patch_size:
            return None, None, None, None

        detach_history = bool(after_cfg.get("detach_history", True))
        use_gt_future_modalities = bool(self.val_cfg.get("rollout_use_gt_future_modalities", False))
        rolled_pred = self._rollout_predict_with_settings(
            context=context,
            total_future_frames=roll_frames,
            mode="block",
            rollout_block_size=roll_frames,
            detach_history=detach_history,
            context_time=context_time,
            future_time=None if target_time is None else target_time[:, :roll_frames],
            future_modalities=target,
            use_gt_future_modalities=use_gt_future_modalities,
        )

        rolled_modalities = rolled_pred
        if use_gt_future_modalities:
            rolled_modalities = {
                "radar": target["radar"][:, :, :roll_frames],
                "satellite": target["satellite"][:, :, :roll_frames],
                "rain": rolled_pred["rain"],
            }
        rolled_tensor = self._merge_modalities(
            rolled_modalities["radar"],
            rolled_modalities["satellite"],
            rolled_modalities["rain"],
        )
        rolled_hist = rolled_tensor.detach() if detach_history else rolled_tensor
        rolled_context = torch.cat([context, rolled_hist], dim=2)
        seed = rolled_context[:, :, -frame_patch_size:, :, :]
        rolled_context_time = None
        if context_time is not None and target_time is not None:
            rolled_context_time = torch.cat([context_time, target_time[:, :roll_frames]], dim=1)
            seed_time = rolled_context_time[:, -frame_patch_size:]
        else:
            seed_time = None

        with self.accelerator.autocast():
            forward_kwargs = {
                "context_x": rolled_context,
                "target_x": seed,
                "predict_frames": frame_patch_size,
                "strict_target_isolation": bool(self.train_cfg.strict_target_isolation),
                "return_modality_dict": True,
            }
            if torch.is_tensor(rolled_context_time):
                forward_kwargs["context_time"] = rolled_context_time
                forward_kwargs["target_time"] = seed_time
            pred_next = self.model.forward_ar(**forward_kwargs)

        target_next = {
            "radar": target["radar"][:, :, roll_frames : roll_frames + frame_patch_size],
            "satellite": target["satellite"][:, :, roll_frames : roll_frames + frame_patch_size],
            "rain": target["rain"][:, :, roll_frames : roll_frames + frame_patch_size],
        }
        lw = self.train_cfg.loss_weights
        loss_next = (
            float(lw.radar) * F.mse_loss(pred_next["radar"], target_next["radar"])
            + float(lw.satellite) * F.mse_loss(pred_next["satellite"], target_next["satellite"])
            + float(lw.rain) * F.mse_loss(pred_next["rain"], target_next["rain"])
        )
        return pred_next, target_next, loss_next.detach(), rolled_context

    @torch.no_grad()
    def _val_inference_step(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        context, target, context_time, target_time = self._prepare_val_inference_batch(batch)
        target_frames = int(target["rain"].shape[2])
        if target_frames <= 0:
            raise ValueError("Validation target frames must be > 0.")

        pred_target = self._rollout_predict(
            context=context,
            total_future_frames=target_frames,
            context_time=context_time,
            future_time=target_time,
            future_modalities=target,
        )

        lw = self.train_cfg.loss_weights
        infer_loss = (
            float(lw.radar) * F.mse_loss(pred_target["radar"], target["radar"])
            + float(lw.satellite) * F.mse_loss(pred_target["satellite"], target["satellite"])
            + float(lw.rain) * F.mse_loss(pred_target["rain"], target["rain"])
        )
        extra_logs: dict[str, torch.Tensor] = {}
        pred_next, target_next, next_loss, after_roll_context = self._predict_next_after_roll_block(
            context=context,
            target=target,
            context_time=context_time,
            target_time=target_time,
        )
        if next_loss is not None:
            extra_logs["val/infer_after_roll_next_loss"] = next_loss
        if pred_next is not None and target_next is not None:
            extra_logs["after_roll_next_pred_radar"] = pred_next["radar"]
            extra_logs["after_roll_next_pred_satellite"] = pred_next["satellite"]
            extra_logs["after_roll_next_pred_rain"] = pred_next["rain"]
            extra_logs["after_roll_next_target_radar"] = target_next["radar"]
            extra_logs["after_roll_next_target_satellite"] = target_next["satellite"]
            extra_logs["after_roll_next_target_rain"] = target_next["rain"]
        if after_roll_context is not None:
            extra_logs["after_roll_next_context"] = after_roll_context

        return pred_target, target, infer_loss.detach(), extra_logs

    @staticmethod
    def _psnr_ssim_sums(
        pred: torch.Tensor, target: torch.Tensor, data_range: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if pred.shape != target.shape:
            raise ValueError(f"pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
        if pred.ndim != 5:
            raise ValueError(f"pred/target must be [B,C,T,H,W], got shape={tuple(pred.shape)}")
        if data_range <= 0:
            raise ValueError(f"data_range must be > 0, got {data_range}")

        b, c, t, h, w = pred.shape
        pred_bt = pred.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        target_bt = target.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)

        psnr_mean = peak_signal_noise_ratio(
            preds=pred_bt,
            target=target_bt,
            data_range=data_range,
            reduction="elementwise_mean",
        )
        ssim_mean = structural_similarity_index_measure(
            preds=pred_bt,
            target=target_bt,
            data_range=data_range,
            reduction="elementwise_mean",
        )
        count = torch.tensor(float(pred_bt.shape[0]), device=pred.device)
        return psnr_mean * count, ssim_mean * count, count

    @staticmethod
    def _ensure_rgb_uint8(img: object) -> torch.Tensor:
        if not isinstance(img, torch.Tensor):
            img_np = torch.as_tensor(img)
        else:
            img_np = img
        if img_np.ndim != 3 or img_np.shape[-1] != 3:
            raise ValueError(f"Expected RGB image [H,W,3], got shape={tuple(img_np.shape)}")
        if img_np.dtype != torch.uint8:
            img_np = img_np.clamp(0, 255).to(torch.uint8)
        return img_np

    @staticmethod
    def _sequence_header_labels(history_frames: int, pred_frames: int, gt_frames: int) -> list[str]:
        labels: list[str] = []
        for idx in range(history_frames):
            labels.append(f"history {idx + 1}")
        for idx in range(pred_frames):
            labels.append(f"pred t+{idx + 1}")
        for idx in range(gt_frames):
            labels.append(f"gt t+{idx + 1}")
        return labels

    def _build_modality_sequence_grid(
        self,
        context: dict[str, torch.Tensor],
        pred_target: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        sample_idx: int,
    ) -> Image.Image:
        font = ImageFont.load_default()
        modality_names = ["radar", "satellite", "rain"]

        labels = self._sequence_header_labels(
            history_frames=int(context["rain"].shape[2]),
            pred_frames=int(pred_target["rain"].shape[2]),
            gt_frames=int(target["rain"].shape[2]),
        )
        first_tile = plot_any_modality(context["radar"][sample_idx, :, 0], modality_name="radar", to_PIL=False)
        tile = self._ensure_rgb_uint8(first_tile)
        tile_h, tile_w = int(tile.shape[0]), int(tile.shape[1])

        left_label_w = 84
        cell_gap = 6
        top_header_h = 26
        row_gap = 8
        total_cols = len(labels)
        total_rows = len(modality_names)
        canvas_w = left_label_w + total_cols * tile_w + max(total_cols - 1, 0) * cell_gap
        canvas_h = top_header_h + total_rows * tile_h + max(total_rows - 1, 0) * row_gap
        canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        for col_idx, label in enumerate(labels):
            x = left_label_w + col_idx * (tile_w + cell_gap)
            draw.text((x + 4, 6), label, fill=(0, 0, 0), font=font)

        history_cols = int(context["rain"].shape[2])
        pred_cols = int(pred_target["rain"].shape[2])
        separators = [history_cols, history_cols + pred_cols]
        for separator in separators:
            if separator <= 0 or separator >= total_cols:
                continue
            x = left_label_w + separator * tile_w + max(separator - 1, 0) * cell_gap + (cell_gap // 2)
            draw.line((x, 0, x, canvas_h), fill=(160, 160, 160), width=2)

        for row_idx, modality_name in enumerate(modality_names):
            y = top_header_h + row_idx * (tile_h + row_gap)
            draw.text((8, y + 8), modality_name, fill=(0, 0, 0), font=font)

            frame_tensors = []
            for frame_idx in range(int(context[modality_name].shape[2])):
                frame_tensors.append(context[modality_name][sample_idx, :, frame_idx])
            for frame_idx in range(int(pred_target[modality_name].shape[2])):
                frame_tensors.append(pred_target[modality_name][sample_idx, :, frame_idx])
            for frame_idx in range(int(target[modality_name].shape[2])):
                frame_tensors.append(target[modality_name][sample_idx, :, frame_idx])

            for col_idx, frame_tensor in enumerate(frame_tensors):
                x = left_label_w + col_idx * (tile_w + cell_gap)
                frame_img = plot_any_modality(frame_tensor, modality_name=modality_name, to_PIL=False)
                frame_rgb = self._ensure_rgb_uint8(frame_img).cpu().numpy()
                canvas.paste(Image.fromarray(frame_rgb), (x, y))

        return canvas

    def _save_val_visualizations(
        self,
        context: torch.Tensor,
        pred_target: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        output_prefix: str = "rollout",
    ) -> None:
        if not bool(self.val_cfg.get("save_visuals", True)):
            return
        if not self.accelerator.is_main_process:
            return
        sample_idx = int(self.val_cfg.get("viz_sample_index", 0))
        viz_dir = self.proj_dir / "val_viz" / f"step_{self.global_step:08d}"
        viz_dir.mkdir(parents=True, exist_ok=True)
        context_cpu = self._split_modalities(context.detach().float().cpu().clamp_min(0.0))
        pred_cpu = {
            k: pred_target[k].detach().float().cpu().clamp_min(0.0)
            for k in ("radar", "satellite", "rain")
        }
        gt_cpu = {k: v.detach().float().cpu().clamp_min(0.0) for k, v in target.items()}

        batch_size = int(pred_cpu["rain"].shape[0])
        if batch_size <= 0:
            return
        if sample_idx < 0:
            sample_id = int(torch.randint(low=0, high=batch_size, size=(1,)).item())
        else:
            sample_id = max(0, min(sample_idx, batch_size - 1))
        grid = self._build_modality_sequence_grid(
            context=context_cpu,
            pred_target=pred_cpu,
            target=gt_cpu,
            sample_idx=sample_id,
        )
        out_path = viz_dir / f"{output_prefix}_sample{sample_id}_timeline.jpg"
        grid.save(out_path, quality=85)

    def _save_checkpoint(self) -> None:
        ckpt_dir = self.proj_dir / f"checkpoint-{self.global_step:08d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.accelerator.save_state(str(ckpt_dir))
        if self.accelerator.is_main_process:
            (ckpt_dir / "meta.json").write_text(json.dumps({"global_step": self.global_step}, indent=2))
            if self.ema_model is not None:
                # Keep only one EMA file for the whole run to avoid per-checkpoint duplication.
                ema_dir = self.proj_dir / "ema"
                ema_dir.mkdir(parents=True, exist_ok=True)
                torch.save(self.ema_model.state_dict(), ema_dir / "ema.pt")
        self.log_msg(f"[Checkpoint] saved to {ckpt_dir}")

    def _resume_if_needed(self) -> None:
        resume_path = self.train_cfg.resume_path
        if resume_path is None:
            return
        resume_dir = Path(resume_path)
        self.accelerator.load_state(str(resume_dir))
        meta = resume_dir / "meta.json"
        if meta.exists():
            payload = json.loads(meta.read_text())
            self.global_step = int(payload.get("global_step", 0))
        if self.ema_model is not None:
            ema_candidates = [
                resume_dir / "ema" / "ema.pt",
                resume_dir.parent / "ema" / "ema.pt",
                self.proj_dir / "ema" / "ema.pt",
            ]
            for ema_path in ema_candidates:
                if ema_path.exists():
                    self.ema_model.load_state_dict(torch.load(ema_path, map_location=self.device))
                    break
        self.log_msg(f"[Resume] loaded from {resume_dir}, global_step={self.global_step}")

    def _log_metrics(self, logs: dict[str, torch.Tensor]) -> None:
        if self.global_step % int(self.train_cfg.log.log_every) != 0:
            return
        scalar_logs: dict[str, float] = {}
        for k, v in logs.items():
            if torch.is_tensor(v):
                scalar_logs[k] = float(v.detach().item())
            else:
                scalar_logs[k] = float(v)
        scalar_logs["lr"] = float(self.optim.param_groups[0]["lr"])
        if self.use_gan and self.disc_optim is not None:
            scalar_logs["lr/discriminator"] = float(self.disc_optim.param_groups[0]["lr"])
        msg = " | ".join([f"{k}: {v:.6f}" for k, v in scalar_logs.items()])
        self.log_msg(f"[Train][{self.global_step}/{self.train_cfg.max_steps}] {msg}")
        if not self.train_cfg.debug:
            self._log_tensorboard_scalars(scalar_logs, step=self.global_step)

    @torch.no_grad()
    def _run_val(self) -> None:
        max_iters = int(self.val_cfg.max_val_iters)
        if max_iters <= 0:
            return
        self.model.eval()
        self.log_msg(f"[Val] start at step={self.global_step}, max_val_iters={max_iters}")

        loss_sum = torch.tensor(0.0, device=self.device)
        infer_loss_sum = torch.tensor(0.0, device=self.device)
        batch_count = torch.tensor(0.0, device=self.device)

        csi_thresholds = [float(v) for v in self.val_cfg.get("csi_thresholds", [0.5])]
        if len(csi_thresholds) == 0:
            raise ValueError("val.csi_thresholds should contain at least one threshold.")
        csi_tp = torch.zeros(len(csi_thresholds), device=self.device)
        csi_fp = torch.zeros(len(csi_thresholds), device=self.device)
        csi_fn = torch.zeros(len(csi_thresholds), device=self.device)

        radar_psnr_sum = torch.tensor(0.0, device=self.device)
        radar_ssim_sum = torch.tensor(0.0, device=self.device)
        radar_count = torch.tensor(0.0, device=self.device)
        satellite_psnr_sum = torch.tensor(0.0, device=self.device)
        satellite_ssim_sum = torch.tensor(0.0, device=self.device)
        satellite_count = torch.tensor(0.0, device=self.device)
        infer_after_roll_next_sum = torch.tensor(0.0, device=self.device)
        infer_after_roll_next_count = torch.tensor(0.0, device=self.device)
        first_context: torch.Tensor | None = None
        first_pred_target: dict[str, torch.Tensor] | None = None
        first_target: dict[str, torch.Tensor] | None = None
        first_after_roll_next_context: torch.Tensor | None = None
        first_after_roll_next_pred: dict[str, torch.Tensor] | None = None
        first_after_roll_next_target: dict[str, torch.Tensor] | None = None

        iterator = iter(self.val_dataloader)
        if self.accelerator.is_main_process:
            val_iter = tqdm(range(max_iters), desc=f"val[{self.global_step}]", leave=False, dynamic_ncols=True)
        else:
            val_iter = range(max_iters)

        for _ in val_iter:
            try:
                batch = next(iterator)
            except StopIteration:
                break

            logs = self.val_step(batch)
            pred_target, target, infer_loss, extra_logs = self._val_inference_step(batch)
            if first_pred_target is None:
                first_context, _, _, _ = self._prepare_val_inference_batch(batch)
                first_context = first_context.detach()
                first_pred_target = {k: v.detach() for k, v in pred_target.items()}
                first_target = {k: v.detach() for k, v in target.items()}
            if (
                first_after_roll_next_pred is None
                and "after_roll_next_pred_rain" in extra_logs
                and "after_roll_next_target_rain" in extra_logs
            ):
                first_after_roll_next_context = extra_logs["after_roll_next_context"].detach()
                first_after_roll_next_pred = {
                    "radar": extra_logs["after_roll_next_pred_radar"].detach(),
                    "satellite": extra_logs["after_roll_next_pred_satellite"].detach(),
                    "rain": extra_logs["after_roll_next_pred_rain"].detach(),
                }
                first_after_roll_next_target = {
                    "radar": extra_logs["after_roll_next_target_radar"].detach(),
                    "satellite": extra_logs["after_roll_next_target_satellite"].detach(),
                    "rain": extra_logs["after_roll_next_target_rain"].detach(),
                }

            loss_sum += logs["loss"].detach().float()
            infer_loss_sum += infer_loss
            batch_count += 1.0
            if "val/infer_after_roll_next_loss" in extra_logs:
                infer_after_roll_next_sum += extra_logs["val/infer_after_roll_next_loss"].detach().float()
                infer_after_roll_next_count += 1.0

            rain_pred = self._denormalize_rain_for_metrics(pred_target["rain"].detach())
            rain_target = self._denormalize_rain_for_metrics(target["rain"].detach())
            for idx, threshold in enumerate(csi_thresholds):
                pred_bin = rain_pred >= threshold
                target_bin = rain_target >= threshold
                csi_tp[idx] += (pred_bin & target_bin).sum(dtype=torch.float32)
                csi_fp[idx] += (pred_bin & ~target_bin).sum(dtype=torch.float32)
                csi_fn[idx] += (~pred_bin & target_bin).sum(dtype=torch.float32)

            data_range = float(self.val_cfg.get("metric_data_range", 1.0))
            if data_range <= 0:
                target_all = self._merge_modalities(target["radar"], target["satellite"], target["rain"])
                data_range = float((target_all.amax() - target_all.amin()).item())
                if data_range <= 0:
                    data_range = 1.0

            radar_psnr_delta, radar_ssim_delta, radar_count_delta = self._psnr_ssim_sums(
                pred_target["radar"], target["radar"], data_range=data_range
            )
            satellite_psnr_delta, satellite_ssim_delta, satellite_count_delta = self._psnr_ssim_sums(
                pred_target["satellite"], target["satellite"], data_range=data_range
            )
            radar_psnr_sum += radar_psnr_delta
            radar_ssim_sum += radar_ssim_delta
            radar_count += radar_count_delta
            satellite_psnr_sum += satellite_psnr_delta
            satellite_ssim_sum += satellite_ssim_delta
            satellite_count += satellite_count_delta

        if float(batch_count.item()) == 0.0:
            self.model.train()
            return

        loss_sum = self.accelerator.reduce(loss_sum, reduction="sum")
        infer_loss_sum = self.accelerator.reduce(infer_loss_sum, reduction="sum")
        batch_count = self.accelerator.reduce(batch_count, reduction="sum")
        csi_tp = self.accelerator.reduce(csi_tp, reduction="sum")
        csi_fp = self.accelerator.reduce(csi_fp, reduction="sum")
        csi_fn = self.accelerator.reduce(csi_fn, reduction="sum")
        radar_psnr_sum = self.accelerator.reduce(radar_psnr_sum, reduction="sum")
        radar_ssim_sum = self.accelerator.reduce(radar_ssim_sum, reduction="sum")
        radar_count = self.accelerator.reduce(radar_count, reduction="sum")
        satellite_psnr_sum = self.accelerator.reduce(satellite_psnr_sum, reduction="sum")
        satellite_ssim_sum = self.accelerator.reduce(satellite_ssim_sum, reduction="sum")
        satellite_count = self.accelerator.reduce(satellite_count, reduction="sum")
        infer_after_roll_next_sum = self.accelerator.reduce(infer_after_roll_next_sum, reduction="sum")
        infer_after_roll_next_count = self.accelerator.reduce(infer_after_roll_next_count, reduction="sum")

        val_loss = float((loss_sum / batch_count.clamp_min(1.0)).item())
        val_infer_loss = float((infer_loss_sum / batch_count.clamp_min(1.0)).item())
        radar_psnr = float((radar_psnr_sum / radar_count.clamp_min(1.0)).item())
        radar_ssim = float((radar_ssim_sum / radar_count.clamp_min(1.0)).item())
        satellite_psnr = float((satellite_psnr_sum / satellite_count.clamp_min(1.0)).item())
        satellite_ssim = float((satellite_ssim_sum / satellite_count.clamp_min(1.0)).item())
        infer_after_roll_next_loss = None
        if float(infer_after_roll_next_count.item()) > 0.0:
            infer_after_roll_next_loss = float(
                (infer_after_roll_next_sum / infer_after_roll_next_count.clamp_min(1.0)).item()
            )

        csi_logs: dict[str, float] = {}
        csi_msg_parts = []
        for idx, threshold in enumerate(csi_thresholds):
            denom = csi_tp[idx] + csi_fp[idx] + csi_fn[idx]
            csi_value = float((csi_tp[idx] / denom.clamp_min(1e-8)).item())
            key = f"val/csi@{threshold:g}"
            csi_logs[key] = csi_value
            csi_msg_parts.append(f"csi@{threshold:g}={csi_value:.6f}")

        metric_msg = (
            f"[Val][{self.global_step}] "
            f"loss={val_loss:.6f} | infer_loss={val_infer_loss:.6f} | "
            f"radar_psnr={radar_psnr:.6f} | radar_ssim={radar_ssim:.6f} | "
            f"satellite_psnr={satellite_psnr:.6f} | satellite_ssim={satellite_ssim:.6f} | "
            f"{' | '.join(csi_msg_parts)}"
        )
        if infer_after_roll_next_loss is not None:
            metric_msg = f"{metric_msg} | infer_after_roll_next_loss={infer_after_roll_next_loss:.6f}"
        self.log_msg(metric_msg)
        if not self.train_cfg.debug:
            log_payload = {
                "val/loss": val_loss,
                "val/infer_loss": val_infer_loss,
                "val/radar_psnr": radar_psnr,
                "val/radar_ssim": radar_ssim,
                "val/satellite_psnr": satellite_psnr,
                "val/satellite_ssim": satellite_ssim,
                **csi_logs,
            }
            if infer_after_roll_next_loss is not None:
                log_payload["val/infer_after_roll_next_loss"] = infer_after_roll_next_loss
            if not self.train_cfg.debug:
                self._log_tensorboard_scalars(log_payload, step=self.global_step)
        if first_context is not None and first_pred_target is not None and first_target is not None:
            self._save_val_visualizations(
                context=first_context,
                pred_target=first_pred_target,
                target=first_target,
                output_prefix="rollout",
            )
        if (
            first_after_roll_next_context is not None
            and first_after_roll_next_pred is not None
            and first_after_roll_next_target is not None
        ):
            self._save_val_visualizations(
                context=first_after_roll_next_context,
                pred_target=first_after_roll_next_pred,
                target=first_after_roll_next_target,
                output_prefix="after_roll_next",
            )
        self.model.train()

    def train(self) -> None:
        self._resume_if_needed()
        self.model.train()

        stop = False
        while not stop:
            for batch in self.train_dataloader:
                logs, did_step = self.train_step(batch)
                if not did_step:
                    continue
                self._log_metrics(logs)

                if self.global_step % int(self.val_cfg.val_duration) == 0:
                    self._run_val()
                if self.global_step % int(self.train_cfg.save_every) == 0:
                    self._save_checkpoint()

                if self.global_step >= int(self.train_cfg.max_steps):
                    stop = True
                    break

        self._save_checkpoint()
        self.log_msg("Training finished.")

    def run(self) -> None:
        try:
            self.train()
        finally:
            self._close_tensorboard_writer()


@hydra.main(
    config_path="../config/ts_rain_train",
    #config_name="rain_trainer_ts_next_frame",
    config_name="rain_trainer_ts_next_frame_delta_filter",

    version_base=None,
)
def main(cfg: DictConfig) -> None:
    catcher = logger.catch if PartialState().is_main_process else nullcontext
    with catcher():
        trainer = RainTSNextFrameTrainer(cfg)
        trainer.run()


if __name__ == "__main__":
    main()
