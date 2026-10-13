"""
Time-series diffusion trainer (Accelerate) for rain prediction.

Stage-1 target (causal-forcing style):
- autoregressive teacher-forcing diffusion training
- model conditions on clean history frames and denoises current noisy target frame
"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import accelerate
import hydra
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.state import PartialState
from ema_pytorch import EMA
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
from tqdm import tqdm

from src.dataset.rain_ts_litdata import denormalize_rain_linear
from src.networks.time_series.causal_forcing.scheduler import FlowMatchScheduler
from src.networks.time_series.diffusion.gaussian_scheduler import GaussianDiffusionScheduler
from src.networks.time_series.diffusion.fm_solvers import FlowDPMSolverMultistepScheduler
from src.networks.time_series.diffusion.fm_solvers_unipc import FlowUniPCMultistepScheduler
from src.utils.visualization.plot import plot_any_modality

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

try:
    import colored_traceback

    colored_traceback.add_hook()
except Exception:
    colored_traceback = None

try:
    from accelerate.utils import DummyOptim, DummyScheduler
except Exception:

    class DummyOptim:  # pragma: no cover - fallback for old accelerate versions
        def __init__(self, *args, **kwargs):
            self.param_groups = [{"lr": 0.0}]

        def step(self):
            return None

        def zero_grad(self, *args, **kwargs):
            return None

    class DummyScheduler:  # pragma: no cover - fallback for old accelerate versions
        def __init__(self, *args, **kwargs):
            pass

        def step(self):
            return None


class RainTSDiffusionTrainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.train_cfg = cfg.train
        self.val_cfg = cfg.val
        self.dataset_cfg = cfg.dataset
        self.ema_cfg = cfg.ema

        self.accelerator: Accelerator = hydra.utils.instantiate(cfg.accelerator)
        seed = int(getattr(self.train_cfg, "seed", 2025))
        accelerate.utils.set_seed(seed)

        self.log_file = self._configure_logger()
        self.device = self.accelerator.device
        self.dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "no": torch.float32,
        }[self.accelerator.mixed_precision]

        self.log_msg(f"Log file: {self.log_file}")
        self.log_msg(f"Project dir: {self.proj_dir}")

        self.train_dataset, self.train_dataloader = hydra.utils.instantiate(self.dataset_cfg.train)
        self.val_dataset, self.val_dataloader = hydra.utils.instantiate(self.dataset_cfg.val)
        self._init_rain_norm_params()

        self.model = hydra.utils.instantiate(cfg.rain_prediction_model)
        self._validate_data_model_contract()
        self.optim, self.sched = self._build_optim_sched()

        self.diffusion_mode = self._resolve_diffusion_mode()
        self.prediction_target = self._resolve_prediction_target()
        self.noise_schedule = self._build_noise_scheduler()

        self.model, self.optim, self.train_dataloader, self.val_dataloader, self.sched = self.accelerator.prepare(
            self.model,
            self.optim,
            self.train_dataloader,
            self.val_dataloader,
            self.sched,
        )

        self.no_ema = False
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
        self.min_t = int(self.train_cfg.diffusion.min_timestep)
        self.max_t = int(self.train_cfg.diffusion.max_timestep)

        self.radar_c = int(getattr(self.model, "radar_out_channels", 1))
        self.satellite_c = int(getattr(self.model, "satellite_out_channels", 10))
        self.rain_c = int(getattr(self.model, "rain_out_channels", 1))

        self.log_msg(f"Model channels: radar={self.radar_c}, satellite={self.satellite_c}, rain={self.rain_c}")
        self.log_msg(f"Diffusion mode: {self.diffusion_mode}, prediction_target: {self.prediction_target}")
        self.log_msg("Stage-1 objective: autoregressive teacher-forcing diffusion")

    def _init_rain_norm_params(self) -> None:
        train_mzc = bool(self.dataset_cfg.train.get("modality_zero_centering", False))
        val_mzc = bool(self.dataset_cfg.val.get("modality_zero_centering", False))
        if train_mzc != val_mzc:
            raise ValueError(
                "dataset.train and dataset.val modality_zero_centering should be identical."
            )
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
            "Modality zero-centering enabled. "
            f"rain mean={self.rain_norm_mean:.12f}, std={self.rain_norm_std:.12f}"
        )

    def _validate_data_model_contract(self) -> None:
        n_past = int(self.dataset_cfg.n_past)
        n_futures = int(self.dataset_cfg.n_futures)
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

    def _configure_logger(self) -> Path:
        logger.remove()
        logger.add(
            sys.stdout,
            format="{time:HH:mm:ss} - {level.icon} <level>[{level}:{file.name}:{line}]</level> - <level>{message}</level>",
            level="DEBUG",
            colorize=True,
        )

        configured_proj_dir = getattr(self.train_cfg, "proj_dir", None)
        use_configured_proj_dir = configured_proj_dir not in (None, "")
        if use_configured_proj_dir:
            log_root = Path(str(configured_proj_dir))
            if self.train_cfg.log.log_with_time:
                stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
                log_root = log_root / stamp
            if self.train_cfg.log.run_comment:
                log_root = Path(f"{log_root.as_posix()}_{self.train_cfg.log.run_comment}")
        else:
            # Use hydra runtime output dir (set by hydra.run.dir / output_subdir).
            hydra_cfg = HydraConfig.get()
            log_root = Path(hydra_cfg.runtime.output_dir)
            logger.info(f"[Hydra] use runtime output dir as log root: {log_root}")
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
                    enqueue=True,
                    colorize=False,
                )
            cfg_dump = self.proj_dir / "config" / "config_total.yaml"
            cfg_dump.parent.mkdir(parents=True, exist_ok=True)
            cfg_dump.write_text(OmegaConf.to_yaml(self.cfg, resolve=True))

        self.accelerator.project_configuration.project_dir = str(self.proj_dir)
        self.accelerator.project_configuration.logging_dir = str(self.proj_dir / "tensorboard")
        if self.accelerator.is_main_process and not self.train_cfg.debug:
            self.accelerator.init_trackers(
                "rain_ts_diffusion",
                config={},  # OmegaConf.to_container(self.cfg, resolve=True),
            )
        return log_file

    def log_msg(self, msg: str, level: str = "info", only_rank_zero: bool = True) -> None:
        fn = getattr(logger, level.lower())
        if only_rank_zero:
            if self.accelerator.is_main_process:
                fn(msg)
        else:
            with self.accelerator.main_process_first():
                fn(f"rank-{self.accelerator.process_index} | {msg}")

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

    def _resolve_diffusion_mode(self) -> str:
        configured = str(self.train_cfg.diffusion.get("scheduler_type", "auto")).lower()
        if configured in {"gaussian", "ddpm"}:
            return "gaussian"
        if configured in {"fm", "flow", "flow_match"}:
            return "fm"
        if configured not in {"", "auto"}:
            raise ValueError(f"Unsupported train.diffusion.scheduler_type={configured}")

        raw_target = str(self.train_cfg.get("prediction_target", "epsilon")).lower()
        if raw_target in {"flow", "flow_prediction"}:
            return "fm"
        return "gaussian"

    def _resolve_prediction_target(self) -> str:
        configured = str(self.train_cfg.get("prediction_target", "epsilon")).lower()
        if configured in {"epsilon", "eps"}:
            configured = "noise"
        if configured == "flow_prediction":
            configured = "flow"

        if self.diffusion_mode == "gaussian":
            if configured not in {"noise", "x0"}:
                raise ValueError(
                    "Gaussian scheduler requires prediction_target in ['epsilon'/'noise', 'x0'], "
                    f"got {self.train_cfg.get('prediction_target')}"
                )
            return configured

        if configured not in {"flow", "noise", "x0"}:
            raise ValueError(
                "Flow-Matching scheduler requires prediction_target in ['flow', 'noise', 'x0'], "
                f"got {self.train_cfg.get('prediction_target')}"
            )
        return configured

    def _build_noise_scheduler(self) -> GaussianDiffusionScheduler | FlowMatchScheduler:
        num_train_timesteps = int(self.train_cfg.diffusion.num_train_timesteps)
        if self.diffusion_mode == "gaussian":
            return GaussianDiffusionScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=float(self.train_cfg.diffusion.beta_start),
                beta_end=float(self.train_cfg.diffusion.beta_end),
                device=self.device,
            )

        fm_scheduler = FlowMatchScheduler(
            num_inference_steps=num_train_timesteps,
            num_train_timesteps=num_train_timesteps,
            shift=float(self.train_cfg.diffusion.get("fm_shift", 3.0)),
            sigma_max=float(self.train_cfg.diffusion.get("fm_sigma_max", 1.0)),
            sigma_min=float(self.train_cfg.diffusion.get("fm_sigma_min", 0.003 / 1.002)),
            inverse_timesteps=bool(self.train_cfg.diffusion.get("fm_inverse_timesteps", False)),
            extra_one_step=bool(self.train_cfg.diffusion.get("fm_extra_one_step", False)),
            reverse_sigmas=bool(self.train_cfg.diffusion.get("fm_reverse_sigmas", False)),
        )
        fm_scheduler.set_timesteps(num_inference_steps=num_train_timesteps, training=True)
        return fm_scheduler

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
        """
        Canonicalize tensor to [B,C,T,H,W].
        Supported inputs:
        - [B,T,H,W]             (single-channel)
        - [B,C,T,H,W]
        - [B,T,C,H,W]
        """
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
        return denormalize_rain_linear(
            rain,
            mean=self.rain_norm_mean,
            std=self.rain_norm_std,
        )

    @staticmethod
    def _flatten_bcthw_to_btchw(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        if x.ndim != 5:
            raise ValueError(f"Expected [B,C,T,H,W], got {tuple(x.shape)}")
        b, c, t, h, w = x.shape
        x_bt = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        return x_bt, b, t

    @staticmethod
    def _unflatten_btchw_to_bcthw(x_bt: torch.Tensor, batch: int, frames: int) -> torch.Tensor:
        if x_bt.ndim != 4:
            raise ValueError(f"Expected [BT,C,H,W], got {tuple(x_bt.shape)}")
        bt, channels, height, width = x_bt.shape
        if bt != batch * frames:
            raise ValueError(f"Invalid BT shape={bt}, expected {batch * frames}")
        return x_bt.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)

    @staticmethod
    def _expand_timestep_to_bt(
        timestep: torch.Tensor,
        batch: int,
        frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        t = timestep.to(device=device)
        if t.ndim == 0:
            t = t.expand(batch * frames)
        elif t.ndim == 1:
            if t.numel() == 1:
                t = t.expand(batch * frames)
            elif t.numel() == batch:
                t = t[:, None].expand(batch, frames).reshape(batch * frames)
            elif t.numel() == batch * frames:
                t = t.reshape(batch * frames)
            else:
                raise ValueError(
                    f"Invalid timestep shape {tuple(t.shape)} for batch={batch}, frames={frames}"
                )
        elif t.ndim == 2:
            if t.shape != (batch, frames):
                raise ValueError(
                    f"Invalid timestep shape {tuple(t.shape)}, expected {(batch, frames)}"
                )
            t = t.reshape(batch * frames)
        else:
            raise ValueError(f"Unsupported timestep ndim={t.ndim}")
        return t.to(dtype=dtype)

    def _scheduler_add_noise(self, clean: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        if self.diffusion_mode == "gaussian":
            return self.noise_schedule.add_noise(clean, noise, timestep)
        if not isinstance(self.noise_schedule, FlowMatchScheduler):
            raise TypeError(f"Expected FlowMatchScheduler, got {type(self.noise_schedule).__name__}")

        clean_bt, batch, frames = self._flatten_bcthw_to_btchw(clean)
        noise_bt, _, _ = self._flatten_bcthw_to_btchw(noise)
        timestep_bt = self._expand_timestep_to_bt(
            timestep=timestep,
            batch=batch,
            frames=frames,
            device=clean.device,
            dtype=torch.float32,
        )
        noisy_bt = self.noise_schedule.add_noise(clean_bt, noise_bt, timestep_bt)
        return self._unflatten_btchw_to_bcthw(noisy_bt, batch=batch, frames=frames)

    def _form_teacher_forcing_batch(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
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

        bsz = radar_past.shape[0]
        n_future = radar_future.shape[2]
        if n_future <= 0:
            raise ValueError("n_future must be > 0 for diffusion training.")

        tf_cfg = self.train_cfg.teacher_forcing
        target_mode = str(tf_cfg.get("target_mode", "next_frame")).lower()
        if target_mode == "next_frame":
            target_idx = 0
            target_frames = 1
        elif target_mode == "random_frame":
            target_idx = int(torch.randint(0, n_future, (1,), device=self.device).item())
            target_frames = 1
        elif target_mode == "block":
            target_idx = 0
            configured_block = tf_cfg.get("block_size", None)
            target_frames = n_future if configured_block is None else int(configured_block)
            if target_frames <= 0 or target_frames > n_future:
                raise ValueError(
                    f"teacher_forcing.block_size must be in [1, {n_future}] for block mode, got {target_frames}."
                )
        else:
            raise ValueError(
                "teacher_forcing.target_mode must be one of "
                f"['next_frame', 'random_frame', 'block'], got {target_mode}."
            )

        if target_idx > 0:
            radar_context = torch.cat([radar_past, radar_future[:, :, :target_idx]], dim=2)
            satellite_context = torch.cat([satellite_past, satellite_future[:, :, :target_idx]], dim=2)
            rain_context = torch.cat([rain_past, rain_future[:, :, :target_idx]], dim=2)
        else:
            radar_context, satellite_context, rain_context = radar_past, satellite_past, rain_past

        target_end = target_idx + target_frames
        radar_target = radar_future[:, :, target_idx:target_end]
        satellite_target = satellite_future[:, :, target_idx:target_end]
        rain_target = rain_future[:, :, target_idx:target_end]
        x0_target = self._merge_modalities(radar_target, satellite_target, rain_target)

        t_shape = (bsz, target_frames) if target_frames > 1 else (bsz,)
        t = torch.randint(low=self.min_t, high=self.max_t + 1, size=t_shape, device=self.device, dtype=torch.long)
        target_noise = torch.randn_like(x0_target)
        noisy_target = self._scheduler_add_noise(x0_target, target_noise, t)

        # Optional: add tiny noise to context to reduce train-test mismatch.
        context_max_t = int(self.train_cfg.teacher_forcing.context_noise_max_timestep)
        context = self._merge_modalities(radar_context, satellite_context, rain_context)
        context_t = torch.zeros((bsz, context.shape[2]), device=self.device, dtype=torch.long)
        if context_max_t > 0 and context.shape[2] > 0:
            context_t = torch.randint(
                low=0,
                high=context_max_t + 1,
                size=(bsz, context.shape[2]),
                device=self.device,
                dtype=torch.long,
            )
            context_noise = torch.randn_like(context)
            context = self._scheduler_add_noise(context, context_noise, context_t)

        target_noise_dict = self._split_modalities(target_noise)
        target_x0_dict = self._split_modalities(x0_target)
        aux = {
            "target_mode": target_mode,
            "target_idx": target_idx,
            "target_frames": target_frames,
            "context_frames": int(context.shape[2]),
            "t_mean": float(t.float().mean().item()),
        }
        return context, noisy_target, context_t, t, target_noise_dict, {**aux, "target_x0": target_x0_dict}

    def _diffusion_loss(
        self,
        pred: dict[str, torch.Tensor],
        target_noise: dict[str, torch.Tensor],
        target_x0: dict[str, torch.Tensor],
        target_t: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pred_tensor = self._merge_modalities(pred["radar"], pred["satellite"], pred["rain"])
        target_noise_tensor = self._merge_modalities(
            target_noise["radar"], target_noise["satellite"], target_noise["rain"]
        )
        target_x0_tensor = self._merge_modalities(target_x0["radar"], target_x0["satellite"], target_x0["rain"])

        if self.prediction_target == "noise":
            ref_tensor = target_noise_tensor
        elif self.prediction_target == "x0":
            ref_tensor = target_x0_tensor
        elif self.prediction_target == "flow":
            if not isinstance(self.noise_schedule, FlowMatchScheduler):
                raise TypeError("prediction_target='flow' requires FlowMatchScheduler.")
            x0_bt, batch, frames = self._flatten_bcthw_to_btchw(target_x0_tensor)
            noise_bt, _, _ = self._flatten_bcthw_to_btchw(target_noise_tensor)
            timestep_bt = self._expand_timestep_to_bt(
                timestep=target_t,
                batch=batch,
                frames=frames,
                device=target_x0_tensor.device,
                dtype=torch.float32,
            )
            flow_bt = self.noise_schedule.training_target(sample=x0_bt, noise=noise_bt, timestep=timestep_bt)
            ref_tensor = self._unflatten_btchw_to_bcthw(flow_bt, batch=batch, frames=frames)
        else:
            raise ValueError(f"Unsupported prediction_target={self.prediction_target}")

        loss_map = F.mse_loss(pred_tensor, ref_tensor, reduction="none")
        if isinstance(self.noise_schedule, FlowMatchScheduler) and hasattr(self.noise_schedule, "training_weight"):
            batch = loss_map.shape[0]
            frames = loss_map.shape[2]
            timestep_bt = self._expand_timestep_to_bt(
                timestep=target_t,
                batch=batch,
                frames=frames,
                device=loss_map.device,
                dtype=torch.float32,
            )
            weight_bt = self.noise_schedule.training_weight(timestep_bt).to(device=loss_map.device, dtype=loss_map.dtype)
            weight = weight_bt.reshape(batch, frames, 1, 1, 1).permute(0, 2, 1, 3, 4)
            loss_map = loss_map * weight

        lw = self.train_cfg.loss_weights
        l_radar = loss_map[:, : self.radar_c].mean()
        l_satellite = loss_map[:, self.radar_c : self.radar_c + self.satellite_c].mean()
        l_rain = loss_map[:, self.radar_c + self.satellite_c :].mean()
        loss = float(lw.radar) * l_radar + float(lw.satellite) * l_satellite + float(lw.rain) * l_rain
        logs = {
            "loss": loss.detach(),
            "loss/radar": l_radar.detach(),
            "loss/satellite": l_satellite.detach(),
            "loss/rain": l_rain.detach(),
        }
        return loss, logs

    def train_step(self, batch: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], bool]:
        context, noisy_target, context_t, target_t, target_noise_dict, aux = self._form_teacher_forcing_batch(batch)
        target_x0_dict = aux["target_x0"]
        target_frames = int(aux["target_frames"])

        with self.accelerator.accumulate(self.model):
            with self.accelerator.autocast():
                pred = self.model.forward_ar(
                    context_x=context,
                    target_x=noisy_target,
                    context_timestep=context_t,
                    target_timestep=target_t,
                    predict_frames=target_frames,
                    strict_target_isolation=bool(self.train_cfg.strict_target_isolation),
                    return_modality_dict=True,
                )
                loss, logs = self._diffusion_loss(
                    pred,
                    target_noise=target_noise_dict,
                    target_x0=target_x0_dict,
                    target_t=target_t,
                )

            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients and float(self.train_cfg.max_grad_norm) > 0:
                self.accelerator.clip_grad_norm_(self.model.parameters(), float(self.train_cfg.max_grad_norm))
            if self.accelerator.sync_gradients:
                self.optim.step()
                self.sched.step()
                self.optim.zero_grad(set_to_none=True)

        did_step = bool(self.accelerator.sync_gradients)
        if did_step:
            if self.ema_model is not None:
                self.ema_model.update()
            self.global_step += 1
        logs["meta/target_mode"] = torch.tensor(
            {"next_frame": 0.0, "random_frame": 1.0, "block": 2.0}[str(aux["target_mode"])], device=self.device
        )
        logs["meta/target_idx"] = torch.tensor(float(aux["target_idx"]), device=self.device)
        logs["meta/target_frames"] = torch.tensor(float(aux["target_frames"]), device=self.device)
        logs["meta/context_frames"] = torch.tensor(float(aux["context_frames"]), device=self.device)
        logs["meta/t_mean"] = torch.tensor(float(aux["t_mean"]), device=self.device)
        return logs, did_step

    @torch.no_grad()
    def val_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        context, noisy_target, context_t, target_t, target_noise_dict, aux = self._form_teacher_forcing_batch(batch)
        target_x0_dict = aux["target_x0"]
        target_frames = int(aux["target_frames"])
        with self.accelerator.autocast():
            pred = self.model.forward_ar(
                context_x=context,
                target_x=noisy_target,
                context_timestep=context_t,
                target_timestep=target_t,
                predict_frames=target_frames,
                strict_target_isolation=bool(self.train_cfg.strict_target_isolation),
                return_modality_dict=True,
            )
            _, logs = self._diffusion_loss(
                pred,
                target_noise=target_noise_dict,
                target_x0=target_x0_dict,
                target_t=target_t,
            )
        return logs

    def _prepare_val_inference_batch(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
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
        context_t = torch.zeros((context.shape[0], context.shape[2]), device=self.device, dtype=torch.long)
        target = {
            "radar": radar_future,
            "satellite": satellite_future,
            "rain": rain_future,
        }
        return context, context_t, target

    @staticmethod
    def _psnr_ssim_sums(pred: torch.Tensor, target: torch.Tensor, data_range: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def _default_inference_sampler(self) -> str:
        if self.diffusion_mode == "fm":
            return "fm_dpmpp"
        return "ddpm"

    def _build_fm_inference_scheduler(self, sampler: str) -> FlowMatchScheduler | FlowDPMSolverMultistepScheduler | FlowUniPCMultistepScheduler:
        num_train_timesteps = int(self.train_cfg.diffusion.num_train_timesteps)
        shift = float(self.train_cfg.diffusion.get("fm_shift", 3.0))

        if sampler in {"fm", "flow", "fm_euler", "euler"}:
            return FlowMatchScheduler(
                num_inference_steps=int(self.val_cfg.get("num_inference_steps", 50)),
                num_train_timesteps=num_train_timesteps,
                shift=shift,
                sigma_max=float(self.train_cfg.diffusion.get("fm_sigma_max", 1.0)),
                sigma_min=float(self.train_cfg.diffusion.get("fm_sigma_min", 0.003 / 1.002)),
                inverse_timesteps=bool(self.train_cfg.diffusion.get("fm_inverse_timesteps", False)),
                extra_one_step=bool(self.train_cfg.diffusion.get("fm_extra_one_step", False)),
                reverse_sigmas=bool(self.train_cfg.diffusion.get("fm_reverse_sigmas", False)),
            )
        if sampler in {"fm_dpmpp", "dpmpp", "dpm_solver", "dpm"}:
            return FlowDPMSolverMultistepScheduler(
                num_train_timesteps=num_train_timesteps,
                prediction_type="flow_prediction",
                shift=shift,
            )
        if sampler in {"fm_unipc", "unipc"}:
            return FlowUniPCMultistepScheduler(
                num_train_timesteps=num_train_timesteps,
                prediction_type="flow_prediction",
                shift=shift,
            )
        raise ValueError(f"Unsupported FM sampler={sampler}")

    @staticmethod
    def _extract_solver_sample(step_output: Any) -> torch.Tensor:
        if isinstance(step_output, tuple):
            return step_output[0]
        if hasattr(step_output, "prev_sample"):
            return step_output.prev_sample
        if torch.is_tensor(step_output):
            return step_output
        raise TypeError(f"Unsupported scheduler step output type: {type(step_output).__name__}")

    @staticmethod
    def _scheduler_sigma_from_timestep(
        scheduler: FlowMatchScheduler | FlowDPMSolverMultistepScheduler | FlowUniPCMultistepScheduler,
        timestep: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(scheduler, FlowMatchScheduler):
            return scheduler.sigma_from_timestep(timestep=timestep, device=device)
        timestep_flat = timestep.reshape(-1).to(device=device, dtype=torch.float32)
        solver_timesteps = scheduler.timesteps.to(device=device, dtype=torch.float32)
        timestep_id = torch.argmin((solver_timesteps.unsqueeze(0) - timestep_flat.unsqueeze(1)).abs(), dim=1)
        sigmas = scheduler.sigmas.to(device=device, dtype=torch.float32)
        return sigmas[timestep_id].reshape(-1, 1, 1, 1)

    def _x0_to_flow_pred(
        self,
        xt: torch.Tensor,
        x0: torch.Tensor,
        scheduler: FlowMatchScheduler | FlowDPMSolverMultistepScheduler | FlowUniPCMultistepScheduler,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        if xt.shape != x0.shape:
            raise ValueError(f"xt/x0 shape mismatch: {tuple(xt.shape)} vs {tuple(x0.shape)}")
        xt_bt, batch, frames = self._flatten_bcthw_to_btchw(xt)
        x0_bt, _, _ = self._flatten_bcthw_to_btchw(x0)
        timestep_bt = self._expand_timestep_to_bt(
            timestep=timestep,
            batch=batch,
            frames=frames,
            device=xt.device,
            dtype=torch.float32,
        )
        sigma_bt = self._scheduler_sigma_from_timestep(scheduler=scheduler, timestep=timestep_bt, device=xt.device)
        flow_bt = (xt_bt - x0_bt) / sigma_bt.clamp(min=1e-12)
        return self._unflatten_btchw_to_bcthw(flow_bt, batch=batch, frames=frames)

    def _flow_pred_to_x0(
        self,
        xt: torch.Tensor,
        flow_pred: torch.Tensor,
        scheduler: FlowMatchScheduler | FlowDPMSolverMultistepScheduler | FlowUniPCMultistepScheduler,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        if xt.shape != flow_pred.shape:
            raise ValueError(f"xt/flow shape mismatch: {tuple(xt.shape)} vs {tuple(flow_pred.shape)}")
        xt_bt, batch, frames = self._flatten_bcthw_to_btchw(xt)
        flow_bt, _, _ = self._flatten_bcthw_to_btchw(flow_pred)
        timestep_bt = self._expand_timestep_to_bt(
            timestep=timestep,
            batch=batch,
            frames=frames,
            device=xt.device,
            dtype=torch.float32,
        )
        sigma_bt = self._scheduler_sigma_from_timestep(scheduler=scheduler, timestep=timestep_bt, device=xt.device)
        x0_bt = xt_bt - sigma_bt * flow_bt
        return self._unflatten_btchw_to_bcthw(x0_bt, batch=batch, frames=frames)

    def _noise_to_x0_fm(
        self,
        xt: torch.Tensor,
        noise_pred: torch.Tensor,
        scheduler: FlowMatchScheduler | FlowDPMSolverMultistepScheduler | FlowUniPCMultistepScheduler,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        if xt.shape != noise_pred.shape:
            raise ValueError(f"xt/noise shape mismatch: {tuple(xt.shape)} vs {tuple(noise_pred.shape)}")
        xt_bt, batch, frames = self._flatten_bcthw_to_btchw(xt)
        noise_bt, _, _ = self._flatten_bcthw_to_btchw(noise_pred)
        timestep_bt = self._expand_timestep_to_bt(
            timestep=timestep,
            batch=batch,
            frames=frames,
            device=xt.device,
            dtype=torch.float32,
        )
        sigma_bt = self._scheduler_sigma_from_timestep(scheduler=scheduler, timestep=timestep_bt, device=xt.device)
        x0_bt = (xt_bt - sigma_bt * noise_bt) / (1.0 - sigma_bt).clamp(min=1e-12)
        return self._unflatten_btchw_to_bcthw(x0_bt, batch=batch, frames=frames)

    def _fm_denoise(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,
        context_t: torch.Tensor,
        target_frames: int,
        sampler: str,
        num_inference_steps: int,
        clip_x0: bool,
        show_progress: bool = False,
    ) -> torch.Tensor:
        scheduler = self._build_fm_inference_scheduler(sampler=sampler)
        if isinstance(scheduler, FlowMatchScheduler):
            scheduler.set_timesteps(num_inference_steps=num_inference_steps, training=False)
        else:
            scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=latents.device)

        batch_size = latents.shape[0]
        if show_progress and self.accelerator.is_main_process:
            timestep_iter = tqdm(
                scheduler.timesteps,
                desc=f"val-sample[{sampler}]",
                leave=False,
                dynamic_ncols=True,
            )
        else:
            timestep_iter = scheduler.timesteps

        for current_timestep in timestep_iter:
            target_timestep = torch.full(
                (batch_size, target_frames),
                fill_value=float(current_timestep),
                device=latents.device,
                dtype=torch.float32,
            )
            with self.accelerator.autocast():
                pred = self.model.forward_ar(
                    context_x=context,
                    target_x=latents,
                    context_timestep=context_t,
                    target_timestep=target_timestep,
                    predict_frames=target_frames,
                    strict_target_isolation=bool(self.train_cfg.strict_target_isolation),
                    return_modality_dict=True,
                )
            pred_tensor = self._merge_modalities(pred["radar"], pred["satellite"], pred["rain"])

            if self.prediction_target == "flow":
                flow_pred = pred_tensor
            elif self.prediction_target == "x0":
                x0_pred = pred_tensor.clamp(0.0, 1.0) if clip_x0 else pred_tensor
                flow_pred = self._x0_to_flow_pred(
                    xt=latents,
                    x0=x0_pred,
                    scheduler=scheduler,
                    timestep=target_timestep,
                )
            elif self.prediction_target == "noise":
                x0_pred = self._noise_to_x0_fm(
                    xt=latents,
                    noise_pred=pred_tensor,
                    scheduler=scheduler,
                    timestep=target_timestep,
                )
                if clip_x0:
                    x0_pred = x0_pred.clamp(0.0, 1.0)
                flow_pred = self._x0_to_flow_pred(
                    xt=latents,
                    x0=x0_pred,
                    scheduler=scheduler,
                    timestep=target_timestep,
                )
            else:
                raise ValueError(f"Unsupported prediction_target={self.prediction_target}")

            if clip_x0 and self.prediction_target == "flow":
                x0_pred = self._flow_pred_to_x0(
                    xt=latents,
                    flow_pred=flow_pred,
                    scheduler=scheduler,
                    timestep=target_timestep,
                ).clamp(0.0, 1.0)
                flow_pred = self._x0_to_flow_pred(
                    xt=latents,
                    x0=x0_pred,
                    scheduler=scheduler,
                    timestep=target_timestep,
                )

            if isinstance(scheduler, FlowMatchScheduler):
                latents_bt, batch, frames = self._flatten_bcthw_to_btchw(latents)
                flow_bt, _, _ = self._flatten_bcthw_to_btchw(flow_pred)
                timestep_bt = self._expand_timestep_to_bt(
                    timestep=target_timestep,
                    batch=batch,
                    frames=frames,
                    device=latents.device,
                    dtype=torch.float32,
                )
                next_bt = scheduler.step(model_output=flow_bt, timestep=timestep_bt, sample=latents_bt)
                latents = self._unflatten_btchw_to_bcthw(next_bt, batch=batch, frames=frames)
                continue

            step_output = scheduler.step(
                model_output=flow_pred,
                timestep=current_timestep,
                sample=latents,
                return_dict=False,
            )
            latents = self._extract_solver_sample(step_output)

        return latents

    @torch.no_grad()
    def _val_inference_step(
        self,
        batch: dict[str, torch.Tensor],
        show_sample_progress: bool = False,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
        context, context_t, target = self._prepare_val_inference_batch(batch)
        target_frames = int(target["rain"].shape[2])
        if target_frames <= 0:
            raise ValueError("Validation target frames must be > 0.")

        total_c = self.radar_c + self.satellite_c + self.rain_c
        _, _, _, h, w = target["rain"].shape
        bsz = context.shape[0]
        init_latents = torch.randn((bsz, total_c, target_frames, h, w), device=self.device, dtype=torch.float32)

        sampler = str(self.val_cfg.get("inference_sampler", self._default_inference_sampler())).lower()

        if self.diffusion_mode == "gaussian":
            if not isinstance(self.noise_schedule, GaussianDiffusionScheduler):
                raise TypeError(f"Expected GaussianDiffusionScheduler, got {type(self.noise_schedule).__name__}")

            def predict_fn(current_latents: torch.Tensor, target_timestep: torch.Tensor) -> torch.Tensor:
                with self.accelerator.autocast():
                    pred = self.model.forward_ar(
                        context_x=context,
                        target_x=current_latents,
                        context_timestep=context_t,
                        target_timestep=target_timestep,
                        predict_frames=target_frames,
                        strict_target_isolation=bool(self.train_cfg.strict_target_isolation),
                        return_modality_dict=True,
                    )
                return self._merge_modalities(pred["radar"], pred["satellite"], pred["rain"])

            gaussian_prediction_type = "epsilon" if self.prediction_target == "noise" else "x0"
            latents = self.noise_schedule.denoise(
                latents=init_latents,
                predict_fn=predict_fn,
                prediction_type=gaussian_prediction_type,
                sampler=sampler,
                num_inference_steps=int(self.val_cfg.get("num_inference_steps", 50)),
                min_timestep=self.min_t,
                max_timestep=self.max_t,
                clip_x0=bool(self.val_cfg.get("clip_pred_x0", False)),
            )
        else:
            latents = self._fm_denoise(
                latents=init_latents,
                context=context,
                context_t=context_t,
                target_frames=target_frames,
                sampler=sampler,
                num_inference_steps=int(self.val_cfg.get("num_inference_steps", 50)),
                clip_x0=bool(self.val_cfg.get("clip_pred_x0", False)),
                show_progress=show_sample_progress,
            )

        pred_target = self._split_modalities(latents)
        lw = self.train_cfg.loss_weights
        infer_loss = (
            float(lw.radar) * F.mse_loss(pred_target["radar"], target["radar"])
            + float(lw.satellite) * F.mse_loss(pred_target["satellite"], target["satellite"])
            + float(lw.rain) * F.mse_loss(pred_target["rain"], target["rain"])
        )
        return pred_target, target, infer_loss.detach()

    def _resolve_visual_frame_indices(self, total_frames: int) -> list[int]:
        configured = self.val_cfg.get("viz_frame_indices", [0, -1])
        try:
            configured_list = list(configured)
        except TypeError:
            configured_list = [configured]

        flattened: list[object] = []
        for item in configured_list:
            if isinstance(item, (list, tuple)):
                flattened.extend(list(item))
                continue
            if isinstance(item, (str, bytes)):
                flattened.append(item)
                continue
            try:
                nested = list(item)
            except TypeError:
                flattened.append(item)
            else:
                flattened.extend(nested)

        resolved: list[int] = []
        for idx_value in flattened:
            try:
                idx_int = int(idx_value)
            except (TypeError, ValueError):
                continue
            if idx_int < 0:
                idx_int = total_frames + idx_int
            if 0 <= idx_int < total_frames:
                resolved.append(idx_int)
        if len(resolved) == 0:
            resolved.append(0)
        unique_sorted = sorted(set(resolved))
        return unique_sorted

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
    def _build_compare_strip(pred_img: object, gt_img: object, err_img: object) -> Image.Image:
        pred = RainTSDiffusionTrainer._ensure_rgb_uint8(pred_img)
        gt = RainTSDiffusionTrainer._ensure_rgb_uint8(gt_img)
        err = RainTSDiffusionTrainer._ensure_rgb_uint8(err_img)
        height = int(pred.shape[0])
        spacer = torch.full((height, 6, 3), 255, dtype=torch.uint8)
        strip = torch.cat([pred, spacer, gt, spacer, err], dim=1).cpu().numpy()
        return Image.fromarray(strip)

    def _save_val_visualizations(self, pred_target: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> None:
        if not bool(self.val_cfg.get("save_visuals", True)):
            return
        if not self.accelerator.is_main_process:
            return
        sample_idx = int(self.val_cfg.get("viz_sample_index", 0))
        viz_dir = self.proj_dir / "val_viz" / f"step_{self.global_step:08d}"
        viz_dir.mkdir(parents=True, exist_ok=True)

        modal_pairs = [
            ("radar", pred_target["radar"], target["radar"]),
            ("satellite", pred_target["satellite"], target["satellite"]),
            ("rain", pred_target["rain"], target["rain"]),
        ]

        for modality_name, pred_tensor, gt_tensor in modal_pairs:
            pred_cpu = pred_tensor.detach().float().cpu().clamp_min(0.0)
            gt_cpu = gt_tensor.detach().float().cpu().clamp_min(0.0)
            if pred_cpu.ndim != 5:
                raise ValueError(f"Expected [B,C,T,H,W], got {tuple(pred_cpu.shape)}")

            batch_size = pred_cpu.shape[0]
            total_frames = pred_cpu.shape[2]
            if batch_size <= 0 or total_frames <= 0:
                continue
            sample_id = max(0, min(sample_idx, batch_size - 1))
            frame_indices = self._resolve_visual_frame_indices(total_frames=total_frames)

            for frame_idx in frame_indices:
                pred_frame = pred_cpu[sample_id, :, frame_idx]
                gt_frame = gt_cpu[sample_id, :, frame_idx]
                err_frame = (pred_frame - gt_frame).abs()

                pred_img = plot_any_modality(pred_frame, modality_name=modality_name, to_PIL=False)
                gt_img = plot_any_modality(gt_frame, modality_name=modality_name, to_PIL=False)
                err_img = plot_any_modality(err_frame, modality_name=modality_name, to_PIL=False)
                strip = self._build_compare_strip(pred_img=pred_img, gt_img=gt_img, err_img=err_img)
                out_path = viz_dir / f"{modality_name}_sample{sample_id}_frame{frame_idx}.png"
                strip.save(out_path)

    def _save_checkpoint(self) -> None:
        ckpt_dir = self.proj_dir / f"checkpoint-{self.global_step:08d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.accelerator.save_state(str(ckpt_dir))
        if self.accelerator.is_main_process:
            (ckpt_dir / "meta.json").write_text(json.dumps({"global_step": self.global_step}, indent=2))
            if self.ema_model is not None:
                ema_dir = ckpt_dir / "ema"
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
            ema_path = resume_dir / "ema" / "ema.pt"
            if ema_path.exists():
                self.ema_model.load_state_dict(torch.load(ema_path, map_location=self.device))
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
        msg = " | ".join([f"{k}: {v:.6f}" for k, v in scalar_logs.items()])
        self.log_msg(f"[Train][{self.global_step}/{self.train_cfg.max_steps}] {msg}")
        if not self.train_cfg.debug:
            self.accelerator.log(scalar_logs, step=self.global_step)

    @torch.no_grad()
    def _run_val(self) -> None:
        max_iters = int(self.val_cfg.max_val_iters)
        if max_iters <= 0:
            return
        self.model.eval()
        self.log_msg(
            f"[Val] start at step={self.global_step}, max_val_iters={max_iters}, "
            f"num_inference_steps={int(self.val_cfg.get('num_inference_steps', 50))}",
        )
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
        first_pred_target: dict[str, torch.Tensor] | None = None
        first_target: dict[str, torch.Tensor] | None = None

        iterator = iter(self.val_dataloader)
        if self.accelerator.is_main_process:
            val_iter = tqdm(range(max_iters), desc=f"val[{self.global_step}]", leave=False, dynamic_ncols=True)
        else:
            val_iter = range(max_iters)

        for val_iter_idx in val_iter:
            try:
                batch = next(iterator)
            except StopIteration:
                break
            logs = self.val_step(batch)
            pred_target, target, infer_loss = self._val_inference_step(
                batch,
                show_sample_progress=bool(self.val_cfg.get("show_sample_progress", True)) and val_iter_idx == 0,
            )
            if first_pred_target is None:
                first_pred_target = {k: v.detach() for k, v in pred_target.items()}
                first_target = {k: v.detach() for k, v in target.items()}
            loss_sum += logs["loss"].detach().float()
            infer_loss_sum += infer_loss
            batch_count += 1.0

            if self.accelerator.is_main_process and hasattr(val_iter, "set_postfix"):
                val_iter.set_postfix(
                    {
                        "loss": f"{float(logs['loss'].detach().item()):.4f}",
                        "infer": f"{float(infer_loss.item()):.4f}",
                    }
                )

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

        val_loss = float((loss_sum / batch_count.clamp_min(1.0)).item())
        val_infer_loss = float((infer_loss_sum / batch_count.clamp_min(1.0)).item())
        radar_psnr = float((radar_psnr_sum / radar_count.clamp_min(1.0)).item())
        radar_ssim = float((radar_ssim_sum / radar_count.clamp_min(1.0)).item())
        satellite_psnr = float((satellite_psnr_sum / satellite_count.clamp_min(1.0)).item())
        satellite_ssim = float((satellite_ssim_sum / satellite_count.clamp_min(1.0)).item())

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
        self.log_msg(metric_msg)
        if not self.train_cfg.debug:
            self.accelerator.log(
                {
                    "val/loss": val_loss,
                    "val/infer_loss": val_infer_loss,
                    "val/radar_psnr": radar_psnr,
                    "val/radar_ssim": radar_ssim,
                    "val/satellite_psnr": satellite_psnr,
                    "val/satellite_ssim": satellite_ssim,
                    **csi_logs,
                },
                step=self.global_step,
            )
        if first_pred_target is not None and first_target is not None:
            self._save_val_visualizations(first_pred_target, first_target)
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
        self.train()


@hydra.main(
    config_path="../config/ts_rain_train",
    config_name="rain_trainer_ts_diffusion",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    catcher = logger.catch if PartialState().is_main_process else nullcontext
    with catcher():
        trainer = RainTSDiffusionTrainer(cfg)
        trainer.run()


if __name__ == "__main__":
    main()
