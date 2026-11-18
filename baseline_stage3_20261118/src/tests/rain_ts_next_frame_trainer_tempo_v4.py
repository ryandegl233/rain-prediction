"""
TEMPO-Rain v4 training driver (Accelerate) for multimodal precipitation nowcasting.

This trainer keeps the existing causal Patch Transformer runnable while adding:
- staged self-rollout curricula with independently scheduled length, forcing, and weight
- differentiable multi-threshold Soft-CSI and multi-scale Soft-FSS objectives
- a lightweight spectral-amplitude objective for sharper deterministic forecasts
- contracts for TEMPO model outputs (event logits, direct anchors,
  transport/source decomposition, and flow fields)
- strict configuration validation and per-lead operational verification

The transport/evolution network is implemented by the v4 causal Patch Transformer;
the output contracts below supervise its physical decomposition during rollout.
"""

import json
import math
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

from src.dataset.rain_ts_litdata_tempo_v4 import denormalize_rain_linear
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
        self._validate_training_config_contract()

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
        else:
            self.model, self.optim, self.train_dataloader, self.val_dataloader, self.sched = self.accelerator.prepare(
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
        self._last_rollout_stage_name: str | None = None
        self.log_msg(f"Model channels: radar={self.radar_c}, satellite={self.satellite_c}, rain={self.rain_c}")
        self.log_msg("Objective: autoregressive next-token style prediction")
        if self.use_gan:
            self.log_msg("GAN training enabled for next-frame trainer.")

    def _validate_training_config_contract(self) -> None:
        """Fail fast on configuration keys that the old trainer silently ignored."""
        loss_cfg = self.train_cfg.get("loss", {})
        next_cfg = self.train_cfg.get("next_pred", {})
        rollout_cfg = next_cfg.get("rollout_branch", {})

        misplaced: list[str] = []
        for key in ("sequence_loss_enabled", "sequence_context_frames"):
            if key in loss_cfg:
                misplaced.append(
                    f"train.loss.{key} is not supported; use train.next_pred.sequence_loss instead"
                )
        if "scheduled_sampling" in loss_cfg:
            misplaced.append(
                "train.loss.scheduled_sampling is not supported; use "
                "train.next_pred.rollout_branch.scheduled_sampling"
            )
        if misplaced:
            raise ValueError("Invalid training config:\n- " + "\n- ".join(misplaced))

        mode = str(loss_cfg.get("mode", "mse")).lower()
        if mode not in {"mse", "mae", "huber", "enhanced"}:
            raise ValueError(
                "train.loss.mode must be one of 'mse', 'mae', 'huber', or 'enhanced', "
                f"got {mode}"
            )

        # Keep the inline model contract synchronized with loss heads.  This is
        # checked before model/data construction so a stale YAML fails quickly.
        model_cfg = self.cfg.get("rain_prediction_model", {})
        model_aux_cfg = loss_cfg.get("model_auxiliary", {})
        if bool(model_aux_cfg.get("enabled", False)) and bool(model_aux_cfg.get("require_outputs", False)):
            if not bool(model_cfg.get("tempo_enabled", False)):
                raise ValueError(
                    "model_auxiliary.require_outputs=True requires "
                    "rain_prediction_model.tempo_enabled=True"
                )
        event_cfg = loss_cfg.get("rain_event_aux", {})
        if bool(event_cfg.get("enabled", False)) and str(event_cfg.get("head_mode", "auto")).lower() == "model":
            model_thresholds = [float(v) for v in model_cfg.get("tempo_event_thresholds", [])]
            loss_thresholds = [float(v) for v in event_cfg.get("thresholds", [])]
            if model_thresholds != loss_thresholds:
                raise ValueError(
                    "rain_prediction_model.tempo_event_thresholds must exactly match "
                    "train.loss.rain_event_aux.thresholds; "
                    f"got model={model_thresholds}, loss={loss_thresholds}"
                )
        model_max_frames = int(model_cfg.get("max_frames", 0))
        required_frames = int(self.dataset_cfg.get("n_past", 0)) + int(self.dataset_cfg.get("n_futures", 0))
        if model_max_frames < required_frames:
            raise ValueError(
                "rain_prediction_model.max_frames must cover n_past + n_futures; "
                f"got max_frames={model_max_frames}, required={required_frames}"
            )

        seq_cfg = next_cfg.get("sequence_loss", {})
        if bool(seq_cfg.get("enabled", False)) and bool(self.train_cfg.get("strict_target_isolation", False)):
            raise ValueError(
                "train.next_pred.sequence_loss.enabled=True requires "
                "train.strict_target_isolation=false"
            )

        rollout_mode = str(rollout_cfg.get("mode", "frame")).lower()
        if rollout_mode not in {"frame", "block"}:
            raise ValueError(
                "train.next_pred.rollout_branch.mode must be 'frame' or 'block', "
                f"got {rollout_mode}"
            )

        sampling_cfg = rollout_cfg.get("scheduled_sampling", {})
        if bool(sampling_cfg.get("enabled", False)):
            if rollout_mode != "frame":
                raise ValueError(
                    "Scheduled sampling requires train.next_pred.rollout_branch.mode='frame' "
                    "so feedback is selected at every rollout step."
                )
            start_prob = float(sampling_cfg.get("start_prob", 1.0))
            end_prob = float(sampling_cfg.get("end_prob", 0.0))
            if not 0.0 <= start_prob <= 1.0 or not 0.0 <= end_prob <= 1.0:
                raise ValueError(
                    "scheduled_sampling start_prob/end_prob must both be in [0, 1], "
                    f"got start={start_prob}, end={end_prob}"
                )
            modalities = [str(v) for v in sampling_cfg.get("modalities", ["radar", "satellite", "rain"])]
            invalid_modalities = sorted(set(modalities) - {"radar", "satellite", "rain"})
            if invalid_modalities:
                raise ValueError(
                    "scheduled_sampling.modalities contains invalid values: "
                    f"{invalid_modalities}"
                )

        stages_cfg = rollout_cfg.get("stages", [])
        if stages_cfg:
            if bool(rollout_cfg.get("curriculum", {}).get("enabled", False)):
                raise ValueError(
                    "train.next_pred.rollout_branch.stages and rollout_branch.curriculum "
                    "cannot both be enabled. Use stages for the TEMPO curriculum."
                )
            previous_start = -1
            for stage_idx, stage_cfg in enumerate(stages_cfg):
                start_step = int(stage_cfg.get("start_step", 0))
                end_step = int(stage_cfg.get("end_step", start_step))
                if start_step < 0 or end_step < start_step:
                    raise ValueError(
                        f"rollout stage {stage_idx} requires 0 <= start_step <= end_step, "
                        f"got {start_step}, {end_step}"
                    )
                if start_step < previous_start:
                    raise ValueError("rollout stages must be sorted by start_step.")
                previous_start = start_step
                start_frames = int(stage_cfg.get("start_frames", 1))
                end_frames = int(stage_cfg.get("end_frames", start_frames))
                if start_frames <= 0 or end_frames <= 0:
                    raise ValueError(
                        f"rollout stage {stage_idx} frame counts must be > 0, "
                        f"got {start_frames}, {end_frames}"
                    )
                start_prob = float(stage_cfg.get("start_prob", 0.0))
                end_prob = float(stage_cfg.get("end_prob", start_prob))
                if not 0.0 <= start_prob <= 1.0 or not 0.0 <= end_prob <= 1.0:
                    raise ValueError(
                        f"rollout stage {stage_idx} forcing probabilities must be in [0, 1], "
                        f"got {start_prob}, {end_prob}"
                    )
                start_weight = float(stage_cfg.get("start_weight", rollout_cfg.get("weight", 0.0)))
                end_weight = float(stage_cfg.get("end_weight", start_weight))
                if start_weight < 0 or end_weight < 0:
                    raise ValueError(
                        f"rollout stage {stage_idx} weights must be >= 0, "
                        f"got {start_weight}, {end_weight}"
                    )
                schedule = str(stage_cfg.get("schedule", "linear")).lower()
                if schedule not in {"linear", "cosine"}:
                    raise ValueError(
                        f"rollout stage {stage_idx} schedule must be linear/cosine, got {schedule}"
                    )

        gradient_leads_cfg = rollout_cfg.get("gradient_leads", [])
        if gradient_leads_cfg:
            gradient_leads = [int(value) for value in gradient_leads_cfg]
            if any(value <= 0 for value in gradient_leads):
                raise ValueError("rollout_branch.gradient_leads must contain positive lead indices.")
            max_future = int(self.dataset_cfg.get("n_futures", 0))
            if any(value > max_future for value in gradient_leads):
                raise ValueError(
                    "rollout_branch.gradient_leads cannot exceed dataset.n_futures, "
                    f"got leads={gradient_leads}, n_futures={max_future}."
                )

        for objective_name in ("soft_csi", "soft_fss", "spectral_amplitude", "model_auxiliary"):
            objective_cfg = loss_cfg.get(objective_name, {})
            if bool(objective_cfg.get("enabled", False)) and float(objective_cfg.get("weight", 0.0)) < 0:
                raise ValueError(f"train.loss.{objective_name}.weight must be >= 0.")
            apply_on = str(objective_cfg.get("apply_on", "all")).lower()
            if apply_on not in {"all", "rollout", "teacher"}:
                raise ValueError(
                    f"train.loss.{objective_name}.apply_on must be all/rollout/teacher, got {apply_on}"
                )

        model_aux_cfg = loss_cfg.get("model_auxiliary", {})
        for weight_name in (
            "direct_weight",
            "advection_weight",
            "transport_weight",
            "composition_weight",
            "source_sparsity_weight",
            "flow_smoothness_weight",
        ):
            if float(model_aux_cfg.get(weight_name, 0.0)) < 0:
                raise ValueError(
                    f"train.loss.model_auxiliary.{weight_name} must be >= 0."
                )

        threshold_space = str(loss_cfg.get("threshold_space", "model")).lower()
        if threshold_space not in {"model", "physical"}:
            raise ValueError(
                "train.loss.threshold_space must be 'model' or 'physical', "
                f"got {threshold_space}"
            )

        delta_cfg = rollout_cfg.get("delta_loss", {})
        rain_residual_cfg = loss_cfg.get("rain_residual", {})
        if bool(delta_cfg.get("enabled", False)) and not bool(rain_residual_cfg.get("enabled", False)):
            raise ValueError(
                "rollout_branch.delta_loss predicts rain increments and therefore requires "
                "train.loss.rain_residual.enabled=true"
            )

        diff_cfg = loss_cfg.get("temporal_diff", {})
        if "lambda_diff" in diff_cfg and "weight_init" not in diff_cfg and "weight_max" not in diff_cfg:
            self.log_msg(
                "Deprecated config train.loss.temporal_diff.lambda_diff detected. "
                "It will be used as a fixed temporal-difference weight; prefer weight_init/weight_max."
            )

        if bool(self.dataset_cfg.val.get("rain_ratio_filter_enabled", False)):
            self.log_msg(
                "Validation rain-ratio filtering is enabled. Report it as a conditional rainy-event "
                "validation set, not as operational all-weather performance."
            )
        if bool(self.dataset_cfg.val.get("shuffle", False)):
            self.log_msg("Validation loader shuffle=True makes checkpoint comparisons noisy; prefer shuffle=False.")
        if int(self.val_cfg.get("max_val_iters", 0)) <= 5:
            self.log_msg("val.max_val_iters<=5 is too small for stable CSI/FAR/POD estimates.")

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

        frame_patch_size = int(getattr(self.model, "frame_patch_size", 1))
        if frame_patch_size <= 0:
            raise ValueError(f"model.frame_patch_size must be > 0, got {frame_patch_size}")
        rollout_enabled = bool(
            self.train_cfg.get("next_pred", {}).get("rollout_branch", {}).get("enabled", False)
        )
        if rollout_enabled or int(self.val_cfg.get("max_val_iters", 0)) > 0:
            if n_past % frame_patch_size != 0 or n_futures % frame_patch_size != 0:
                raise ValueError(
                    "Autoregressive rollout requires dataset.n_past and dataset.n_futures to be "
                    "divisible by model.frame_patch_size. "
                    f"got n_past={n_past}, n_futures={n_futures}, frame_patch_size={frame_patch_size}"
                )
        sequence_enabled = bool(
            self.train_cfg.get("next_pred", {}).get("sequence_loss", {}).get("enabled", False)
        )
        if sequence_enabled and total_frames % frame_patch_size != 0:
            raise ValueError(
                "Sequence right-shift supervision requires n_past+n_futures divisible by "
                f"frame_patch_size. got total_frames={total_frames}, frame_patch_size={frame_patch_size}"
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

            legacy_lambda = diff_cfg.get("lambda_diff", None)
            if legacy_lambda is not None and "weight_init" not in diff_cfg and "weight_max" not in diff_cfg:
                weight_init = float(legacy_lambda)
                weight_max = float(legacy_lambda)
            else:
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

        def build_lead_time_weights(
            *,
            total_frames: int,
            context_frames: int,
            device: torch.device,
            dtype: torch.dtype,
        ) -> torch.Tensor:
            if total_frames <= 0:
                raise ValueError(f"total_frames must be > 0, got {total_frames}")
            if context_frames < 0 or context_frames > total_frames:
                raise ValueError(
                    f"context_frames must be in [0, total_frames], got {context_frames}"
                )
            weights = torch.ones(total_frames, device=device, dtype=dtype)
            lead_cfg = self.train_cfg.get("loss", {}).get("lead_time_weight", {})
            if not bool(lead_cfg.get("enabled", False)):
                return weights

            start_weight = float(lead_cfg.get("start_weight", 1.0))
            end_weight = float(lead_cfg.get("end_weight", 2.0))
            power = float(lead_cfg.get("power", 1.0))
            if start_weight <= 0 or end_weight <= 0:
                raise ValueError(
                    "lead_time_weight start_weight/end_weight must be > 0, "
                    f"got {start_weight}, {end_weight}"
                )
            if power <= 0:
                raise ValueError(f"lead_time_weight.power must be > 0, got {power}")

            future_frames = total_frames - context_frames
            if future_frames > 0:
                progress = torch.linspace(0.0, 1.0, future_frames, device=device, dtype=dtype)
                weights[context_frames:] = start_weight + (end_weight - start_weight) * progress.pow(power)
            return weights

        def weighted_time_mean(loss_chunk: torch.Tensor, time_weights: torch.Tensor | None) -> torch.Tensor:
            if time_weights is None:
                return loss_chunk.mean()
            if loss_chunk.ndim != 5:
                raise ValueError(f"loss_chunk must be [B,C,T,H,W], got {tuple(loss_chunk.shape)}")
            if int(loss_chunk.shape[2]) != int(time_weights.numel()):
                raise ValueError(
                    "time-weight length mismatch: "
                    f"loss_frames={int(loss_chunk.shape[2])}, weights={int(time_weights.numel())}"
                )
            per_frame = loss_chunk.mean(dim=(0, 1, 3, 4))
            weights = time_weights.to(device=loss_chunk.device, dtype=loss_chunk.dtype)
            return (per_frame * weights).sum() / weights.sum().clamp_min(1.0e-8)

        def rain_in_threshold_space(rain_tensor: torch.Tensor) -> torch.Tensor:
            threshold_space = str(self.train_cfg.get("loss", {}).get("threshold_space", "model")).lower()
            if threshold_space == "model":
                return rain_tensor
            if threshold_space == "physical":
                return self._denormalize_rain_for_metrics(rain_tensor)
            raise ValueError(f"Unsupported rain threshold space: {threshold_space}")

        def future_skill_slice(rain_tensor: torch.Tensor) -> torch.Tensor:
            start = 0
            if aux is not None and bool(aux.get("sequence_loss_enabled", 0)):
                start = int(aux.get("sequence_context_frames", 0))
            if start < 0 or start >= int(rain_tensor.shape[2]):
                if start == int(rain_tensor.shape[2]):
                    raise ValueError("Skill losses require at least one future frame.")
                raise ValueError(
                    f"Invalid future skill slice start={start} for T={int(rain_tensor.shape[2])}."
                )
            return rain_tensor[:, :, start:]

        def threshold_tag(threshold: float) -> str:
            return (f"{threshold:g}").replace("-", "m").replace(".", "p")

        def compute_tempo_skill_losses(
            pred_rain: torch.Tensor,
            target_rain: torch.Tensor,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            loss_cfg = self.train_cfg.get("loss", {})
            target_mode = str(aux.get("target_mode", "unknown")) if aux is not None else "unknown"

            def objective_applies(objective_cfg: dict) -> bool:
                apply_on = str(objective_cfg.get("apply_on", "all")).lower()
                if apply_on not in {"all", "rollout", "teacher"}:
                    raise ValueError(
                        f"TEMPO objective apply_on must be all/rollout/teacher, got {apply_on}"
                    )
                if apply_on == "all":
                    return True
                is_rollout = target_mode == "rollout_branch"
                return is_rollout if apply_on == "rollout" else not is_rollout

            pred_skill = future_skill_slice(rain_in_threshold_space(pred_rain)).float()
            target_skill = future_skill_slice(rain_in_threshold_space(target_rain)).float()
            if pred_skill.shape != target_skill.shape:
                raise ValueError(
                    "TEMPO skill loss shape mismatch: "
                    f"pred={tuple(pred_skill.shape)}, target={tuple(target_skill.shape)}"
                )

            lead_weights = build_lead_time_weights(
                total_frames=int(pred_skill.shape[2]),
                context_frames=0,
                device=pred_skill.device,
                dtype=pred_skill.dtype,
            )
            lead_weights = lead_weights / lead_weights.sum().clamp_min(1.0e-8)
            total = torch.zeros((), device=pred_skill.device, dtype=pred_skill.dtype)
            tempo_logs: dict[str, torch.Tensor] = {}

            soft_csi_cfg = loss_cfg.get("soft_csi", {})
            if bool(soft_csi_cfg.get("enabled", False)) and objective_applies(soft_csi_cfg):
                weight = float(soft_csi_cfg.get("weight", 0.0))
                thresholds = [float(v) for v in soft_csi_cfg.get("thresholds", [0.1, 0.3, 0.5])]
                temperature = float(soft_csi_cfg.get("temperature", 0.05))
                eps = float(soft_csi_cfg.get("eps", 1.0e-6))
                if not thresholds:
                    raise ValueError("train.loss.soft_csi.thresholds must not be empty.")
                if temperature <= 0 or eps <= 0:
                    raise ValueError("soft_csi temperature and eps must be > 0.")
                threshold_losses: list[torch.Tensor] = []
                for threshold in thresholds:
                    probability = torch.sigmoid((pred_skill - threshold) / temperature)
                    label = (target_skill >= threshold).to(dtype=probability.dtype)
                    reduce_dims = (1, 3, 4)
                    tp = (probability * label).sum(dim=reduce_dims)
                    fp = (probability * (1.0 - label)).sum(dim=reduce_dims)
                    fn = ((1.0 - probability) * label).sum(dim=reduce_dims)
                    soft_csi = (tp + eps) / (tp + fp + fn + eps)
                    per_lead_loss = (1.0 - soft_csi).mean(dim=0)
                    threshold_loss = (per_lead_loss * lead_weights).sum()
                    threshold_losses.append(threshold_loss)
                    tempo_logs[f"loss/soft_csi_t{threshold_tag(threshold)}_raw"] = threshold_loss.detach()
                soft_csi_raw = torch.stack(threshold_losses).mean()
                soft_csi_weighted = weight * soft_csi_raw
                total = total + soft_csi_weighted
                tempo_logs["loss/soft_csi_raw"] = soft_csi_raw.detach()
                tempo_logs["loss/soft_csi"] = soft_csi_weighted.detach()

            soft_fss_cfg = loss_cfg.get("soft_fss", {})
            if bool(soft_fss_cfg.get("enabled", False)) and objective_applies(soft_fss_cfg):
                weight = float(soft_fss_cfg.get("weight", 0.0))
                thresholds = [float(v) for v in soft_fss_cfg.get("thresholds", [0.1, 0.3])]
                windows = [int(v) for v in soft_fss_cfg.get("windows", [3, 9])]
                temperature = float(soft_fss_cfg.get("temperature", 0.05))
                eps = float(soft_fss_cfg.get("eps", 1.0e-6))
                if not thresholds or not windows:
                    raise ValueError("soft_fss thresholds/windows must not be empty.")
                if temperature <= 0 or eps <= 0 or any(v <= 0 or v % 2 == 0 for v in windows):
                    raise ValueError("soft_fss requires temperature/eps>0 and positive odd windows.")

                b, _, t, h, w = pred_skill.shape
                pred_scalar = pred_skill.mean(dim=1).reshape(b * t, 1, h, w)
                target_scalar = target_skill.mean(dim=1).reshape(b * t, 1, h, w)
                scale_losses: list[torch.Tensor] = []
                for threshold in thresholds:
                    probability = torch.sigmoid((pred_scalar - threshold) / temperature)
                    label = (target_scalar >= threshold).to(dtype=probability.dtype)
                    for window in windows:
                        pred_fraction = F.avg_pool2d(
                            probability, kernel_size=window, stride=1, padding=window // 2
                        )
                        target_fraction = F.avg_pool2d(
                            label, kernel_size=window, stride=1, padding=window // 2
                        )
                        mse = (pred_fraction - target_fraction).pow(2).mean(dim=(1, 2, 3)).view(b, t)
                        reference = (
                            pred_fraction.pow(2) + target_fraction.pow(2)
                        ).mean(dim=(1, 2, 3)).view(b, t)
                        fss_loss_per_lead = (mse / (reference + eps)).mean(dim=0)
                        scale_loss = (fss_loss_per_lead * lead_weights).sum()
                        scale_losses.append(scale_loss)
                        tempo_logs[
                            f"loss/soft_fss_t{threshold_tag(threshold)}_w{window}_raw"
                        ] = scale_loss.detach()
                soft_fss_raw = torch.stack(scale_losses).mean()
                soft_fss_weighted = weight * soft_fss_raw
                total = total + soft_fss_weighted
                tempo_logs["loss/soft_fss_raw"] = soft_fss_raw.detach()
                tempo_logs["loss/soft_fss"] = soft_fss_weighted.detach()

            spectral_cfg = loss_cfg.get("spectral_amplitude", {})
            if bool(spectral_cfg.get("enabled", False)) and objective_applies(spectral_cfg):
                weight = float(spectral_cfg.get("weight", 0.0))
                max_size = int(spectral_cfg.get("max_size", 128))
                use_log1p = bool(spectral_cfg.get("log1p", True))
                if max_size <= 0:
                    raise ValueError("spectral_amplitude.max_size must be > 0.")
                b, c, t, h, w = pred_skill.shape
                pred_fft_input = pred_skill.permute(0, 2, 1, 3, 4).reshape(b * t * c, 1, h, w)
                target_fft_input = target_skill.permute(0, 2, 1, 3, 4).reshape(b * t * c, 1, h, w)
                if h > max_size or w > max_size:
                    out_h, out_w = min(h, max_size), min(w, max_size)
                    pred_fft_input = F.adaptive_avg_pool2d(pred_fft_input, (out_h, out_w))
                    target_fft_input = F.adaptive_avg_pool2d(target_fft_input, (out_h, out_w))
                pred_amp = torch.fft.rfft2(pred_fft_input.float(), norm="ortho").abs()
                target_amp = torch.fft.rfft2(target_fft_input.float(), norm="ortho").abs()
                if use_log1p:
                    pred_amp = torch.log1p(pred_amp)
                    target_amp = torch.log1p(target_amp)
                spectral_raw = F.l1_loss(pred_amp, target_amp) / target_amp.mean().clamp_min(1.0e-6)
                spectral_weighted = weight * spectral_raw
                total = total + spectral_weighted
                tempo_logs["loss/spectral_amplitude_raw"] = spectral_raw.detach()
                tempo_logs["loss/spectral_amplitude"] = spectral_weighted.detach()

            return total, tempo_logs

        def compute_tempo_model_auxiliary_losses(
            pred_dict: dict[str, torch.Tensor],
            target_rain: torch.Tensor,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            aux_cfg = self.train_cfg.get("loss", {}).get("model_auxiliary", {})
            zero = torch.zeros((), device=target_rain.device, dtype=target_rain.dtype)
            if not bool(aux_cfg.get("enabled", False)):
                return zero, {}

            apply_on = str(aux_cfg.get("apply_on", "all")).lower()
            if apply_on not in {"all", "rollout", "teacher"}:
                raise ValueError("model_auxiliary.apply_on must be all/rollout/teacher.")
            target_mode = str(aux.get("target_mode", "unknown")) if aux is not None else "unknown"
            is_rollout = target_mode == "rollout_branch"
            if (apply_on == "rollout" and not is_rollout) or (apply_on == "teacher" and is_rollout):
                return zero, {}

            scale = float(aux_cfg.get("weight", 1.0))
            require_outputs = bool(aux_cfg.get("require_outputs", False))
            expected = [str(v) for v in aux_cfg.get(
                "expected_outputs", ["rain_direct", "rain_advected", "rain_source", "flow"]
            )]
            missing = [key for key in expected if key not in pred_dict]
            if missing and require_outputs:
                raise ValueError(
                    "TEMPO model auxiliary loss requires missing model outputs: " + ", ".join(missing)
                )

            target_future = future_skill_slice(target_rain)
            model_aux_total = zero
            model_aux_logs: dict[str, torch.Tensor] = {
                "meta/tempo_model_aux_outputs": torch.tensor(
                    float(len(expected) - len(missing)), device=zero.device, dtype=zero.dtype
                )
            }

            direct = pred_dict.get("rain_direct")
            direct_weight = float(aux_cfg.get("direct_weight", 0.0))
            if torch.is_tensor(direct) and direct_weight > 0:
                direct_future = future_skill_slice(direct)
                if direct_future.shape != target_future.shape:
                    raise ValueError("rain_direct must match future rain target shape.")
                direct_raw = F.smooth_l1_loss(direct_future, target_future, beta=0.1)
                model_aux_total = model_aux_total + direct_weight * direct_raw
                model_aux_logs["loss/tempo_direct_raw"] = direct_raw.detach()

            advected = pred_dict.get("rain_advected")
            source = pred_dict.get("rain_source")
            confidence = pred_dict.get("flow_confidence")
            transport_output = pred_dict.get("rain_transport")

            advection_weight = float(aux_cfg.get("advection_weight", 0.0))
            if torch.is_tensor(advected) and advection_weight > 0:
                advected_future = future_skill_slice(advected)
                if advected_future.shape != target_future.shape:
                    raise ValueError("rain_advected must match the future rain target shape.")
                advection_raw = F.smooth_l1_loss(
                    advected_future,
                    target_future,
                    beta=0.1,
                )
                model_aux_total = model_aux_total + advection_weight * advection_raw
                model_aux_logs["loss/tempo_advection_raw"] = advection_raw.detach()

            transport: torch.Tensor | None = None
            if torch.is_tensor(transport_output):
                transport = future_skill_slice(transport_output)
            elif torch.is_tensor(advected) and torch.is_tensor(source):
                transport = future_skill_slice(advected) + future_skill_slice(source)

            transport_weight = float(aux_cfg.get("transport_weight", 0.0))
            if torch.is_tensor(transport) and transport_weight > 0:
                if transport.shape != target_future.shape:
                    raise ValueError("rain_transport must match the future rain target shape.")
                transport_raw = F.smooth_l1_loss(transport, target_future, beta=0.1)
                model_aux_total = model_aux_total + transport_weight * transport_raw
                model_aux_logs["loss/tempo_transport_raw"] = transport_raw.detach()

            composition_weight = float(aux_cfg.get("composition_weight", 0.0))
            if torch.is_tensor(transport) and composition_weight > 0:
                direct_for_gate = pred_dict.get("rain_direct")
                if torch.is_tensor(confidence) and torch.is_tensor(direct_for_gate):
                    confidence_future = future_skill_slice(confidence)
                    direct_future_for_gate = future_skill_slice(direct_for_gate)
                    if int(confidence_future.shape[1]) not in {
                        1,
                        int(direct_future_for_gate.shape[1]),
                    }:
                        raise ValueError(
                            "flow_confidence channels must be 1 or match rain_direct channels."
                        )
                    if (
                        int(confidence_future.shape[0]) != int(transport.shape[0])
                        or int(confidence_future.shape[2]) != int(transport.shape[2])
                        or tuple(confidence_future.shape[-2:]) != tuple(transport.shape[-2:])
                    ):
                        raise ValueError(
                            "flow_confidence B/T/H/W dimensions must match the transport forecast."
                        )
                    composed = (
                        confidence_future * transport
                        + (1.0 - confidence_future) * direct_future_for_gate
                    )
                else:
                    composed = transport
                final_future = future_skill_slice(pred_dict["rain"])
                if composed.shape != final_future.shape:
                    raise ValueError("TEMPO composed rain must match final rain shape.")
                composition_raw = F.smooth_l1_loss(composed, final_future, beta=0.05)
                model_aux_total = model_aux_total + composition_weight * composition_raw
                model_aux_logs["loss/tempo_composition_raw"] = composition_raw.detach()
                if torch.is_tensor(confidence):
                    model_aux_logs["meta/tempo_transport_confidence_mean"] = (
                        future_skill_slice(confidence).mean().detach()
                    )

            source_weight = float(aux_cfg.get("source_sparsity_weight", 0.0))
            if torch.is_tensor(source) and source_weight > 0:
                source_raw = future_skill_slice(source).abs().mean()
                model_aux_total = model_aux_total + source_weight * source_raw
                model_aux_logs["loss/tempo_source_sparsity_raw"] = source_raw.detach()
                model_aux_logs["meta/tempo_source_abs_mean"] = source_raw.detach()

            flow = pred_dict.get("flow")
            flow_weight = float(aux_cfg.get("flow_smoothness_weight", 0.0))
            if torch.is_tensor(flow) and flow_weight > 0:
                flow_future = future_skill_slice(flow)
                dx = flow_future[..., 1:] - flow_future[..., :-1]
                dy = flow_future[..., 1:, :] - flow_future[..., :-1, :]
                dt = flow_future[:, :, 1:] - flow_future[:, :, :-1]
                flow_raw = dx.abs().mean() + dy.abs().mean()
                if int(flow_future.shape[2]) > 1:
                    flow_raw = flow_raw + 0.25 * dt.abs().mean()
                model_aux_total = model_aux_total + flow_weight * flow_raw
                model_aux_logs["loss/tempo_flow_smoothness_raw"] = flow_raw.detach()
                model_aux_logs["meta/tempo_flow_abs_mean"] = flow_future.abs().mean().detach()

            model_aux_total = scale * model_aux_total
            model_aux_logs["loss/tempo_model_auxiliary"] = model_aux_total.detach()
            return model_aux_total, model_aux_logs

        def compute_weighted_segment_loss(
            loss_chunk: torch.Tensor,
            context_frames: int,
            future_frames: int,
            context_weight: float,
            future_weight: float,
            time_weights: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            zero = torch.zeros((), device=loss_chunk.device, dtype=loss_chunk.dtype)
            weighted_sum = zero
            weight_sum = 0.0

            if context_frames > 0:
                context_time_weights = None if time_weights is None else time_weights[:context_frames]
                context_loss = weighted_time_mean(
                    loss_chunk[:, :, :context_frames], context_time_weights
                )
                if context_weight > 0:
                    weighted_sum = weighted_sum + context_weight * context_loss
                    weight_sum += context_weight
            else:
                context_loss = zero

            if future_frames > 0:
                future_time_weights = (
                    None
                    if time_weights is None
                    else time_weights[context_frames : context_frames + future_frames]
                )
                future_loss = weighted_time_mean(
                    loss_chunk[:, :, context_frames : context_frames + future_frames],
                    future_time_weights,
                )
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
            loss_cfg = self.train_cfg.get("loss", {})
            mode = str(loss_cfg.get("mode", "mse")).lower()
            base_mode = str(loss_cfg.get("regression_base", "huber")).lower() if mode == "enhanced" else mode
            if base_mode == "mse":
                base_error = F.mse_loss(pred_rain, target_rain, reduction="none")
            elif base_mode == "mae":
                base_error = F.l1_loss(pred_rain, target_rain, reduction="none")
            elif base_mode == "huber":
                huber_beta = float(loss_cfg.get("huber_beta", 0.1))
                if huber_beta <= 0:
                    raise ValueError(f"train.loss.huber_beta must be > 0, got {huber_beta}")
                base_error = F.smooth_l1_loss(
                    pred_rain,
                    target_rain,
                    reduction="none",
                    beta=huber_beta,
                )
            else:
                raise ValueError(
                    "train.loss regression base must be 'mse', 'mae', or 'huber', "
                    f"got {base_mode}"
                )

            rain_cfg = loss_cfg.get("rain_region_weight", {})
            if not bool(rain_cfg.get("enabled", False)):
                return base_error, False

            alpha = float(rain_cfg.get("alpha", 0.0))
            r0 = float(rain_cfg.get("r0", 0.0))
            gamma = float(rain_cfg.get("gamma", 1.0))
            max_weight = float(rain_cfg.get("max_weight", 0.0))
            normalize_weight = bool(rain_cfg.get("normalize", True))
            if alpha < 0:
                raise ValueError(f"train.loss.rain_region_weight.alpha must be >= 0, got {alpha}")
            if gamma <= 0:
                raise ValueError(f"train.loss.rain_region_weight.gamma must be > 0, got {gamma}")
            if max_weight < 0:
                raise ValueError(
                    f"train.loss.rain_region_weight.max_weight must be >= 0, got {max_weight}"
                )

            target_for_weight = rain_in_threshold_space(target_rain)
            weight = 1.0 + alpha * torch.clamp(target_for_weight - r0, min=0.0).pow(gamma)
            if max_weight > 0:
                weight = weight.clamp(max=max_weight)
            if normalize_weight:
                weight = weight / weight.mean().clamp_min(1.0e-6)
            return base_error * weight, True

        def build_rain_event_loss_map(
            pred_rain: torch.Tensor,
            target_rain: torch.Tensor,
            event_logits: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor | None, float]:
            loss_cfg = self.train_cfg.get("loss", {})
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

            head_mode = str(event_cfg.get("head_mode", "auto")).lower()
            if head_mode not in {"auto", "derived", "model"}:
                raise ValueError(
                    "rain_event_aux.head_mode must be auto/derived/model, "
                    f"got {head_mode}"
                )
            use_model_logits = head_mode == "model" or (head_mode == "auto" and torch.is_tensor(event_logits))
            if head_mode == "model" and not torch.is_tensor(event_logits):
                raise ValueError(
                    "rain_event_aux.head_mode='model' requires model output 'rain_event_logits'."
                )
            if use_model_logits:
                if event_logits is None or event_logits.ndim != 5:
                    raise ValueError("rain_event_logits must be [B,K,T,H,W].")
                if int(event_logits.shape[1]) != len(thresholds):
                    raise ValueError(
                        "rain_event_logits channel count must equal number of thresholds, "
                        f"got K={int(event_logits.shape[1])}, thresholds={len(thresholds)}"
                    )
                if (
                    int(event_logits.shape[0]) != int(target_rain.shape[0])
                    or int(event_logits.shape[2]) != int(target_rain.shape[2])
                    or tuple(event_logits.shape[-2:]) != tuple(target_rain.shape[-2:])
                ):
                    raise ValueError(
                        "rain_event_logits B/T/H/W dimensions must match rain target."
                    )

            pred_threshold_space = rain_in_threshold_space(pred_rain)
            target_threshold_space = rain_in_threshold_space(target_rain)
            per_threshold_maps: list[torch.Tensor] = []
            for threshold_idx, threshold in enumerate(thresholds):
                threshold_tensor = torch.tensor(
                    threshold, device=pred_rain.device, dtype=pred_rain.dtype
                )
                if use_model_logits:
                    logits = event_logits[:, threshold_idx : threshold_idx + 1].to(dtype=pred_rain.dtype)
                else:
                    logits = (pred_threshold_space - threshold_tensor) * logit_scale
                labels = (target_threshold_space >= threshold_tensor).to(dtype=pred_rain.dtype)
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
            ordinal_weight = float(event_cfg.get("ordinal_weight", 0.0))
            if ordinal_weight < 0:
                raise ValueError("rain_event_aux.ordinal_weight must be >= 0.")
            if use_model_logits and ordinal_weight > 0 and event_logits is not None and len(thresholds) > 1:
                probabilities = torch.sigmoid(event_logits)
                # Exceedance probabilities must decrease as the rain threshold rises.
                ordinal_violation = F.relu(probabilities[:, 1:] - probabilities[:, :-1])
                event_map = event_map + ordinal_weight * ordinal_violation.mean(dim=1, keepdim=True)
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
            time_weights = build_lead_time_weights(
                total_frames=int(loss_map.shape[2]),
                context_frames=0,
                device=loss_map.device,
                dtype=loss_map.dtype,
            )
            l_radar = weighted_time_mean(loss_map[:, : self.radar_c], time_weights)
            l_satellite = weighted_time_mean(
                loss_map[:, self.radar_c : self.radar_c + self.satellite_c], time_weights
            )
            rain_loss_map, rain_weighted_enabled = build_rain_regression_loss_map(
                pred_for_loss["rain"], target_gt["rain"]
            )
            l_rain = weighted_time_mean(rain_loss_map, time_weights)
            event_map, event_weight = build_rain_event_loss_map(
                pred_for_loss["rain"],
                target_gt["rain"],
                pred_for_loss.get("rain_event_logits"),
            )
            if event_map is None:
                event_loss = torch.zeros((), device=loss_map.device, dtype=loss_map.dtype)
            else:
                event_loss = event_weight * weighted_time_mean(event_map, time_weights)
            rain_delta_loss = (
                torch.zeros((), device=loss_map.device, dtype=loss_map.dtype)
                if rain_delta_loss_map is None
                else weighted_time_mean(rain_delta_loss_map, time_weights)
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
                        diff_radar = weighted_time_mean(radar_diff_map, time_weights)
                    if satellite_diff_map is not None:
                        diff_satellite = weighted_time_mean(satellite_diff_map, time_weights)
                    if rain_diff_map is not None:
                        diff_rain = weighted_time_mean(rain_diff_map, time_weights)
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
                        diff_rain = weighted_time_mean(rain_diff_map, time_weights)
                        diff_raw = diff_rain

                diff_weighted = temporal_diff_weight * diff_raw
                loss = loss + diff_weighted

            tempo_skill_loss, tempo_skill_logs = compute_tempo_skill_losses(
                pred_for_loss["rain"], target_gt["rain"]
            )
            tempo_model_aux_loss, tempo_model_aux_logs = compute_tempo_model_auxiliary_losses(
                pred_for_loss, target_gt["rain"]
            )
            loss = loss + tempo_skill_loss + tempo_model_aux_loss

            logs = {
                "loss": loss.detach(),
                "loss/radar": l_radar.detach(),
                "loss/satellite": l_satellite.detach(),
                "loss/rain": l_rain.detach(),
                "meta/lead_time_weight_first": time_weights[0].detach(),
                "meta/lead_time_weight_last": time_weights[-1].detach(),
            }
            logs.update(tempo_skill_logs)
            logs.update(tempo_model_aux_logs)
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
        time_weights = build_lead_time_weights(
            total_frames=total_frames,
            context_frames=context_frames,
            device=loss_map.device,
            dtype=loss_map.dtype,
        )

        radar_map = loss_map[:, : self.radar_c]
        satellite_map = loss_map[:, self.radar_c : self.radar_c + self.satellite_c]
        rain_map, rain_weighted_enabled = build_rain_regression_loss_map(pred_for_loss["rain"], target_gt["rain"])
        rain_event_map, event_weight = build_rain_event_loss_map(
            pred_for_loss["rain"],
            target_gt["rain"],
            pred_for_loss.get("rain_event_logits"),
        )

        l_radar, l_radar_context, l_radar_future = compute_weighted_segment_loss(
            radar_map, context_frames, future_frames, context_weight, future_weight, time_weights
        )
        l_satellite, l_satellite_context, l_satellite_future = compute_weighted_segment_loss(
            satellite_map, context_frames, future_frames, context_weight, future_weight, time_weights
        )
        l_rain, l_rain_context, l_rain_future = compute_weighted_segment_loss(
            rain_map, context_frames, future_frames, context_weight, future_weight, time_weights
        )
        if rain_event_map is None:
            rain_event = torch.zeros((), device=loss_map.device, dtype=loss_map.dtype)
            rain_event_context = torch.zeros((), device=loss_map.device, dtype=loss_map.dtype)
            rain_event_future = torch.zeros((), device=loss_map.device, dtype=loss_map.dtype)
        else:
            rain_event, rain_event_context, rain_event_future = compute_weighted_segment_loss(
                rain_event_map,
                context_frames,
                future_frames,
                context_weight,
                future_weight,
                time_weights,
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
                rain_delta_loss_map,
                context_frames,
                future_frames,
                context_weight,
                future_weight,
                time_weights,
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
            diff_total_frames = diff_context_frames + diff_future_frames
            diff_time_weights = (
                build_lead_time_weights(
                    total_frames=diff_total_frames,
                    context_frames=diff_context_frames,
                    device=loss_map.device,
                    dtype=loss_map.dtype,
                )
                if diff_total_frames > 0
                else None
            )

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
                        radar_diff_map,
                        diff_context_frames,
                        diff_future_frames,
                        context_weight,
                        future_weight,
                        diff_time_weights,
                    )
                if satellite_diff_map is not None and (diff_context_frames + diff_future_frames) > 0:
                    diff_satellite, _, _ = compute_weighted_segment_loss(
                        satellite_diff_map,
                        diff_context_frames,
                        diff_future_frames,
                        context_weight,
                        future_weight,
                        diff_time_weights,
                    )
                if rain_diff_map is not None and (diff_context_frames + diff_future_frames) > 0:
                    diff_rain, _, _ = compute_weighted_segment_loss(
                        rain_diff_map,
                        diff_context_frames,
                        diff_future_frames,
                        context_weight,
                        future_weight,
                        diff_time_weights,
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
                        rain_diff_map,
                        diff_context_frames,
                        diff_future_frames,
                        context_weight,
                        future_weight,
                        diff_time_weights,
                    )
                    diff_raw = diff_rain
            diff_weighted = temporal_diff_weight * diff_raw
            loss = loss + diff_weighted

        tempo_skill_loss, tempo_skill_logs = compute_tempo_skill_losses(
            pred_for_loss["rain"], target_gt["rain"]
        )
        tempo_model_aux_loss, tempo_model_aux_logs = compute_tempo_model_auxiliary_losses(
            pred_for_loss, target_gt["rain"]
        )
        loss = loss + tempo_skill_loss + tempo_model_aux_loss

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
            "meta/lead_time_weight_first": time_weights[context_frames].detach()
            if future_frames > 0
            else time_weights[0].detach(),
            "meta/lead_time_weight_last": time_weights[-1].detach(),
        }
        logs.update(tempo_skill_logs)
        logs.update(tempo_model_aux_logs)
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
            logs["meta/rollout_branch_frames"] = torch.zeros((), device=self.device)
            logs["meta/scheduled_sampling_prob"] = torch.zeros((), device=self.device)
            logs["meta/rollout_stage_index"] = torch.full((), -1.0, device=self.device)
            logs["meta/rollout_stage_progress"] = torch.zeros((), device=self.device)
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
        sampling_bucket = batch.get("sampling_bucket_id")
        if torch.is_tensor(sampling_bucket):
            sampling_bucket = sampling_bucket.to(self.device).reshape(-1)
            for bucket_id, bucket_name in enumerate(("dry", "light", "moderate", "heavy")):
                logs[f"data/sampling_{bucket_name}_fraction"] = (
                    sampling_bucket == bucket_id
                ).to(dtype=torch.float32).mean()
        sampling_score = batch.get("sampling_window_score")
        if torch.is_tensor(sampling_score):
            sampling_score = sampling_score.to(self.device, dtype=torch.float32).reshape(-1)
            finite_score = sampling_score[torch.isfinite(sampling_score)]
            if int(finite_score.numel()) > 0:
                logs["data/sampling_window_score"] = finite_score.mean()
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

    @staticmethod
    def _interpolate_schedule(
        *,
        start_value: float,
        end_value: float,
        step: int,
        start_step: int,
        end_step: int,
        schedule: str,
    ) -> float:
        if start_step < 0 or end_step < start_step:
            raise ValueError(
                f"schedule requires 0 <= start_step <= end_step, got {start_step}, {end_step}"
            )
        if step <= start_step:
            progress = 0.0
        elif step >= end_step or end_step == start_step:
            progress = 1.0
        else:
            progress = float(step - start_step) / float(end_step - start_step)

        schedule = schedule.lower()
        if schedule == "linear":
            ratio = progress
        elif schedule == "cosine":
            ratio = 0.5 - 0.5 * math.cos(math.pi * progress)
        else:
            raise ValueError(f"schedule must be 'linear' or 'cosine', got {schedule}")
        return start_value + (end_value - start_value) * ratio

    def _resolve_train_rollout_branch(
        self,
        total_future_frames: int,
    ) -> dict[
        str,
        float | int | bool | str | tuple[str, ...] | tuple[int, ...],
    ] | None:
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

        configured_frames = rollout_cfg.get("rollout_frames", None)
        rollout_frames = total_future_frames if configured_frames in (None, "") else int(configured_frames)
        if rollout_frames <= 0:
            raise ValueError(f"train.next_pred.rollout_branch.rollout_frames must be > 0, got {rollout_frames}")
        if rollout_frames > total_future_frames:
            raise ValueError(
                "train.next_pred.rollout_branch.rollout_frames cannot exceed available future frames, "
                f"got rollout_frames={rollout_frames}, total_future_frames={total_future_frames}"
            )

        frame_patch_size = self._resolve_frame_patch_size()
        global_step = int(getattr(self, "global_step", 0))
        stage_index = -1
        stage_name = "legacy"
        stage_progress = 0.0
        stage_teacher_forcing_prob: float | None = None

        stages_cfg = rollout_cfg.get("stages", [])
        if stages_cfg:
            active_idx = 0
            for idx, candidate in enumerate(stages_cfg):
                if global_step >= int(candidate.get("start_step", 0)):
                    active_idx = idx
                else:
                    break
            active_stage = stages_cfg[active_idx]
            stage_index = active_idx
            stage_name = str(active_stage.get("name", f"stage_{active_idx}"))
            start_step = int(active_stage.get("start_step", 0))
            end_step = int(active_stage.get("end_step", start_step))
            schedule = str(active_stage.get("schedule", "linear"))

            if end_step <= start_step:
                stage_progress = 1.0 if global_step >= end_step else 0.0
            else:
                stage_progress = min(
                    max(float(global_step - start_step) / float(end_step - start_step), 0.0),
                    1.0,
                )

            start_frames = int(active_stage.get("start_frames", frame_patch_size))
            end_frames = int(active_stage.get("end_frames", start_frames))
            interpolated = self._interpolate_schedule(
                start_value=float(start_frames),
                end_value=float(end_frames),
                step=global_step,
                start_step=start_step,
                end_step=end_step,
                schedule=schedule,
            )
            rollout_frames = int(round(interpolated / frame_patch_size)) * frame_patch_size
            rollout_frames = max(frame_patch_size, min(rollout_frames, total_future_frames))

            weight = self._interpolate_schedule(
                start_value=float(active_stage.get("start_weight", weight)),
                end_value=float(active_stage.get("end_weight", active_stage.get("start_weight", weight))),
                step=global_step,
                start_step=start_step,
                end_step=end_step,
                schedule=schedule,
            )
            stage_teacher_forcing_prob = self._interpolate_schedule(
                start_value=float(active_stage.get("start_prob", 0.0)),
                end_value=float(active_stage.get("end_prob", active_stage.get("start_prob", 0.0))),
                step=global_step,
                start_step=start_step,
                end_step=end_step,
                schedule=schedule,
            )
            if getattr(self, "_last_rollout_stage_name", None) != stage_name:
                self.log_msg(
                    "TEMPO rollout stage -> "
                    f"{stage_name} (index={stage_index}, step={global_step}, "
                    f"frames={rollout_frames}, forcing={stage_teacher_forcing_prob:.3f}, "
                    f"weight={weight:.3f})"
                )
                self._last_rollout_stage_name = stage_name
        else:
            curriculum_cfg = rollout_cfg.get("curriculum", {})
            if bool(curriculum_cfg.get("enabled", False)):
                start_frames = int(curriculum_cfg.get("start_frames", frame_patch_size))
                end_frames = int(curriculum_cfg.get("end_frames", rollout_frames))
                start_step = int(curriculum_cfg.get("start_step", 0))
                end_step = int(curriculum_cfg.get("end_step", 30000))
                interpolated = self._interpolate_schedule(
                    start_value=float(start_frames),
                    end_value=float(end_frames),
                    step=global_step,
                    start_step=start_step,
                    end_step=end_step,
                    schedule=str(curriculum_cfg.get("schedule", "linear")),
                )
                rollout_frames = int(round(interpolated / frame_patch_size)) * frame_patch_size
                rollout_frames = max(frame_patch_size, min(rollout_frames, total_future_frames))

        if weight < 0:
            raise ValueError(f"resolved rollout weight must be >= 0, got {weight}")
        if weight == 0:
            return None

        if rollout_frames % frame_patch_size != 0:
            raise ValueError(
                "rollout_frames must be divisible by model.frame_patch_size, "
                f"got rollout_frames={rollout_frames}, frame_patch_size={frame_patch_size}"
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

        scheduled_sampling_cfg = rollout_cfg.get("scheduled_sampling", {})
        scheduled_sampling_enabled = bool(scheduled_sampling_cfg.get("enabled", False))
        teacher_forcing_prob = 0.0
        teacher_forcing_modalities: tuple[str, ...] = ()
        if scheduled_sampling_enabled:
            start_step = int(scheduled_sampling_cfg.get("start_step", 0))
            if "end_step" in scheduled_sampling_cfg:
                end_step = int(scheduled_sampling_cfg.get("end_step"))
            else:
                end_step = start_step + int(scheduled_sampling_cfg.get("decay_steps", 30000))
            teacher_forcing_prob = self._interpolate_schedule(
                start_value=float(scheduled_sampling_cfg.get("start_prob", 1.0)),
                end_value=float(scheduled_sampling_cfg.get("end_prob", 0.0)),
                step=int(getattr(self, "global_step", 0)),
                start_step=start_step,
                end_step=end_step,
                schedule=str(scheduled_sampling_cfg.get("schedule", "cosine")),
            )
            teacher_forcing_prob = min(max(teacher_forcing_prob, 0.0), 1.0)
            if stage_teacher_forcing_prob is not None:
                teacher_forcing_prob = min(max(stage_teacher_forcing_prob, 0.0), 1.0)
            teacher_forcing_modalities = tuple(
                str(v)
                for v in scheduled_sampling_cfg.get(
                    "modalities", ["radar", "satellite", "rain"]
                )
            )

        use_gt_future_modalities = bool(rollout_cfg.get("use_gt_future_modalities", False))
        if use_gt_future_modalities and scheduled_sampling_enabled:
            raise ValueError(
                "use_gt_future_modalities and scheduled_sampling cannot be enabled together. "
                "Use scheduled_sampling with start_prob=1.0 for a curriculum."
            )

        configured_gradient_leads = tuple(
            sorted(
                {
                    int(value)
                    for value in rollout_cfg.get("gradient_leads", [])
                    if int(value) <= rollout_frames
                }
                | {rollout_frames}
            )
        )

        return {
            "mode": mode,
            "weight": weight,
            "rollout_frames": rollout_frames,
            "rollout_block_size": rollout_block_size,
            "detach_history": bool(rollout_cfg.get("detach_history", True)),
            "use_gt_future_modalities": use_gt_future_modalities,
            "scheduled_sampling_enabled": scheduled_sampling_enabled,
            "teacher_forcing_prob": teacher_forcing_prob,
            "teacher_forcing_modalities": teacher_forcing_modalities,
            "gradient_leads": configured_gradient_leads,
            "loss_on": loss_on,
            "delta_loss_enabled": delta_loss_enabled,
            "delta_loss_weight": delta_loss_weight,
            "stage_index": stage_index,
            "stage_name": stage_name,
            "stage_progress": stage_progress,
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
        teacher_forcing_prob: float = 0.0,
        teacher_forcing_modalities: tuple[str, ...] = (),
        gradient_leads: tuple[int, ...] | None = None,
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
        if not 0.0 <= teacher_forcing_prob <= 1.0:
            raise ValueError(
                f"teacher_forcing_prob must be in [0, 1], got {teacher_forcing_prob}"
            )
        invalid_modalities = sorted(
            set(teacher_forcing_modalities) - {"radar", "satellite", "rain"}
        )
        if invalid_modalities:
            raise ValueError(f"Invalid teacher-forcing modalities: {invalid_modalities}")
        if gradient_leads is not None:
            if any(int(lead) <= 0 or int(lead) > total_future_frames for lead in gradient_leads):
                raise ValueError(
                    "gradient_leads must fall inside [1,total_future_frames], "
                    f"got {gradient_leads}."
                )
        if teacher_forcing_prob > 0 and use_gt_future_modalities:
            raise ValueError(
                "teacher_forcing_prob and use_gt_future_modalities cannot both be active"
            )
        if teacher_forcing_prob > 0 and mode != "frame":
            raise ValueError(
                "Scheduled sampling requires rollout mode='frame' so feedback is mixed every step"
            )
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
        if use_gt_future_modalities or teacher_forcing_prob > 0:
            if future_modalities is None:
                raise ValueError("future_modalities are required when teacher forcing is active.")
            required_modalities = (
                {"radar", "satellite"}
                if use_gt_future_modalities
                else set(teacher_forcing_modalities)
            )
            missing_modalities = sorted(required_modalities - set(future_modalities))
            if missing_modalities:
                raise ValueError(
                    f"future_modalities is missing teacher-forced modalities: {missing_modalities}"
                )
            radar_future = future_modalities["radar"]
            satellite_future = future_modalities["satellite"]
            rain_future = future_modalities["rain"]
            if radar_future.ndim != 5 or satellite_future.ndim != 5 or rain_future.ndim != 5:
                raise ValueError(
                    "future_modalities tensors must be [B,C,T,H,W], "
                    f"got radar={tuple(radar_future.shape)}, satellite={tuple(satellite_future.shape)}, "
                    f"rain={tuple(rain_future.shape)}"
                )
            if (
                int(radar_future.shape[0]) != int(context.shape[0])
                or int(satellite_future.shape[0]) != int(context.shape[0])
                or int(rain_future.shape[0]) != int(context.shape[0])
            ):
                raise ValueError(
                    "future_modalities batch mismatch with context: "
                    f"context_batch={int(context.shape[0])}, radar_batch={int(radar_future.shape[0])}, "
                    f"satellite_batch={int(satellite_future.shape[0])}, rain_batch={int(rain_future.shape[0])}"
                )
            if (
                int(radar_future.shape[2]) != total_future_frames
                or int(satellite_future.shape[2]) != total_future_frames
                or int(rain_future.shape[2]) != total_future_frames
            ):
                raise ValueError(
                    "future_modalities time length mismatch with total_future_frames: "
                    f"radar_frames={int(radar_future.shape[2])}, satellite_frames={int(satellite_future.shape[2])}, "
                    f"rain_frames={int(rain_future.shape[2])}, total_future_frames={total_future_frames}"
                )

        remaining = total_future_frames
        context_cur = context
        context_time_cur = context_time
        produced = 0

        pred_radar: list[torch.Tensor] = []
        pred_satellite: list[torch.Tensor] = []
        pred_rain: list[torch.Tensor] = []
        pred_rain_delta: list[torch.Tensor] = []
        # Optional TEMPO model contract. Every tensor must be [B,C,T,H,W]
        # and must be returned consistently at every rollout step.
        optional_rollout_keys = (
            "rain_event_logits",
            "rain_direct",
            "rain_advected",
            "rain_source",
            "rain_transport",
            "flow",
            "flow_confidence",
        )
        pred_optional: dict[str, list[torch.Tensor]] = {}

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
                    "rain": future_modalities["rain"][:, :, produced : produced + chunk],
                }

            seed_block = self._build_self_rolled_seed(
                context=context_cur,
                seed_frames=chunk,
                detach_history=detach_history,
                context_time=context_time_cur,
                future_time=chunk_future_time,
                future_modality_forcing=chunk_future_modalities if use_gt_future_modalities else None,
            )
            chunk_leads = range(produced + 1, produced + chunk + 1)
            track_chunk_grad = gradient_leads is None or any(
                lead in gradient_leads for lead in chunk_leads
            )
            gradient_context = nullcontext() if track_chunk_grad else torch.no_grad()
            with gradient_context:
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
            for optional_key in optional_rollout_keys:
                optional_value = pred_block.get(optional_key)
                if optional_value is None:
                    if optional_key in pred_optional:
                        raise ValueError(
                            f"Model stopped returning optional rollout output '{optional_key}' "
                            f"after {produced} generated frames."
                        )
                    continue
                if produced > 0 and optional_key not in pred_optional:
                    raise ValueError(
                        f"Model started returning optional rollout output '{optional_key}' only "
                        f"after {produced} generated frames. Return it at every step or never."
                    )
                if not torch.is_tensor(optional_value) or optional_value.ndim != 5:
                    raise ValueError(
                        f"Optional model output '{optional_key}' must be [B,C,T,H,W], "
                        f"got {type(optional_value)} / "
                        f"{tuple(optional_value.shape) if torch.is_tensor(optional_value) else 'non-tensor'}"
                    )
                if int(optional_value.shape[2]) != chunk:
                    raise ValueError(
                        f"Optional model output '{optional_key}' has T={int(optional_value.shape[2])}, "
                        f"expected rollout chunk={chunk}."
                    )
                pred_optional.setdefault(optional_key, []).append(optional_value)

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
            elif teacher_forcing_prob > 0:
                if chunk_future_modalities is None:
                    raise ValueError(
                        "chunk_future_modalities should be available for scheduled sampling"
                    )
                feedback = self._split_modalities(pred_hist)
                batch_size = int(pred_hist.shape[0])
                for modality_name in teacher_forcing_modalities:
                    use_gt = torch.rand(
                        (batch_size, 1, 1, 1, 1),
                        device=pred_hist.device,
                    ) < teacher_forcing_prob
                    gt_value = chunk_future_modalities[modality_name].to(
                        device=pred_hist.device,
                        dtype=feedback[modality_name].dtype,
                    )
                    feedback[modality_name] = torch.where(
                        use_gt,
                        gt_value,
                        feedback[modality_name],
                    )
                pred_hist = self._merge_modalities(
                    feedback["radar"], feedback["satellite"], feedback["rain"]
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
        for optional_key, optional_values in pred_optional.items():
            output[optional_key] = torch.cat(optional_values, dim=2)
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
            teacher_forcing_prob=float(rollout_branch_cfg["teacher_forcing_prob"]),
            teacher_forcing_modalities=tuple(rollout_branch_cfg["teacher_forcing_modalities"]),
            gradient_leads=tuple(rollout_branch_cfg["gradient_leads"]),
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
            "meta/rollout_branch_frames": torch.tensor(
                float(rollout_frames), device=rollout_raw_loss.device, dtype=rollout_raw_loss.dtype
            ),
            "meta/scheduled_sampling_prob": torch.tensor(
                float(rollout_branch_cfg["teacher_forcing_prob"]),
                device=rollout_raw_loss.device,
                dtype=rollout_raw_loss.dtype,
            ),
            "meta/rollout_stage_index": torch.tensor(
                float(rollout_branch_cfg.get("stage_index", -1)),
                device=rollout_raw_loss.device,
                dtype=rollout_raw_loss.dtype,
            ),
            "meta/rollout_stage_progress": torch.tensor(
                float(rollout_branch_cfg.get("stage_progress", 0.0)),
                device=rollout_raw_loss.device,
                dtype=rollout_raw_loss.dtype,
            ),
            "meta/rollout_gradient_leads": torch.tensor(
                float(len(tuple(rollout_branch_cfg["gradient_leads"]))),
                device=rollout_raw_loss.device,
                dtype=rollout_raw_loss.dtype,
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
                torch.save(self.ema_model.state_dict(), ckpt_dir / "ema.pt")
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
            ema_path = resume_dir / "ema.pt"
            if ema_path.exists():
                self.ema_model.load_state_dict(torch.load(ema_path, map_location=self.device))
            else:
                online_state = self.accelerator.unwrap_model(self.model).state_dict()
                self.ema_model.ema_model.load_state_dict(online_state, strict=True)
                self.log_msg(
                    "[Resume] checkpoint has no ema.pt; initialized EMA weights from the loaded online model."
                )
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
        training_model = self.model
        use_ema = (
            self.ema_model is not None
            and bool(self.val_cfg.get("use_ema", True))
            and self.global_step >= int(self.ema_cfg.update_after_step)
        )
        if use_ema:
            self.model = self.ema_model.ema_model
        self.model.eval()
        validation_model_name = "ema" if use_ema else "online"
        self.log_msg(
            f"[Val] start at step={self.global_step}, max_val_iters={max_iters}, "
            f"model={validation_model_name}"
        )

        loss_sum = torch.tensor(0.0, device=self.device)
        infer_loss_sum = torch.tensor(0.0, device=self.device)
        batch_count = torch.tensor(0.0, device=self.device)

        csi_thresholds = [float(v) for v in self.val_cfg.get("csi_thresholds", [0.5])]
        if len(csi_thresholds) == 0:
            raise ValueError("val.csi_thresholds should contain at least one threshold.")
        fss_thresholds = [float(v) for v in self.val_cfg.get("fss_thresholds", csi_thresholds)]
        fss_windows = [int(v) for v in self.val_cfg.get("fss_windows", [3, 9])]
        if not fss_thresholds or not fss_windows:
            raise ValueError("val.fss_thresholds and val.fss_windows must not be empty.")
        if any(window <= 0 or window % 2 == 0 for window in fss_windows):
            raise ValueError("val.fss_windows must contain positive odd integers.")
        csi_tp: torch.Tensor | None = None
        csi_fp: torch.Tensor | None = None
        csi_fn: torch.Tensor | None = None
        fss_numerator: torch.Tensor | None = None
        fss_reference: torch.Tensor | None = None
        rain_abs_error_sum: torch.Tensor | None = None
        rain_pixel_count: torch.Tensor | None = None

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
            lead_frames = int(rain_pred.shape[2])
            if csi_tp is None:
                metric_shape = (lead_frames, len(csi_thresholds))
                csi_tp = torch.zeros(metric_shape, device=self.device)
                csi_fp = torch.zeros(metric_shape, device=self.device)
                csi_fn = torch.zeros(metric_shape, device=self.device)
                fss_shape = (lead_frames, len(fss_thresholds), len(fss_windows))
                fss_numerator = torch.zeros(fss_shape, device=self.device)
                fss_reference = torch.zeros(fss_shape, device=self.device)
                rain_abs_error_sum = torch.zeros(lead_frames, device=self.device)
                rain_pixel_count = torch.zeros(lead_frames, device=self.device)
            elif int(csi_tp.shape[0]) != lead_frames:
                raise ValueError(
                    "Validation batches have inconsistent future lengths: "
                    f"expected {int(csi_tp.shape[0])}, got {lead_frames}"
                )

            assert csi_fp is not None and csi_fn is not None
            assert fss_numerator is not None and fss_reference is not None
            assert rain_abs_error_sum is not None and rain_pixel_count is not None
            for idx, threshold in enumerate(csi_thresholds):
                pred_bin = rain_pred >= threshold
                target_bin = rain_target >= threshold
                reduce_dims = (0, 1, 3, 4)
                csi_tp[:, idx] += (pred_bin & target_bin).sum(dim=reduce_dims, dtype=torch.float32)
                csi_fp[:, idx] += (pred_bin & ~target_bin).sum(dim=reduce_dims, dtype=torch.float32)
                csi_fn[:, idx] += (~pred_bin & target_bin).sum(dim=reduce_dims, dtype=torch.float32)

            b, c, t, h, w = rain_pred.shape
            for threshold_idx, threshold in enumerate(fss_thresholds):
                pred_binary = (rain_pred >= threshold).to(dtype=torch.float32)
                target_binary = (rain_target >= threshold).to(dtype=torch.float32)
                pred_bt = pred_binary.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
                target_bt = target_binary.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
                for window_idx, window in enumerate(fss_windows):
                    pred_fraction = F.avg_pool2d(
                        pred_bt, kernel_size=window, stride=1, padding=window // 2
                    ).view(b, t, c, h, w)
                    target_fraction = F.avg_pool2d(
                        target_bt, kernel_size=window, stride=1, padding=window // 2
                    ).view(b, t, c, h, w)
                    fss_numerator[:, threshold_idx, window_idx] += (
                        pred_fraction - target_fraction
                    ).pow(2).sum(dim=(0, 2, 3, 4))
                    fss_reference[:, threshold_idx, window_idx] += (
                        pred_fraction.pow(2) + target_fraction.pow(2)
                    ).sum(dim=(0, 2, 3, 4))

            rain_abs_error_sum += (rain_pred - rain_target).abs().sum(
                dim=(0, 1, 3, 4), dtype=torch.float32
            )
            pixels_per_lead = (
                int(rain_pred.shape[0])
                * int(rain_pred.shape[1])
                * int(rain_pred.shape[3])
                * int(rain_pred.shape[4])
            )
            rain_pixel_count += float(pixels_per_lead)

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
            self.model = training_model
            self.model.train()
            return

        if csi_tp is None or csi_fp is None or csi_fn is None:
            raise RuntimeError("Validation produced batches but no rain contingency statistics.")
        if fss_numerator is None or fss_reference is None:
            raise RuntimeError("Validation produced batches but no FSS statistics.")
        if rain_abs_error_sum is None or rain_pixel_count is None:
            raise RuntimeError("Validation produced batches but no per-lead rain error statistics.")

        loss_sum = self.accelerator.reduce(loss_sum, reduction="sum")
        infer_loss_sum = self.accelerator.reduce(infer_loss_sum, reduction="sum")
        batch_count = self.accelerator.reduce(batch_count, reduction="sum")
        csi_tp = self.accelerator.reduce(csi_tp, reduction="sum")
        csi_fp = self.accelerator.reduce(csi_fp, reduction="sum")
        csi_fn = self.accelerator.reduce(csi_fn, reduction="sum")
        fss_numerator = self.accelerator.reduce(fss_numerator, reduction="sum")
        fss_reference = self.accelerator.reduce(fss_reference, reduction="sum")
        rain_abs_error_sum = self.accelerator.reduce(rain_abs_error_sum, reduction="sum")
        rain_pixel_count = self.accelerator.reduce(rain_pixel_count, reduction="sum")
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

        metric_logs: dict[str, float] = {}
        csi_msg_parts = []
        total_tp = csi_tp.sum(dim=0)
        total_fp = csi_fp.sum(dim=0)
        total_fn = csi_fn.sum(dim=0)
        time_interval = int(self.dataset_cfg.get("time_interval", 1))
        for idx, threshold in enumerate(csi_thresholds):
            denom = total_tp[idx] + total_fp[idx] + total_fn[idx]
            csi_value = float((total_tp[idx] / denom.clamp_min(1e-8)).item())
            key = f"val/csi@{threshold:g}"
            metric_logs[key] = csi_value
            csi_msg_parts.append(f"csi@{threshold:g}={csi_value:.6f}")

            for lead_idx in range(int(csi_tp.shape[0])):
                tp = csi_tp[lead_idx, idx]
                fp = csi_fp[lead_idx, idx]
                fn = csi_fn[lead_idx, idx]
                lead_minutes = (lead_idx + 1) * time_interval
                prefix = f"val/lead_{lead_minutes:03d}m"
                lead_csi = tp / (tp + fp + fn).clamp_min(1.0e-8)
                lead_pod = tp / (tp + fn).clamp_min(1.0e-8)
                lead_far = fp / (tp + fp).clamp_min(1.0e-8)
                lead_bias = (tp + fp) / (tp + fn).clamp_min(1.0e-8)
                metric_logs[f"{prefix}/csi@{threshold:g}"] = float(lead_csi.item())
                metric_logs[f"{prefix}/pod@{threshold:g}"] = float(lead_pod.item())
                metric_logs[f"{prefix}/far@{threshold:g}"] = float(lead_far.item())
                metric_logs[f"{prefix}/frequency_bias@{threshold:g}"] = float(lead_bias.item())

        for threshold_idx, threshold in enumerate(fss_thresholds):
            for window_idx, window in enumerate(fss_windows):
                total_num = fss_numerator[:, threshold_idx, window_idx].sum()
                total_ref = fss_reference[:, threshold_idx, window_idx].sum()
                total_fss = 1.0 - total_num / total_ref.clamp_min(1.0e-8)
                metric_logs[f"val/fss@{threshold:g}/w{window}"] = float(total_fss.item())
                for lead_idx in range(int(fss_numerator.shape[0])):
                    lead_minutes = (lead_idx + 1) * time_interval
                    lead_fss = 1.0 - (
                        fss_numerator[lead_idx, threshold_idx, window_idx]
                        / fss_reference[lead_idx, threshold_idx, window_idx].clamp_min(1.0e-8)
                    )
                    metric_logs[
                        f"val/lead_{lead_minutes:03d}m/fss@{threshold:g}/w{window}"
                    ] = float(lead_fss.item())

        rain_mae_per_lead = rain_abs_error_sum / rain_pixel_count.clamp_min(1.0)
        for lead_idx in range(int(rain_mae_per_lead.numel())):
            lead_minutes = (lead_idx + 1) * time_interval
            metric_logs[f"val/lead_{lead_minutes:03d}m/rain_mae"] = float(
                rain_mae_per_lead[lead_idx].item()
            )

        last_lead_minutes = int(csi_tp.shape[0]) * time_interval
        last_threshold = csi_thresholds[0]
        last_lead_csi = metric_logs[
            f"val/lead_{last_lead_minutes:03d}m/csi@{last_threshold:g}"
        ]

        metric_msg = (
            f"[Val][{self.global_step}] "
            f"loss={val_loss:.6f} | infer_loss={val_infer_loss:.6f} | "
            f"radar_psnr={radar_psnr:.6f} | radar_ssim={radar_ssim:.6f} | "
            f"satellite_psnr={satellite_psnr:.6f} | satellite_ssim={satellite_ssim:.6f} | "
            f"{' | '.join(csi_msg_parts)} | "
            f"last_lead_csi@{last_threshold:g}={last_lead_csi:.6f}"
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
                "val/used_ema": 1.0 if use_ema else 0.0,
                **metric_logs,
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
        self.model = training_model
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
    config_name="rain_trainer_tempo_rollout_v4",

    version_base=None,
)
def main(cfg: DictConfig) -> None:
    catcher = logger.catch if PartialState().is_main_process else nullcontext
    with catcher():
        trainer = RainTSNextFrameTrainer(cfg)
        trainer.run()


if __name__ == "__main__":
    main()
