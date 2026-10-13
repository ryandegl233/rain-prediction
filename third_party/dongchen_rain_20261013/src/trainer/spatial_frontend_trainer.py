"""
Standalone trainer for the 1024 multimodal spatial enhancement frontend.

This trainer intentionally does not instantiate or call the RainPred backend.
It reuses the time-series rain dataloader contract and trains only the frontend
with pseudo-HR spatial enhancement targets.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Mapping

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import accelerate
import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.state import PartialState
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.networks.spatial_rain_upsample.upsampler import (
    MultimodalSpatialEnhancementFrontend,
    SpatialFrontendOutput,
    frontend_supervised_loss,
    frontend_unsupervised_loss,
    psnr,
    resize_bcthw,
    ssim_global,
)
from src.utils.visualization.plot import plot_any_modality


def get_frontend_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    reduced_factor: float = 2.0,
    last_epoch: int = -1,
    min_lr: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Local scheduler to avoid importing src.utils.train_utils on Python 3.11."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cycle_num = int(progress * num_cycles)
        cycle_progress = (progress * num_cycles) % 1.0
        max_lr = 1.0 / (float(reduced_factor) ** cycle_num)

        if cycle_num == 0:
            return min_lr + (max_lr - min_lr) * (0.5 * (1.0 + math.cos(math.pi * cycle_progress)))
        if cycle_progress < 0.5:
            return min_lr + (max_lr - min_lr) * (0.5 + 0.5 * math.cos(math.pi * (1 - 2 * cycle_progress)))
        return min_lr + (max_lr - min_lr) * (0.5 * (1.0 + math.cos(math.pi * (2 * cycle_progress - 1))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def _ensure_bcthw(x: torch.Tensor, *, expected_channels: int, name: str) -> torch.Tensor:
    if x.ndim == 4:
        x = x.unsqueeze(1)
    if x.ndim != 5:
        raise ValueError(f"{name} must be 4D/5D tensor, got shape={tuple(x.shape)}")
    if int(x.shape[1]) == int(expected_channels):
        return x
    if int(x.shape[2]) == int(expected_channels):
        return x.permute(0, 2, 1, 3, 4).contiguous()
    raise ValueError(f"{name} channel mismatch, expected C={expected_channels}, got shape={tuple(x.shape)}")


class SpatialFrontendTrainer:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.train_cfg = cfg.train
        self.val_cfg = cfg.val
        self.frontend_cfg = cfg.frontend
        self.loss_cfg = cfg.loss
        self.metric_cfg = cfg.metrics

        self.accelerator: Accelerator = hydra.utils.instantiate(cfg.accelerator)
        accelerate.utils.set_seed(int(self.train_cfg.get("seed", 2025)))
        self.device = self.accelerator.device

        self.log_file = self._configure_logger()
        self.log_msg(f"Log file: {self.log_file}")
        self.log_msg(f"Project dir: {self.proj_dir}")

        self.train_dataset, self.train_dataloader = hydra.utils.instantiate(cfg.dataset.train)
        self.val_dataset, self.val_dataloader = hydra.utils.instantiate(cfg.dataset.val)

        self.frontend: MultimodalSpatialEnhancementFrontend = hydra.utils.instantiate(cfg.frontend.model)
        self.optim = hydra.utils.instantiate(cfg.train.optim)(self.frontend.parameters())
        self.sched = hydra.utils.instantiate(cfg.train.scheduler)(optimizer=self.optim)

        self.frontend, self.optim, self.train_dataloader, self.val_dataloader, self.sched = self.accelerator.prepare(
            self.frontend,
            self.optim,
            self.train_dataloader,
            self.val_dataloader,
            self.sched,
        )
        self.global_step = 0
        self._resume_if_needed()

        target_size = tuple(int(v) for v in self.frontend_cfg.model.output_size)
        if target_size != (1024, 1024):
            raise ValueError(f"V1 frontend target size is fixed to (1024, 1024), got {target_size}")
        input_size = int(self.frontend_cfg.get("input_size", 448))
        output_h, output_w = target_size
        self.log_msg(
            f"Objective: standalone {input_size}x{input_size}->{output_h}x{output_w} "
            "multimodal spatial enhancement frontend"
        )

    def _configure_logger(self) -> Path:
        logger.remove()
        logger.add(
            sys.stdout,
            format="{time:HH:mm:ss} - {level.icon} <level>[{level}:{file.name}:{line}]</level> - <level>{message}</level>",
            level="DEBUG",
            colorize=True,
        )

        configured_proj_dir = self.train_cfg.get("proj_dir")
        if configured_proj_dir not in (None, ""):
            log_root = Path(str(configured_proj_dir))
            if bool(self.train_cfg.log.get("log_with_time", True)):
                log_root = log_root / time.strftime("%Y-%m-%d_%H-%M-%S")
            run_comment = self.train_cfg.log.get("run_comment", "")
            if run_comment:
                log_root = Path(f"{log_root.as_posix()}_{run_comment}")
        else:
            hydra_cfg = HydraConfig.get()
            log_root = Path(hydra_cfg.runtime.output_dir)

        log_file = log_root / "log.log"
        self.proj_dir = log_file.parent
        if self.accelerator.is_main_process:
            self.proj_dir.mkdir(parents=True, exist_ok=True)
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
            cfg_dump.write_text(OmegaConf.to_yaml(self.cfg, resolve=True), encoding="utf-8")

        self.accelerator.project_configuration.project_dir = str(self.proj_dir)
        self.accelerator.project_configuration.logging_dir = str(self.proj_dir / "tensorboard")
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers("rain_spatial_frontend_1024", config={})
        return log_file

    def log_msg(self, msg: str, level: str = "info") -> None:
        if self.accelerator.is_main_process:
            getattr(logger, level.lower())(msg)

    def _extract_inputs(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        radar = _ensure_bcthw(
            batch["radar_past"].to(self.device, dtype=torch.float32),
            expected_channels=int(self.frontend_cfg.model.radar_channels),
            name="radar_past",
        )
        satellite = _ensure_bcthw(
            batch["satellite_past"].to(self.device, dtype=torch.float32),
            expected_channels=int(self.frontend_cfg.model.satellite_channels),
            name="satellite_past",
        )
        rain = _ensure_bcthw(
            batch["rain_past"].to(self.device, dtype=torch.float32),
            expected_channels=int(self.frontend_cfg.model.rain_channels),
            name="rain_past",
        )
        expected_size = int(self.frontend_cfg.get("input_size", 448))
        if int(radar.shape[-2]) != expected_size or int(radar.shape[-1]) != expected_size:
            raise ValueError(
                f"V1 expects {expected_size}x{expected_size} frontend inputs, got radar={tuple(radar.shape)}"
            )
        return {"radar": radar, "satellite": satellite, "rain": rain}

    def _extract_hr_targets(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if "rain_past_hr" not in batch:
            raise KeyError("batch must include rain_past_hr for supervised spatial frontend training")
        rain = _ensure_bcthw(
            batch["rain_past_hr"].to(self.device, dtype=torch.float32),
            expected_channels=int(self.frontend_cfg.model.rain_channels),
            name="rain_past_hr",
        )
        output_h, output_w = (int(v) for v in self.frontend_cfg.model.output_size)
        if int(rain.shape[-2]) != output_h or int(rain.shape[-1]) != output_w:
            raise ValueError(f"rain_past_hr must be {output_h}x{output_w}, got shape={tuple(rain.shape)}")
        return {"rain": rain}

    def _degraded_metrics(
        self,
        outputs: Mapping[str, torch.Tensor],
        inputs: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        psnr_values: list[torch.Tensor] = []
        ssim_values: list[torch.Tensor] = []
        data_range = float(self.metric_cfg.get("data_range", 1.0))
        for name in ("radar", "satellite", "rain"):
            ref = inputs[name]
            degraded = resize_bcthw(outputs[name], size=(int(ref.shape[-2]), int(ref.shape[-1])), mode="area")
            psnr_values.append(psnr(degraded, ref, data_range=data_range))
            ssim_values.append(ssim_global(degraded, ref, data_range=data_range))
        return {
            "frontend/degraded_psnr": torch.stack(psnr_values).mean().detach(),
            "frontend/degraded_ssim": torch.stack(ssim_values).mean().detach(),
        }

    def _hr_baseline_metrics(
        self,
        output: SpatialFrontendOutput,
        high_targets: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        rain_target = high_targets["rain"]
        data_range = float(self.metric_cfg.get("data_range", 1.0))
        base_psnr = psnr(output.rain_base, rain_target, data_range=data_range).detach()
        enhanced_psnr = psnr(output.rain, rain_target, data_range=data_range).detach()
        base_ssim = ssim_global(output.rain_base, rain_target, data_range=data_range).detach()
        enhanced_ssim = ssim_global(output.rain, rain_target, data_range=data_range).detach()
        target_detail = rain_target - output.rain_base
        pred_detail = output.rain - output.rain_base
        logs = {
            "frontend/rain_base_hr_psnr": base_psnr,
            "frontend/rain_enhanced_hr_psnr": enhanced_psnr,
            "frontend/rain_base_hr_ssim": base_ssim,
            "frontend/rain_enhanced_hr_ssim": enhanced_ssim,
            "frontend/rain_base_hr_l1": F.l1_loss(output.rain_base, rain_target).detach(),
            "frontend/rain_enhanced_hr_l1": F.l1_loss(output.rain, rain_target).detach(),
            "frontend/rain_psnr_gain": (enhanced_psnr - base_psnr).detach(),
            "frontend/rain_detail_l1": F.l1_loss(pred_detail, target_detail).detach(),
            "frontend/rain_residual_abs_mean": output.rain_residual.detach().abs().mean(),
        }
        if output.rain_gate is not None:
            logs["frontend/rain_gate_mean"] = output.rain_gate.detach().mean()
            logs["frontend/rain_gate_std"] = output.rain_gate.detach().float().std(unbiased=False)
        return logs

    def _compute_loss_logs_and_output(
        self, batch: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor], SpatialFrontendOutput]:
        inputs = self._extract_inputs(batch)
        high_targets = self._extract_hr_targets(batch)
        output = self.frontend(**inputs)
        loss, logs = frontend_supervised_loss(
            output,
            inputs,
            high_targets,
            rain_hr_weight=float(self.loss_cfg.get("rain_hr_weight", 1.0)),
            rain_detail_weight=float(self.loss_cfg.get("rain_detail_weight", 0.25)),
            degradation_weight=float(self.loss_cfg.get("degradation_weight", 0.1)),
            residual_weight=float(self.loss_cfg.get("residual_weight", 1.0e-4)),
        )
        logs.update(self._degraded_metrics(output.enhanced(), inputs))
        logs.update(self._hr_baseline_metrics(output, high_targets))
        return loss, logs, inputs, output

    def _compute_loss_and_logs(self, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss, logs, _inputs, _output = self._compute_loss_logs_and_output(batch)
        return loss, logs

    def train_step(self, batch: Mapping[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], bool]:
        self.frontend.train()
        with self.accelerator.accumulate(self.frontend):
            with self.accelerator.autocast():
                loss, logs = self._compute_loss_and_logs(batch)
            self.accelerator.backward(loss)
            max_grad_norm = float(self.train_cfg.get("max_grad_norm", 0.0))
            if self.accelerator.sync_gradients and max_grad_norm > 0:
                self.accelerator.clip_grad_norm_(self.frontend.parameters(), max_grad_norm)
            if self.accelerator.sync_gradients:
                self.optim.step()
                self.sched.step()
                self.optim.zero_grad(set_to_none=True)

        did_step = bool(self.accelerator.sync_gradients)
        if did_step:
            self.global_step += 1
        return logs, did_step

    @torch.no_grad()
    def val_step(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.frontend.eval()
        with self.accelerator.autocast():
            _loss, logs = self._compute_loss_and_logs(batch)
        return logs

    def _visual_tensor(self, x: torch.Tensor, *, frame_idx: int) -> torch.Tensor:
        x = x.detach().float().cpu()
        frame_idx = max(-int(x.shape[2]), min(int(frame_idx), int(x.shape[2]) - 1))
        return x[:1, :, frame_idx]

    def _plot_modality(self, ax: plt.Axes, x: torch.Tensor, *, modality_name: str, title: str) -> None:
        image = plot_any_modality(
            x,
            modality_name=modality_name,  # type: ignore[arg-type]
            to_PIL=False,
        )
        ax.imshow(image)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    @torch.no_grad()
    def _save_val_visualization(
        self,
        inputs: Mapping[str, torch.Tensor],
        output: SpatialFrontendOutput,
    ) -> None:
        if not self.accelerator.is_main_process or not bool(self.val_cfg.get("save_visuals", True)):
            return

        frame_idx = int(self.val_cfg.get("visual_frame_idx", -1))
        visual_dir = self.proj_dir / "val_visualizations"
        visual_dir.mkdir(parents=True, exist_ok=True)

        enhanced = output.enhanced()
        bases = output.bases()
        fig, axes = plt.subplots(3, 4, figsize=(13.5, 10.0), squeeze=False)
        input_size = int(self.frontend_cfg.get("input_size", 448))
        output_h, output_w = (int(v) for v in self.frontend_cfg.model.output_size)
        output_label = str(output_h) if output_h == output_w else f"{output_h}x{output_w}"
        columns = (
            f"input_{input_size}",
            f"base_{output_label}",
            f"enhanced_{output_label}",
            f"degraded_{input_size}",
        )
        for row, modality in enumerate(("radar", "satellite", "rain")):
            low_ref = inputs[modality]
            degraded = resize_bcthw(
                enhanced[modality],
                size=(int(low_ref.shape[-2]), int(low_ref.shape[-1])),
                mode="area",
            )
            tensors = (low_ref, bases[modality], enhanced[modality], degraded)
            for col, (name, tensor) in enumerate(zip(columns, tensors)):
                self._plot_modality(
                    axes[row, col],
                    self._visual_tensor(tensor, frame_idx=frame_idx),
                    modality_name=modality,
                    title=f"{modality} {name}",
                )

        fig.suptitle(f"Validation visualization at step {self.global_step}", fontsize=12)
        fig.tight_layout()
        path = visual_dir / f"val_step_{self.global_step:08d}.png"
        fig.savefig(path, dpi=int(self.val_cfg.get("visual_dpi", 160)), bbox_inches="tight")
        plt.close(fig)
        self.log_msg(f"Saved validation visualization: {path}")

    def _log_metrics(self, logs: Mapping[str, torch.Tensor], prefix: str = "train") -> None:
        if not self.accelerator.is_main_process:
            return
        payload: dict[str, float] = {}
        for key, value in logs.items():
            log_key = key if key.startswith("loss/") or key.startswith("frontend/") else f"{prefix}/{key}"
            payload[log_key] = float(value.detach().float().item())
        self.accelerator.log(payload, step=self.global_step)
        if self.global_step % int(self.train_cfg.log.get("log_every", 10)) == 0:
            msg = " | ".join(f"{k}={v:.6f}" for k, v in payload.items())
            self.log_msg(f"[{prefix}][{self.global_step}] {msg}")

    @torch.no_grad()
    def _run_val(self) -> None:
        self.frontend.eval()
        max_iters = int(self.val_cfg.get("max_val_iters", 10))
        sums: dict[str, torch.Tensor] = {}
        count = 0
        for batch in tqdm(self.val_dataloader, desc="Val", disable=not self.accelerator.is_main_process):
            with self.accelerator.autocast():
                _loss, logs, inputs, output = self._compute_loss_logs_and_output(batch)
            if count == 0:
                self._save_val_visualization(inputs, output)
            for key, value in logs.items():
                sums[key] = sums.get(key, torch.zeros_like(value.detach().float())) + value.detach().float()
            count += 1
            if count >= max_iters:
                break
        if count <= 0:
            self.frontend.train()
            return
        count_tensor = torch.as_tensor(float(count), device=self.device)
        if hasattr(self.accelerator, "reduce"):
            count_tensor = self.accelerator.reduce(count_tensor, reduction="sum")
            sums = {key: self.accelerator.reduce(value.to(self.device), reduction="sum") for key, value in sums.items()}
        averaged = {f"val/{key}": value / count_tensor.clamp_min(1.0) for key, value in sums.items()}
        if self.accelerator.is_main_process:
            payload = {key: float(value.item()) for key, value in averaged.items()}
            self.accelerator.log(payload, step=self.global_step)
            msg = " | ".join(f"{k}={v:.6f}" for k, v in payload.items())
            self.log_msg(f"[Val][{self.global_step}] {msg}")
        self.frontend.train()

    def _save_checkpoint(self) -> None:
        if not self.accelerator.is_main_process:
            return
        ckpt_dir = self.proj_dir / f"checkpoint-{self.global_step:08d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "frontend.pt"
        state = {
            "frontend": self.accelerator.unwrap_model(self.frontend).state_dict(),
            "optimizer": self.optim.state_dict(),
            "scheduler": self.sched.state_dict(),
            "global_step": int(self.global_step),
            "config": OmegaConf.to_container(self.cfg, resolve=True),
        }
        torch.save(state, ckpt_path)
        meta = {"global_step": int(self.global_step), "checkpoint": str(ckpt_path)}
        (ckpt_dir / "frontend_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self.log_msg(f"Saved checkpoint: {ckpt_path}")

    def _resume_if_needed(self) -> None:
        resume_path = self.train_cfg.get("resume_path")
        if resume_path in (None, ""):
            return
        ckpt_path = Path(str(resume_path))
        if ckpt_path.is_dir():
            ckpt_path = ckpt_path / "frontend.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"frontend checkpoint not found: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        frontend = self.accelerator.unwrap_model(self.frontend)
        frontend.load_state_dict(state["frontend"], strict=True)
        self.optim.load_state_dict(state["optimizer"])
        self.sched.load_state_dict(state["scheduler"])
        self.global_step = int(state.get("global_step", 0))
        self.log_msg(f"Resumed frontend checkpoint: {ckpt_path}, global_step={self.global_step}")

    def train(self) -> None:
        self.frontend.train()
        stop = False
        while not stop:
            for batch in self.train_dataloader:
                logs, did_step = self.train_step(batch)
                if not did_step:
                    continue
                self._log_metrics(logs, prefix="train")
                if self.global_step % int(self.val_cfg.get("val_duration", 200)) == 0:
                    self._run_val()
                if self.global_step % int(self.train_cfg.get("save_every", 1000)) == 0:
                    self._save_checkpoint()
                if self.global_step >= int(self.train_cfg.max_steps):
                    stop = True
                    break
        self._save_checkpoint()
        self.log_msg("Frontend training finished.")

    def run(self) -> None:
        self.train()


@hydra.main(
    config_path="../config/ts_rain_train",
    config_name="rain_frontend_1024",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    catcher = logger.catch if PartialState().is_main_process else nullcontext
    with catcher():
        trainer = SpatialFrontendTrainer(cfg)
        trainer.run()


if __name__ == "__main__":
    main()
