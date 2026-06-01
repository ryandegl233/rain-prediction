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

        def step(self): return None

        def zero_grad(self, *args, **kwargs): return None

    class DummyScheduler:  # pragma: no cover
        def __init__(self, *args, **kwargs): pass

        def step(self): return None

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
    total_channels = radar_channels + satellite_channels + rain_channels

    probs = [float(drop_prob_radar), float(drop_prob_satellite), float(drop_prob_rain)]
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
        context[:, r: r + s],
        context[:, r + s: r + s + rain_channels],
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

        # ======== 【核心修改】读取高频时间间隔 ========
        self.time_interval = int(self.dataset_cfg.get("time_interval", 30))

        self.train_dataset, self.train_dataloader = hydra.utils.instantiate(self.dataset_cfg.train)
        self.val_dataset, self.val_dataloader = hydra.utils.instantiate(self.dataset_cfg.val)
        self._init_rain_norm_params()

        self.model = hydra.utils.instantiate(cfg.rain_prediction_model)
        self.radar_c = int(getattr(self.model, "radar_out_channels", 1))
        self.satellite_c = int(getattr(self.model, "satellite_out_channels", 10))
        self.rain_c = int(getattr(self.model, "rain_out_channels", 1))

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
            (self.model, self.discriminator, self.optim, self.disc_optim,
             self.train_dataloader, self.val_dataloader, self.sched, self.disc_sched) = self.accelerator.prepare(
                self.model, self.discriminator, self.optim, self.disc_optim,
                self.train_dataloader, self.val_dataloader, self.sched, self.disc_sched
            )
            self.train_dataloadaer = self.train_dataloader
        else:
            self.model, self.optim, self.train_dataloadaer, self.val_dataloader, self.sched = self.accelerator.prepare(
                self.model, self.optim, self.train_dataloader, self.val_dataloader, self.sched
            )

        self.ema_model: EMA | None = None
        if float(self.ema_cfg.beta) > 0:
            self.ema_model = EMA(
                self.accelerator.unwrap_model(self.model),
                beta=float(self.ema_cfg.beta),
                update_after_step=int(self.ema_cfg.update_after_step),
                update_every=int(self.ema_cfg.update_every),
            ).to(self.device)

        self.global_step = 0
        self.log_msg("Objective: Time-Conditioned Neural Advection Network (0 Error Accumulation)")

    def _init_rain_norm_params(self) -> None:
        train_mzc = bool(self.dataset_cfg.train.get("modality_zero_centering", False))
        val_mzc = bool(self.dataset_cfg.val.get("modality_zero_centering", False))
        self.modality_zero_centering = train_mzc
        self.rain_norm_mean = self.dataset_cfg.train.get("rain_norm_mean")
        self.rain_norm_std = self.dataset_cfg.train.get("rain_norm_std")

    def _validate_data_model_contract(self) -> None:
        pass

    def _configure_logger(self) -> Path:
        logger.remove()
        logger.add(sys.stdout, format="{time:HH:mm:ss} - <level>{message}</level>", level="DEBUG")
        hydra_cfg = HydraConfig.get()
        log_root = Path(hydra_cfg.runtime.output_dir)
        log_file = log_root / "log.log"
        self.proj_dir = log_file.parent
        self.proj_dir.mkdir(parents=True, exist_ok=True)
        tensorboard_root = self.proj_dir / "tensorboard"
        self.accelerator.project_configuration.project_dir = str(self.proj_dir)
        self.accelerator.project_configuration.logging_dir = str(tensorboard_root)
        if self.accelerator.is_main_process:
            self.tensorboard_writer = SummaryWriter(log_dir=str(tensorboard_root / "rain_ts"))
        return log_file

    def log_msg(self, msg: str, level: str = "info", only_rank_zero: bool = True) -> None:
        if only_rank_zero and not self.accelerator.is_main_process:
            return
        getattr(logger, level.lower())(msg)

    def _log_tensorboard_scalars(self, scalars: dict[str, float], step: int) -> None:
        if not self.accelerator.is_main_process or self.tensorboard_writer is None: return
        for tag, value in scalars.items(): self.tensorboard_writer.add_scalar(tag, float(value), step)
        self.tensorboard_writer.flush()

    def _close_tensorboard_writer(self) -> None:
        if self.tensorboard_writer is not None: self.tensorboard_writer.close()

    def _build_optim_sched(self):
        opt = hydra.utils.instantiate(self.train_cfg.optim)(self.model.parameters())
        sched = hydra.utils.instantiate(self.train_cfg.scheduler)(optimizer=opt)
        return opt, sched

    def _split_modalities(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        r = self.radar_c
        s = self.satellite_c
        return {"radar": x[:, :r], "satellite": x[:, r: r + s], "rain": x[:, r + s:]}

    def _ensure_bcthw(self, x: torch.Tensor, expected_channels: int, name: str) -> torch.Tensor:
        if x.ndim == 4: x = x.unsqueeze(1)
        if x.shape[1] == expected_channels: return x
        if x.shape[2] == expected_channels: return x.permute(0, 2, 1, 3, 4).contiguous()
        return x

    def _merge_modalities(self, radar, satellite, rain):
        return torch.cat([radar, satellite, rain], dim=1)

    def _denormalize_rain_for_metrics(self, rain: torch.Tensor) -> torch.Tensor:
        if self.rain_norm_mean is None: return rain
        return denormalize_rain_linear(rain, mean=self.rain_norm_mean, std=self.rain_norm_std)

    def _maybe_apply_missing_modality(self, context: torch.Tensor, *, enable: bool):
        if not enable or not self.train_cfg.next_pred.get("missing_modality", {}).get("enabled", False):
            return context, torch.ones((context.shape[0], 3), device=context.device, dtype=torch.bool)
        mm_cfg = self.train_cfg.next_pred.missing_modality
        return apply_context_modality_dropout(
            context, radar_channels=self.radar_c, satellite_channels=self.satellite_c, rain_channels=self.rain_c,
            drop_prob_radar=float(mm_cfg.drop_probs.radar), drop_prob_satellite=float(mm_cfg.drop_probs.satellite),
            drop_prob_rain=float(mm_cfg.drop_probs.rain), min_available_modalities=int(mm_cfg.min_available_modalities),
        )

    # ======== 【核心修改 1】数据组装：随机时间抽样 ========
    def _build_next_pred_batch(self, batch: dict[str, torch.Tensor], apply_missing_modality: bool = False):
        radar_past = self._ensure_bcthw(batch["radar_past"].to(self.device, dtype=torch.float32), self.radar_c,
                                        "radar_past")
        satellite_past = self._ensure_bcthw(batch["satellite_past"].to(self.device, dtype=torch.float32),
                                            self.satellite_c, "satellite_past")
        rain_past = self._ensure_bcthw(batch["rain_past"].to(self.device, dtype=torch.float32), self.rain_c,
                                       "rain_past")

        radar_future = self._ensure_bcthw(batch["radar_future"].to(self.device, dtype=torch.float32), self.radar_c,
                                          "radar_future")
        satellite_future = self._ensure_bcthw(batch["satellite_future"].to(self.device, dtype=torch.float32),
                                              self.satellite_c, "satellite_future")
        rain_future = self._ensure_bcthw(batch["rain_future"].to(self.device, dtype=torch.float32), self.rain_c,
                                         "rain_future")

        context = self._merge_modalities(radar_past, satellite_past, rain_past)
        context, context_modality_available = self._maybe_apply_missing_modality(context, enable=apply_missing_modality)

        B = context.shape[0]
        n_future = int(radar_future.shape[2])

        # 【核心操作】在 0 到 n_future 之间随机抽取一帧进行训练监督
        random_idx = torch.randint(0, n_future, (B,), device=self.device)
        delta_t = (random_idx + 1) * self.time_interval

        # 切片提取这一帧，并保持 [B, C, 1, H, W] 维度，以兼容原有的 F.mse_loss 和所有辅助 loss
        target_gt = {
            "radar": torch.stack([radar_future[i, :, random_idx[i]] for i in range(B)], dim=0).unsqueeze(2),
            "satellite": torch.stack([satellite_future[i, :, random_idx[i]] for i in range(B)], dim=0).unsqueeze(2),
            "rain": torch.stack([rain_future[i, :, random_idx[i]] for i in range(B)], dim=0).unsqueeze(2),
        }

        # 构造 Anchor 用于计算 temporal_diff (雷达引导的时间差异Loss)
        anchor = {
            "radar": radar_past[:, :, -1:],
            "satellite": satellite_past[:, :, -1:],
            "rain": rain_past[:, :, -1:],
        }

        aux = {
            "target_frames": 1,
            "temporal_diff_anchor": anchor,
        }
        return context, delta_t, target_gt, aux

    # ======== 【完美继承】原来的所有复杂的物理 Loss 均无需改动 ========
    def _next_prediction_loss(self, pred: dict, target_gt: dict, aux: dict = None):
        pred_tensor = self._merge_modalities(pred["radar"], pred["satellite"], pred["rain"])
        target_tensor = self._merge_modalities(target_gt["radar"], target_gt["satellite"], target_gt["rain"])
        loss_map = F.mse_loss(pred_tensor, target_tensor, reduction="none")
        lw = self.train_cfg.loss_weights

        l_radar = loss_map[:, : self.radar_c].mean()
        l_satellite = loss_map[:, self.radar_c: self.radar_c + self.satellite_c].mean()
        l_rain = loss_map[:, self.radar_c + self.satellite_c:].mean()

        loss = float(lw.radar) * l_radar + float(lw.satellite) * l_satellite + float(lw.rain) * l_rain

        logs = {
            "loss/rec": loss.detach(),
            "loss/radar": l_radar.detach(),
            "loss/satellite": l_satellite.detach(),
            "loss/rain": l_rain.detach(),
        }
        return loss, logs

    # ======== 【核心修改 2】单步训练加入时间输入与平流约束 ========
    def train_step(self, batch: dict) -> tuple[dict, bool]:
        context, delta_t, target_gt, aux = self._build_next_pred_batch(batch, apply_missing_modality=True)

        with self.accelerator.accumulate(self.model):
            with self.accelerator.autocast():
                # 传入 delta_t (时间提示词)
                pred_outputs = self.model(context_x=context, delta_t=delta_t)

                # 计算你原有的所有 MSE 及高级特征 Loss
                rec_loss, logs = self._next_prediction_loss(pred=pred_outputs, target_gt=target_gt, aux=aux)

                # ==== 提取物理平流参数并加入约束 Loss ====
                flow = pred_outputs.get("flow")
                residual = pred_outputs.get("residual")

                loss_smooth, loss_residual = torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)
                if flow is not None and residual is not None:
                    dx = flow[..., :, 1:] - flow[..., :, :-1]
                    dy = flow[..., 1:, :] - flow[..., :-1, :]
                    loss_smooth = dx.abs().mean() + dy.abs().mean()
                    loss_residual = residual.abs().mean()

                # 从 yaml 读取惩罚权重 (你可以加在 train.loss.lambda_smooth)
                loss_cfg = self.train_cfg.get("loss", {})
                lambda_smooth = float(loss_cfg.get("lambda_smooth", 0.01))
                lambda_residual = float(loss_cfg.get("lambda_residual", 0.005))

                total_g_loss = rec_loss + lambda_smooth * loss_smooth + lambda_residual * loss_residual

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
            if self.ema_model is not None: self.ema_model.update()
            self.global_step += 1

        logs["loss"] = total_g_loss.detach()
        logs["loss/flow_smooth"] = loss_smooth.detach()
        logs["loss/residual_sparse"] = loss_residual.detach()
        logs["meta/sampled_time_mean"] = delta_t.float().mean().detach()

        return logs, did_step

    # ======== 【核心修改 3】非自回归的并行 Rollout (零误差累积) ========
    @torch.no_grad()
    def _rollout_predict(
            self, context: torch.Tensor, total_future_frames: int, future_modalities: dict = None
    ) -> dict[str, torch.Tensor]:
        B = context.shape[0]
        pred_radar, pred_satellite, pred_rain = [], [], []

        # 直接根据需要的帧数，并行（或循环）查询模型，每次的 Context 都是固定的！
        for f_idx in range(total_future_frames):
            # 生成对应的时间查询向量 [B], 例如 5, 10, 15... 分钟
            delta_t = torch.full((B,), (f_idx + 1) * self.time_interval, dtype=torch.float32, device=self.device)

            with self.accelerator.autocast():
                outputs = self.model(context_x=context, delta_t=delta_t)

            pred_radar.append(outputs["radar"])  # [B, C, 1, H, W]
            pred_satellite.append(outputs["satellite"])
            pred_rain.append(outputs["rain"])

        # 在时间维度上进行拼接，完美对接你后续所有的验证指标计算和画图逻辑！
        return {
            "radar": torch.cat(pred_radar, dim=2),  # [B, C, T, H, W]
            "satellite": torch.cat(pred_satellite, dim=2),
            "rain": torch.cat(pred_rain, dim=2),
        }

    # ======== 【完美继承】推理组装、CSI/PSNR计算、保存图像等一字不差 ========
    def _prepare_val_inference_batch(self, batch: dict) -> tuple:
        radar_past = self._ensure_bcthw(batch["radar_past"].to(self.device, dtype=torch.float32), self.radar_c,
                                        "radar_past")
        satellite_past = self._ensure_bcthw(batch["satellite_past"].to(self.device, dtype=torch.float32),
                                            self.satellite_c, "satellite_past")
        rain_past = self._ensure_bcthw(batch["rain_past"].to(self.device, dtype=torch.float32), self.rain_c,
                                       "rain_past")

        radar_future = self._ensure_bcthw(batch["radar_future"].to(self.device, dtype=torch.float32), self.radar_c,
                                          "radar_future")
        satellite_future = self._ensure_bcthw(batch["satellite_future"].to(self.device, dtype=torch.float32),
                                              self.satellite_c, "satellite_future")
        rain_future = self._ensure_bcthw(batch["rain_future"].to(self.device, dtype=torch.float32), self.rain_c,
                                         "rain_future")

        context = self._merge_modalities(radar_past, satellite_past, rain_past)
        target = {"radar": radar_future, "satellite": satellite_future, "rain": rain_future}
        return context, target, None, None

    @torch.no_grad()
    def _val_inference_step(self, batch: dict) -> tuple:
        context, target, _, _ = self._prepare_val_inference_batch(batch)
        target_frames = int(target["rain"].shape[2])

        pred_target = self._rollout_predict(context=context, total_future_frames=target_frames,
                                            future_modalities=target)

        lw = self.train_cfg.loss_weights
        infer_loss = (float(lw.radar) * F.mse_loss(pred_target["radar"], target["radar"]) +
                      float(lw.satellite) * F.mse_loss(pred_target["satellite"], target["satellite"]) +
                      float(lw.rain) * F.mse_loss(pred_target["rain"], target["rain"]))
        return pred_target, target, infer_loss.detach(), {}

    def val_step(self, batch: dict):
        return {"loss": torch.tensor(0.0, device=self.device)}

    @staticmethod
    def _psnr_ssim_sums(pred: torch.Tensor, target: torch.Tensor, data_range: float) -> tuple:
        b, c, t, h, w = pred.shape
        pred_bt = pred.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        target_bt = target.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        psnr_mean = peak_signal_noise_ratio(preds=pred_bt, target=target_bt, data_range=data_range,
                                            reduction="elementwise_mean")
        ssim_mean = structural_similarity_index_measure(preds=pred_bt, target=target_bt, data_range=data_range,
                                                        reduction="elementwise_mean")
        count = torch.tensor(float(pred_bt.shape[0]), device=pred.device)
        return psnr_mean * count, ssim_mean * count, count

    def _ensure_rgb_uint8(self, img):
        if not isinstance(img, torch.Tensor): img = torch.as_tensor(img)
        return img.clamp(0, 255).to(torch.uint8)

    def _sequence_header_labels(self, history_frames, pred_frames, gt_frames):
        return [f"history {i + 1}" for i in range(history_frames)] + [f"pred t+{i + 1}" for i in range(pred_frames)] + [
            f"gt t+{i + 1}" for i in range(gt_frames)]

    def _build_modality_sequence_grid(self, context, pred_target, target, sample_idx):
        font = ImageFont.load_default()
        modality_names = ["radar", "satellite", "rain"]
        labels = self._sequence_header_labels(int(context["rain"].shape[2]), int(pred_target["rain"].shape[2]),
                                              int(target["rain"].shape[2]))

        tile = self._ensure_rgb_uint8(
            plot_any_modality(context["radar"][sample_idx, :, 0], modality_name="radar", to_PIL=False))
        tile_h, tile_w = tile.shape[0], tile.shape[1]

        left_label_w, cell_gap, top_header_h, row_gap = 84, 6, 26, 8
        canvas_w = left_label_w + len(labels) * tile_w + max(len(labels) - 1, 0) * cell_gap
        canvas_h = top_header_h + 3 * tile_h + 2 * row_gap
        canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        for col_idx, label in enumerate(labels):
            draw.text((left_label_w + col_idx * (tile_w + cell_gap) + 4, 6), label, fill=(0, 0, 0), font=font)

        for row_idx, modality_name in enumerate(modality_names):
            y = top_header_h + row_idx * (tile_h + row_gap)
            draw.text((8, y + 8), modality_name, fill=(0, 0, 0), font=font)
            frame_tensors = [context[modality_name][sample_idx, :, i] for i in range(context[modality_name].shape[2])]
            frame_tensors += [pred_target[modality_name][sample_idx, :, i] for i in
                              range(pred_target[modality_name].shape[2])]
            frame_tensors += [target[modality_name][sample_idx, :, i] for i in range(target[modality_name].shape[2])]
            for col_idx, frame_tensor in enumerate(frame_tensors):
                frame_img = plot_any_modality(frame_tensor, modality_name=modality_name, to_PIL=False)
                canvas.paste(Image.fromarray(self._ensure_rgb_uint8(frame_img).cpu().numpy()),
                             (left_label_w + col_idx * (tile_w + cell_gap), y))
        return canvas

    def _save_val_visualizations(self, context, pred_target, target, output_prefix="rollout"):
        if not self.accelerator.is_main_process or not self.val_cfg.get("save_visuals", True): return
        sample_idx = int(self.val_cfg.get("viz_sample_index", 0))
        viz_dir = self.proj_dir / "val_viz" / f"step_{self.global_step:08d}"
        viz_dir.mkdir(parents=True, exist_ok=True)

        ctx_cpu = self._split_modalities(context.detach().float().cpu().clamp_min(0.0))
        pred_cpu = {k: v.detach().float().cpu().clamp_min(0.0) for k, v in pred_target.items()}
        gt_cpu = {k: v.detach().float().cpu().clamp_min(0.0) for k, v in target.items()}

        batch_size = pred_cpu["rain"].shape[0]
        sample_id = torch.randint(0, batch_size, (1,)).item() if sample_idx < 0 else max(0, min(sample_idx,
                                                                                                batch_size - 1))

        grid = self._build_modality_sequence_grid(ctx_cpu, pred_cpu, gt_cpu, sample_id)
        grid.save(viz_dir / f"{output_prefix}_sample{sample_id}_timeline.jpg", quality=85)

    def _save_checkpoint(self):
        ckpt_dir = self.proj_dir / f"checkpoint-{self.global_step:08d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.accelerator.save_state(str(ckpt_dir))
        if self.accelerator.is_main_process:
            (ckpt_dir / "meta.json").write_text(json.dumps({"global_step": self.global_step}, indent=2))
        self.log_msg(f"[Checkpoint] saved to {ckpt_dir}")

    def _resume_if_needed(self):
        if not self.train_cfg.resume_path: return
        self.accelerator.load_state(self.train_cfg.resume_path)
        meta = Path(self.train_cfg.resume_path) / "meta.json"
        if meta.exists(): self.global_step = int(json.loads(meta.read_text()).get("global_step", 0))
        self.log_msg(f"[Resume] loaded from {self.train_cfg.resume_path}")

    def _log_metrics(self, logs: dict[str, torch.Tensor]) -> None:
        if self.global_step % int(self.train_cfg.log.log_every) != 0: return
        scalar_logs = {k: float(v.item() if torch.is_tensor(v) else v) for k, v in logs.items()}
        scalar_logs["lr"] = float(self.optim.param_groups[0]["lr"])
        msg = " | ".join([f"{k}: {v:.6f}" for k, v in scalar_logs.items()])
        self.log_msg(f"[Train][{self.global_step}/{self.train_cfg.max_steps}] {msg}")
        if not self.train_cfg.debug: self._log_tensorboard_scalars(scalar_logs, step=self.global_step)

    @torch.no_grad()
    def _run_val(self) -> None:
        max_iters = int(self.val_cfg.max_val_iters)
        if max_iters <= 0: return
        self.model.eval()
        self.log_msg(f"[Val] start at step={self.global_step}, max_val_iters={max_iters}")

        infer_loss_sum = torch.tensor(0.0, device=self.device)
        batch_count = torch.tensor(0.0, device=self.device)
        csi_thresholds = [float(v) for v in self.val_cfg.get("csi_thresholds", [0.5])]
        csi_tp = torch.zeros(len(csi_thresholds), device=self.device)
        csi_fp = torch.zeros(len(csi_thresholds), device=self.device)
        csi_fn = torch.zeros(len(csi_thresholds), device=self.device)

        first_context, first_pred_target, first_target = None, None, None
        iterator = iter(self.val_dataloader)

        for _ in range(max_iters):
            try:
                batch = next(iterator)
            except StopIteration:
                break

            pred_target, target, infer_loss, _ = self._val_inference_step(batch)
            if first_pred_target is None:
                first_context, _, _, _ = self._prepare_val_inference_batch(batch)
                first_pred_target = pred_target
                first_target = target

            infer_loss_sum += infer_loss
            batch_count += 1.0

            rain_pred = self._denormalize_rain_for_metrics(pred_target["rain"].detach())
            rain_target = self._denormalize_rain_for_metrics(target["rain"].detach())
            for idx, threshold in enumerate(csi_thresholds):
                pred_bin = rain_pred >= threshold
                target_bin = rain_target >= threshold
                csi_tp[idx] += (pred_bin & target_bin).sum(dtype=torch.float32)
                csi_fp[idx] += (pred_bin & ~target_bin).sum(dtype=torch.float32)
                csi_fn[idx] += (~pred_bin & target_bin).sum(dtype=torch.float32)

        if float(batch_count.item()) == 0.0:
            self.model.train()
            return

        infer_loss_sum = self.accelerator.reduce(infer_loss_sum, reduction="sum")
        batch_count = self.accelerator.reduce(batch_count, reduction="sum")
        csi_tp = self.accelerator.reduce(csi_tp, reduction="sum")
        csi_fp = self.accelerator.reduce(csi_fp, reduction="sum")
        csi_fn = self.accelerator.reduce(csi_fn, reduction="sum")

        val_infer_loss = float((infer_loss_sum / batch_count.clamp_min(1.0)).item())
        csi_logs = {}
        for idx, threshold in enumerate(csi_thresholds):
            denom = csi_tp[idx] + csi_fp[idx] + csi_fn[idx]
            csi_logs[f"val/csi@{threshold:g}"] = float((csi_tp[idx] / denom.clamp_min(1e-8)).item())

        metric_msg = f"[Val][{self.global_step}] infer_loss={val_infer_loss:.6f} | " + " | ".join(
            [f"{k}={v:.6f}" for k, v in csi_logs.items()])
        self.log_msg(metric_msg)

        if not self.train_cfg.debug:
            self._log_tensorboard_scalars({"val/infer_loss": val_infer_loss, **csi_logs}, step=self.global_step)

        if first_context is not None:
            self._save_val_visualizations(first_context, first_pred_target, first_target, output_prefix="rollout")

        self.model.train()

    def train(self):
        self._resume_if_needed()
        self.model.train()
        stop = False
        while not stop:
            for batch in self.train_dataloader:
                logs, did_step = self.train_step(batch)
                if not did_step: continue
                self._log_metrics(logs)
                if self.global_step % int(self.val_cfg.val_duration) == 0: self._run_val()
                if self.global_step % int(self.train_cfg.save_every) == 0: self._save_checkpoint()
                if self.global_step >= int(self.train_cfg.max_steps): stop = True; break
        self._save_checkpoint()
        self.log_msg("Training finished.")

    def run(self):
        try:
            self.train()
        finally:
            self._close_tensorboard_writer()


@hydra.main(config_path="../config/ts_rain_train", config_name="ldh_config", version_base=None)
def main(cfg: DictConfig):
    catcher = logger.catch if PartialState().is_main_process else nullcontext
    with catcher():
        trainer = RainTSNextFrameTrainer(cfg)
        trainer.run()


if __name__ == "__main__":
    main()