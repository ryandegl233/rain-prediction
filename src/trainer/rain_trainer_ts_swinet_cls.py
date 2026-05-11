"""
Rain prediction trainer for non-time series data.
This module is designed to handle rain prediction tasks using non-time series data.

Author: Zihan Cao
Date: 2025-08-08
"""

import sys
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Literal, Sequence, cast

import accelerate
import colored_traceback
import hydra
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.state import PartialState
from accelerate.tracking import TensorBoardTracker
from accelerate.utils.deepspeed import DummyOptim, DummyScheduler
from einops import rearrange
from ema_pytorch import EMA
from kornia.utils.image import make_grid, tensor_to_image
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor
from torchmetrics.aggregation import MeanMetric
from tqdm import trange
import matplotlib.colors as mcolors

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.network_utils import safe_dtensor_operation
from src.utils.train_utils import StepsCounter
from src.utils.visualization.plot import plot_any_modality
from src.utils.metrics.compute_metrics_new import RainGlobalMetricsAccumulator
from src.utils.metrics.compute_metrics_cls import GlobalMetricsAccumulator 

colored_traceback.add_hook()

class FocalLoss(nn.Module):
    def __init__(self, gamma=3, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight

    def forward(self, inputs, targets):
        """
        inputs: [B, C, H, W] or [B, C] logits (unnormalized)
        targets: [B, H, W] or [B] labels
        """
        # 1. 确保输入是 float32
        if inputs.dtype != torch.float32:
            inputs = inputs.float()
            
        # 2. 处理权重
        if self.weight is not None:
            if self.weight.device != inputs.device:
                self.weight = self.weight.to(inputs.device)
        
        # 3. 计算 Log Softmax (数值最稳定)
        # log_pt = log(softmax(x))
        log_pt = F.log_softmax(inputs, dim=1)
        
        # 4. 获取目标类别的 log_pt
        # target 需要 unsqueeze 才能 gather: [B, H, W] -> [B, 1, H, W]
        if targets.ndim == inputs.ndim - 1:
            targets = targets.unsqueeze(1)
            
        # gather: 提取正确标签对应的 log_prob
        target_log_pt = log_pt.gather(1, targets.long())
        target_log_pt = target_log_pt.squeeze(1) # [B, H, W]
        
        # 5. 计算 pt
        pt = target_log_pt.exp()
        
        # 6. 计算 Focal Loss
        # L = - (1-pt)^gamma * log(pt)
        # 这里的 log(pt) 就是 target_log_pt，直接用，无需再 log
        focal_term = (1 - pt) ** self.gamma
        loss = -focal_term * target_log_pt

        # 7. 应用类别权重 (如果有)
        if self.weight is not None:
            # 根据 target 索引取权重
            if targets.shape[1] == 1: targets = targets.squeeze(1)
            loss_weight = self.weight[targets.long()]
            loss = loss * loss_weight

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class RainPredictionTrainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.train_cfg = cfg.train
        self.dataset_cfg = cfg.dataset
        self.ema_cfg = cfg.ema
        self.val_cfg = cfg.val

        # accelerator
        self.accelerator: Accelerator = hydra.utils.instantiate(cfg.accelerator)
        accelerate.utils.set_seed(2025)

        # logger
        log_file = self.configure_logger()

        # attributes
        self.device = self.accelerator.device
        torch.cuda.set_device(self.accelerator.local_process_index)
        self.dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "no": torch.float32,
        }[self.accelerator.mixed_precision]
        self.log_msg("Log file is saved at: {}".format(log_file))
        self.log_msg("Weights will be saved at: {}".format(self.proj_dir))
        self.log_msg("Training is configured and ready to start.")

        # is zero 2 or 3, not EMA
        _dpsp_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        _fsdp_plugin: accelerate.utils.FullyShardedDataParallelPlugin = getattr(  # type: ignore
            self.accelerator.state, "fsdp_plugin", None
        )
        self.no_ema = False
        self._is_ds = _dpsp_plugin is not None
        if self._is_ds:
            self.log_msg("[Deepspeed]: using deepspeed plugin")
            self.no_ema = _dpsp_plugin.deepspeed_config["zero_optimization"][  # type: ignore
                "stage"
            ] in [2, 3]

        self._is_fsdp = _fsdp_plugin is not None
        if self._is_fsdp:
            self.log_msg("[FSDP]: using Fully Sharded Data Parallel plugin")
            self.no_ema = True

        # dataloader
        # used_dataset = self.dataset_cfg.cfgs.used
        # self.log_msg(f"[Data]: using dataset {used_dataset}")
        self.train_dataset, self.train_dataloader = hydra.utils.instantiate(
            self.dataset_cfg.train
        )
        self.val_dataset, self.val_dataloader = hydra.utils.instantiate(
            self.dataset_cfg.val
        )
        if _dpsp_plugin is not None:
            self.accelerator.deepspeed_plugin.deepspeed_config[  # type: ignore
                "train_micro_batch_size_per_gpu"
            ] = self.dataset_cfg.batch_size_train

        # setup the tokenizer
        self.setup_rain_predict_model()  # setup the rain prediction model

        # optimizers and lr schedulers
        self.optim, self.sched = self.get_optimizer_lr_scheduler()

        # EMA models and accelerator prepare
        self.prepare_for_training()
        self.prepare_ema_models()

        # loss
        #self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn = FocalLoss()
        self.log_msg(f"use rain prediction loss: {self.loss_fn.__class__.__name__}")

        # training state counter
        self.train_state = StepsCounter(["train"])

        # clear GPU memory
        torch.cuda.empty_cache()

    def setup_rain_predict_model(self):
        self.model = hydra.utils.instantiate(self.cfg.rain_prediction_model)
        
       

    def prepare_ema_models(self):
        if self.no_ema:
            return

        self.ema_model = EMA(
            self.model,
            beta=self.ema_cfg.beta,
            update_every=self.ema_cfg.update_every,
        ).to(self.device)
        self.log_msg(f"create EMA model for rain prediction")

    def configure_logger(self):
        self.logger = logger

        log_file = Path(self.train_cfg.proj_dir)
        if self.train_cfg.log.log_with_time:
            str_time = time.strftime("%Y-%m-%d_%H-%M-%S")
            log_file = log_file / str_time
        if self.train_cfg.log.run_comment is not None:
            log_file = Path(log_file.as_posix() + "_" + self.train_cfg.log.run_comment)
        log_file = log_file / "log.log"

        # when distributed, there should be the same log_file
        if self.accelerator.use_distributed:
            if self.accelerator.is_main_process:
                input_lst = [log_file] * self.accelerator.num_processes
            else:
                input_lst = [None] * self.accelerator.num_processes
            output_lst = [None]
            torch.distributed.scatter_object_list(output_lst, input_lst, src=0)
            log_file: Path = output_lst[0]
            assert isinstance(log_file, Path), "log_file type should be Path"

        # logger
        self.logger.remove()
        log_format_in_file = (
            "<green>[{time:MM-DD HH:mm:ss}]</green> "
            "- <level>[{level}]</level> "
            "- <cyan>{file}:{line}</cyan> - <level>{message}</level>"
        )
        log_format_in_cmd = (
            "{time:HH:mm:ss} "
            "- {level.icon} <level>[{level}:{file.name}:{line}]</level>"
            "- <level>{message}</level>"
        )
        if not self.train_cfg.debug:
            self.logger.add(
                log_file,
                format=log_format_in_file,
                level="INFO",
                rotation="10 MB",
                enqueue=True,
                backtrace=True,
                colorize=False,
            )
        self.logger.add(
            sys.stdout,
            format=log_format_in_cmd,
            level="DEBUG",
            backtrace=True,
            colorize=True,
        )

        # make log dir
        log_dir = log_file.parent
        if not self.train_cfg.debug:
            log_dir.mkdir(parents=True, exist_ok=True)

        # copy cfg
        if not self.train_cfg.debug:
            yaml_cfg = OmegaConf.to_yaml(self.cfg, resolve=True)
            cfg_cp_path = log_file.parent / "config" / "config_total.yaml"
            cfg_cp_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cfg_cp_path, "w") as f:
                f.write(yaml_cfg)
            self.logger.info(f"[Cfg]: configuration saved to {cfg_cp_path}")

        # accelerate project configuration
        self.proj_dir = log_dir
        self.accelerator.project_configuration.project_dir = str(self.proj_dir)

        # tensorboard logger
        if not self.train_cfg.debug:
            tenb_dir = log_dir / "tensorboard"
            self.accelerator.project_configuration.logging_dir = tenb_dir
            if self.accelerator.is_main_process:
                self.logger.info(f"[Tensorboard]: tensorboard saved to {tenb_dir}")
                self.accelerator.init_trackers("train")
                self.tb_logger: TensorBoardTracker = self.accelerator.get_tracker(  # type: ignore
                    "tensorboard"
                )

        return log_file

    def tenb_log_any(
        self,
        log_type: Literal["metric", "image", "grad_norm_per_param", "grad_norm_sum"],
        logs: dict,
        step: int,
        **kwargs,
    ):
        assert log_type in [
            "metric",
            "image",
            "grad_norm_per_param",
            "grad_norm_sum",
        ], "log_type must be one of [metric, image, grad_norm_per_param, grad_norm_sum]"

        if log_type == "metric":
            if hasattr(self, "tb_logger"):
                self.tb_logger.log(logs, step=step)
        elif log_type == "image":
            if hasattr(self, "tb_logger"):
                self.tb_logger.log_images(logs, step=step)
        elif log_type in ("grad_norm_per_param", "grad_norm_sum"):
            assert "model" in logs, "model name must be in logs"
            model = logs.pop("model")
            # take out the grad of norms
            model_cls_n = model.__class__.__name__
            norms = {}
            if log_type == "grad_norm_sum":
                norms[f"{model_cls_n}_grad_norm"] = 0
                _n_params_sumed = 0
            for n, p in model.named_parameters():
                if p.grad is not None:
                    # must sync grad here, `is_main_process` would cause the ranks do not sync
                    if isinstance(p.grad, DTensor):
                        _grad: torch.Tensor = p.grad._local_tensor
                        if p.grad._local_tensor.device == torch.device("cpu"):
                            self.log_msg(
                                "p.grad is on cpu, this should not happen",
                                level="WARNING",
                            )
                            # ensure the corss rank does not involve cpu bankend
                            _grad = _grad.cuda()
                        # _p_grad = p.grad.full_tensor()  # across all ranks
                        _p_grad = safe_dtensor_operation(p.grad)
                    _grad_norm = (_p_grad.data**2).sum() ** 0.5
                    if log_type == "grad_norm_per_param":
                        norms[f"{model_cls_n}/{n}"] = _grad_norm
                    else:
                        norms[f"{model_cls_n}_grad_norm"] += _grad_norm
                        _n_params_sumed += 1
            # log
            if log_type == "grad_norm_sum":
                norms[f"{model_cls_n}_grad_norm"] /= _n_params_sumed
            if hasattr(self, "tb_logger"):
                self.tb_logger.log(
                    norms,
                    step=step,
                )
        else:
            raise NotImplementedError(f"Unknown log_type {log_type}")

    def log_msg(self, *msgs, only_rank_zero=True, level="INFO", sep=",", **kwargs):
        assert level.lower() in [
            "info",
            "warning",
            "error",
            "debug",
            "critical",
        ], f"Unknown level {level}"

        def str_msg(*msg):
            return sep.join([str(m) for m in msg])

        log_fn = getattr(self.logger, level.lower())

        if only_rank_zero:
            if self.accelerator.is_main_process:
                log_fn(str_msg(*msgs), **kwargs)
        else:  # not only rank zero
            with self.accelerator.main_process_first():
                msg_string = str_msg(*msgs)
                # prefix rank info
                msg_string = f"rank-{self.accelerator.process_index} | {msg_string}"
                log_fn(msg_string, **kwargs)

    def get_optimizer_lr_scheduler(
        self,
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        # optimizers
        if (
            self.accelerator.state.deepspeed_plugin is None
            or "optimizer"
            not in self.accelerator.state.deepspeed_plugin.deepspeed_config
        ):

            def _optimizer_creater(optimizer_cfg, params_getter):
                if "get_muon_optimizer" in optimizer_cfg._target_:
                    self.log_msg("[Optimizer]: using muon optimizer")
                    # is muon optimizer function
                    named_params = params_getter(with_name=True)
                    return hydra.utils.instantiate(optimizer_cfg)(
                        named_parameters=named_params
                    )
                else:
                    self.log_msg(
                        f"[Optimizer]: using optimizer: {optimizer_cfg._target_}"
                    )
                    params = params_getter(with_name=False)
                    return hydra.utils.instantiate(optimizer_cfg)(params)

            _get_panshap_model_params = (
                lambda with_name: self.model.named_parameters()
                if with_name
                else self.model.parameters()
            )
            opt = _optimizer_creater(self.train_cfg.optim, _get_panshap_model_params)
        else:
            opt = DummyOptim([{"params": list(self.model.parameters())}])

        # schedulers
        if (
            self.accelerator.state.deepspeed_plugin is None
            or "scheduler"
            not in self.accelerator.state.deepspeed_plugin.deepspeed_config
        ):
            sched = hydra.utils.instantiate(self.train_cfg.scheduler)(optimizer=opt)
        else:
            sched = DummyScheduler(opt)

        # set the heavyball optimizer without torch compiling
        is_heavyball_opt = lambda opt: opt.__class__.__module__.startswith("heavyball")
        if is_heavyball_opt(opt):
            import heavyball

            heavyball.utils.compile_mode = None

            self.log_msg(
                f"use heavyball optimizer, it will compile the optimizer, "
                "for efficience testing the scripts, disable the compilation."
            )

        return opt, sched

    def set_fsdp_cpu_local_tensor_to_each_rank(self, model: nn.Module):
        if not self._is_fsdp:
            return model

        self.log_msg(
            "FSDP module seems do not move the original parameter (_local_tensor) on the"
            "correct rank, we need to manually move them on cuda while using `to_local` or `redistributed` methods",
            level="WARNING",
        )
        _cpu_device = torch.device("cpu")
        for name, param in model.named_parameters():
            if isinstance(param, DTensor) and param.device == _cpu_device:
                param._local_tensor = param._local_tensor.to(self.device)
                self.log_msg(f"set {name} local_tensor on cuda", level="DEBUG")

        return model

    def prepare_for_training(self):
        # FIXME: FSDP2 seems do not support the sync_bn, find a way to fix it.
        if self._is_fsdp or self.accelerator.distributed_type in (
            accelerate.utils.DistributedType.MULTI_GPU,
            accelerate.utils.DistributedType.FSDP,
        ):  # seems that FSDP does not support synchronized batchnorm
            # discriminator may have batch norm layer
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.log_msg("[Model] convert discriminator to sync batch norm")

        # if use FSDP2
        if self._is_fsdp and self.accelerator.is_fsdp2:
            # set models with property dtype
            _get_model_dtype = lambda model: next(model.parameters()).dtype
            self.model.dtype = torch.float32

        # rain model
        self.model = self.accelerator.prepare(self.model)

        self.train_dataloader, self.val_dataloader = self.accelerator.prepare(
            self.train_dataloader, self.val_dataloader
        )

    def step_train_state(self):
        self.train_state.update("train")

    def ema_update(self, mode="rain"):
        assert mode == "rain"
        if self.no_ema:
            # not support ema when is deepspeed zero2 or zero3
            return

        self.ema_model.update()

    def get_global_step(self, mode="train"):
        # TODO: add val state
        assert mode in ("train",), "Only train mode is supported for now."

        return self.train_state[mode]

    @property
    def global_step(self):
        return self.get_global_step("train")

    def may_freeze(self, model, freeze=True):
        for p in model.parameters():
            p.requires_grad = not freeze

    def gradient_check(self, model: nn.Module):
        # check nan gradient
        if self.accelerator.sync_gradients and getattr(
            self.train_cfg, "grad_check", True
        ):
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if param.grad is None:
                        self.log_msg(
                            f"step {self.global_step} - {name} has None gradient, shaped as {param.shape}",
                            only_rank_zero=False,
                            level="WARNING",
                        )
                    elif torch.isnan(param.grad).any():
                        self.log_msg(
                            f"step {self.global_step} - {name} has nan gradient, shaped as {param.shape}",
                            only_rank_zero=False,
                            level="WARNING",
                        )
                        torch.nan_to_num(
                            param.grad, nan=0.0, posinf=1e5, neginf=-1e5, out=param.grad
                        )

        # clip gradient by norm
        _max_grad_norm = self.train_cfg.max_grad_norm
        if _max_grad_norm is not None and _max_grad_norm > 0:
            if self.dtype != torch.float16 and not self.accelerator.is_fsdp2:
                self.accelerator.clip_grad_norm_(model.parameters(), _max_grad_norm)
            elif (
                self.accelerator.distributed_type
                == accelerate.utils.DistributedType.FSDP
                or self.accelerator.is_fsdp2
            ) and isinstance(model, FSDP):
                FSDP.clip_grad_norm_(model.parameters(), max_norm=_max_grad_norm)

    def forward_rain_model(
        self, pasts: list[Tensor], futures: list[Tensor], times: list[Tensor]
    ):
        with self.accelerator.autocast():
            radar_past, sat_past, rain_past = pasts
            (rain_future,) = futures
            # pred_future = self.model(radar_past, sat_past, rain_past, times)
            pred_future = self.model(radar_past, sat_past, rain_past)

        return pred_future

    def train_rain_step(
        self, pasts: list[Tensor], futures: list[Tensor], times: list[Tensor]
    ):
        pred_future = self.forward_rain_model(pasts=pasts, futures=futures, times=times)


        # loss
        rain_future = futures[-1]
        # l0_rain_mask = rain_future > 0.01
        # fg_loss = self.loss_fn(
        #     rain_future[l0_rain_mask], pred_future[l0_rain_mask]
        # ).mean()
        # bg_loss = self.loss_fn(
        #     rain_future[torch.logical_not(l0_rain_mask)],
        #     pred_future[torch.logical_not(l0_rain_mask)],
        # ).mean()
        # # bg_loss = torch.tensor(0.0).to(self.device)
        # loss = fg_loss + bg_loss
        
        B, C, T_pred, H, W = pred_future.shape

        # [B, 1, T, H, W] -> [B, T, H, W]
        rain_future = rain_future.squeeze(1)
        B_tgt, T_tgt, H_tgt, W_tgt = rain_future.shape

        if (B_tgt != B) or (H_tgt != H) or (W_tgt != W) or (T_tgt != T_pred):
            raise RuntimeError(
                "Prediction/target shape mismatch in train_rain_step: "
                f"pred=[B={B}, C={C}, T={T_pred}, H={H}, W={W}] vs "
                f"target=[B={B_tgt}, T={T_tgt}, H={H_tgt}, W={W_tgt}]. "
                "Likely cause: model outputs one future frame while dataset config sets n_futures>1. "
                "Please align `dataset.n_futures` with model output temporal length."
            )

        loss = self.loss_fn(pred_future, rain_future)
        pred_probs = torch.softmax(pred_future, dim=1)  # [B, num_classes, H, W]

        
        log_losses = {
            # "fg_loss": fg_loss.detach(),
            # "bg_loss": bg_loss.detach(),
            'loss': loss.detach(),
            'pred_min_prob': pred_probs.min().item(),
            'pred_max_prob': pred_probs.max().item(),
        }

        if self.accelerator.sync_gradients:
            # backward
            self.optim.zero_grad()
            self.accelerator.backward(loss)
            self.gradient_check(self.model)
            self.optim.step()
            self.sched.step()
            
            # ema update
            self.ema_update(mode="rain")

        return dict(
            pred_future=pred_future,
            rain_loss=loss.detach(),
            rain_log_losses=log_losses,
        )

    def form_pasts_futures(self, batch: dict):
        """
        Form the pasts and futures from the batch.
        This is a helper function to prepare the data for training or validation.

        Args:
            batch (dict): The batch of data containing past and future data.

        Returns:
            tuple: A tuple containing pasts and futures.
        """
        pasts = [
            batch["radar_past"].to(self.device),
            batch["satellite_past"].to(self.device),
            batch["rain_past"].to(self.device),
        ]
        futures = [batch["rain_future_cls"].to(self.device)]
        times = [batch["time_past"], batch["time_future"]]
        return pasts, futures, times

    def train_step(self, batch: dict):
        # form the pasts and futures
        pasts, futures, times = self.form_pasts_futures(batch)

        with self.accelerator.accumulate(self.model):
            # train rain model
            train_out = self.train_rain_step(pasts=pasts, futures=futures, times=times)
            pred_img = train_out["pred_future"]

        self.step_train_state()

        # log losses
        if self.global_step % self.train_cfg.log.log_every == 0:
            _log_losses = self.format_log(train_out["rain_log_losses"])

            self.log_msg(
                f"[Train State]: lr {self.optim.param_groups[0]['lr']:1.4e} | "
                f"[Step]: {self.global_step}/{self.train_cfg.max_steps}"
            )
            self.log_msg(f"[Train loss]: {_log_losses}")

            # tensorboard log
            self.tenb_log_any("metric", train_out["rain_log_losses"], self.global_step)

    def format_log(self, log_sr_loss: dict) -> str:
        def dict_round_to_list_str(
            d: dict, n_round: int = 5, select: list[str] | None = None
        ):
            strings = []
            for k, v in d.items():
                if select is not None and k not in select:
                    continue

                if isinstance(v, (float, torch.Tensor)):
                    if torch.is_tensor(v):
                        if v.numel() > 1:
                            self.log_msg(
                                f'logs has non-scalar tensor "{k}", skip it',
                                level="WARNING",
                            )
                            continue
                        v = v.item()
                    strings.append(f"{k}: {v:.{n_round}f}")
                else:
                    strings.append(f"{k}: {v}")
            return strings

        strings = dict_round_to_list_str(log_sr_loss, select=list(log_sr_loss.keys()))

        return " - ".join(strings)

    def infinity_train_loader(self):
        while True:
            for batch in self.train_dataloader:
                yield batch

    def train_loop(self):
        _stop_train_and_save = False
        self.accelerator.wait_for_everyone()

        self.log_msg("[Train]: start training", only_rank_zero=False)
        for batch in self.infinity_train_loader():

            # train step
            try:
                self.train_step(batch)
            except Exception as e:
                self.log_msg(
                    f"Training failed, batch keys are {batch.keys()}. {e}",
                    level="critical",
                )
                for k, v in batch.items():
                    self.log_msg(
                        f"{k}: {v.shape if hasattr(v, 'shape') else (v.__class__.__name__, v)}, "
                    )
                raise e

            if self.global_step % self.val_cfg.val_duration == 0:
                self.val_loop()

            if self.global_step >= self.train_cfg.max_steps:
                _stop_train_and_save = True

            if (
                self.global_step % self.train_cfg.save_every == 0
                or _stop_train_and_save
            ):
                self.save_state()
                self.save_ema()

            if _stop_train_and_save:
                self.log_msg(
                    "[Train]: max training step budget reached, stop training and save"
                )
                break

    def infinite_val_loader(self):
        if self.val_dataloader is None:
            raise ValueError("No validation dataloader found")

        while True:
            for batch in self.val_dataloader:
                yield batch

    def val_step(self, batch: dict):
        # form the pasts and futures from the batch
        pasts, futures, times = self.form_pasts_futures(batch)

        with torch.no_grad():
            pred_future = self.forward_rain_model(
                pasts=pasts, futures=futures, times=times
            )

        return {"pred_rain": pred_future}
    
    def val_loop(self):
        self.model.eval()
        torch.cuda.empty_cache()

        # --- 1. 准备验证集迭代器 ---
        if self.val_cfg.max_val_iters > 0:
            if not hasattr(self, "_val_loader_iter"):
                self._val_loader_iter = iter(self.infinite_val_loader())

            tbar = trange(
                self.val_cfg.max_val_iters,
                desc="validating ...",
                leave=False,
                disable=not self.accelerator.is_main_process,
            )
            self.log_msg(
                f"[Val]: start validating with only {self.val_cfg.max_val_iters} batches",
                only_rank_zero=False,
            )
        else:
            tbar = self.infinite_val_loader()
            self.log_msg(f"[Val]: start validating with the whole val set", only_rank_zero=False)

        loss_metrics = MeanMetric().to(device=self.device)

        # --- 2. 动态获取 Dataset 中的分级标准 (Bounds) ---
        # 尝试从 val_dataset 获取 BOUNDS，用于后续将 Class ID 翻译回 mm
        if hasattr(self.val_dataset, "BOUNDS"):
            real_bounds = self.val_dataset.BOUNDS
            self.log_msg(f"[Val] Detected dynamic bounds from dataset: {real_bounds}", level="DEBUG")
        else:
            # 默认 Fallback (对应你代码里的默认值)
            real_bounds = [0, 0.01, 0.1, 0.2, 0.5, 10]
            self.log_msg(f"[Val] Using default bounds: {real_bounds}", level="WARNING")

        # 计算类别数 (N个边界 -> N-1个类别)
        num_classes = len(real_bounds) - 1
        self.log_msg(f"[Val] Metrics initialized with {num_classes} classes", level="DEBUG")

        # --- 3. 初始化两种指标计算器 ---
        
        # A. 分类独立指标 (保持严格匹配，用于监控纯分类精度)
        metric_acc_cls = GlobalMetricsAccumulator(num_classes=num_classes, device=self.device)

        # B. 累积阈值指标 (业务指标，读取 YAML 配置的 tolerance_px)
        # 优先从配置读取，如果没有则默认为 0
        tolerance_px = getattr(self.val_cfg, "tolerance_px", 0) 
        
        self.log_msg(f"[Val] Cumulative Metrics Tolerance: {tolerance_px} px", level="DEBUG")

        # 构造阈值列表
        cumulative_thresholds = [-1.0] + [i + 0.5 for i in range(num_classes - 1)]
        
        metric_acc_cumulative = RainGlobalMetricsAccumulator(
            bounds=cumulative_thresholds, 
            device=self.device,
            tolerance_px=tolerance_px 
        )

        # --- 4. 验证循环 ---
        vis_samples = int(self.val_cfg.get("vis_samples", 4))
        vis_n_pasts = int(self.val_cfg.get("vis_n_pasts", 2))
        vis_n_futures = int(self.val_cfg.get("vis_n_futures", 1))
        self.log_msg(
            f"[Val Visual Config] vis_samples={vis_samples} / vis_n_pasts={vis_n_pasts} / vis_n_futures={vis_n_futures}"
        )

        for batch_or_idx in tbar:
            if self.val_cfg.max_val_iters > 0:
                batch = next(self._val_loader_iter)
            else:
                batch = batch_or_idx

            batch = cast(dict[str, torch.Tensor], batch)

            gt_cls = batch["rain_future_cls"].to(self.device)        # [B, 1, T, H, W]
            val_out = self.val_step(batch)
            pred_rain = val_out["pred_rain"]                         # [B, num_classes, T, H, W]

            # ---- 计算 Loss ---- #
            B, C, T, H, W = pred_rain.shape
            gt_loss = gt_cls.squeeze(1).long()                  
            pred_logits = pred_rain.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            gt_flat = gt_loss.reshape(B * T, H, W)

            loss_val = self.loss_fn(pred_logits, gt_flat)
            loss_metrics.update(loss_val)

            # ---- 更新指标 A (独立分类) ---- #
            metric_acc_cls.update(pred_rain, gt_cls)

            # ---- 更新指标 B (累积阈值) ---- #
            # 需要先拿到预测的类别索引，并转为 float 用于比较
            pred_cls_idx = torch.argmax(pred_rain, dim=1, keepdim=True).float() # [B, 1, T, H, W]
            metric_acc_cumulative.update(pred_cls_idx, gt_cls.float())


        # --- 5. 计算结果与日志输出 ---
        loss_val = loss_metrics.compute()
        metrics_cls = metric_acc_cls.compute()
        metrics_cumulative = metric_acc_cumulative.compute()

        if self.accelerator.is_main_process:
            self.log_msg(f"[Val] Loss={loss_val:.4f}")
            
            # (1) 打印分类独立指标
            self.log_msg("--- Per-Class Metrics (独立类别) ---")
            for cls_key, vals in metrics_cls.items():
                msg = " ".join([f"{m}={v.item():.4f}" for m, v in vals.items()])
                self.log_msg(f"  {cls_key}: {msg}")

            # (2) 打印累积阈值指标 (自动翻译名字)
            self.log_msg("--- Cumulative Metrics (累积阈值) ---")
            
            flat_metrics = {}
            flat_metrics["val/loss"] = loss_val.item()

            # 处理累积指标的 Key 映射
            for th_key, vals in metrics_cumulative.items():
                # th_key 类似于 ">=0.5mm"
                try:
                    # 解析内部阈值: ">=1.5mm" -> 1.5
                    th_val_internal = float(th_key.replace(">=", "").replace("mm", ""))
                    
                    # 找到对应的 Class ID: int(1.5 + 0.5) = 2
                    cls_id = int(th_val_internal + 0.5)
                    
                    # 映射到真实的物理雨量阈值
                    # real_bounds[1] 是 Class 1 的下界 (如 0.01)
                    if cls_id < len(real_bounds):
                        phys_bound = real_bounds[cls_id]
                        readable_key = f"Rain >= {phys_bound} (Class {cls_id})"
                        tb_suffix = f"rain_ge_{phys_bound}"
                    else:
                        readable_key = f"Class >= {cls_id}"
                        tb_suffix = f"class_ge_{cls_id}"
                        
                except Exception:
                    readable_key = th_key
                    tb_suffix = th_key

                # 打印日志
                msg = " ".join([f"{m}={v.item():.4f}" for m, v in vals.items()])
                self.log_msg(f"  {readable_key}: {msg}")
                
                # 收集到 Tensorboard (Cumulative)
                for m_name, v in vals.items():
                    flat_metrics[f"val/cumulative/{tb_suffix}/{m_name}"] = v.item()

            # 收集到 Tensorboard (Per-Class)
            for cls_key, metrics_dict in metrics_cls.items():
                for m_name, v in metrics_dict.items():
                    flat_metrics[f"val/per_class/{cls_key}/{m_name}"] = v.item()

            # 写 Tensorboard
            self.tenb_log_any("metric", flat_metrics, step=self.global_step)

            # 可视化
            try:
                self.visualize_prediction(
                    batch,
                    pred_rain,
                    add_step=True,
                    img_name="val/rain_cls",
                    only_vis_n=vis_samples,
                    vis_n_pasts=vis_n_pasts,
                    vis_n_futures=vis_n_futures,
                )
            except Exception as e:
                self.log_msg(f"[Visualize]: failed to visualize prediction: {e}", level="WARNING")

    def save_state(self):
        self.accelerator.save_state()
        self.log_msg("[State]: save states")

    def save_ema(self):
        if self.no_ema:
            self.log_msg(f"use deepspeed or FSDP, do have EMA model to save")
            return

        ema_path = self.proj_dir / "ema"
        if self.accelerator.is_main_process:
            ema_path.parent.mkdir(parents=True, exist_ok=True)

        self.accelerator.save_model(self.ema_model.ema_model, ema_path / "rain_model")
        # train state
        _ema_path_state_train = ema_path / "train_state.pth"
        _ema_path_state_train.parent.mkdir(parents=True, exist_ok=True)
        accelerate.utils.save(self.train_state.state_dict(), _ema_path_state_train)
        self.log_msg(f"[Ckpt]: save ema at {ema_path}")

    def load_from_ema(self, ema_path: str | Path, strict: bool = True):
        ema_path = Path(ema_path)

        accelerate.load_checkpoint_in_model(
            self.model, ema_path / "rain_model", strict=strict
        )

        # Prepare models
        self.prepare_ema_models()  # This will update EMA models with online models' weights

        # clear the accelerator model registration
        self.log_msg(
            f"[Load EMA]: clear the accelerator registrations and re-prepare training"
        )

    def resume(self, path: str):
        self.log_msg("[Resume]: resume training")
        self.accelerator.load_state(path)
        self.accelerator.wait_for_everyone()

    def visualize_prediction(
        self,
        x: dict[str, torch.Tensor],
        pred: torch.Tensor,
        img_name: str = "train_original_predict",
        add_step: bool = False,
        only_vis_n: int | None = None,
        vis_n_pasts: int = 2,
        vis_n_futures: int = 1,
    ):
        """
            pred: [B, num_class, T, H, W]
            x["rain_future_cls"]: [B, 1, T, H, W]
        """
        B, C, T, H, W = pred.shape

        # 预测类别 [B,1,T,H,W]，GT 保持 [B,1,T,H,W]
        pred_cls = torch.argmax(pred, dim=1, keepdim=True)
        gt_cls = x["rain_future_cls"].long()

        # 颜色映射配置
        num_classes = pred.shape[1]
        #precipitation_colors = ["white", "lightblue", "blue", "green", "yellow", "orange", "red"][:num_classes]
        precipitation_colors = [
    "white",        # Class 0: < 0.001
    "lightblue",    # Class 1: 0.05 - 0.10
    "blue",    # Class 2: 0.10 - 0.20
    "green",  # Class 3: 0.20 - 0.30
    "yellow",         # Class 4: 0.30 - 0.40
    "orange",        # Class 5: 0.40 - 0.50
    "red",       # Class 6: 0.50 - 0.60
    "purple",       # Class 7: 0.60 - 0.70
    "black"           # Class 8: > 0.70
][:num_classes]
        cmap = mcolors.ListedColormap(precipitation_colors)
        norm = mcolors.BoundaryNorm(np.arange(num_classes + 1), cmap.N)

        def colorize_mask(mask_tensor):
            mask_np = mask_tensor.squeeze().cpu().numpy()
            mask_color = cmap(norm(mask_np))
            return (mask_color[..., :3] * 255).astype(np.uint8)

        # ====== 绘图逻辑开始 (严格按照顺序执行) ======
        
        # 1. 计算可视化 past/future 帧数（受数据实际长度约束）
        n_pasts = min(max(1, int(vis_n_pasts)), x["radar_past"].shape[-3])
        n_futures = min(max(1, int(vis_n_futures)), pred.shape[2], x["rain_future_cls"].shape[2])

        # 2. 计算实际 batch size 并限制绘图数量
        actual_bs = x["rain_past"].shape[0]
        if only_vis_n is None:
            only_vis_n = actual_bs
        else:
            only_vis_n = min(only_vis_n, actual_bs)
            
        ncols = n_pasts * 3 + n_futures * 2

        fig, axes = plt.subplots(only_vis_n, ncols, figsize=(20, 5 * only_vis_n))
        
        # 5. 维度修正：如果只画一张图，必须把 axes 变成二维数组，否则后面 axes[i, m] 会报错
        if only_vis_n == 1:
            axes = axes.reshape(1, -1)

        # ====== 循环绘图 ======
        for i in range(only_vis_n):
            # 过去帧：雷达、卫星、雨
            for j in range(n_pasts):
                m = j * 3
                radar_img = plot_any_modality(x["radar_past"][i:i+1, :, j], "radar", False)
                sat_img = plot_any_modality(x["satellite_past"][i:i+1, :, j], "satellite", False)
                rain_img = plot_any_modality(x["rain_past"][i:i+1, :, j], "rain", False)
                
                # 绘制过去帧
                axes[i, m].imshow(radar_img)
                axes[i, m].set_title(f"Radar Past {j}")
                axes[i, m+1].imshow(sat_img)
                axes[i, m+1].set_title(f"Satellite Past {j}")
                axes[i, m+2].imshow(rain_img)
                axes[i, m+2].set_title(f"Rain Past {j}")

            # 绘制 future 帧 GT & Pred 类别
            for t in range(n_futures):
                c = n_pasts * 3 + t * 2
                gt_img = colorize_mask(gt_cls[i, :, t])
                pred_img = colorize_mask(pred_cls[i, :, t])

                axes[i, c].imshow(gt_img)
                axes[i, c].set_title(f"GT Rain Class t={t}")
                axes[i, c + 1].imshow(pred_img)
                axes[i, c + 1].set_title(f"Pred Rain Class t={t}")

        plt.tight_layout()
        
        # 保存图片
        save_path = Path(self.proj_dir) / "vis" / (
            f"{img_name}_{self.global_step:06d}.jpg" if add_step else f"{img_name}.jpg"
        )
        if self.accelerator.is_main_process:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=200)

        plt.close(fig)
        plt.clf()
        plt.cla()
        self.log_msg(f"[Visualize]: save visualization at {save_path}")

    def run(self):
        if self.train_cfg.resume_path is not None:
            self.resume(self.train_cfg.resume_path)
        elif self.train_cfg.ema_load_path is not None:
            self.load_from_ema(self.train_cfg.ema_load_path)

        # train !
        self.train_loop()


_key = "rain_ts"
_configs = {
    "rain_ts": "rain_train_ts_swinnet_all_data"
}[_key]


@hydra.main(
    config_path="../config/ts_rain_train",
    config_name=_configs,
    version_base=None,
)
def main(cfg):
    catcher = logger.catch if PartialState().is_main_process else nullcontext

    with catcher():
        trainer = RainPredictionTrainer(cfg)
        trainer.run()


if __name__ == "__main__":
    main()

import sys
from pathlib import Path

# 动态添加 src 目录到模块搜索路径
project_root = Path("/home/rainpred/RainPrediction")
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))
