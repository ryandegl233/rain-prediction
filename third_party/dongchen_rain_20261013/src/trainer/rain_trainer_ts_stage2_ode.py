"""
Stage-2 trainer: causal ODE-style distillation for time-series diffusion.

Core idea:
- freeze a stage-1 AR teacher model
- feed the same teacher-forcing AR inputs (clean context + noisy target) to teacher and student
- train student to predict teacher's x0 mapping from x_t
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import accelerate
import hydra
import torch
import torch.nn.functional as F
from accelerate.state import PartialState
from loguru import logger
from omegaconf import DictConfig

from src.trainer.rain_trainer_ts_diffusion import RainTSDiffusionTrainer


class RainTSStage2ODETrainer(RainTSDiffusionTrainer):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        self.teacher_prediction_target = str(self.train_cfg.stage2.teacher_prediction_target).lower()
        self.student_prediction_target = str(self.train_cfg.stage2.student_prediction_target).lower()
        self.gt_reg_weight = float(self.train_cfg.stage2.gt_reg_weight)

        if self.teacher_prediction_target not in {"epsilon", "x0"}:
            raise ValueError(
                f"train.stage2.teacher_prediction_target must be 'epsilon' or 'x0', got {self.teacher_prediction_target}"
            )
        if self.student_prediction_target not in {"epsilon", "x0"}:
            raise ValueError(
                f"train.stage2.student_prediction_target must be 'epsilon' or 'x0', got {self.student_prediction_target}"
            )

        self.teacher_model = self._build_teacher_model()
        self.log_msg(
            "Stage-2 objective: distill x_t->x0 mapping from frozen AR teacher "
            f"(teacher={self.teacher_prediction_target}, student={self.student_prediction_target})"
        )

    def _build_teacher_model(self) -> torch.nn.Module:
        if "teacher_model" in self.cfg and self.cfg.teacher_model is not None:
            teacher = hydra.utils.instantiate(self.cfg.teacher_model)
        else:
            teacher = hydra.utils.instantiate(self.cfg.rain_prediction_model)
        teacher.to(self.device)
        teacher.eval()
        teacher.requires_grad_(False)

        ckpt = self.train_cfg.stage2.teacher_ckpt_path
        if ckpt is None:
            raise ValueError("train.stage2.teacher_ckpt_path is required for stage-2 distillation.")
        self._load_model_checkpoint(
            teacher,
            ckpt_path=Path(ckpt),
            strict=bool(self.train_cfg.stage2.teacher_strict_load),
        )
        teacher.eval()
        teacher.requires_grad_(False)
        return teacher

    def _load_model_checkpoint(self, model: torch.nn.Module, ckpt_path: Path, strict: bool) -> None:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"teacher checkpoint not found: {ckpt_path}")

        # First try Accelerate-native loading (supports directory checkpoints and sharded files).
        try:
            accelerate.load_checkpoint_in_model(model, str(ckpt_path), strict=strict)
            self.log_msg(f"[Teacher] loaded by accelerate.load_checkpoint_in_model from {ckpt_path}")
            return
        except Exception as e:
            self.log_msg(f"[Teacher] accelerate loading failed ({e}), fallback to torch.load", level="warning")

        load_path = ckpt_path
        if ckpt_path.is_dir():
            candidates = [
                "model.pt",
                "pytorch_model.bin",
                "diffusion_pytorch_model.bin",
                "model.safetensors",
                "ema/ema.pt",
            ]
            found = None
            for c in candidates:
                p = ckpt_path / c
                if p.exists():
                    found = p
                    break
            if found is None:
                raise FileNotFoundError(f"No recognized model file in checkpoint directory: {ckpt_path}")
            load_path = found

        state = torch.load(load_path, map_location="cpu")
        if isinstance(state, dict):
            if "state_dict" in state and isinstance(state["state_dict"], dict):
                state = state["state_dict"]
            elif "model" in state and isinstance(state["model"], dict):
                state = state["model"]
            elif "generator" in state and isinstance(state["generator"], dict):
                state = state["generator"]

        if not isinstance(state, dict):
            raise ValueError(f"Unsupported checkpoint format at {load_path}")

        cleaned = {}
        for k, v in state.items():
            nk = str(k)
            for prefix in ("module.", "_orig_mod.", "model.", "_fsdp_wrapped_module."):
                if nk.startswith(prefix):
                    nk = nk[len(prefix) :]
            cleaned[nk] = v

        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if strict and (len(missing) > 0 or len(unexpected) > 0):
            raise RuntimeError(
                f"Strict teacher load failed. Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}"
            )
        self.log_msg(
            f"[Teacher] loaded from {load_path}, missing={len(missing)}, unexpected={len(unexpected)}",
            level="warning" if (len(missing) > 0 or len(unexpected) > 0) else "info",
        )

    def _pred_to_x0(
        self,
        pred: dict[str, torch.Tensor],
        noisy_target: dict[str, torch.Tensor],
        t: torch.Tensor,
        prediction_target: str,
    ) -> dict[str, torch.Tensor]:
        if prediction_target == "x0":
            return pred

        if prediction_target != "epsilon":
            raise ValueError(f"Unsupported prediction_target={prediction_target}")

        out = {}
        for k in ("radar", "satellite", "rain"):
            out[k] = self.noise_schedule.convert_noise_to_x0(
                noise=pred[k],
                xt=noisy_target[k],
                timestep=t,
            )
        return out

    def _ode_distill_loss(
        self,
        student_x0: dict[str, torch.Tensor],
        teacher_x0: dict[str, torch.Tensor],
        gt_x0: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        lw = self.train_cfg.loss_weights
        d_r = F.mse_loss(student_x0["radar"], teacher_x0["radar"])
        d_s = F.mse_loss(student_x0["satellite"], teacher_x0["satellite"])
        d_y = F.mse_loss(student_x0["rain"], teacher_x0["rain"])
        distill = float(lw.radar) * d_r + float(lw.satellite) * d_s + float(lw.rain) * d_y

        if self.gt_reg_weight > 0:
            g_r = F.mse_loss(student_x0["radar"], gt_x0["radar"])
            g_s = F.mse_loss(student_x0["satellite"], gt_x0["satellite"])
            g_y = F.mse_loss(student_x0["rain"], gt_x0["rain"])
            gt_reg = float(lw.radar) * g_r + float(lw.satellite) * g_s + float(lw.rain) * g_y
        else:
            g_r = g_s = g_y = torch.zeros_like(d_r)
            gt_reg = torch.zeros_like(distill)

        loss = distill + self.gt_reg_weight * gt_reg
        logs = {
            "loss": loss.detach(),
            "loss/distill": distill.detach(),
            "loss/gt_reg": gt_reg.detach(),
            "loss/distill_radar": d_r.detach(),
            "loss/distill_satellite": d_s.detach(),
            "loss/distill_rain": d_y.detach(),
            "loss/gt_radar": g_r.detach(),
            "loss/gt_satellite": g_s.detach(),
            "loss/gt_rain": g_y.detach(),
        }
        return loss, logs

    def train_step(self, batch: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], bool]:
        context, noisy_target, context_t, target_t, _target_noise_dict, aux = self._form_teacher_forcing_batch(batch)
        target_x0_dict = aux["target_x0"]
        noisy_target_dict = self._split_modalities(noisy_target)
        t = target_t.long()

        with self.accelerator.accumulate(self.model):
            with torch.no_grad():
                with self.accelerator.autocast():
                    teacher_pred = self.teacher_model.forward_ar(
                        context_x=context,
                        target_x=noisy_target,
                        context_timestep=context_t,
                        target_timestep=target_t,
                        predict_frames=1,
                        strict_target_isolation=bool(self.train_cfg.strict_target_isolation),
                        return_modality_dict=True,
                    )
                    teacher_x0 = self._pred_to_x0(
                        teacher_pred,
                        noisy_target=noisy_target_dict,
                        t=t,
                        prediction_target=self.teacher_prediction_target,
                    )

            with self.accelerator.autocast():
                student_pred = self.model.forward_ar(
                    context_x=context,
                    target_x=noisy_target,
                    context_timestep=context_t,
                    target_timestep=target_t,
                    predict_frames=1,
                    strict_target_isolation=bool(self.train_cfg.strict_target_isolation),
                    return_modality_dict=True,
                )
                student_x0 = self._pred_to_x0(
                    student_pred,
                    noisy_target=noisy_target_dict,
                    t=t,
                    prediction_target=self.student_prediction_target,
                )
                loss, logs = self._ode_distill_loss(
                    student_x0=student_x0,
                    teacher_x0={k: v.detach() for k, v in teacher_x0.items()},
                    gt_x0=target_x0_dict,
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
        logs["meta/target_idx"] = torch.tensor(float(aux["target_idx"]), device=self.device)
        logs["meta/context_frames"] = torch.tensor(float(aux["context_frames"]), device=self.device)
        logs["meta/t_mean"] = torch.tensor(float(aux["t_mean"]), device=self.device)
        return logs, did_step

    @torch.no_grad()
    def val_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        context, noisy_target, context_t, target_t, _target_noise_dict, aux = self._form_teacher_forcing_batch(batch)
        target_x0_dict = aux["target_x0"]
        noisy_target_dict = self._split_modalities(noisy_target)
        t = target_t.long()

        with self.accelerator.autocast():
            teacher_pred = self.teacher_model.forward_ar(
                context_x=context,
                target_x=noisy_target,
                context_timestep=context_t,
                target_timestep=target_t,
                predict_frames=1,
                strict_target_isolation=bool(self.train_cfg.strict_target_isolation),
                return_modality_dict=True,
            )
            teacher_x0 = self._pred_to_x0(
                teacher_pred,
                noisy_target=noisy_target_dict,
                t=t,
                prediction_target=self.teacher_prediction_target,
            )

            student_pred = self.model.forward_ar(
                context_x=context,
                target_x=noisy_target,
                context_timestep=context_t,
                target_timestep=target_t,
                predict_frames=1,
                strict_target_isolation=bool(self.train_cfg.strict_target_isolation),
                return_modality_dict=True,
            )
            student_x0 = self._pred_to_x0(
                student_pred,
                noisy_target=noisy_target_dict,
                t=t,
                prediction_target=self.student_prediction_target,
            )
            _, logs = self._ode_distill_loss(
                student_x0=student_x0,
                teacher_x0={k: v.detach() for k, v in teacher_x0.items()},
                gt_x0=target_x0_dict,
            )
        return logs

    def _save_checkpoint(self) -> None:
        ckpt_dir = self.proj_dir / f"checkpoint-{self.global_step:08d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.accelerator.save_state(str(ckpt_dir))
        if self.accelerator.is_main_process:
            meta = {
                "global_step": self.global_step,
                "stage": 2,
                "teacher_ckpt_path": str(self.train_cfg.stage2.teacher_ckpt_path),
                "teacher_prediction_target": self.teacher_prediction_target,
                "student_prediction_target": self.student_prediction_target,
                "gt_reg_weight": self.gt_reg_weight,
            }
            (ckpt_dir / "meta.json").write_text(json.dumps(meta, indent=2))
            if self.ema_model is not None:
                ema_dir = ckpt_dir / "ema"
                ema_dir.mkdir(parents=True, exist_ok=True)
                torch.save(self.ema_model.state_dict(), ema_dir / "ema.pt")
        self.log_msg(f"[Checkpoint] saved to {ckpt_dir}")


@hydra.main(
    config_path="../config/ts_rain_train",
    config_name="rain_trainer_ts_stage2_ode",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    catcher = logger.catch if PartialState().is_main_process else nullcontext
    with catcher():
        trainer = RainTSStage2ODETrainer(cfg)
        trainer.run()


if __name__ == "__main__":
    main()
