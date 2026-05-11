"""
Stage-2 trainer for next-block causal forcing with roll-history supervision.

Core idea:
- load stage-1 model weights only
- keep teacher-forcing next-block loss for stability
- add roll-history next-frame loss to align training with self-conditioned rollout
"""

import json
from contextlib import nullcontext
from pathlib import Path

import accelerate
import hydra
import torch
import torch.nn.functional as F
from accelerate.state import PartialState
from loguru import logger
from omegaconf import DictConfig

from src.trainer.rain_trainer_ts_next_frame import RainTSNextFrameTrainer


class RainTSNextFrameStage2Trainer(RainTSNextFrameTrainer):
    def __init__(self, cfg: DictConfig):
        self.stage2_cfg = cfg.train.get("stage2", {})
        super().__init__(cfg)
        self.stage2_block_size = int(self.stage2_cfg.get("block_size", self.train_cfg.next_pred.get("block_size", 1)))
        self.stage2_roll_n = int(self.stage2_cfg.get("roll_n", 1))
        self.stage2_detach_history = bool(self.stage2_cfg.get("detach_history", True))
        self.stage2_lambda_tf = float(self.stage2_cfg.get("lambda_teacher_forcing", 1.0))
        self.stage2_lambda_roll_next = float(self.stage2_cfg.get("lambda_roll_next", 1.0))
        self.stage2_ckpt_path = self.stage2_cfg.get("stage1_ckpt_path")
        self.stage2_strict_load = bool(self.stage2_cfg.get("strict_load", True))

        if self.stage2_block_size <= 0:
            raise ValueError(f"train.stage2.block_size must be > 0, got {self.stage2_block_size}")
        if self.stage2_roll_n <= 0:
            raise ValueError(f"train.stage2.roll_n must be > 0, got {self.stage2_roll_n}")
        if self.stage2_lambda_tf < 0 or self.stage2_lambda_roll_next < 0:
            raise ValueError(
                "train.stage2.lambda_teacher_forcing and train.stage2.lambda_roll_next must be >= 0, "
                f"got tf={self.stage2_lambda_tf}, roll_next={self.stage2_lambda_roll_next}"
            )
        if self.stage2_ckpt_path is None:
            raise ValueError("train.stage2.stage1_ckpt_path is required for stage-2 training.")

        target_mode = str(self.train_cfg.next_pred.get("target_mode", "next_frame")).lower()
        if target_mode != "block":
            raise ValueError(
                "Stage-2 requires train.next_pred.target_mode=block, "
                f"got {target_mode}."
            )
        configured_block_size = int(self.train_cfg.next_pred.get("block_size", 1))
        if configured_block_size != self.stage2_block_size:
            raise ValueError(
                "train.next_pred.block_size must equal train.stage2.block_size for stage-2. "
                f"got next_pred.block_size={configured_block_size}, stage2.block_size={self.stage2_block_size}"
            )

        self._load_stage1_model_weights()
        self.log_msg(
            "Stage-2 objective: mixed next-block + roll-history-next-frame "
            f"(roll_n={self.stage2_roll_n}, block_size={self.stage2_block_size}, "
            f"detach_history={self.stage2_detach_history}, "
            f"lambda_tf={self.stage2_lambda_tf:.4f}, lambda_roll_next={self.stage2_lambda_roll_next:.4f})"
        )

    def _resolve_stage2_roll_frames(self) -> int:
        return self.stage2_roll_n * self.stage2_block_size

    def _required_future_frames(self) -> int:
        return self._resolve_stage2_roll_frames() + 1

    def _check_stage2_future_requirement(self, n_future: int) -> None:
        required = self._required_future_frames()
        if n_future < required:
            raise ValueError(
                "Stage-2 requires dataset.n_futures >= roll_n * block_size + 1. "
                f"got n_futures={n_future}, roll_n={self.stage2_roll_n}, "
                f"block_size={self.stage2_block_size}, required={required}"
            )

    def _validate_data_model_contract(self) -> None:
        super()._validate_data_model_contract()
        block_size = int(self.stage2_cfg.get("block_size", self.train_cfg.next_pred.get("block_size", 1)))
        roll_n = int(self.stage2_cfg.get("roll_n", 1))
        if block_size <= 0 or roll_n <= 0:
            raise ValueError(f"stage2 block_size/roll_n must be > 0, got block_size={block_size}, roll_n={roll_n}")
        n_future = int(self.dataset_cfg.n_futures)
        required = roll_n * block_size + 1
        if n_future < required:
            raise ValueError(
                "Stage-2 requires dataset.n_futures >= roll_n * block_size + 1. "
                f"got n_futures={n_future}, roll_n={roll_n}, block_size={block_size}, required={required}"
            )

    def _load_stage1_model_weights(self) -> None:
        ckpt_path = Path(str(self.stage2_ckpt_path))
        if not ckpt_path.exists():
            raise FileNotFoundError(f"stage1 checkpoint not found: {ckpt_path}")

        model = self.accelerator.unwrap_model(self.model)
        try:
            accelerate.load_checkpoint_in_model(model, str(ckpt_path), strict=self.stage2_strict_load)
            self.accelerator.wait_for_everyone()
            self.log_msg(f"[Stage2] loaded model weights with accelerate from {ckpt_path}")
            return
        except Exception as e:
            self.log_msg(f"[Stage2] accelerate model loading failed ({e}), fallback to torch.load", level="warning")

        load_path = ckpt_path
        if ckpt_path.is_dir():
            candidates = [
                "model.safetensors",
                "pytorch_model.bin",
                "model.bin",
                "model.pt",
                "ema/ema.pt",
            ]
            found = None
            for candidate in candidates:
                candidate_path = ckpt_path / candidate
                if candidate_path.exists():
                    found = candidate_path
                    break
            if found is None:
                raise FileNotFoundError(f"No recognized model file in stage1 checkpoint directory: {ckpt_path}")
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
            raise ValueError(f"Unsupported stage1 checkpoint format at {load_path}")

        cleaned: dict[str, torch.Tensor] = {}
        for k, v in state.items():
            key = str(k)
            for prefix in ("module.", "_orig_mod.", "model.", "_fsdp_wrapped_module."):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
            cleaned[key] = v

        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if self.stage2_strict_load and (len(missing) > 0 or len(unexpected) > 0):
            raise RuntimeError(
                f"Strict stage1 weight loading failed. missing={len(missing)}, unexpected={len(unexpected)}"
            )
        self.accelerator.wait_for_everyone()
        self.log_msg(
            f"[Stage2] loaded model weights from {load_path}, missing={len(missing)}, unexpected={len(unexpected)}",
            level="warning" if (len(missing) > 0 or len(unexpected) > 0) else "info",
        )

    def _build_stage2_roll_seed(
        self,
        context: torch.Tensor,
        seed_frames: int,
        detach_history: bool,
    ) -> torch.Tensor:
        if seed_frames <= 0:
            raise ValueError(f"seed_frames must be > 0, got {seed_frames}")

        seed_list: list[torch.Tensor] = [context[:, :, -1:, :, :]]
        context_cur = context

        for _ in range(1, seed_frames):
            pred_one = self.model.forward_ar(
                context_x=context_cur,
                target_x=seed_list[-1],
                predict_frames=1,
                strict_target_isolation=bool(self.train_cfg.strict_target_isolation),
                return_modality_dict=True,
            )
            pred_one_tensor = self._merge_modalities(pred_one["radar"], pred_one["satellite"], pred_one["rain"])
            pred_hist = pred_one_tensor.detach() if detach_history else pred_one_tensor
            seed_list.append(pred_hist)
            context_cur = torch.cat([context_cur, pred_hist], dim=2)

        return torch.cat(seed_list, dim=2)

    def _rollout_blocks_for_stage2(
        self,
        context: torch.Tensor,
        total_future_frames: int,
        block_size: int,
        detach_history: bool,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        if context.ndim != 5:
            raise ValueError(f"context must be [B,C,T,H,W], got {tuple(context.shape)}")
        if total_future_frames <= 0:
            raise ValueError(f"total_future_frames must be > 0, got {total_future_frames}")
        if block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {block_size}")

        remaining = total_future_frames
        context_cur = context

        pred_radar: list[torch.Tensor] = []
        pred_satellite: list[torch.Tensor] = []
        pred_rain: list[torch.Tensor] = []

        while remaining > 0:
            chunk = min(block_size, remaining)
            seed_block = self._build_stage2_roll_seed(
                context=context_cur,
                seed_frames=chunk,
                detach_history=detach_history,
            )
            pred_block = self.model.forward_ar(
                context_x=context_cur,
                target_x=seed_block,
                predict_frames=chunk,
                strict_target_isolation=bool(self.train_cfg.strict_target_isolation),
                return_modality_dict=True,
            )
            pred_radar.append(pred_block["radar"])
            pred_satellite.append(pred_block["satellite"])
            pred_rain.append(pred_block["rain"])

            pred_tensor = self._merge_modalities(pred_block["radar"], pred_block["satellite"], pred_block["rain"])
            pred_hist = pred_tensor.detach() if detach_history else pred_tensor
            context_cur = torch.cat([context_cur, pred_hist], dim=2)
            remaining -= chunk

        pred = {
            "radar": torch.cat(pred_radar, dim=2),
            "satellite": torch.cat(pred_satellite, dim=2),
            "rain": torch.cat(pred_rain, dim=2),
        }
        return pred, context_cur

    def _compute_roll_next_loss(
        self,
        context: torch.Tensor,
        target: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        n_future = int(target["rain"].shape[2])
        self._check_stage2_future_requirement(n_future)
        roll_frames = self._resolve_stage2_roll_frames()

        _rolled_pred, rolled_context = self._rollout_blocks_for_stage2(
            context=context,
            total_future_frames=roll_frames,
            block_size=self.stage2_block_size,
            detach_history=self.stage2_detach_history,
        )
        seed = rolled_context[:, :, -1:, :, :]
        pred_next = self.model.forward_ar(
            context_x=rolled_context,
            target_x=seed,
            predict_frames=1,
            strict_target_isolation=bool(self.train_cfg.strict_target_isolation),
            return_modality_dict=True,
        )
        target_next = {
            "radar": target["radar"][:, :, roll_frames : roll_frames + 1],
            "satellite": target["satellite"][:, :, roll_frames : roll_frames + 1],
            "rain": target["rain"][:, :, roll_frames : roll_frames + 1],
        }
        lw = self.train_cfg.loss_weights
        loss_roll_next = (
            float(lw.radar) * F.mse_loss(pred_next["radar"], target_next["radar"])
            + float(lw.satellite) * F.mse_loss(pred_next["satellite"], target_next["satellite"])
            + float(lw.rain) * F.mse_loss(pred_next["rain"], target_next["rain"])
        )
        return loss_roll_next, pred_next, target_next

    def _mixed_stage2_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        context_tf, target_seed, target_gt, aux = self._build_next_pred_batch(batch)
        if str(aux["target_mode"]) != "block" or int(aux["target_frames"]) != self.stage2_block_size:
            raise ValueError(
                "Stage-2 teacher-forcing path expects block mode with target_frames == stage2.block_size. "
                f"got mode={aux['target_mode']}, target_frames={aux['target_frames']}, "
                f"stage2.block_size={self.stage2_block_size}"
            )
        pred_tf = self.model.forward_ar(
            context_x=context_tf,
            target_x=target_seed,
            predict_frames=self.stage2_block_size,
            strict_target_isolation=bool(self.train_cfg.strict_target_isolation),
            return_modality_dict=True,
        )
        loss_tf, tf_logs = self._next_prediction_loss(pred=pred_tf, target_gt=target_gt)

        context_roll, target_roll = self._prepare_val_inference_batch(batch)
        loss_roll_next, _pred_next, _target_next = self._compute_roll_next_loss(context=context_roll, target=target_roll)

        loss = self.stage2_lambda_tf * loss_tf + self.stage2_lambda_roll_next * loss_roll_next
        logs = {
            "loss": loss.detach(),
            "loss/tf_block": loss_tf.detach(),
            "loss/roll_next": loss_roll_next.detach(),
            "loss/lambda_tf": torch.tensor(self.stage2_lambda_tf, device=self.device),
            "loss/lambda_roll_next": torch.tensor(self.stage2_lambda_roll_next, device=self.device),
            "meta/target_frames": torch.tensor(float(aux["target_frames"]), device=self.device),
            "meta/context_frames": torch.tensor(float(aux["context_frames"]), device=self.device),
            "meta/roll_frames": torch.tensor(float(self._resolve_stage2_roll_frames()), device=self.device),
            "meta/detach_history": torch.tensor(1.0 if self.stage2_detach_history else 0.0, device=self.device),
        }
        logs.update(
            {
                "loss/tf_radar": tf_logs["loss/radar"],
                "loss/tf_satellite": tf_logs["loss/satellite"],
                "loss/tf_rain": tf_logs["loss/rain"],
            }
        )
        return loss, logs

    def train_step(self, batch: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], bool]:
        with self.accelerator.accumulate(self.model):
            with self.accelerator.autocast():
                loss, logs = self._mixed_stage2_loss(batch)

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
        return logs, did_step

    @torch.no_grad()
    def val_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        with self.accelerator.autocast():
            _loss, logs = self._mixed_stage2_loss(batch)
        return logs

    def _save_checkpoint(self) -> None:
        super()._save_checkpoint()
        if not self.accelerator.is_main_process:
            return
        ckpt_dir = self.proj_dir / f"checkpoint-{self.global_step:08d}"
        stage2_meta = {
            "stage": 2,
            "stage1_ckpt_path": str(self.stage2_ckpt_path),
            "roll_n": int(self.stage2_roll_n),
            "block_size": int(self.stage2_block_size),
            "detach_history": bool(self.stage2_detach_history),
            "lambda_teacher_forcing": float(self.stage2_lambda_tf),
            "lambda_roll_next": float(self.stage2_lambda_roll_next),
        }
        (ckpt_dir / "stage2_meta.json").write_text(json.dumps(stage2_meta, indent=2))


@hydra.main(
    config_path="../config/ts_rain_train",
    config_name="rain_trainer_ts_next_frame_stage2",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    catcher = logger.catch if PartialState().is_main_process else nullcontext
    with catcher():
        trainer = RainTSNextFrameStage2Trainer(cfg)
        trainer.run()


if __name__ == "__main__":
    main()
