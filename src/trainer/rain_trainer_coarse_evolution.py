"""Isolated coarse fine-tuning with sequential backward and paired validation."""

import hashlib
import json
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import default_collate

from src.residual_diffusion_stage.compare_coarse_smoke import save_preview
from src.residual_diffusion_stage.diagnose_coarse_evolution import evolution_statistics, save_changes, summarize
from src.residual_diffusion_stage.metrics import DeterministicMetricAccumulator
from src.trainer.rain_trainer_ts_next_frame import RainTSNextFrameTrainer


class CoarseEvolutionTrainer(RainTSNextFrameTrainer):
    def train_step(self, batch: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], bool]:
        if self.use_gan or self.accelerator.num_processes != 1:
            raise ValueError("Sequential-backward experiment requires one GPU and GAN disabled")
        context, seed, target, aux = self._build_next_pred_batch(batch, apply_missing_modality=True)
        with self.accelerator.accumulate(self.model):
            with self.accelerator.autocast():
                prediction = self.model.forward_ar(
                    context_x=context, target_x=seed, predict_frames=int(aux["target_frames"]),
                    strict_target_isolation=bool(self.train_cfg.strict_target_isolation),
                    return_modality_dict=True, context_modality_available=aux.get("context_modality_available"),
                    context_time=aux.get("context_time"), target_time=aux.get("target_seed_time"),
                )
                rec_loss, logs = self._next_prediction_loss(pred=prediction, target_gt=target, aux=aux)
            self.accelerator.backward(rec_loss)
            rec_value = rec_loss.detach()
            del prediction, rec_loss, context, seed, target, aux
            with self.accelerator.autocast():
                rollout_loss, rollout_logs = self._train_rollout_branch_loss(batch)
            rollout_value = torch.zeros_like(rec_value)
            if rollout_loss is not None:
                self.accelerator.backward(rollout_loss)
                rollout_value = rollout_loss.detach()
            logs.update(rollout_logs)
            if self.accelerator.sync_gradients:
                max_norm = float(self.train_cfg.get("max_grad_norm", 0.0))
                if max_norm > 0:
                    norm = self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm)
                    if not torch.isfinite(norm):
                        raise FloatingPointError("Nonfinite gradient norm")
                self.optim.step()
                self.sched.step()
                self.optim.zero_grad(set_to_none=True)
        did_step = bool(self.accelerator.sync_gradients)
        if did_step:
            if self.ema_model is not None:
                self.ema_model.update()
            self.global_step += 1
        logs["loss/rec_teacher_forced"] = rec_value
        logs["loss/rec"] = rec_value + rollout_value
        logs["loss"] = logs["loss/rec"]
        if self.device.type == "cuda":
            logs["memory/peak_allocated_gib"] = torch.tensor(
                torch.cuda.max_memory_allocated(self.device) / 2**30, device=self.device,
            )
            logs["memory/peak_reserved_gib"] = torch.tensor(
                torch.cuda.max_memory_reserved(self.device) / 2**30, device=self.device,
            )
        return logs, did_step

    @torch.no_grad()
    def _run_val(self) -> None:
        if int(self.val_cfg.max_val_iters) <= 0:
            return
        manifest_path = Path(self.val_cfg.fixed_comparison)
        source = json.loads(manifest_path.read_text())
        manifest = source["manifest"][:int(self.val_cfg.max_val_iters)]
        accumulator = DeterministicMetricAccumulator([0.01, 0.1, 0.2, 0.5], 4, torch.device("cpu"))
        sums = {}
        output = self.proj_dir / "fixed_validation" / f"step_{self.global_step:08d}"
        output.mkdir(parents=True, exist_ok=True)
        self.model.eval()
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            for ordinal, item in enumerate(manifest):
                batch = default_collate([self.val_dataset[item["index"]]])
                digest = hashlib.sha256()
                for key in sorted(batch):
                    if torch.is_tensor(batch[key]):
                        digest.update(key.encode())
                        digest.update(batch[key].contiguous().numpy().tobytes())
                if digest.hexdigest() != item["sha256"]:
                    raise ValueError(f"Validation input mismatch: {item['index']}")
                with self.accelerator.autocast():
                    context, target, context_time, future_time = self._prepare_val_inference_batch(batch)
                    prediction = self._rollout_predict_with_settings(
                        context=context, total_future_frames=4, mode="frame", rollout_block_size=1,
                        detach_history=True, context_time=context_time, future_time=future_time,
                        future_modalities=None, use_gt_future_modalities=False,
                    )["rain"]
                rain = self._denormalize_rain_for_metrics(prediction.float()).clamp_min(0).cpu()
                truth = self._denormalize_rain_for_metrics(target["rain"].float()).clamp_min(0).cpu()
                anchor = self._denormalize_rain_for_metrics(context[:, -self.rain_c:, -1:].float()).clamp_min(0).cpu()
                if not torch.isfinite(rain).all():
                    raise FloatingPointError("Nonfinite free-rollout prediction")
                accumulator.update(rain, truth)
                for lead, row in evolution_statistics(rain, truth, anchor).items():
                    total = sums.setdefault(lead, {key: 0.0 for key in row})
                    for key, value in row.items():
                        total[key] += value
                if ordinal < 3:
                    fields = {"GT": truth, "Persistence": anchor.expand_as(truth), "Coarse": rain}
                    save_preview(
                        output / f"rain_index_{item['index']}.png",
                        {name: value[0, 0].numpy() for name, value in fields.items()}, item["index"],
                    )
                    save_changes(output / f"changes_index_{item['index']}.png", fields, anchor)
        report = {
            "step": self.global_step, "samples": len(manifest), "unit": "mm/min",
            "manifest_source": str(manifest_path), "inference": "free rollout; no observed future modalities",
            "deterministic": accumulator.result(), "evolution": summarize(sums),
        }
        (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        with (self.proj_dir / "fixed_validation_metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(report) + "\n")
        self.log_msg(f"[FixedVal][{self.global_step}] {report['deterministic']['overall']}")
        self.model.train()


@hydra.main(config_path="../config/ts_rain_train", config_name="rain_trainer_coarse_evolution", version_base=None)
def main(cfg: DictConfig) -> None:
    trainer = CoarseEvolutionTrainer(cfg)
    if cfg.train.get("validate_only", False):
        try:
            trainer._resume_if_needed()
            trainer._run_val()
        finally:
            trainer._close_tensorboard_writer()
        return
    if cfg.train.get("validate_before_training", False) and not cfg.train.get("resume_path"):
        trainer._run_val()
    trainer.run()


if __name__ == "__main__":
    main()
