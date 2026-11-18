from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from src.baselines_stage3.evaluation import evaluate, write_report
from src.baselines_stage3.models import Stage3Reference
from src.baselines_stage3.runtime import (
    check_data,
    checkpoint_state,
    classical_models,
    complete_report,
    validate_config,
)
from src.trainer.rain_trainer_ts_next_frame_missing_modality import RainTSNextFrameMissingModalityTrainer


class Stage3ComparisonTrainer(RainTSNextFrameMissingModalityTrainer):
    def __init__(self, cfg: DictConfig) -> None:
        cfg = OmegaConf.merge(cfg)
        if cfg.train.get("resume_path"):
            cfg.train.init_model_path = None
        for split in ("train", "val"):
            if cfg.dataset[split].get("inp_dirs") is not None:
                cfg.dataset[f"{split}_inp_dirs"] = list(cfg.dataset[split].inp_dirs)
        validate_config(cfg)
        check_data(cfg, ("train", "val"))
        initialization = cfg.train.get("init_model_path")
        if initialization and not Path(initialization).is_file():
            raise FileNotFoundError(f"Stage 2 initialization weights are required: {initialization}")
        self.baseline_models = classical_models(cfg)
        super().__init__(cfg)
        if self.accelerator.num_processes != 1:
            raise ValueError("Use one process for Stage 3 baseline comparison.")
        if initialization:
            self.accelerator.unwrap_model(self.model).load_state_dict(
                checkpoint_state(Path(initialization), "model"), strict=True
            )

    def _run_val(self) -> None:
        kind = self.cfg.evaluation.checkpoint_kind
        if kind == "ema":
            if self.ema_model is None:
                raise ValueError("EMA validation requested but EMA is disabled.")
            network = self.ema_model.ema_model
        else:
            network = self.accelerator.unwrap_model(self.model)
        models = {"reference": Stage3Reference(network, self.cfg), **self.baseline_models}
        report = evaluate(models, self.val_dataloader, self.cfg)
        complete_report(report, self.cfg, len(self.val_dataset))
        report.update(global_step=self.global_step, reference_weights=kind)
        destination = self.proj_dir / "baseline_validation" / f"step_{self.global_step:08d}"
        write_report(report, destination)
        scalars = {
            f"baseline/{scenario}/{method}/{metric}": value
            for scenario, methods in report["results"].items()
            for method, result in methods.items()
            for metric, value in result["metrics"].items()
            if value is not None
        }
        self.last_val_metrics = {"step": self.global_step, "values": scalars}
        self._log_tensorboard_scalars(scalars, self.global_step)
        self.log_msg(f"[Baseline val] {report['samples']} windows; report={destination}")

    def _save_checkpoint(self) -> None:
        if self.last_saved_checkpoint_step == self.global_step:
            return
        destination = self._checkpoint_dir()
        if (destination / "metadata.json").exists():
            self.last_saved_checkpoint_step = self.global_step
            return
        self.accelerator.save_state(str(destination), safe_serialization=True)
        if self.ema_model is not None:
            torch.save(self.ema_model.state_dict(), destination / "ema.pt")
        metadata = self._checkpoint_metadata(destination)
        metadata["retention"] = {"mode": "keep_all", "automatic_deletion": False}
        self._write_json(destination / "metadata.json", metadata)
        self._write_json(
            destination.parent / "latest.json",
            {"global_step": self.global_step, "checkpoint": destination.name, "retention": metadata["retention"]},
        )
        self.last_saved_checkpoint_step = self.global_step
        self.log_msg(f"[Checkpoint] {destination}; previous checkpoints retained")

    def train(self) -> None:
        self._resume_if_needed()
        self.model.train()
        while self.global_step < self.train_cfg.max_steps:
            for batch in self.train_dataloader:
                logs, did_step = self.train_step(batch)
                if not did_step:
                    continue
                self._log_metrics(logs)
                if self.global_step % self.val_cfg.val_duration == 0:
                    self._run_val()
                if self.global_step % self.train_cfg.save_every == 0:
                    self._save_checkpoint()
                if self.global_step >= self.train_cfg.max_steps:
                    break
        self._save_checkpoint()
        self.log_msg("Training finished.")
