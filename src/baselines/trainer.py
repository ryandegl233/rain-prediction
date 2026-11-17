import torch
from omegaconf import DictConfig

from src.baselines.evaluation import evaluate_forecasts, write_report
from src.baselines.runtime import build_classical, complete_report, validate_protocol, wrap_reference
from src.trainer.rain_trainer_ts_next_frame import RainTSNextFrameTrainer


class BaselineComparisonTrainer(RainTSNextFrameTrainer):
    def __init__(self, cfg: DictConfig) -> None:
        validate_protocol(cfg)
        self.baseline_models = build_classical(cfg)
        super().__init__(cfg)
        if self.accelerator.num_processes != 1:
            raise ValueError("Baseline comparison training currently supports a single process.")
        self.train_dataloader = self.train_dataloadaer
        self.log_msg("Baseline validation: fixed origin, history only; persistence and LK require no training.")

    def train_step(self, batch: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], bool]:
        batch = {key: value.to(self.device, non_blocking=self.accelerator.non_blocking) for key, value in batch.items()}
        return super().train_step(batch)

    def _run_val(self) -> None:
        weights = self.cfg.evaluation.checkpoint_kind
        if weights == "ema":
            if self.ema_model is None:
                raise ValueError("EMA validation requested but EMA is disabled.")
            network = self.ema_model.ema_model
        elif weights == "model":
            network = self.accelerator.unwrap_model(self.model)
        else:
            raise ValueError("checkpoint_kind must be model or ema.")
        models = {"reference": wrap_reference(network, self.cfg), **self.baseline_models}
        self.log_msg(f"[Baseline val] step={self.global_step}, weights={weights}")
        with self.accelerator.autocast():
            report = evaluate_forecasts(
                models,
                self.val_dataloader,
                [tuple(names) for names in self.cfg.baseline.combinations],
                list(self.val_cfg.csi_thresholds),
                max_batches=self.cfg.evaluation.max_batches,
                progress_every=self.cfg.evaluation.progress_every,
            )
        report = complete_report(report, self.cfg, len(self.val_dataset))
        report.update(global_step=self.global_step, reference_weights=weights)
        destination = self.proj_dir / "baseline_validation" / f"step-{self.global_step:08d}"
        write_report(report, destination)
        scalars = {
            f"baseline/{combination}/{method}/{metric}": value
            for combination, methods in report["results"].items()
            for method, result in methods.items()
            for metric, value in result.get("metrics", {}).items()
            if value is not None
        }
        self._log_tensorboard_scalars(scalars, self.global_step)
        self.log_msg(f"[Baseline val] {report['samples']} windows; report={destination}")
