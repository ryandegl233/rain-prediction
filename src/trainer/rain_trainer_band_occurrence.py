"""Train one weak-rain occurrence head, then sweep suppression at fixed weights."""

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig
import torch

from src.trainer.rain_trainer_coarse_evolution import CoarseEvolutionTrainer
from src.trainer.rain_trainer_soft_occurrence import SoftOccurrenceTrainer


class BandOccurrenceTrainer(SoftOccurrenceTrainer):
    @torch.no_grad()
    def run_suppression_sweep(self) -> None:
        model = self.accelerator.unwrap_model(self.model)
        original_dir = Path(self.proj_dir)
        original_suppression = float(model.occurrence_max_suppression)
        original_strength = float(model.soft_gate_strength)
        values = [float(value) for value in self.train_cfg.soft_occurrence.sweep_suppressions]
        if not values or any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("soft_occurrence.sweep_suppressions must contain values in [0, 1]")
        try:
            model.set_soft_gate_strength(1.0)
            for value in values:
                name = f"lambda{value:.2f}".replace(".", "p")
                model.occurrence_max_suppression = value
                self.proj_dir = original_dir / name
                self.proj_dir.mkdir(parents=True, exist_ok=False)
                CoarseEvolutionTrainer._run_val(self)
        finally:
            model.occurrence_max_suppression = original_suppression
            model.set_soft_gate_strength(original_strength)
            self.proj_dir = original_dir
        (original_dir / "suppression_sweep.json").write_text(json.dumps({
            "step": self.global_step,
            "values": values,
            "shared_weights": True,
            "rollout_history": "ungated",
        }, indent=2))


@hydra.main(config_path="../config/ts_rain_train", config_name="rain_trainer_band_occurrence", version_base=None)
def main(cfg: DictConfig) -> None:
    trainer = BandOccurrenceTrainer(cfg)
    try:
        trainer._run_val()
        trainer.run()
        trainer.run_suppression_sweep()
    finally:
        trainer._close_tensorboard_writer()


if __name__ == "__main__":
    main()
