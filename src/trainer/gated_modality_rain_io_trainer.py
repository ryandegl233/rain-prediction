import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import hydra
import torch
from omegaconf import DictConfig

from src.gated_modality_rain.io_trainer import GatedModalityRainIOTrainer


@hydra.main(config_path="../config/ts_rain_train", config_name="gated_modality_rain_io", version_base=None)
def main(cfg: DictConfig) -> None:
    trainer = GatedModalityRainIOTrainer(cfg)
    if trainer.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(trainer.device)

    try:
        trainer.run()
    finally:
        if trainer.device.type == "cuda":
            allocated_gib = torch.cuda.max_memory_allocated(trainer.device) / 1024**3
            reserved_gib = torch.cuda.max_memory_reserved(trainer.device) / 1024**3
            trainer.log_msg(
                f"[CUDA memory] peak allocated={allocated_gib:.2f} GiB, peak reserved={reserved_gib:.2f} GiB"
            )


if __name__ == "__main__":
    main()
