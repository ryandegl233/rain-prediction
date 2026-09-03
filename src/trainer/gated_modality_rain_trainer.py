import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import hydra
from omegaconf import DictConfig

from src.gated_modality_rain.trainer import GatedModalityRainTrainer


@hydra.main(config_path="../config/ts_rain_train", config_name="gated_modality_rain_trainer", version_base=None)
def main(cfg: DictConfig) -> None:
    trainer = GatedModalityRainTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
