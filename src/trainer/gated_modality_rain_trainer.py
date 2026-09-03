import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import hydra
from omegaconf import DictConfig

from src.trainer.rain_trainer_ts_next_frame import RainTSNextFrameTrainer


@hydra.main(config_path="../config/ts_rain_train", config_name="gated_modality_rain_trainer", version_base=None)
def main(cfg: DictConfig) -> None:
    trainer = RainTSNextFrameTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
