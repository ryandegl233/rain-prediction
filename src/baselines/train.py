import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../config/ts_rain_train", config_name="baselines_train_reference", version_base=None)
def main(cfg: DictConfig) -> None:
    from src.baselines.trainer import BaselineComparisonTrainer

    BaselineComparisonTrainer(cfg).run()


if __name__ == "__main__":
    main()
