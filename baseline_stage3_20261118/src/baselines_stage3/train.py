import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../config/ts_rain_train", config_name="baseline_stage3_train", version_base=None)
def main(cfg: DictConfig) -> None:
    from src.baselines_stage3.trainer import Stage3ComparisonTrainer

    Stage3ComparisonTrainer(cfg).run()


if __name__ == "__main__":
    main()
