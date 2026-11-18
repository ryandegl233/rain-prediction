from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from src.baselines_stage3.deep.training import DeepTrainer


@hydra.main(config_path="../config/ts_rain_train", config_name="baseline_deep_train", version_base=None)
def main(cfg: DictConfig) -> None:
    directory = Path(cfg.train.proj_dir or HydraConfig.get().runtime.output_dir)
    trainer = DeepTrainer(cfg, directory)
    print(
        f"method={cfg.baseline_model.name} parameters={sum(p.numel() for p in trainer.model.parameters()):,} output={directory}",
        flush=True,
    )
    trainer.run()


if __name__ == "__main__":
    main()
