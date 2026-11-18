from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.baselines_stage3.deep.training import load_deep, prepare_config
from src.baselines_stage3.evaluation import evaluate, write_report
from src.baselines_stage3.runtime import check_data, classical_models, complete_report


@hydra.main(config_path="../config/ts_rain_train", config_name="baseline_deep_train", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg = prepare_config(cfg)
    if not cfg.evaluation.reference_checkpoint:
        raise ValueError("Supply evaluation.reference_checkpoint for the selected baseline.")
    check_data(cfg, ("val",))
    model = load_deep(cfg)
    dataset, loader = hydra.utils.instantiate(cfg.dataset.val)
    report = complete_report(
        evaluate({cfg.baseline_model.name: model, **classical_models(cfg)}, loader, cfg), cfg, len(dataset)
    )
    directory = Path(cfg.evaluation.output_dir or HydraConfig.get().runtime.output_dir)
    write_report(report, directory)
    OmegaConf.save(cfg, directory / "config_resolved.yaml", resolve=True)
    print(f"Evaluation saved: {directory}", flush=True)


if __name__ == "__main__":
    main()
