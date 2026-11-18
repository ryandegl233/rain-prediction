from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.baselines_stage3.evaluation import evaluate, write_report
from src.baselines_stage3.runtime import check_data, classical_models, complete_report, load_reference, validate_config


@hydra.main(config_path="../config/ts_rain_train", config_name="baseline_stage3_eval", version_base=None)
def main(cfg: DictConfig) -> None:
    validate_config(cfg)
    check_data(cfg, ("val",))
    models = classical_models(cfg)
    if cfg.evaluation.reference_checkpoint:
        models["reference"] = load_reference(cfg)
    dataset, loader = hydra.utils.instantiate(cfg.dataset.val)
    report = complete_report(evaluate(models, loader, cfg), cfg, len(dataset))
    destination = Path(cfg.evaluation.output_dir or HydraConfig.get().runtime.output_dir)
    write_report(report, destination)
    OmegaConf.save(cfg, destination / "config_resolved.yaml", resolve=True)
    print(f"Evaluation saved: {destination.resolve()}", flush=True)


if __name__ == "__main__":
    main()
