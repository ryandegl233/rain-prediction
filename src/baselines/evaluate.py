from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.baselines.evaluation import evaluate_forecasts, write_report
from src.baselines.runtime import build_classical, complete_report, load_reference, validate_protocol


@hydra.main(config_path="../config/ts_rain_train", config_name="baselines_eval", version_base=None)
def main(cfg: DictConfig) -> None:
    validate_protocol(cfg)
    models = build_classical(cfg)
    if cfg.evaluation.reference_checkpoint:
        models["reference"] = load_reference(cfg)
    dataset, loader = hydra.utils.instantiate(cfg.dataset.val)
    print(f"Evaluating {len(dataset)} windows; methods={list(models)}", flush=True)
    report = evaluate_forecasts(
        models,
        loader,
        [tuple(names) for names in cfg.baseline.combinations],
        list(cfg.val.csi_thresholds),
        max_batches=cfg.evaluation.max_batches,
        progress_every=cfg.evaluation.progress_every,
    )
    report = complete_report(report, cfg, len(dataset))
    destination = cfg.evaluation.output_dir or HydraConfig.get().runtime.output_dir
    output = Path(destination)
    write_report(report, output)
    OmegaConf.save(cfg, output / "config_resolved.yaml", resolve=True)
    print(f"Evaluation saved: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
