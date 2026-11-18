import json
import random
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import torch
from ema_pytorch import EMA
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file, save_file

from src.baselines_stage3.deep.common import DeepForecaster, objective_terms
from src.baselines_stage3.evaluation import evaluate, write_report
from src.baselines_stage3.runtime import ROOT, check_data, classical_models, complete_report, validate_config


def prepare_config(cfg: DictConfig) -> DictConfig:
    cfg = OmegaConf.merge(cfg)
    validate_config(cfg)
    if cfg.train.get("init_model_path"):
        raise ValueError(
            "Deep baselines cannot load the reference Stage 2 initialization; use their own resume checkpoint."
        )
    if cfg.train.max_steps < 1 or cfg.train.save_every < 1 or cfg.val.val_duration < 1:
        raise ValueError("Training, checkpoint and evaluation intervals must be positive.")
    if cfg.accelerator.mixed_precision not in ("no", "bf16"):
        raise ValueError("Deep baseline training supports mixed_precision=no or bf16.")
    if cfg.baseline_model.name == "npm":
        if (
            cfg.baseline.scenarios != ["satellite_only"]
            or cfg.val.rollout_use_gt_future_modalities
            or cfg.train.next_pred.rollout_branch.use_gt_future_modalities
        ):
            raise ValueError("NPM requires its satellite-only configuration with future observations disabled.")
    elif cfg.baseline_model.options.context_frames != cfg.train.next_pred.rollout_branch.history_window_frames:
        raise ValueError("Baseline context capacity must match the configured rollout history window.")
    if cfg.baseline_model.name != "npm":
        branch = cfg.train.next_pred.rollout_branch
        missing = cfg.train.next_pred.missing_modality
        if not branch.enabled or not cfg.train.next_pred.sequence_loss.enabled or not missing.enabled:
            raise ValueError("This comparison requires sequence, rollout and missing-modality branches enabled.")
        if (
            branch.loss_on != "rain"
            or branch.delta_loss.enabled
            or branch.rollout_frames != 4
            or branch.mode != "frame"
        ):
            raise ValueError("Supported rollout objective is four-frame rain MSE without delta loss.")
    if (
        cfg.train.loss.mode != "mse"
        or any(cfg.train.loss[name].enabled for name in ("temporal_diff", "rain_region_weight", "rain_event_aux"))
        or cfg.train.gan.enabled
    ):
        raise ValueError(
            "Deep comparison supports the configured MSE objective without extra GAN/event/region/temporal losses."
        )
    for split in ("train", "val"):
        data = cfg.dataset[split]
        if data.get("inp_dirs") is not None:
            data.inp_dirs = [str(Path(name) if Path(name).is_absolute() else ROOT / name) for name in data.inp_dirs]
    return cfg


def training_signature(cfg: DictConfig) -> dict:
    sections = {
        name: OmegaConf.to_container(cfg.train[name], resolve=True)
        for name in ("loss", "loss_weights", "next_pred", "optim", "scheduler")
    }
    return {
        **sections,
        "seed": cfg.train.seed,
        "ema": OmegaConf.to_container(cfg.ema, resolve=True),
        "gradient_accumulation_steps": cfg.accelerator.gradient_accumulation_plugin.num_steps,
        "mixed_precision": cfg.accelerator.mixed_precision,
        "training_scenario": cfg.deep.training_scenario,
        "auxiliary_weights": OmegaConf.to_container(cfg.deep.auxiliary_weights, resolve=True),
    }


def load_deep(cfg: DictConfig) -> DeepForecaster:
    folder = Path(cfg.evaluation.reference_checkpoint)
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    architecture = OmegaConf.to_container(cfg.baseline_model, resolve=True)
    if metadata["baseline_model"] != architecture:
        raise ValueError("Checkpoint model configuration differs from the selected baseline.")
    if metadata["training_signature"]["loss"]["rain_residual"] != OmegaConf.to_container(
        cfg.train.loss.rain_residual, resolve=True
    ):
        raise ValueError("Checkpoint rain residual output convention differs from evaluation configuration.")
    model = DeepForecaster(cfg)
    filename = "ema.safetensors" if cfg.evaluation.checkpoint_kind == "ema" else "model.safetensors"
    model.load_state_dict(load_file(str(folder / filename)), strict=True)
    model.to(cfg.evaluation.device).eval()
    model.checkpoint_metadata = metadata
    return model


class DeepTrainer:
    def __init__(self, cfg: DictConfig, output_dir: Path) -> None:
        self.cfg = prepare_config(cfg)
        cfg = self.cfg
        check_data(cfg, ("train", "val"))
        self.directory = output_dir
        self.directory.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, self.directory / "config_resolved.yaml", resolve=True)
        random.seed(cfg.train.seed)
        np.random.seed(cfg.train.seed)
        torch.manual_seed(cfg.train.seed)
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if cfg.deep.device == "auto"
            else torch.device(cfg.deep.device)
        )
        self.model = DeepForecaster(cfg).to(self.device)
        self.optimizer = hydra.utils.instantiate(cfg.train.optim)(params=self.model.parameters())
        self.scheduler = hydra.utils.instantiate(cfg.train.scheduler)(optimizer=self.optimizer)
        self.ema = (
            EMA(
                self.model,
                beta=cfg.ema.beta,
                update_after_step=cfg.ema.update_after_step,
                update_every=cfg.ema.update_every,
            )
            if cfg.ema.beta > 0
            else None
        )
        self.train_dataset, self.train_loader = hydra.utils.instantiate(cfg.dataset.train)
        self.val_dataset, self.val_loader = hydra.utils.instantiate(cfg.dataset.val)
        self.classical = classical_models(cfg)
        self.step = 0
        if cfg.train.resume_path:
            folder = Path(cfg.train.resume_path)
            if (folder / "latest.json").is_file():
                folder = folder / json.loads((folder / "latest.json").read_text())["checkpoint"]
            metadata = json.loads((folder / "metadata.json").read_text())
            if metadata["baseline_model"] != OmegaConf.to_container(cfg.baseline_model, resolve=True):
                raise ValueError("Resume checkpoint belongs to a different model configuration.")
            if metadata["max_steps"] != cfg.train.max_steps:
                raise ValueError(
                    "Resume must keep the original max_steps so the learning-rate schedule remains unchanged."
                )
            if metadata["training_signature"] != training_signature(cfg):
                raise ValueError(
                    "Resume training signature differs: keep loss, scheduler, optimizer, EMA and sampling settings unchanged."
                )
            self.model.load_state_dict(load_file(str(folder / "model.safetensors")), strict=True)
            state = torch.load(folder / "training.pt", map_location="cpu", weights_only=True)
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            if self.ema is not None:
                self.ema.load_state_dict(state["ema"])
            self.step = metadata["step"]
            random.setstate(state["python_rng"])
            numpy_rng = state["numpy_rng"]
            np.random.set_state((numpy_rng[0], np.array(numpy_rng[1], dtype=np.uint32), *numpy_rng[2:]))
            torch.set_rng_state(state["torch_rng"])
            if self.device.type == "cuda":
                torch.cuda.set_rng_state_all(state["cuda_rng"])
        self.last_logs: dict[str, float] = {}

    def save(self) -> Path:
        folder = self.directory / "checkpoints" / f"step_{self.step:08d}"
        if (folder / "metadata.json").is_file():
            return folder
        folder.mkdir(parents=True, exist_ok=True)
        save_file(
            {name: value.detach().cpu().contiguous() for name, value in self.model.state_dict().items()},
            str(folder / "model.safetensors"),
        )
        if self.ema is not None:
            save_file(
                {name: value.detach().cpu().contiguous() for name, value in self.ema.ema_model.state_dict().items()},
                str(folder / "ema.safetensors"),
            )
        numpy_rng = np.random.get_state()
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "ema": self.ema.state_dict() if self.ema is not None else None,
                "python_rng": random.getstate(),
                "numpy_rng": (numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
            },
            folder / "training.pt",
        )
        metadata = {
            "kind": "deep_baseline",
            "step": self.step,
            "max_steps": self.cfg.train.max_steps,
            "baseline_model": OmegaConf.to_container(self.cfg.baseline_model, resolve=True),
            "training_signature": training_signature(self.cfg),
            "automatic_deletion": False,
        }
        (folder / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        (folder.parent / "latest.json").write_text(
            json.dumps({"checkpoint": folder.name, "step": self.step}) + "\n", encoding="utf-8"
        )
        return folder

    def validate(self) -> dict:
        network = self.model
        if self.cfg.evaluation.checkpoint_kind == "ema":
            if self.ema is None:
                raise ValueError("EMA validation requested with EMA disabled.")
            network = self.ema.ema_model
        models = {self.cfg.baseline_model.name: network, **self.classical}
        report = complete_report(evaluate(models, self.val_loader, self.cfg), self.cfg, len(self.val_dataset))
        report.update(global_step=self.step, reference_weights=self.cfg.evaluation.checkpoint_kind)
        write_report(report, self.directory / "baseline_validation" / f"step_{self.step:08d}")
        print(f"validation step={self.step} samples={report['samples']} complete={report['complete']}", flush=True)
        return report

    def run(self, stop_after: int | None = None) -> None:
        target = self.cfg.train.max_steps if stop_after is None else min(stop_after, self.cfg.train.max_steps)
        if self.step >= target:
            return
        accumulation = self.cfg.accelerator.gradient_accumulation_plugin.num_steps
        if accumulation < 1:
            raise ValueError("Gradient accumulation must be positive.")
        batches = iter(self.train_loader)
        self.model.train()
        precision = self.cfg.accelerator.mixed_precision
        while self.step < target:
            self.optimizer.zero_grad(set_to_none=True)
            logs = defaultdict(float)
            for _ in range(accumulation):
                try:
                    batch = next(batches)
                except StopIteration:
                    batches = iter(self.train_loader)
                    batch = next(batches, None)
                    if batch is None:
                        raise ValueError("Training dataset yielded no batches.")
                with torch.autocast(
                    self.device.type,
                    dtype=torch.bfloat16 if precision == "bf16" else torch.float16,
                    enabled=precision != "no",
                ):
                    for loss, values in objective_terms(self.model, batch, self.step):
                        if not torch.isfinite(loss):
                            raise FloatingPointError(f"Nonfinite training loss at step {self.step}.")
                        (loss / accumulation).backward()
                        for name, value in values.items():
                            logs[name] += value / accumulation
            norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.train.max_grad_norm, error_if_nonfinite=True
            )
            self.optimizer.step()
            self.scheduler.step()
            self.step += 1
            if self.ema is not None:
                self.ema.update()
            logs.update(step=self.step, lr=self.optimizer.param_groups[0]["lr"], grad_norm=norm.item())
            self.last_logs = dict(logs)
            with (self.directory / "train_metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(self.last_logs, allow_nan=False) + "\n")
            if self.step % self.cfg.train.log.log_every == 0:
                print(json.dumps(self.last_logs), flush=True)
            if self.step % self.cfg.val.val_duration == 0:
                self.validate()
            if self.step % self.cfg.train.save_every == 0:
                self.save()
        self.save()
