"""Paired CCC experiment with the pretrained backbone and decoders frozen."""

import hashlib
import json
from pathlib import Path

import accelerate
import hydra
import torch
from omegaconf import DictConfig
from safetensors import safe_open
from torch import nn

from src.trainer.rain_trainer_coarse_evolution import CoarseEvolutionTrainer


def freeze_pretrained_parameters(model: nn.Module) -> dict:
    prefixes = ("cross_modal_adapter.", "local_window_refiner.")
    if model.cross_modal_adapter is None or model.local_window_refiner is None:
        raise ValueError("Both experimental arms require cross attention and the local branch")
    groups = {"frozen": 0, "cross_attention": 0, "local": 0, "ccc": 0}
    trainable_names = []
    for name, parameter in model.named_parameters():
        trainable = name.startswith(prefixes)
        parameter.requires_grad_(trainable)
        group = "frozen"
        if trainable:
            trainable_names.append(name)
            group = "local" if name.startswith("local_window_refiner.") else "cross_attention"
            if ".consensus_calibration." in name:
                group = "ccc"
        groups[group] += parameter.numel()
    return {"parameter_counts": groups, "trainable_names": trainable_names}


def frozen_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        if not value.requires_grad:
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class CCCFrozenAblationTrainer(CoarseEvolutionTrainer):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        # Extra CCC initialization must not shift the training RNG stream.
        accelerate.utils.set_seed(int(self.train_cfg.seed))
        model = self.accelerator.unwrap_model(self.model)
        self.initial_frozen_digest = frozen_digest(model)
        self.gradient_audit = {}
        self.batch_fingerprints = []
        tracked = ["cross_modal_adapter.gate", "local_window_refiner.gate"]
        calibration = model.cross_modal_adapter.consensus_calibration
        if calibration is not None:
            for index in range(3):
                tracked += [
                    f"cross_modal_adapter.consensus_calibration.conditional.{index}.net.1.weight",
                    f"cross_modal_adapter.consensus_calibration.conditional.{index}.net.3.weight",
                ]
            tracked.append("cross_modal_adapter.consensus_calibration.condition_scale")
        parameters = dict(model.named_parameters())
        for name in tracked:
            if name not in parameters:
                raise ValueError(f"Missing gradient audit parameter: {name}")
            self.gradient_audit[name] = {"calls": 0, "nonzero_calls": 0, "max_abs": 0.0}

            def record(gradient: torch.Tensor, parameter_name: str = name) -> None:
                if self.global_step >= 2:
                    return
                peak = float(gradient.detach().abs().max())
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError(f"Nonfinite gradient: {parameter_name}")
                row = self.gradient_audit[parameter_name]
                row["calls"] += 1
                row["nonzero_calls"] += int(peak > 0)
                row["max_abs"] = max(row["max_abs"], peak)

            parameters[name].register_hook(record)
        self.scope_audit["initial_frozen_sha256"] = self.initial_frozen_digest
        (self.proj_dir / "trainable_scope.json").write_text(json.dumps(self.scope_audit, indent=2))
        self.log_msg(f"[FrozenScope] {self.scope_audit['parameter_counts']}")

    def _build_optim_sched(self) -> tuple:
        if self.accelerator.state.deepspeed_plugin is not None:
            raise ValueError("This ablation requires a single GPU without DeepSpeed")
        path = Path(self.train_cfg.init_model_path) / "model.safetensors"
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            source_keys = set(checkpoint.keys())
            model_state = self.model.state_dict()
            missing = set(model_state) - source_keys
            if source_keys - set(model_state) or any(".consensus_calibration." not in key for key in missing):
                raise ValueError("Initialization checkpoint must match all non-CCC model tensors")
            if any(".consensus_calibration." in key for key in source_keys):
                raise ValueError("Start both arms from the same checkpoint without CCC")
            for key in source_keys:
                if not torch.equal(model_state[key].cpu(), checkpoint.get_tensor(key)):
                    raise ValueError(f"Initialization did not load checkpoint tensor: {key}")
        self.scope_audit = freeze_pretrained_parameters(self.model)
        named = [(name, value) for name, value in self.model.named_parameters() if value.requires_grad]
        params = named if "muon" in self.train_cfg.optim._target_ else [value for _, value in named]
        optimizer = hydra.utils.instantiate(self.train_cfg.optim)(params)
        scheduler = hydra.utils.instantiate(self.train_cfg.scheduler)(optimizer=optimizer)
        return optimizer, scheduler

    def train_step(self, batch: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], bool]:
        if self.global_step < 2:
            digest = hashlib.sha256()
            for name in sorted(batch):
                value = batch[name]
                if torch.is_tensor(value):
                    digest.update(name.encode())
                    digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
            self.batch_fingerprints.append(digest.hexdigest())
        return super().train_step(batch)

    @torch.no_grad()
    def _run_val(self) -> None:
        model = self.accelerator.unwrap_model(self.model)
        current = frozen_digest(model)
        if current != self.initial_frozen_digest:
            raise RuntimeError("Frozen pretrained parameters changed")
        if any(value.grad is not None for value in model.parameters() if not value.requires_grad):
            raise RuntimeError("Frozen parameter unexpectedly received gradients")
        super()._run_val()
        report = {"step": self.global_step, "frozen_unchanged": True, "gradients": self.gradient_audit,
                  "first_two_updates_batches": self.batch_fingerprints}
        with (self.proj_dir / "gradient_audit.jsonl").open("a") as handle:
            handle.write(json.dumps(report) + "\n")
        if self.global_step >= 2:
            inactive = [name for name, row in self.gradient_audit.items() if row["nonzero_calls"] == 0]
            if inactive:
                raise RuntimeError(f"No gradient in the first two updates: {inactive}")


@hydra.main(config_path="../config/ts_rain_train", config_name="rain_trainer_ccc_frozen_ablation", version_base=None)
def main(cfg: DictConfig) -> None:
    trainer = CCCFrozenAblationTrainer(cfg)
    try:
        if not cfg.train.get("resume_path"):
            trainer._run_val()
        trainer.run()
    finally:
        trainer._close_tensorboard_writer()


if __name__ == "__main__":
    main()
