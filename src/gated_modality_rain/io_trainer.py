import torch
from omegaconf import DictConfig

from src.gated_modality_rain.trainer import GatedModalityRainTrainer


class GatedModalityRainIOTrainer(GatedModalityRainTrainer):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.train_dataloader = self.train_dataloadaer
        del self.train_dataloadaer

        model = self.accelerator.unwrap_model(self.model)
        cross_modal_adapter = getattr(model, "cross_modal_adapter", None)
        active_modules = {
            "spatial_modality_gate": getattr(model, "spatial_modality_gate", None) is not None,
            "cross_modal_adapter": cross_modal_adapter is not None,
            "local_window_refiner": getattr(model, "local_window_refiner", None) is not None,
            "consensus_calibration": getattr(cross_modal_adapter, "consensus_calibration", None) is not None,
        }
        self.log_msg(
            "Active model modules: " + ", ".join(f"{name}={enabled}" for name, enabled in active_modules.items())
        )
        self.log_msg(
            f"I/O: prepared_loader={type(self.train_dataloader).__name__}, "
            f"non_blocking={self.accelerator.non_blocking}, shared_batch_conversion=True"
        )

    def train_step(self, batch: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], bool]:
        model_fields = {
            f"{modality}_{period}"
            for modality in ("radar", "satellite", "rain", "time")
            for period in ("past", "future")
        }
        device_batch = {
            name: tensor.to(
                device=self.device,
                dtype=torch.float32 if name in model_fields else tensor.dtype,
                non_blocking=self.accelerator.non_blocking,
            )
            for name, tensor in batch.items()
        }
        return super().train_step(device_batch)
