from omegaconf import DictConfig

from src.rain_local_fusion.model import RainLocalFusionModel
from src.rain_priority.trainer import RainPriorityTrainer


class RainLocalFusionTrainer(RainPriorityTrainer):
    def __init__(self, cfg: DictConfig) -> None:
        if cfg.train.priority.horizon.enabled:
            raise ValueError("Keep horizon.enabled=false for the A0/R0/R1 comparison.")
        expected = {"radar": 0.1, "satellite": 0.1, "rain": 1.0}
        if dict(cfg.train.loss_weights) != expected or cfg.train.next_pred.rollout_branch.weight != 0.2:
            raise ValueError("Keep A coefficients: radar=0.1, satellite=0.1, rain=1.0, rollout=0.2.")
        if cfg.rain_prediction_model.get("spatial_modality_gate_enabled", False):
            raise ValueError("The shared input gate must be disabled.")
        super().__init__(cfg)
        model = self.accelerator.unwrap_model(self.model)
        if not isinstance(model, RainLocalFusionModel):
            raise ValueError("RainLocalFusionTrainer requires RainLocalFusionModel.")
        branch = model.local_fusion
        count = sum(parameter.numel() for parameter in branch.parameters()) if branch is not None else 0
        kernel = branch.spatial[0].kernel_size[0] if branch is not None else 0
        variant = "A0" if branch is None else ("R0" if kernel == 1 else "R1")
        self.log_msg(f"Rain local fusion: variant={variant}, kernel={kernel}, added_parameters={count}, fixed_q=0.25.")
        self.log_msg("Same labels and A objective; only the rain decoder receives the new residual.")
