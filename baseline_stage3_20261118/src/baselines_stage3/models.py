import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn

from src.baselines_stage3.contracts import ForecastInput, latest_rain


class Persistence(nn.Module):
    uses_auxiliary_observations = False

    def forward(self, inputs: ForecastInput) -> torch.Tensor:
        rain, _ = latest_rain(inputs)
        return rain.expand(-1, -1, 4, -1, -1).clone()


class LucasKanade(nn.Module):
    uses_auxiliary_observations = False

    def __init__(self, rain_threshold: float = 0.02, size_opening: int = 3) -> None:
        super().__init__()
        if not np.isfinite(rain_threshold) or rain_threshold <= 0 or size_opening < 0:
            raise ValueError("Invalid LK rain threshold or opening size.")
        self.rain_threshold = rain_threshold
        self.size_opening = size_opening

    def forward(self, inputs: ForecastInput) -> torch.Tensor:
        from pysteps.extrapolation.semilagrangian import extrapolate
        from pysteps.motion.lucaskanade import dense_lucaskanade

        predictions = []
        for frames, availability in zip(inputs.history[:, 11], inputs.available[:, 2], strict=True):
            observed = torch.where(availability)[0].tolist()
            rain = frames.numpy()
            if np.any(rain[observed] < 0):
                raise ValueError("LK requires nonnegative rain in dataset units.")
            if len(observed) == 1:
                prediction = np.repeat(rain[observed[-1]][None], 4, axis=0)
            else:
                selected = observed if len(observed) == 4 else observed[-2:]
                gap = 1 if len(observed) == 4 else selected[-1] - selected[-2]
                motion = 10 * np.log10(np.maximum(rain[selected], self.rain_threshold))
                velocity = dense_lucaskanade(motion, size_opening=self.size_opening) / gap
                horizons = [step - observed[-1] for step in range(4, 8)]
                prediction = extrapolate(rain[observed[-1]], velocity, horizons, outval=0.0, interp_order=1)
            if not np.isfinite(prediction).all():
                raise ValueError("LK produced nonfinite output.")
            predictions.append(prediction)
        return torch.from_numpy(np.stack(predictions).astype(np.float32))[:, None]


class Stage3Reference(nn.Module):
    def __init__(self, network: nn.Module, cfg: DictConfig) -> None:
        super().__init__()
        self.network = network
        self.uses_auxiliary_observations = cfg.val.rollout_use_gt_future_modalities
        self.residual = cfg.train.loss.rain_residual
        self.strict_isolation = cfg.train.strict_target_isolation
        self.history_window = cfg.train.next_pred.rollout_branch.history_window_frames
        if network.frame_patch_size != 1:
            raise ValueError("Stage 3 adapter requires frame_patch_size=1.")
        self.train(network.training)

    def forward(self, inputs: ForecastInput) -> torch.Tensor:
        device = next(self.network.parameters()).device
        history = inputs.history.to(device)
        mask = inputs.available.to(device)
        time = inputs.time.to(device) if inputs.time is not None else None
        if self.network.use_time_embedding and time is None:
            raise ValueError("Reference network requires historical time features.")
        if self.uses_auxiliary_observations and inputs.future_aux is None:
            raise ValueError("Stage 3 requires progressively arriving radar/satellite observations.")
        chunks = []
        for index, (name, section) in enumerate(
            (("radar", slice(0, 1)), ("satellite", slice(1, 11)), ("rain", slice(11, 12)))
        ):
            values = history[:, section]
            fill = getattr(self.network, f"{name}_mask_token").to(values) if self.network.use_modality_mask_token else 0
            chunks.append(torch.where(mask[:, index, None, :, None, None], values, fill))
        context = torch.cat(chunks, dim=1)
        anchor = latest_rain(inputs)[0].to(device)
        predictions = []
        for step in range(4):
            if self.history_window:
                context = context[:, :, -self.history_window :]
                if time is not None:
                    time = time[:, -self.history_window :]
            output = self.network.forward_ar(
                context_x=context,
                target_x=context[:, :, -1:],
                predict_frames=1,
                strict_target_isolation=self.strict_isolation,
                return_modality_dict=True,
                context_time=time,
                target_time=time[:, -1:] if time is not None else None,
            )
            rain = output["rain"]
            if self.residual.enabled:
                rain = rain + anchor
                if self.residual.clamp_output:
                    rain = rain.clamp_min(self.residual.min_value)
            predictions.append(rain)
            if step < 3:
                auxiliary = (
                    inputs.future_aux[:, :, step : step + 1].to(device)
                    if self.uses_auxiliary_observations
                    else torch.cat([output["radar"], output["satellite"]], dim=1)
                )
                context = torch.cat([context, torch.cat([auxiliary, rain], dim=1)], dim=2)
                anchor = rain
                if time is not None:
                    time = torch.cat([time, (time[:, -1:] + 1 / 48) % 1], dim=1)
        return torch.cat(predictions, dim=2)
