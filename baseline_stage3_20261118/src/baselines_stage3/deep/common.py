from collections.abc import Iterator
from datetime import datetime

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.checkpoint import checkpoint

from src.baselines_stage3.contracts import COMBINATIONS, ForecastInput, latest_rain, make_inputs


def build_core(cfg: DictConfig) -> nn.Module:
    method = cfg.baseline_model.name
    options = OmegaConf.to_container(cfg.baseline_model.options, resolve=True)
    if method in ("convlstm", "predrnn_v2", "simvp"):
        from src.baselines_stage3.deep.openstl import build_openstl

        return build_openstl(method, options)
    if method in ("earthformer", "alphapre"):
        from src.baselines_stage3.deep.earth_alpha import build_earth_alpha

        return build_earth_alpha(method, options)
    if method == "npm":
        from src.baselines_stage3.deep.npm import build_npm

        return build_npm(options)
    raise ValueError(f"Unknown deep baseline: {method}")


def apply_mask(history: torch.Tensor, available: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
    channels = available.repeat_interleave(torch.tensor([1, 10, 1], device=available.device), dim=1)
    return torch.where(channels[..., None, None], history, token)


def training_masks(batch_size: int, frames: int, cfg: DictConfig, device: torch.device) -> torch.Tensor:
    probability = cfg.train.next_pred.missing_modality.full_modality_prob
    if cfg.deep.training_scenario == "sampled":
        probabilities = torch.tensor([probability] + [(1 - probability) / 6] * 6, device=device)
        indices = torch.multinomial(probabilities, batch_size * frames, replacement=True).view(batch_size, frames)
        masks = torch.tensor(list(COMBINATIONS.values()), device=device)[indices].transpose(1, 2).contiguous()
    else:
        masks = (
            torch.tensor(COMBINATIONS[cfg.deep.training_scenario], device=device)[None, :, None]
            .expand(batch_size, 3, frames)
            .clone()
        )
    masks[:, :, 0] = True
    if not cfg.train.next_pred.missing_modality.mask_future_seed_in_sequence_loss:
        masks[:, :, 4:] = True
    return masks


def npm_times(stamps: torch.Tensor | None, device: torch.device) -> torch.Tensor:
    if stamps is None:
        raise ValueError("NPM requires historical calendar timestamps for day/hour conditioning.")
    dates = [[datetime.fromtimestamp(stamp) for stamp in window] for window in stamps.detach().cpu().tolist()]
    return torch.tensor(
        [
            [[date.timetuple().tm_yday, date.hour + date.minute / 60 + date.second / 3600] for date in window]
            for window in dates
        ],
        dtype=torch.float32,
        device=device,
    )


class DeepForecaster(nn.Module):
    def __init__(self, cfg: DictConfig, core: nn.Module | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.core = build_core(cfg) if core is None else core
        self.satellite_only = cfg.baseline_model.name == "npm"
        self.uses_auxiliary_observations = not self.satellite_only and cfg.val.rollout_use_gt_future_modalities
        if not self.satellite_only:
            self.mask_token = nn.Parameter(torch.zeros(1, 12, 1, 1, 1))

    def next_frame(self, context: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        frames = self.cfg.baseline_model.options.context_frames
        context = context[:, :, -frames:]
        if context.shape[2] < frames:
            context = F.pad(context, (0, 0, 0, 0, frames - context.shape[2], 0))
        if self.training and self.cfg.deep.activation_checkpoint:
            return checkpoint(self.core, context, use_reentrant=False)
        return self.core(context)

    def rollout(
        self, history: torch.Tensor, available: torch.Tensor, future_aux: torch.Tensor | None, *, training: bool
    ) -> torch.Tensor:
        context = apply_mask(history, available, self.mask_token)
        indices = torch.arange(4, device=available.device).expand_as(available[:, 2])
        last = torch.where(available[:, 2], indices, -1).amax(dim=1)
        if (last < 0).any():
            raise ValueError("Multimodal residual rollout requires observed historical rain.")
        anchor = history[:, 11:12].gather(2, last[:, None, None, None, None].expand(-1, 1, 1, *history.shape[-2:]))
        branch = self.cfg.train.next_pred.rollout_branch
        use_gt = branch.use_gt_future_modalities if training else self.uses_auxiliary_observations
        if use_gt and future_aux is None:
            raise ValueError("GT auxiliary rollout requires radar/satellite observations.")
        predictions = []
        for step in range(4):
            raw, _ = self.next_frame(context)
            rain = raw[:, 11:12]
            if self.cfg.train.loss.rain_residual.enabled:
                rain = rain + anchor
                if self.cfg.train.loss.rain_residual.clamp_output:
                    rain = rain.clamp_min(self.cfg.train.loss.rain_residual.min_value)
            predictions.append(rain)
            if step < 3:
                auxiliary = future_aux[:, :, step : step + 1] if use_gt else raw[:, :11]
                new_frame = torch.cat([auxiliary, rain], dim=1)
                if training and branch.detach_history:
                    new_frame = new_frame.detach()
                context = torch.cat([context, new_frame], dim=2)
                anchor = rain.detach() if training and branch.detach_rain_anchor else rain
        return torch.cat(predictions, dim=2)

    def forward(self, inputs: ForecastInput) -> torch.Tensor:
        device = next(self.parameters()).device
        if self.satellite_only:
            if inputs.available[:, (0, 2)].any() or not inputs.available[:, 1].all() or inputs.future_aux is not None:
                raise ValueError("NPM requires strict satellite-only historical input.")
            result = self.core(inputs.history[:, 1:11].to(device), npm_times(inputs.timestamps, device))
            return result["rain"]
        return self.rollout(
            inputs.history.to(device),
            inputs.available.to(device),
            inputs.future_aux.to(device) if inputs.future_aux is not None else None,
            training=False,
        )


def objective_terms(
    model: DeepForecaster, batch: dict[str, torch.Tensor], step: int
) -> Iterator[tuple[torch.Tensor, dict[str, float]]]:
    cfg, device = model.cfg, next(model.parameters()).device
    if model.satellite_only:
        satellite = batch["satellite_past"].to(device).float()
        result = model.core(satellite, npm_times(batch.get("time_past_timestamp"), device))
        rain_loss = F.mse_loss(result["rain"].float(), batch["rain_future"].to(device).float())
        satellite_loss = F.mse_loss(result["satellite"].float(), batch["satellite_future"].to(device).float())
        loss = cfg.train.loss_weights.rain * rain_loss + cfg.train.loss_weights.satellite * satellite_loss
        yield (
            loss,
            {
                "loss/npm_rain": rain_loss.detach().item(),
                "loss/npm_satellite": satellite_loss.detach().item(),
                "loss": loss.detach().item(),
            },
        )
        return

    sequence = (
        torch.cat(
            [
                torch.cat([batch[f"{name}_past"], batch[f"{name}_future"]], dim=2)
                for name in ("radar", "satellite", "rain")
            ],
            dim=1,
        )
        .to(device)
        .float()
    )
    if not torch.isfinite(sequence).all():
        raise ValueError("Training observations and targets must be finite.")
    masks = training_masks(sequence.shape[0], 7, cfg, device)
    anchor = sequence[:, 11:12, :1]
    weights = cfg.train.next_pred.sequence_loss
    if not weights.enabled:
        raise ValueError("Deep comparison requires sequence loss enabled.")
    denominator = weights.context_weight + weights.future_weight
    for index in range(7):
        context = apply_mask(sequence[:, :, : index + 1], masks[:, :, : index + 1], model.mask_token)
        raw, auxiliary = model.next_frame(context)
        current = sequence[:, 11:12, index : index + 1]
        anchor = torch.where(masks[:, 2:3, index : index + 1, None, None], current, anchor)
        rain = raw[:, 11:12] + anchor if cfg.train.loss.rain_residual.enabled else raw[:, 11:12]
        if cfg.train.loss.rain_residual.enabled and cfg.train.loss.rain_residual.clamp_output:
            rain = rain.clamp_min(cfg.train.loss.rain_residual.min_value)
        predicted = torch.cat([raw[:, :11], rain], dim=1)
        target = sequence[:, :, index + 1 : index + 2]
        coefficient = (
            weights.context_weight / (3 * denominator) if index < 3 else weights.future_weight / (4 * denominator)
        )
        losses = {
            name: F.mse_loss(predicted[:, section].float(), target[:, section])
            for name, section in (("radar", slice(0, 1)), ("satellite", slice(1, 11)), ("rain", slice(11, 12)))
        }
        shared = sum(cfg.train.loss_weights[name] * value for name, value in losses.items()) * coefficient
        regularizers = {
            name: value * cfg.deep.auxiliary_weights.get(name, 1.0)
            for name, value in auxiliary.items()
            if value.ndim == 0
        }
        if hasattr(model.core, "supervised_losses"):
            native_target = (
                torch.cat([target[:, :11], target[:, 11:12] - anchor], dim=1)
                if cfg.train.loss.rain_residual.enabled
                else target
            )
            regularizers.update(model.core.supervised_losses(auxiliary, native_target, step))
        regularization = sum(regularizers.values()) * coefficient
        loss = shared + regularization
        logs = {f"loss/{name}": value.detach().item() * coefficient for name, value in losses.items()}
        logs.update({f"aux/{name}": value.detach().item() * coefficient for name, value in regularizers.items()})
        logs.update({"loss/sequence": shared.detach().item(), "loss": loss.detach().item()})
        yield loss, logs

    future_aux = sequence[:, :11, 4:7] if cfg.train.next_pred.rollout_branch.use_gt_future_modalities else None
    forecast = model.rollout(sequence[:, :, :4], masks[:, :, :4], future_aux, training=True)
    raw_rollout = F.mse_loss(forecast.float(), sequence[:, 11:12, 4:])
    rollout = cfg.train.next_pred.rollout_branch.weight * raw_rollout
    yield (
        rollout,
        {
            "loss/rollout_raw": raw_rollout.detach().item(),
            "loss/rollout": rollout.detach().item(),
            "loss": rollout.detach().item(),
        },
    )
