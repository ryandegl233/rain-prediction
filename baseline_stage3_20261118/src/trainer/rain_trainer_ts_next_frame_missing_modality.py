import sys
from contextlib import nullcontext
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import hydra
import torch
import torch.nn.functional as F
from accelerate.state import PartialState
from einops import rearrange
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from src.trainer.rain_trainer_ts_next_frame_transfer import (
    RainCausalPatchTransformerRainDecode,
    RainTSNextFrameTransferTrainer,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

NONEMPTY_MODALITY_COMBINATIONS: tuple[tuple[bool, bool, bool], ...] = (
    (True, True, True),    # rad_sat_rain (full)
    (True, True, False),   # rad_sat
    (True, False, True),   # rad_rain
    (False, True, True),   # sat_rain
    (True, False, False),  # rad
    (False, True, False),  # sat
    (False, False, True),  # rain
)

_MASK_SAMPLER_CACHE: dict[tuple[str, int | None, float], tuple[torch.Tensor, torch.Tensor]] = {}


def _get_cached_Combinations_and_probs(
    device: torch.device,
    full_modality_prob: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (device.type, device.index, float(full_modality_prob))
    cached = _MASK_SAMPLER_CACHE.get(key)
    if cached is not None:
        return cached
    combos = torch.tensor(NONEMPTY_MODALITY_COMBINATIONS, dtype=torch.bool, device=device)
    incomplete_prob = (1.0 - full_modality_prob) / 6.0
    probs = torch.tensor(
        [full_modality_prob] + [incomplete_prob] * 6,
        dtype=torch.float32,
        device=device,
    )
    _MASK_SAMPLER_CACHE[key] = (combos, probs)
    return combos, probs


def sample_per_frame_modality_mask(
    batch_size: int,
    num_frames: int,
    *,
    device: torch.device,
    full_modality_prob: float = 0.6,
    anchor_first_frame: bool = True,
) -> torch.Tensor:
    """Sample per-frame modality availability mask of shape [B, 3, T] on device without host syncs."""
    if batch_size <= 0 or num_frames <= 0:
        raise ValueError(f"batch_size and num_frames must be > 0, got B={batch_size}, T={num_frames}")
    if not (0.0 <= full_modality_prob <= 1.0):
        raise ValueError(f"full_modality_prob must be in [0, 1], got {full_modality_prob}")

    combos, probs = _get_cached_Combinations_and_probs(device, full_modality_prob)
    sampled_indices = torch.multinomial(
        probs,
        num_samples=batch_size * num_frames,
        replacement=True,
    ).view(batch_size, num_frames)
    mask_bt3 = combos[sampled_indices]  # [B, T, 3]
    if anchor_first_frame:
        mask_bt3[:, 0, :] = True
    return mask_bt3.permute(0, 2, 1).contiguous()  # [B, 3, T]


def compute_carry_forward_rain_ref(
    unmasked_rain: torch.Tensor,
    rain_available: torch.Tensor,
) -> torch.Tensor:
    """Compute per-frame rain residual reference [B, C_rain, T, H, W] by carrying forward
    the most recent available (unmasked) rain frame along time.
    """
    if unmasked_rain.ndim != 5:
        raise ValueError(f"unmasked_rain must be [B,C,T,H,W], got {tuple(unmasked_rain.shape)}")
    b, _, t, _, _ = unmasked_rain.shape
    if rain_available.shape != (b, t):
        raise ValueError(f"rain_available must have shape {(b, t)}, got {tuple(rain_available.shape)}")

    refs: list[torch.Tensor] = []
    last_valid = unmasked_rain[:, :, 0:1]
    for frame_idx in range(t):
        frame_rain = unmasked_rain[:, :, frame_idx : frame_idx + 1]
        if frame_idx == 0:
            last_valid = frame_rain
        else:
            is_avail = rain_available[:, frame_idx].view(b, 1, 1, 1, 1)
            last_valid = torch.where(is_avail, frame_rain, last_valid)
        refs.append(last_valid)
    return torch.cat(refs, dim=2)


class RainCausalPatchTransformerMissingModality(RainCausalPatchTransformerRainDecode):
    """Extends RainCausalPatchTransformerRainDecode so that during rollout_branch with
    `loss_on='rain'` and `detach_history=True`, radar and satellite decoders can run under
    `torch.no_grad()` to avoid storing unused 3D decoder activation graphs.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rollout_aux_no_grad: bool = False

    def _decode_target_tokens(
        self,
        target_tokens: torch.Tensor,
        hp: int,
        wp: int,
        h: int,
        w: int,
        batch_size: int,
        predict_frames: int,
        context_anchor: dict[str, torch.Tensor] | None = None,
        film_condition: dict[str, torch.Tensor] | None = None,
        return_modality_dict: bool = True,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        if not (return_modality_dict and self.rollout_aux_no_grad and self.training):
            return super()._decode_target_tokens(
                target_tokens,
                hp,
                wp,
                h,
                w,
                batch_size,
                predict_frames,
                context_anchor=context_anchor,
                film_condition=film_condition,
                return_modality_dict=return_modality_dict,
            )

        use_film = self.decoder_condition_mode == "film" and film_condition is not None
        if self.decoder_type == "2d":
            feat = rearrange(target_tokens, "b p (hp wp) d -> (b p) d hp wp", hp=hp, wp=wp)
            radar_cond = rearrange(film_condition["radar"], "b c p h w -> (b p) c h w") if use_film else None
            satellite_cond = rearrange(film_condition["satellite"], "b c p h w -> (b p) c h w") if use_film else None
            rain_cond = rearrange(film_condition["rain"], "b c p h w -> (b p) c h w") if use_film else None
            with torch.no_grad():
                feat_detached = feat.detach()
                radar = self.radar_decoder(feat_detached, target_hw=(h, w), cond=radar_cond)
                satellite = self.satellite_decoder(feat_detached, target_hw=(h, w), cond=satellite_cond)
                radar = rearrange(radar, "(b p) c h w -> b c p h w", b=batch_size, p=predict_frames)
                satellite = rearrange(satellite, "(b p) c h w -> b c p h w", b=batch_size, p=predict_frames)
            rain = self.rain_decoder(feat, target_hw=(h, w), cond=rain_cond)
            rain = rearrange(rain, "(b p) c h w -> b c p h w", b=batch_size, p=predict_frames)
        else:
            feat = rearrange(target_tokens, "b p (hp wp) d -> b d p hp wp", hp=hp, wp=wp)
            out_frames = predict_frames * self.frame_patch_size
            radar_cond = film_condition["radar"] if use_film else None
            satellite_cond = film_condition["satellite"] if use_film else None
            rain_cond = film_condition["rain"] if use_film else None
            with torch.no_grad():
                feat_detached = feat.detach()
                radar = self.radar_decoder(feat_detached, target_thw=(out_frames, h, w), cond=radar_cond)
                satellite = self.satellite_decoder(feat_detached, target_thw=(out_frames, h, w), cond=satellite_cond)
            rain = self.rain_decoder(feat, target_thw=(out_frames, h, w), cond=rain_cond)

        if self.decoder_output_mode == "residual" and context_anchor is not None:
            out_frames = rain.shape[2]
            radar = radar + context_anchor["radar"].expand(-1, -1, out_frames, -1, -1)
            satellite = satellite + context_anchor["satellite"].expand(-1, -1, out_frames, -1, -1)
            rain = rain + context_anchor["rain"].expand(-1, -1, out_frames, -1, -1)

        return {"radar": radar, "satellite": satellite, "rain": rain}


def _resolve_repo_relative_paths(cfg: DictConfig) -> DictConfig:
    for split_key in ("train_inp_dirs", "val_inp_dirs"):
        if split_key in cfg.dataset and cfg.dataset[split_key] is not None:
            resolved = [
                str(REPO_ROOT / str(p)) if not Path(str(p)).is_absolute() else str(p)
                for p in cfg.dataset[split_key]
            ]
            cfg.dataset[split_key] = resolved
    for sub_key in ("train", "val"):
        if sub_key in cfg.dataset and "inp_dirs" in cfg.dataset[sub_key] and cfg.dataset[sub_key]["inp_dirs"] is not None:
            resolved = [
                str(REPO_ROOT / str(p)) if not Path(str(p)).is_absolute() else str(p)
                for p in cfg.dataset[sub_key]["inp_dirs"]
            ]
            cfg.dataset[sub_key]["inp_dirs"] = resolved

    rain_filter_cfg = cfg.dataset.get("rain_ratio_filter", {})
    if bool(rain_filter_cfg.get("enabled", False)):
        filter_file = str(rain_filter_cfg.get("file_name", "metadata_rain_ratio.parquet"))
        all_dirs: list[str] = []
        for split_key in ("train_inp_dirs", "val_inp_dirs"):
            if split_key in cfg.dataset and cfg.dataset[split_key] is not None:
                all_dirs.extend(str(p) for p in cfg.dataset[split_key])
        missing_dirs = [d for d in all_dirs if not (Path(d) / filter_file).exists()]
        if missing_dirs:
            logger.warning(
                f"[RainRatioFilter] {filter_file} not found in {len(missing_dirs)}/{len(all_dirs)} dirs "
                f"(e.g. {missing_dirs[0]}); auto-disabling rain_ratio_filter."
            )
            cfg.dataset.rain_ratio_filter.enabled = False
            if "train" in cfg.dataset and "rain_ratio_filter_enabled" in cfg.dataset.train:
                cfg.dataset.train.rain_ratio_filter_enabled = False
            if "val" in cfg.dataset and "rain_ratio_filter_enabled" in cfg.dataset.val:
                cfg.dataset.val.rain_ratio_filter_enabled = False
    return cfg


class RainTSNextFrameMissingModalityTrainer(RainTSNextFrameTransferTrainer):
    """Standalone trainer implementing anchored per-frame missing-modality training
    and carry-forward rain residual references across both sequence_loss and rollout_branch.
    """

    def __init__(self, cfg: DictConfig) -> None:
        cfg = _resolve_repo_relative_paths(cfg)
        target_cls = str(cfg.rain_prediction_model.get("_target_", ""))
        if target_cls in {
            "src.networks.time_series.causal_patch_transformer_next_frame.RainCausalPatchTransformerNextFrame",
            "src.trainer.rain_trainer_ts_next_frame_transfer.RainCausalPatchTransformerRainDecode",
        }:
            cfg = OmegaConf.merge(
                cfg,
                {
                    "rain_prediction_model": {
                        "_target_": (
                            "src.trainer.rain_trainer_ts_next_frame_missing_modality."
                            "RainCausalPatchTransformerMissingModality"
                        )
                    }
                },
            )
        super().__init__(cfg)
        self._current_step_history_mask: torch.Tensor | None = None
        self._current_step_initial_rain_anchor: torch.Tensor | None = None

    def _resolve_missing_modality_sampling_cfg(self) -> tuple[bool, float, bool, bool]:
        mm_cfg = self.train_cfg.next_pred.get("missing_modality", {})
        enabled = bool(mm_cfg.get("enabled", False))
        full_prob = float(mm_cfg.get("full_modality_prob", 0.6))
        anchor_first = bool(mm_cfg.get("anchor_first_frame", True))
        mask_future_seed = bool(mm_cfg.get("mask_future_seed_in_sequence_loss", True))
        return enabled, full_prob, anchor_first, mask_future_seed

    def _apply_per_frame_mask_tokens(
        self,
        sequence_tensor: torch.Tensor,
        frame_modality_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Replace unavailable modalities (False in frame_modality_mask [B, 3, T])
        with the model's learned modality mask tokens.
        """
        if sequence_tensor.ndim != 5:
            raise ValueError(f"sequence_tensor must be [B,C,T,H,W], got {tuple(sequence_tensor.shape)}")
        b, _, t, h, w = sequence_tensor.shape
        if frame_modality_mask.shape != (b, 3, t):
            raise ValueError(
                f"frame_modality_mask shape mismatch: expected {(b, 3, t)}, got {tuple(frame_modality_mask.shape)}"
            )

        model = self.accelerator.unwrap_model(self.model)
        parts = self._split_modalities(sequence_tensor)
        radar = parts["radar"]
        satellite = parts["satellite"]
        rain = parts["rain"]

        use_mask_token = bool(getattr(model, "use_modality_mask_token", False))
        if use_mask_token:
            radar_fill = model.radar_mask_token.to(device=radar.device, dtype=radar.dtype).expand(b, -1, t, h, w)
            sat_fill = model.satellite_mask_token.to(device=satellite.device, dtype=satellite.dtype).expand(
                b, -1, t, h, w
            )
            rain_fill = model.rain_mask_token.to(device=rain.device, dtype=rain.dtype).expand(b, -1, t, h, w)
        else:
            radar_fill = torch.zeros_like(radar)
            sat_fill = torch.zeros_like(satellite)
            rain_fill = torch.zeros_like(rain)

        rad_keep = frame_modality_mask[:, 0].view(b, 1, t, 1, 1)
        sat_keep = frame_modality_mask[:, 1].view(b, 1, t, 1, 1)
        rain_keep = frame_modality_mask[:, 2].view(b, 1, t, 1, 1)

        masked_radar = torch.where(rad_keep, radar, radar_fill)
        masked_satellite = torch.where(sat_keep, satellite, sat_fill)
        masked_rain = torch.where(rain_keep, rain, rain_fill)
        return self._merge_modalities(masked_radar, masked_satellite, masked_rain)

    def _build_next_pred_batch(
        self,
        batch: dict[str, torch.Tensor],
        apply_missing_modality: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, float | int | str | torch.Tensor | dict[str, torch.Tensor]],
    ]:
        sequence_context, target_seed_tensor, target_gt, aux = super()._build_next_pred_batch(
            batch,
            apply_missing_modality=False,
        )
        self._current_step_history_mask = None
        self._current_step_initial_rain_anchor = None

        mm_enabled, full_prob, anchor_first, mask_future_seed = self._resolve_missing_modality_sampling_cfg()
        if not (apply_missing_modality and mm_enabled):
            return sequence_context, target_seed_tensor, target_gt, aux

        n_past = int(batch["rain_past"].shape[2] if batch["rain_past"].ndim == 5 else batch["rain_past"].shape[1])
        sequence_loss_enabled = bool(aux.get("sequence_loss_enabled", 0))

        if sequence_loss_enabled:
            total_seed_frames = int(target_seed_tensor.shape[2])
            unmasked_seed_parts = self._split_modalities(target_seed_tensor)

            seed_mask = sample_per_frame_modality_mask(
                batch_size=int(target_seed_tensor.shape[0]),
                num_frames=total_seed_frames,
                device=target_seed_tensor.device,
                full_modality_prob=full_prob,
                anchor_first_frame=anchor_first,
            )
            if not mask_future_seed and total_seed_frames > n_past:
                seed_mask[:, :, n_past:] = True

            self._current_step_history_mask = seed_mask[:, :, :n_past]
            masked_seed_tensor = self._apply_per_frame_mask_tokens(target_seed_tensor, seed_mask)
            frame_patch_size = self._resolve_frame_patch_size()
            masked_sequence_context = masked_seed_tensor[:, :, :frame_patch_size]

            rain_ref = compute_carry_forward_rain_ref(
                unmasked_seed_parts["rain"],
                seed_mask[:, 2, :],
            )
            self._current_step_initial_rain_anchor = rain_ref[:, :, n_past - 1 : n_past].detach()
            aux["rain_residual_ref"] = rain_ref
            aux["context_modality_available"] = None
            aux["per_frame_modality_mask"] = seed_mask
            return masked_sequence_context, masked_seed_tensor, target_gt, aux

        # Non-sequence-loss path: context has n_past frames
        total_context_frames = int(sequence_context.shape[2])
        unmasked_context_parts = self._split_modalities(sequence_context)
        hist_mask = sample_per_frame_modality_mask(
            batch_size=int(sequence_context.shape[0]),
            num_frames=total_context_frames,
            device=sequence_context.device,
            full_modality_prob=full_prob,
            anchor_first_frame=anchor_first,
        )
        self._current_step_history_mask = hist_mask
        masked_context = self._apply_per_frame_mask_tokens(sequence_context, hist_mask)
        hist_rain_ref = compute_carry_forward_rain_ref(
            unmasked_context_parts["rain"],
            hist_mask[:, 2, :],
        )
        last_valid_hist_rain = hist_rain_ref[:, :, -1:]
        self._current_step_initial_rain_anchor = last_valid_hist_rain.detach()
        if isinstance(aux.get("rain_residual_ref"), torch.Tensor):
            ref_tensor = aux["rain_residual_ref"].clone()
            ref_tensor[:, :, 0:1] = last_valid_hist_rain
            aux["rain_residual_ref"] = ref_tensor
        aux["context_modality_available"] = None
        aux["per_frame_modality_mask"] = hist_mask
        return masked_context, target_seed_tensor, target_gt, aux

    def _rollout_predict_from_masked_context(
        self,
        masked_context: torch.Tensor,
        initial_rain_anchor: torch.Tensor,
        total_future_frames: int,
        *,
        detach_history: bool,
        detach_rain_anchor: bool = True,
        history_window_frames: int | None = None,
        context_time: torch.Tensor | None = None,
        future_time: torch.Tensor | None = None,
        future_modalities: dict[str, torch.Tensor] | None = None,
        use_gt_future_modalities: bool = False,
        return_rain_delta: bool = False,
        rain_grad_only: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Autoregressively roll out `total_future_frames` starting from `masked_context`
        while using `initial_rain_anchor` (`last_valid_rain`) as the residual baseline for step 0.
        When `detach_rain_anchor=False` (Method 1), `rain_anchor` retains gradients across steps
        to suppress multi-step light-rain residual accumulation while `context_cur` still uses
        detached history to avoid extra Transformer encoder BPTT memory.
        """
        frame_patch_size = self._resolve_frame_patch_size()
        context_cur = masked_context
        context_time_cur = context_time
        if history_window_frames is not None and history_window_frames > 0:
            context_cur = context_cur[:, :, -history_window_frames:]
            if torch.is_tensor(context_time_cur):
                context_time_cur = context_time_cur[:, -history_window_frames:]
        rain_anchor = initial_rain_anchor
        residual_enabled = self._resolve_rain_residual_cfg()[0]
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        pred_radar: list[torch.Tensor] = []
        pred_satellite: list[torch.Tensor] = []
        pred_rain: list[torch.Tensor] = []
        pred_rain_delta: list[torch.Tensor] = []

        prev_flag = bool(getattr(unwrapped_model, "rollout_aux_no_grad", False))
        if hasattr(unwrapped_model, "rollout_aux_no_grad"):
            unwrapped_model.rollout_aux_no_grad = bool(rain_grad_only and detach_history)
        try:
            for start in range(0, total_future_frames, frame_patch_size):
                end = start + frame_patch_size
                seed_block = context_cur[:, :, -frame_patch_size:]
                forward_kwargs: dict[str, object] = {
                    "context_x": context_cur,
                    "target_x": seed_block,
                    "predict_frames": frame_patch_size,
                    "strict_target_isolation": bool(self.train_cfg.strict_target_isolation),
                    "return_modality_dict": True,
                }
                if torch.is_tensor(context_time_cur):
                    forward_kwargs["context_time"] = context_time_cur
                    forward_kwargs["target_time"] = context_time_cur[:, -frame_patch_size:]

                with self.accelerator.autocast():
                    pred_block = self.model.forward_ar(**forward_kwargs)

                raw_rain_delta = pred_block["rain"]
                step_rain = raw_rain_delta
                if residual_enabled:
                    step_rain = self._apply_rain_residual_output(raw_rain_delta, rain_anchor)

                pred_radar.append(pred_block["radar"])
                pred_satellite.append(pred_block["satellite"])
                pred_rain.append(step_rain)
                pred_rain_delta.append(raw_rain_delta)

                if end < total_future_frames:
                    if use_gt_future_modalities and future_modalities is not None:
                        next_radar = future_modalities["radar"][:, :, start:end]
                        next_satellite = future_modalities["satellite"][:, :, start:end]
                    else:
                        next_radar = pred_block["radar"].detach() if detach_history else pred_block["radar"]
                        next_satellite = pred_block["satellite"].detach() if detach_history else pred_block["satellite"]
                    next_rain = step_rain.detach() if detach_history else step_rain
                    anchor_rain = step_rain.detach() if detach_rain_anchor else step_rain
                    rain_anchor = anchor_rain[:, :, -frame_patch_size:]
                    next_hist = self._merge_modalities(next_radar, next_satellite, next_rain)
                    context_cur = torch.cat([context_cur, next_hist], dim=2)
                    if history_window_frames is not None and history_window_frames > 0:
                        context_cur = context_cur[:, :, -history_window_frames:]
                    if torch.is_tensor(context_time_cur) and future_time is not None:
                        context_time_cur = torch.cat([context_time_cur, future_time[:, start:end]], dim=1)
                        if history_window_frames is not None and history_window_frames > 0:
                            context_time_cur = context_time_cur[:, -history_window_frames:]
        finally:
            if hasattr(unwrapped_model, "rollout_aux_no_grad"):
                unwrapped_model.rollout_aux_no_grad = prev_flag

        output = {
            "radar": torch.cat(pred_radar, dim=2),
            "satellite": torch.cat(pred_satellite, dim=2),
            "rain": torch.cat(pred_rain, dim=2),
        }
        if return_rain_delta:
            output["rain_delta"] = torch.cat(pred_rain_delta, dim=2)
        return output

    def _train_rollout_branch_loss(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
        raw_context, target, context_time, target_time = self._prepare_val_inference_batch(batch)
        total_future_frames = int(target["rain"].shape[2])
        rollout_branch_cfg = self._resolve_train_rollout_branch(total_future_frames=total_future_frames)
        if rollout_branch_cfg is None:
            return None, {}

        rollout_raw_cfg = self.train_cfg.next_pred.get("rollout_branch", {})
        detach_rain_anchor = bool(rollout_raw_cfg.get("detach_rain_anchor", True))
        raw_hw = rollout_raw_cfg.get("history_window_frames", None)
        history_window_frames = None if raw_hw in (None, "") else int(raw_hw)

        rollout_frames = int(rollout_branch_cfg["rollout_frames"])
        target_slice = {
            "radar": target["radar"][:, :, :rollout_frames],
            "satellite": target["satellite"][:, :, :rollout_frames],
            "rain": target["rain"][:, :, :rollout_frames],
        }
        target_time_slice = None if target_time is None else target_time[:, :rollout_frames]

        mm_enabled, full_prob, anchor_first, _ = self._resolve_missing_modality_sampling_cfg()
        unmasked_context_parts = self._split_modalities(raw_context)
        if mm_enabled:
            hist_mask = self._current_step_history_mask
            if hist_mask is None or hist_mask.shape != (int(raw_context.shape[0]), 3, int(raw_context.shape[2])):
                hist_mask = sample_per_frame_modality_mask(
                    batch_size=int(raw_context.shape[0]),
                    num_frames=int(raw_context.shape[2]),
                    device=raw_context.device,
                    full_modality_prob=full_prob,
                    anchor_first_frame=anchor_first,
                )
            masked_context = self._apply_per_frame_mask_tokens(raw_context, hist_mask)
            if self._current_step_initial_rain_anchor is not None:
                initial_rain_anchor = self._current_step_initial_rain_anchor
            else:
                hist_rain_ref = compute_carry_forward_rain_ref(
                    unmasked_context_parts["rain"],
                    hist_mask[:, 2, :],
                )
                initial_rain_anchor = hist_rain_ref[:, :, -1:]
        else:
            masked_context = raw_context
            initial_rain_anchor = unmasked_context_parts["rain"][:, :, -1:]

        loss_on_rain_only = str(rollout_branch_cfg["loss_on"]) == "rain"
        pred_rollout = self._rollout_predict_from_masked_context(
            masked_context=masked_context,
            initial_rain_anchor=initial_rain_anchor,
            total_future_frames=rollout_frames,
            detach_history=bool(rollout_branch_cfg["detach_history"]),
            detach_rain_anchor=detach_rain_anchor,
            history_window_frames=history_window_frames,
            context_time=context_time,
            future_time=target_time_slice,
            future_modalities=target_slice,
            use_gt_future_modalities=bool(rollout_branch_cfg["use_gt_future_modalities"]),
            return_rain_delta=True,
            rain_grad_only=loss_on_rain_only,
        )

        loss_weight_override = None
        if loss_on_rain_only:
            loss_weight_override = {"radar": 0.0, "satellite": 0.0, "rain": 1.0}

        aux = {
            "target_mode": "rollout_branch",
            "target_frames": rollout_frames,
            "context_frames": int(raw_context.shape[2]),
            "sequence_loss_enabled": 0,
            "sequence_context_frames": 0,
            "sequence_future_frames": rollout_frames,
            "sequence_context_weight": 1.0,
            "sequence_future_weight": 1.0,
            "temporal_diff_anchor": self._split_modalities(raw_context[:, :, -1:]),
            "rain_residual_already_applied": 1,
        }
        rollout_raw_loss, rollout_inner_logs = self._next_prediction_loss(
            pred=pred_rollout,
            target_gt=target_slice,
            aux=aux,
            loss_weight_override=loss_weight_override,
        )

        delta_loss_enabled = bool(rollout_branch_cfg["delta_loss_enabled"])
        delta_loss_weight = float(rollout_branch_cfg["delta_loss_weight"])
        rollout_delta_loss = torch.zeros((), device=rollout_raw_loss.device, dtype=rollout_raw_loss.dtype)
        rollout_delta_loss_weighted = rollout_delta_loss
        delta_mean_ratio = torch.zeros((), device=rollout_raw_loss.device, dtype=rollout_raw_loss.dtype)
        pred_delta_abs_mean = torch.zeros((), device=rollout_raw_loss.device, dtype=rollout_raw_loss.dtype)
        gt_delta_abs_mean = torch.zeros((), device=rollout_raw_loss.device, dtype=rollout_raw_loss.dtype)
        if delta_loss_enabled and delta_loss_weight > 0:
            gt_delta_ref = torch.cat([initial_rain_anchor, target_slice["rain"][:, :, :-1]], dim=2)
            gt_delta = target_slice["rain"] - gt_delta_ref
            pred_delta = pred_rollout["rain_delta"].to(device=gt_delta.device, dtype=gt_delta.dtype)
            rollout_delta_loss = F.mse_loss(pred_delta, gt_delta)
            rollout_delta_loss_weighted = delta_loss_weight * rollout_delta_loss
            rollout_raw_loss = rollout_raw_loss + rollout_delta_loss_weighted
            pred_delta_abs_mean = pred_delta.abs().mean()
            gt_delta_abs_mean = gt_delta.abs().mean()
            delta_mean_ratio = pred_delta_abs_mean / gt_delta_abs_mean.clamp_min(1.0e-8)

        weight_start = float(rollout_branch_cfg["weight_start"])
        weight_end = float(rollout_branch_cfg["weight_end"])
        ramp_steps = int(rollout_branch_cfg["ramp_steps"])
        if ramp_steps > 0:
            progress = min(float(self.global_step) / float(ramp_steps), 1.0)
            rollout_weight = weight_start + (weight_end - weight_start) * progress
        else:
            rollout_weight = weight_end

        lead_weights = rollout_branch_cfg["lead_weights"]
        lead_loss_weight = float(rollout_branch_cfg["lead_loss_weight"])
        lead_weighted_loss = torch.zeros((), device=rollout_raw_loss.device, dtype=rollout_raw_loss.dtype)
        if lead_loss_weight > 0 and lead_weights:
            rain_pred = pred_rollout["rain"]
            rain_target = target_slice["rain"].to(device=rain_pred.device, dtype=rain_pred.dtype)
            weights = torch.ones(int(rain_pred.shape[2]), device=rain_pred.device, dtype=rain_pred.dtype)
            count = min(len(lead_weights), int(rain_pred.shape[2]))
            weights[:count] = torch.as_tensor(lead_weights[:count], device=rain_pred.device, dtype=rain_pred.dtype)
            weights = weights / weights.mean().clamp_min(1.0e-6)
            lead_weighted_loss = (
                F.mse_loss(rain_pred, rain_target, reduction="none") * weights.view(1, 1, -1, 1, 1)
            ).mean()
            rollout_raw_loss = rollout_raw_loss + lead_loss_weight * lead_weighted_loss

        pod_floor = float(rollout_branch_cfg["pod_floor"])
        pod_floor_weight = float(rollout_branch_cfg["pod_floor_weight"])
        pod_floor_loss = torch.zeros((), device=rollout_raw_loss.device, dtype=rollout_raw_loss.dtype)
        if pod_floor > 0 and pod_floor_weight > 0:
            threshold = float(rollout_branch_cfg["pod_threshold"])
            temperature = max(float(rollout_branch_cfg["pod_temperature"]), 1.0e-4)
            probability = torch.sigmoid((pred_rollout["rain"] - threshold) / temperature)
            labels = (target_slice["rain"] >= threshold).to(probability.dtype)
            reduce_dims = (0, 1, 3, 4)
            soft_pod_by_lead = (probability * labels).sum(dim=reduce_dims) / labels.sum(dim=reduce_dims).clamp_min(1.0)
            pod_floor_loss = F.relu(
                torch.as_tensor(pod_floor, device=probability.device) - soft_pod_by_lead
            ).pow(2).mean()
            rollout_raw_loss = rollout_raw_loss + pod_floor_weight * pod_floor_loss

        rollout_scaled_loss = rollout_raw_loss * rollout_weight
        logs = {
            "loss/rollout_branch_raw": rollout_raw_loss.detach(),
            "loss/rollout_branch": rollout_scaled_loss.detach(),
            "loss/rollout_branch_delta": rollout_delta_loss.detach(),
            "loss/rollout_branch_delta_weighted_inner": rollout_delta_loss_weighted.detach(),
            "meta/rollout_branch_weight": torch.tensor(
                rollout_weight, device=rollout_raw_loss.device, dtype=rollout_raw_loss.dtype
            ),
            "loss/rollout_branch_lead_weighted": lead_weighted_loss.detach(),
            "loss/rollout_branch_pod_floor": pod_floor_loss.detach(),
            "meta/rollout_branch_delta_loss_weight": torch.tensor(
                delta_loss_weight, device=rollout_raw_loss.device, dtype=rollout_raw_loss.dtype
            ),
            "meta/rollout_branch_pred_delta_abs_mean": pred_delta_abs_mean.detach(),
            "meta/rollout_branch_gt_delta_abs_mean": gt_delta_abs_mean.detach(),
            "meta/rollout_branch_delta_mean_ratio": delta_mean_ratio.detach(),
        }
        for key, value in rollout_inner_logs.items():
            if key == "loss":
                continue
            logs[f"rollout/{key}"] = value.detach()
        return rollout_scaled_loss, logs


@hydra.main(
    config_path="../config/ts_rain_train",
    config_name="rain_trainer_ts_next_frame_missing_modality",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    catcher = logger.catch if PartialState().is_main_process else nullcontext
    with catcher():
        trainer = RainTSNextFrameMissingModalityTrainer(cfg)
        trainer.run()


if __name__ == "__main__":
    main()
