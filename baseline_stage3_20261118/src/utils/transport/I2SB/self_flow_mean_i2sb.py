import inspect

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .mean_i2sb import MeanI2SB, adaptive_l2_loss, unsqueeze_xdim


class SelfFlowMeanI2SB(MeanI2SB):
    def __init__(
        self,
        n_timestep: int = 1000,
        linear_start: float = 1e-4,
        beta_max: float = 0.3,
        linear_end: float | None = None,
        time_eps: float = 1e-3,
        flow_ratio: float = 0.5,
        aux_x0_weight: float = 0.1,
        interval_eps: float = 1e-6,
        self_flow_weight: float = 1.0,
        distill_gamma: float = 0.8,
        mask_ratio: float = 0.25,
    ) -> None:
        super().__init__(
            n_timestep=n_timestep,
            linear_start=linear_start,
            beta_max=beta_max,
            linear_end=linear_end,
            time_eps=time_eps,
            flow_ratio=flow_ratio,
            aux_x0_weight=aux_x0_weight,
            interval_eps=interval_eps,
        )
        self.self_flow_weight = self_flow_weight
        self.distill_gamma = distill_gamma
        self.mask_ratio = mask_ratio

    @staticmethod
    def _to_token_view(x: Tensor) -> tuple[Tensor, tuple[int, ...]]:
        if x.ndim == 3:
            return x, tuple(x.shape)
        if x.ndim == 4:
            b, c, h, w = x.shape
            return x.permute(0, 2, 3, 1).reshape(b, h * w, c), tuple(x.shape)
        raise ValueError(f"Expected x with shape (B,N,C) or (B,C,H,W), got {tuple(x.shape)}")

    @staticmethod
    def _from_token_view(tokens: Tensor, ref_shape: tuple[int, ...]) -> Tensor:
        if len(ref_shape) == 3:
            return tokens
        if len(ref_shape) == 4:
            b, c, h, w = ref_shape
            return tokens.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        raise ValueError(f"Unsupported reference shape: {ref_shape}")

    @staticmethod
    def _call_model_with_timestep(
        model: nn.Module,
        x: Tensor,
        timestep: Tensor,
        **model_kwargs: object,
    ) -> object:
        target_fn = model.forward if hasattr(model, "forward") else model
        params = inspect.signature(target_fn).parameters
        if "timesteps" in params:
            return model(x, timesteps=timestep, **model_kwargs)
        return model(x, timestep, **model_kwargs)

    @staticmethod
    def _normalize_feature_output(feature: object) -> Tensor | list[Tensor]:
        if torch.is_tensor(feature):
            return feature
        if isinstance(feature, (tuple, list)):
            if not feature:
                raise ValueError("Feature list must not be empty.")
            tensors: list[Tensor] = []
            for item in feature:
                if not torch.is_tensor(item):
                    raise TypeError("Feature list must contain only tensors.")
                tensors.append(item)
            return tensors
        raise TypeError(f"Unsupported feature type: {type(feature)}")

    @staticmethod
    def _split_output_and_feature(model_output: object) -> tuple[Tensor, Tensor | list[Tensor]]:
        if torch.is_tensor(model_output):
            return model_output, model_output
        if isinstance(model_output, (tuple, list)):
            if not model_output:
                raise ValueError("Model output tuple/list must not be empty.")
            pred = model_output[0]
            if not torch.is_tensor(pred):
                raise TypeError("Model output at index 0 must be a tensor.")
            if len(model_output) == 1:
                return pred, pred
            if len(model_output) == 2:
                return pred, SelfFlowMeanI2SB._normalize_feature_output(model_output[1])
            return pred, SelfFlowMeanI2SB._normalize_feature_output(model_output[1:])
        raise TypeError(f"Unsupported model output type: {type(model_output)}")

    @staticmethod
    def _cosine_loss_tensor(student_feat: Tensor, teacher_feat: Tensor, eps: float) -> Tensor:
        if student_feat.shape != teacher_feat.shape:
            raise ValueError(
                f"Student and teacher feature shapes must match, got {student_feat.shape} and {teacher_feat.shape}"
            )
        student_flat = student_feat.reshape(student_feat.shape[0], -1)
        teacher_flat = teacher_feat.reshape(teacher_feat.shape[0], -1)
        cosine = F.cosine_similarity(student_flat, teacher_flat, dim=-1, eps=eps)
        return (1.0 - cosine).mean()

    def _compute_rep_loss(
        self,
        student_feature: Tensor | list[Tensor],
        teacher_feature: Tensor | list[Tensor],
        *,
        eps: float,
    ) -> Tensor:
        if torch.is_tensor(student_feature) and torch.is_tensor(teacher_feature):
            return self._cosine_loss_tensor(student_feature, teacher_feature, eps=eps)

        if isinstance(student_feature, list) and isinstance(teacher_feature, list):
            if len(student_feature) != len(teacher_feature):
                raise ValueError(
                    f"Student and teacher feature counts must match, got {len(student_feature)} and {len(teacher_feature)}"
                )
            losses = [
                self._cosine_loss_tensor(student_feat, teacher_feat, eps=eps)
                for student_feat, teacher_feat in zip(student_feature, teacher_feature)
            ]
            return torch.stack(losses).mean()

        raise TypeError("Student and teacher features must be both tensors or both list[tensor].")

    @staticmethod
    def _expand_time_to_tokens(t: Tensor, *, num_tokens: int, dtype: torch.dtype) -> Tensor:
        if t.ndim != 1:
            raise ValueError(f"Expected scalar batch timesteps with shape (B,), got {tuple(t.shape)}")
        return t.unsqueeze(1).expand(-1, num_tokens).to(dtype=dtype)

    def _sample_time(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        forced: Tensor | None = None,
    ) -> Tensor:
        if forced is not None:
            if forced.ndim != 1:
                raise ValueError(f"Forced timesteps must be 1D, got shape {tuple(forced.shape)}")
            if forced.shape[0] != batch_size:
                raise ValueError(f"Forced timesteps batch mismatch: expected {batch_size}, got {forced.shape[0]}")
            return forced.to(device=device, dtype=dtype)

        t = torch.rand(batch_size, device=device, dtype=dtype)
        return t * (1.0 - 2.0 * self.time_eps) + self.time_eps

    def _predict_x0_tokens(
        self,
        model: nn.Module,
        x_t: Tensor,
        timestep: Tensor,
        *,
        model_kwargs: dict[str, object],
        clip_denoise: bool,
    ) -> tuple[Tensor, object, Tensor | list[Tensor]]:
        raw_output = self._call_model_with_timestep(model, x_t, timestep, **model_kwargs)
        x0_hat, feature = self._split_output_and_feature(raw_output)
        if clip_denoise:
            x0_hat = x0_hat.clamp(-1.0, 1.0)
        return x0_hat, raw_output, feature

    def _sample_bridge_tokenwise(
        self,
        x0_tokens: Tensor,
        x1_tokens: Tensor,
        tau: Tensor,
        *,
        eps: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if x0_tokens.shape != x1_tokens.shape:
            raise ValueError(f"x0_tokens shape {x0_tokens.shape} must match x1_tokens shape {x1_tokens.shape}.")
        if tau.shape != x0_tokens.shape[:2]:
            raise ValueError(f"Token-wise tau shape {tau.shape} must match token grid {x0_tokens.shape[:2]}.")

        mu0, mu1, sigma, dmu0, dmu1, dsigma = self.bridge_stats(tau, xdim=(x0_tokens.shape[-1],))
        eps = torch.randn_like(x0_tokens) if eps is None else eps
        x_tau = mu0 * x0_tokens + mu1 * x1_tokens + sigma * eps
        v_tau = dmu0 * x0_tokens + dmu1 * x1_tokens + dsigma * eps
        return x_tau, v_tau, eps

    def compute_eps_from_x0_tokenwise(
        self,
        x_t: Tensor,
        x0_hat: Tensor,
        x1: Tensor,
        tau: Tensor,
    ) -> Tensor:
        mu0, mu1, sigma, _, _, _ = self.bridge_stats(tau, xdim=(x_t.shape[-1],))
        sigma_safe = sigma.clamp_min(self.interval_eps)
        return (x_t - mu0 * x0_hat - mu1 * x1) / sigma_safe

    def compute_velocity_from_x0_tokenwise(
        self,
        x_t: Tensor,
        x0_hat: Tensor,
        x1: Tensor,
        tau: Tensor,
    ) -> Tensor:
        _, _, _, dmu0, dmu1, dsigma = self.bridge_stats(tau, xdim=(x_t.shape[-1],))
        eps_hat = self.compute_eps_from_x0_tokenwise(x_t, x0_hat, x1, tau)
        return dmu0 * x0_hat + dmu1 * x1 + dsigma * eps_hat

    def _build_dual_timestep(
        self,
        t_primary: Tensor,
        t_secondary: Tensor,
        *,
        num_tokens: int,
        mask_ratio: float,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        if not (0.0 <= mask_ratio <= 1.0):
            raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}")

        if mask_ratio == 0.0:
            mask = torch.zeros((t_primary.shape[0], num_tokens), dtype=torch.bool, device=t_primary.device)
        elif mask_ratio == 1.0:
            mask = torch.ones((t_primary.shape[0], num_tokens), dtype=torch.bool, device=t_primary.device)
        else:
            mask = torch.rand((t_primary.shape[0], num_tokens), device=t_primary.device) < mask_ratio

        tau = torch.where(mask, t_secondary.unsqueeze(1), t_primary.unsqueeze(1)).to(dtype=dtype)
        tau_min = torch.minimum(t_primary, t_secondary).unsqueeze(1).expand(-1, num_tokens).to(dtype=dtype)
        return tau, tau_min

    def _build_meanflow_u_fn(
        self,
        model: nn.Module,
        x1_tokens: Tensor,
        *,
        model_kwargs: dict[str, object],
        clip_denoise: bool,
    ) -> Callable[[Tensor, Tensor, Tensor], Tensor]:
        num_tokens = x1_tokens.shape[1]

        def u_fn(x_t: Tensor, t: Tensor, r: Tensor) -> Tensor:
            tau_t = self._expand_time_to_tokens(t, num_tokens=num_tokens, dtype=x_t.dtype)
            x0_hat, _, _ = self._predict_x0_tokens(
                model,
                x_t,
                tau_t,
                model_kwargs=model_kwargs,
                clip_denoise=clip_denoise,
            )
            u_hat, _, _ = self.compute_mean_velocity_from_x0(x_t, x0_hat, x1_tokens, t, r)
            return u_hat

        return u_fn

    def _compute_main_terms(
        self,
        model: nn.Module,
        x0: Tensor,
        x1: Tensor,
        *,
        model_kwargs: dict[str, object],
    ) -> dict[str, Tensor]:
        x0_tokens, ref_shape = self._to_token_view(x0)
        x1_tokens, _ = self._to_token_view(x1)
        batch_size, num_tokens = x0_tokens.shape[:2]

        t, r = self.sample_t_r(batch_size, device=x0.device, dtype=x0.dtype)
        x_t_tokens, v_t_tokens, _ = self.sample_bridge(x0_tokens, x1_tokens, t)
        tau_t = self._expand_time_to_tokens(t, num_tokens=num_tokens, dtype=x0_tokens.dtype)

        x0_hat_tokens, _, _ = self._predict_x0_tokens(
            model,
            x_t_tokens,
            tau_t,
            model_kwargs=model_kwargs,
            clip_denoise=False,
        )
        u_hat_tokens, _, _ = self.compute_mean_velocity_from_x0(x_t_tokens, x0_hat_tokens, x1_tokens, t, r)

        r_jvp = torch.where((t - r).abs() <= self.interval_eps, (t - self.time_eps).clamp_min(0.0), r)
        u_fn = self._build_meanflow_u_fn(model, x1_tokens, model_kwargs=model_kwargs, clip_denoise=False)
        _, dudt = torch.autograd.functional.jvp(
            u_fn,
            (x_t_tokens, t, r_jvp),
            (v_t_tokens, torch.ones_like(t), torch.zeros_like(r_jvp)),
        )

        delta = unsqueeze_xdim(t - r, tuple(x_t_tokens.shape[1:]))
        u_tgt = v_t_tokens - delta * dudt

        loss_mf = adaptive_l2_loss(u_hat_tokens - u_tgt.detach())
        loss_x0 = F.mse_loss(x0_hat_tokens, x0_tokens)
        loss = loss_mf + self.aux_x0_weight * loss_x0
        return {
            "loss": loss,
            "loss_mf": loss_mf,
            "loss_x0": loss_x0,
            "pred_x0": self._from_token_view(x0_hat_tokens, ref_shape),
            "pred_u": self._from_token_view(u_hat_tokens, ref_shape),
        }

    def training_loss_self_flow(
        self,
        model: nn.Module,
        teacher_model: nn.Module,
        x0: Tensor,
        x1: Tensor,
        model_kwargs: dict[str, object] | None = None,
        teacher_model_kwargs: dict[str, object] | None = None,
        *,
        feature_fn: Callable[[object, object], tuple[object, object]] | None = None,
        gamma: float | None = None,
        mask_ratio: float | None = None,
        t_forced: Tensor | None = None,
        s_forced: Tensor | None = None,
        eps: float = 1e-8,
    ) -> dict[str, Tensor]:
        model_kwargs = {} if model_kwargs is None else model_kwargs
        teacher_model_kwargs = dict(model_kwargs) if teacher_model_kwargs is None else teacher_model_kwargs

        x0_tokens, ref_shape = self._to_token_view(x0)
        x1_tokens, _ = self._to_token_view(x1)
        batch_size, num_tokens = x0_tokens.shape[:2]

        t_primary = self._sample_time(batch_size, device=x0.device, dtype=x0.dtype, forced=t_forced)
        t_secondary = self._sample_time(batch_size, device=x0.device, dtype=x0.dtype, forced=s_forced)
        tau, tau_min = self._build_dual_timestep(
            t_primary,
            t_secondary,
            num_tokens=num_tokens,
            mask_ratio=self.mask_ratio if mask_ratio is None else mask_ratio,
            dtype=x0_tokens.dtype,
        )

        x_tau_tokens, v_tau_tokens, shared_eps = self._sample_bridge_tokenwise(x0_tokens, x1_tokens, tau)
        x_tau_min_tokens, _, _ = self._sample_bridge_tokenwise(x0_tokens, x1_tokens, tau_min, eps=shared_eps)

        student_x0_hat, student_raw, student_feature = self._predict_x0_tokens(
            model,
            x_tau_tokens,
            tau,
            model_kwargs=model_kwargs,
            clip_denoise=False,
        )
        student_v_hat = self.compute_velocity_from_x0_tokenwise(x_tau_tokens, student_x0_hat, x1_tokens, tau)

        with torch.no_grad():
            _, teacher_raw, teacher_feature = self._predict_x0_tokens(
                teacher_model,
                x_tau_min_tokens,
                tau_min,
                model_kwargs=teacher_model_kwargs,
                clip_denoise=False,
            )

        if feature_fn is not None:
            feature_pair = feature_fn(student_raw, teacher_raw)
            if not isinstance(feature_pair, (tuple, list)) or len(feature_pair) != 2:
                raise ValueError("feature_fn must return a tuple/list of (student_feature, teacher_feature).")
            student_feature = self._normalize_feature_output(feature_pair[0])
            teacher_feature = self._normalize_feature_output(feature_pair[1])

        gen_loss = F.mse_loss(student_v_hat, v_tau_tokens.detach())
        rep_loss = self._compute_rep_loss(student_feature, teacher_feature, eps=eps)
        self_flow_loss = gen_loss + (self.distill_gamma if gamma is None else gamma) * rep_loss

        return {
            "self_flow_loss": self_flow_loss,
            "self_flow_gen_loss": gen_loss,
            "self_flow_rep_loss": rep_loss,
            "pred_x0_self_flow": self._from_token_view(student_x0_hat, ref_shape),
            "pred_v_self_flow": self._from_token_view(student_v_hat, ref_shape),
            "t_primary": t_primary,
            "t_secondary": t_secondary,
            "tau": tau,
            "tau_min": tau_min,
        }

    def training_loss(
        self,
        model: nn.Module,
        x0: Tensor,
        x1: Tensor,
        model_kwargs: dict[str, object] | None = None,
        *,
        teacher_model: nn.Module | None = None,
        teacher_model_kwargs: dict[str, object] | None = None,
        feature_fn: Callable[[object, object], tuple[object, object]] | None = None,
        gamma: float | None = None,
        mask_ratio: float | None = None,
    ) -> dict[str, Tensor]:
        model_kwargs = {} if model_kwargs is None else model_kwargs
        main_terms = self._compute_main_terms(model, x0, x1, model_kwargs=model_kwargs)

        if teacher_model is None or self.self_flow_weight <= 0.0:
            return main_terms

        self_flow_terms = self.training_loss_self_flow(
            model,
            teacher_model,
            x0,
            x1,
            model_kwargs=model_kwargs,
            teacher_model_kwargs=teacher_model_kwargs,
            feature_fn=feature_fn,
            gamma=gamma,
            mask_ratio=mask_ratio,
        )

        combined_loss = main_terms["loss"] + self.self_flow_weight * self_flow_terms["self_flow_loss"]
        return {
            **main_terms,
            **self_flow_terms,
            "loss_main": main_terms["loss"],
            "loss": combined_loss,
        }

    def sample(
        self,
        model: nn.Module,
        x1: Tensor,
        *,
        clip_denoise: bool = False,
        nfe: int | None = None,
        model_kwargs: dict[str, object] | None = None,
        log_steps: list[int] | None = None,
        sampling_type: str = "ode",
        log_count: int = 0,
        verbose: bool = False,
    ) -> dict[str, Tensor]:
        del verbose
        model_kwargs = {} if model_kwargs is None else model_kwargs

        if sampling_type not in {"ode", "one_step"}:
            raise ValueError(f"Unknown sampling_type: {sampling_type}")

        if sampling_type == "one_step":
            nfe = 1
        elif nfe is None:
            nfe = min(self.n_timestep - 1, 8)

        if nfe <= 0:
            raise ValueError("nfe must be a positive integer.")

        x1_tokens, ref_shape = self._to_token_view(x1)
        x_curr = x1_tokens
        num_tokens = x1_tokens.shape[1]
        time_grid = torch.linspace(1.0 - self.time_eps, 0.0, nfe + 1, device=x1.device, dtype=x1.dtype)

        states: list[Tensor] = []
        x0_preds: list[Tensor] = []

        for idx in range(nfe):
            t_val = time_grid[idx]
            r_val = time_grid[idx + 1]
            t = torch.full((x1.shape[0],), float(t_val.item()), device=x1.device, dtype=x1.dtype)
            r = torch.full((x1.shape[0],), float(r_val.item()), device=x1.device, dtype=x1.dtype)
            tau_t = self._expand_time_to_tokens(t, num_tokens=num_tokens, dtype=x1_tokens.dtype)

            x0_hat_tokens, _, _ = self._predict_x0_tokens(
                model,
                x_curr,
                tau_t,
                model_kwargs=model_kwargs,
                clip_denoise=clip_denoise,
            )
            _, x_next_tokens, _ = self.compute_mean_velocity_from_x0(x_curr, x0_hat_tokens, x1_tokens, t, r)
            x_curr = x_next_tokens

            states.append(self._from_token_view(x_curr.detach(), ref_shape))
            x0_preds.append(self._from_token_view(x_curr.detach(), ref_shape))

        if not states:
            raise RuntimeError("Sampling produced no states.")

        total_states = len(states)
        if log_steps is not None:
            selected_idx = [idx for idx in log_steps if 0 <= idx < total_states]
            if not selected_idx:
                selected_idx = [total_states - 1]
        elif log_count > 1:
            selected_idx = torch.linspace(0, total_states - 1, log_count).round().to(dtype=torch.long).tolist()
        else:
            selected_idx = [total_states - 1]

        sampled = torch.stack([x0_preds[idx] for idx in selected_idx], dim=1)
        traj = torch.stack([states[idx] for idx in selected_idx], dim=1)
        return {"sampled": sampled, "traj": traj}
