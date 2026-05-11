import torch
import torch.nn as nn
import torch.nn.functional as F

from collections.abc import Callable
import numpy as np
from torch import Tensor

from .diffusion import compute_gaussian_product_coef, make_beta_schedule, space_indices


def unsqueeze_xdim(z: Tensor, xdim: tuple[int, ...]) -> Tensor:
    bc_dim = (...,) + (None,) * len(xdim)
    return z[bc_dim]


def adaptive_l2_loss(error: Tensor, gamma: float = 0.5, c: float = 1e-3) -> Tensor:
    reduce_dims = tuple(range(1, error.ndim))
    delta_sq = torch.mean(error**2, dim=reduce_dims, keepdim=False)
    p = 1.0 - gamma
    weights = 1.0 / (delta_sq + c).pow(p)
    return (weights.detach() * delta_sq).mean()


class MeanI2SB(nn.Module):
    mu0_nodes: Tensor
    mu1_nodes: Tensor
    sigma_nodes: Tensor
    dmu0_nodes: Tensor
    dmu1_nodes: Tensor
    dsigma_nodes: Tensor

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
    ) -> None:
        super().__init__()
        self.n_timestep = n_timestep
        self.time_eps = time_eps
        self.flow_ratio = flow_ratio
        self.aux_x0_weight = aux_x0_weight
        self.interval_eps = interval_eps
        self.num_intervals = n_timestep + 1

        linear_end = linear_end if linear_end is not None else beta_max / n_timestep
        betas = make_beta_schedule(n_timestep=n_timestep, linear_start=linear_start, linear_end=linear_end)
        betas = np.concatenate([betas[: n_timestep // 2], np.flip(betas[: n_timestep // 2])])

        std_fwd = np.sqrt(np.cumsum(betas))
        std_bwd = np.sqrt(np.flip(np.cumsum(np.flip(betas))))
        mu0_disc, mu1_disc, var_disc = compute_gaussian_product_coef(std_fwd, std_bwd)
        sigma_disc = np.sqrt(var_disc)

        to_torch = lambda array: torch.tensor(array, dtype=torch.float32)
        mu0_nodes = torch.cat([torch.ones(1), to_torch(mu0_disc), torch.zeros(1)], dim=0)
        mu1_nodes = torch.cat([torch.zeros(1), to_torch(mu1_disc), torch.ones(1)], dim=0)
        sigma_nodes = torch.cat([torch.zeros(1), to_torch(sigma_disc), torch.zeros(1)], dim=0)

        scale = float(self.num_intervals)
        dmu0_nodes = (mu0_nodes[1:] - mu0_nodes[:-1]) * scale
        dmu1_nodes = (mu1_nodes[1:] - mu1_nodes[:-1]) * scale
        dsigma_nodes = (sigma_nodes[1:] - sigma_nodes[:-1]) * scale

        self.register_buffer("mu0_nodes", mu0_nodes, persistent=False)
        self.register_buffer("mu1_nodes", mu1_nodes, persistent=False)
        self.register_buffer("sigma_nodes", sigma_nodes, persistent=False)
        self.register_buffer("dmu0_nodes", dmu0_nodes, persistent=False)
        self.register_buffer("dmu1_nodes", dmu1_nodes, persistent=False)
        self.register_buffer("dsigma_nodes", dsigma_nodes, persistent=False)

    @property
    def device(self) -> torch.device:
        return self.mu0_nodes.device

    def _extract_model_output(self, output: Tensor | tuple[Tensor, ...]) -> Tensor:
        if isinstance(output, tuple):
            return output[0]
        return output

    def _interpolate_coeff(self, nodes: Tensor, slopes: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        t = t.clamp(0.0, 1.0)
        scaled = t * float(self.num_intervals)
        lower_idx = torch.floor(scaled).to(dtype=torch.long).clamp(max=self.num_intervals - 1)
        frac = scaled - lower_idx.to(dtype=t.dtype)
        coeff_left = nodes[lower_idx]
        coeff_right = nodes[lower_idx + 1]
        coeff = torch.lerp(coeff_left, coeff_right, frac)
        dcoeff = slopes[lower_idx]
        return coeff, dcoeff

    def bridge_stats(
        self,
        t: Tensor,
        *,
        xdim: tuple[int, ...] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        mu0, dmu0 = self._interpolate_coeff(self.mu0_nodes, self.dmu0_nodes, t)
        mu1, dmu1 = self._interpolate_coeff(self.mu1_nodes, self.dmu1_nodes, t)
        sigma, dsigma = self._interpolate_coeff(self.sigma_nodes, self.dsigma_nodes, t)

        if xdim is None:
            return mu0, mu1, sigma, dmu0, dmu1, dsigma

        return (
            unsqueeze_xdim(mu0, xdim),
            unsqueeze_xdim(mu1, xdim),
            unsqueeze_xdim(sigma, xdim),
            unsqueeze_xdim(dmu0, xdim),
            unsqueeze_xdim(dmu1, xdim),
            unsqueeze_xdim(dsigma, xdim),
        )

    def sample_t_r(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        t = torch.rand(batch_size, device=device, dtype=dtype)
        t = t * (1.0 - 2.0 * self.time_eps) + self.time_eps
        r = torch.rand(batch_size, device=device, dtype=dtype) * t

        num_equal = int(self.flow_ratio * batch_size)
        if num_equal > 0:
            equal_idx = torch.randperm(batch_size, device=device)[:num_equal]
            r[equal_idx] = t[equal_idx]
        return t, r

    def sample_bridge(
        self,
        x0: Tensor,
        x1: Tensor,
        t: Tensor,
        *,
        eps: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if x0.shape != x1.shape:
            raise ValueError(f"x0 shape {x0.shape} must match x1 shape {x1.shape}.")

        xdim = tuple(x0.shape[1:])
        mu0, mu1, sigma, dmu0, dmu1, dsigma = self.bridge_stats(t, xdim=xdim)
        eps = torch.randn_like(x0) if eps is None else eps

        x_t = mu0 * x0 + mu1 * x1 + sigma * eps
        v_t = dmu0 * x0 + dmu1 * x1 + dsigma * eps
        return x_t, v_t, eps

    def compute_eps_from_x0(self, x_t: Tensor, x0_hat: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        mu0, mu1, sigma, _, _, _ = self.bridge_stats(t, xdim=tuple(x_t.shape[1:]))
        sigma_safe = sigma.clamp_min(self.interval_eps)
        return (x_t - mu0 * x0_hat - mu1 * x1) / sigma_safe

    def compute_velocity_from_x0(self, x_t: Tensor, x0_hat: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        xdim = tuple(x_t.shape[1:])
        _, _, _, dmu0, dmu1, dsigma = self.bridge_stats(t, xdim=xdim)
        eps_hat = self.compute_eps_from_x0(x_t, x0_hat, x1, t)
        return dmu0 * x0_hat + dmu1 * x1 + dsigma * eps_hat

    def compute_mean_velocity_from_x0(
        self,
        x_t: Tensor,
        x0_hat: Tensor,
        x1: Tensor,
        t: Tensor,
        r: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if x_t.shape != x0_hat.shape or x_t.shape != x1.shape:
            raise ValueError("x_t, x0_hat, and x1 must have identical shapes.")

        xdim = tuple(x_t.shape[1:])
        eps_hat = self.compute_eps_from_x0(x_t, x0_hat, x1, t)
        mu0_r, mu1_r, sigma_r, _, _, _ = self.bridge_stats(r, xdim=xdim)
        x_r_hat = mu0_r * x0_hat + mu1_r * x1 + sigma_r * eps_hat

        delta = t - r
        delta_b = unsqueeze_xdim(delta, xdim)
        safe_delta = torch.where(delta_b.abs() <= self.interval_eps, torch.ones_like(delta_b), delta_b)
        u_hat = (x_t - x_r_hat) / safe_delta

        v_hat = self.compute_velocity_from_x0(x_t, x0_hat, x1, t)
        equal_mask = unsqueeze_xdim(delta.abs() <= self.interval_eps, xdim)
        u_hat = torch.where(equal_mask, v_hat, u_hat)
        x_r_hat = torch.where(equal_mask, x_t, x_r_hat)
        return u_hat, x_r_hat, v_hat

    def _predict_x0(
        self,
        model: nn.Module,
        x_t: Tensor,
        t: Tensor,
        *,
        model_kwargs: dict[str, object],
        clip_denoise: bool,
    ) -> Tensor:
        output = model(x_t, t, **model_kwargs)
        x0_hat = self._extract_model_output(output)
        if clip_denoise:
            x0_hat = x0_hat.clamp(-1.0, 1.0)
        return x0_hat

    def _build_u_fn(
        self,
        model: nn.Module,
        x1: Tensor,
        *,
        model_kwargs: dict[str, object],
        clip_denoise: bool,
    ) -> Callable[[Tensor, Tensor, Tensor], Tensor]:
        def u_fn(x_t: Tensor, t: Tensor, r: Tensor) -> Tensor:
            x0_hat = self._predict_x0(model, x_t, t, model_kwargs=model_kwargs, clip_denoise=clip_denoise)
            u_hat, _, _ = self.compute_mean_velocity_from_x0(x_t, x0_hat, x1, t, r)
            return u_hat

        return u_fn

    def training_loss(
        self,
        model: nn.Module,
        x0: Tensor,
        x1: Tensor,
        model_kwargs: dict[str, object] | None = None,
    ) -> dict[str, Tensor]:
        model_kwargs = {} if model_kwargs is None else model_kwargs
        batch_size = x0.shape[0]
        t, r = self.sample_t_r(batch_size, device=x0.device, dtype=x0.dtype)
        x_t, v_t, _ = self.sample_bridge(x0, x1, t)

        x0_hat = self._predict_x0(model, x_t, t, model_kwargs=model_kwargs, clip_denoise=False)
        u_hat, _, _ = self.compute_mean_velocity_from_x0(x_t, x0_hat, x1, t, r)

        r_jvp = torch.where((t - r).abs() <= self.interval_eps, (t - self.time_eps).clamp_min(0.0), r)
        u_fn = self._build_u_fn(model, x1, model_kwargs=model_kwargs, clip_denoise=False)
        _, dudt = torch.autograd.functional.jvp(
            u_fn,
            (x_t, t, r_jvp),
            (v_t, torch.ones_like(t), torch.zeros_like(r_jvp)),
        )

        delta = unsqueeze_xdim(t - r, tuple(x_t.shape[1:]))
        u_tgt = v_t - delta * dudt

        loss_mf = adaptive_l2_loss(u_hat - u_tgt.detach())
        loss_x0 = F.mse_loss(x0_hat, x0)
        loss = loss_mf + self.aux_x0_weight * loss_x0
        return {
            "loss": loss,
            "loss_mf": loss_mf,
            "loss_x0": loss_x0,
            "pred_x0": x0_hat,
            "pred_u": u_hat,
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

        time_grid = torch.linspace(1.0 - self.time_eps, 0.0, nfe + 1, device=x1.device, dtype=x1.dtype)
        x_curr = x1

        states: list[Tensor] = []
        x0_preds: list[Tensor] = []

        for idx in range(nfe):
            t_val = time_grid[idx]
            r_val = time_grid[idx + 1]
            t = torch.full((x1.shape[0],), float(t_val.item()), device=x1.device, dtype=x1.dtype)
            r = torch.full((x1.shape[0],), float(r_val.item()), device=x1.device, dtype=x1.dtype)

            x0_hat = self._predict_x0(model, x_curr, t, model_kwargs=model_kwargs, clip_denoise=clip_denoise)
            _, x_next, _ = self.compute_mean_velocity_from_x0(x_curr, x0_hat, x1, t, r)
            x_curr = x_next

            states.append(x_curr.detach())
            x0_preds.append(x_curr.detach())

        if not states:
            raise RuntimeError("Sampling produced no states.")

        total_states = len(states)
        if log_steps is not None:
            selected_idx = [idx for idx in log_steps if 0 <= idx < total_states]
            if not selected_idx:
                selected_idx = [total_states - 1]
        elif log_count > 1:
            selected_idx = space_indices(total_states, log_count)
        else:
            selected_idx = [total_states - 1]

        sampled = torch.stack([x0_preds[idx] for idx in selected_idx], dim=1)
        traj = torch.stack([states[idx] for idx in selected_idx], dim=1)
        return {"sampled": sampled, "traj": traj}
