from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class DMD(nn.Module):
    """
    Placeholder copy of causal_forcing DMD interface.

    Notes:
    - Kept under src/networks for future integration.
    - Intentionally standalone (no inheritance chain to third_party classes).
    - This module currently only preserves method signatures.
    """

    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device

    def _compute_kl_grad(
        self,
        noisy_image_or_video: torch.Tensor,
        estimated_clean_image_or_video: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        normalization: bool = True,
    ) -> Tuple[torch.Tensor, dict]:
        raise NotImplementedError("DMD local implementation is reserved for future integration.")

    def compute_distribution_matching_loss(
        self,
        image_or_video: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        gradient_mask: Optional[torch.Tensor] = None,
        denoised_timestep_from: int = 0,
        denoised_timestep_to: int = 0,
    ) -> Tuple[torch.Tensor, dict]:
        raise NotImplementedError("DMD local implementation is reserved for future integration.")

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, dict]:
        raise NotImplementedError("DMD local implementation is reserved for future integration.")

    def critic_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, dict]:
        raise NotImplementedError("DMD local implementation is reserved for future integration.")

