from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class GAN(nn.Module):
    """
    Placeholder copy of causal_forcing GAN interface.

    Notes:
    - Kept under src/networks for future integration.
    - Intentionally standalone (no inheritance chain to third_party classes).
    - This module currently only preserves method signatures.
    """

    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, dict]:
        raise NotImplementedError("GAN local implementation is reserved for future integration.")

    def critic_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        real_image_or_video: torch.Tensor,
        initial_latent: torch.Tensor | None = None,
    ) -> Tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], dict]:
        raise NotImplementedError("GAN local implementation is reserved for future integration.")

