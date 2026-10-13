from __future__ import annotations

from abc import abstractmethod
from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig


class BaseCausalForcingModel(nn.Module):
    """
    Minimal base class aligned with third_party causal_forcing model interface.
    """

    def __init__(self, args: DictConfig | Any, device: torch.device | None = None):
        super().__init__()
        self.args = args
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if bool(getattr(args, "mixed_precision", False)) else torch.float32

        self.num_frame_per_block = int(getattr(args, "num_frame_per_block", 1))
        self.independent_first_frame = bool(getattr(args, "independent_first_frame", False))

        self._initialize_models(args=args, device=self.device)

        if hasattr(self, "scheduler") and hasattr(self.scheduler, "timesteps"):
            try:
                self.scheduler.timesteps = self.scheduler.timesteps.to(self.device)
            except Exception:
                pass

        if hasattr(args, "denoising_step_list"):
            self.denoising_step_list = torch.tensor(args.denoising_step_list, dtype=torch.long, device=self.device)

    @abstractmethod
    def _initialize_models(self, args: DictConfig | Any, device: torch.device) -> None:
        pass

    def _get_timestep(
        self,
        min_timestep: int,
        max_timestep: int,
        batch_size: int,
        num_frame: int,
        num_frame_per_block: int,
        uniform_timestep: bool = False,
    ) -> torch.Tensor:
        """
        Sample timestep tensor [B, F], optionally sharing within each temporal block.
        """
        if uniform_timestep:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, 1],
                device=self.device,
                dtype=torch.long,
            ).repeat(1, num_frame)
            return timestep

        timestep = torch.randint(
            min_timestep,
            max_timestep,
            [batch_size, num_frame],
            device=self.device,
            dtype=torch.long,
        )

        if self.independent_first_frame and num_frame > 1:
            second = timestep[:, 1:]
            second = second.reshape(second.shape[0], -1, num_frame_per_block)
            second[:, :, 1:] = second[:, :, 0:1]
            second = second.reshape(second.shape[0], -1)
            timestep = torch.cat([timestep[:, 0:1], second], dim=1)
        else:
            timestep = timestep.reshape(timestep.shape[0], -1, num_frame_per_block)
            timestep[:, :, 1:] = timestep[:, :, 0:1]
            timestep = timestep.reshape(timestep.shape[0], -1)

        return timestep

