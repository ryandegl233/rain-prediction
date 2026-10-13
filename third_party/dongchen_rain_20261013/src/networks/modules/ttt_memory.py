import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for 2D feature maps [B, C, H, W]."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class LoGerStyleMemory(nn.Module):
    """
    LoGeR-style (not LoGeR-copy) global TTT memory block.

    Borrowed ideas:
    - apply-then-update semantics
    - pre-norm organization for stability
    - global/compressed memory interaction via summary-conditioned modulation

    Not included in this version:
    - SWA local memory branch
    - official LoGeR fast-weight optimizer dynamics
    """

    def __init__(
        self,
        channels: int,
        pred_channels: int = 6,
        apply_scale: float = 0.1,
        update_momentum: float = 0.1,
    ):
        super().__init__()
        self.channels = channels
        self.pred_channels = pred_channels
        self.apply_scale = float(apply_scale)
        self.update_momentum = float(update_momentum)

        # init path
        self.init_norm = LayerNorm2d(channels)
        self.init_proj = nn.Conv2d(channels, channels, kernel_size=1)

        # apply path (pre-norm + global summary conditioning)
        self.apply_mem_norm = LayerNorm2d(channels)
        self.apply_rain_norm = LayerNorm2d(channels)
        self.rain_query_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.mem_key_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.mem_value_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.apply_out_proj = nn.Conv2d(channels, channels, kernel_size=1)

        # update path (pre-norm + gated EMA-like write)
        self.update_mem_norm = LayerNorm2d(channels)
        self.update_rain_norm = LayerNorm2d(channels)
        self.pred_proj = nn.Conv2d(pred_channels, channels, kernel_size=1)
        self.update_gate_proj = nn.Conv2d(channels * 3, channels, kernel_size=1)
        self.update_cand_proj = nn.Conv2d(channels * 3, channels, kernel_size=1)

    def _align_like(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        return x

    def init_memory(self, initial_memory_feat: torch.Tensor) -> torch.Tensor:
        """
        Build runtime memory state from initial global feature.
        """
        x = self.init_norm(initial_memory_feat)
        return self.init_proj(x)

    def apply(self, memory_state: torch.Tensor, rain_context_feat: torch.Tensor) -> torch.Tensor:
        """
        Read/apply stage:
        - Pre-norm memory and current rain context.
        - Build a global modulation signal from memory+rainsummary.
        - Produce step_state for current decoder step.
        """
        rain_context_feat = self._align_like(rain_context_feat, memory_state)
        mem_n = self.apply_mem_norm(memory_state)
        rain_n = self.apply_rain_norm(rain_context_feat)

        q = self.rain_query_proj(rain_n)
        k = self.mem_key_proj(mem_n)
        v = self.mem_value_proj(mem_n)

        # Lightweight associative readout: channel-wise compatibility gate.
        score = (q * k).mean(dim=1, keepdim=True)
        gate = torch.sigmoid(score)
        global_ctx = v * gate

        delta = self.apply_out_proj(global_ctx)
        step_state = memory_state + self.apply_scale * delta
        return step_state

    def update(
        self,
        memory_state: torch.Tensor,
        rain_context_feat: torch.Tensor,
        next_rain_prob: torch.Tensor,
    ) -> torch.Tensor:
        """
        Write/update stage:
        - Pre-norm inputs.
        - Add prediction-aware write signal from current next_rain_prob.
        - Generate gated candidate write.
        - Use EMA-like gated blend for stable dynamic memory evolution.
        """
        rain_context_feat = self._align_like(rain_context_feat, memory_state)
        next_rain_prob = self._align_like(next_rain_prob, memory_state)
        mem_n = self.update_mem_norm(memory_state)
        rain_n = self.update_rain_norm(rain_context_feat)
        if next_rain_prob.shape[1] != self.pred_channels:
            raise ValueError(
                f"next_rain_prob channel mismatch: expected {self.pred_channels}, got {next_rain_prob.shape[1]}"
            )
        pred_n = self.pred_proj(next_rain_prob)
        upd_in = torch.cat([mem_n, rain_n, pred_n], dim=1)

        gate = torch.sigmoid(self.update_gate_proj(upd_in))
        cand = self.update_cand_proj(upd_in)
        eff_gate = self.update_momentum * gate
        new_memory = (1.0 - eff_gate) * memory_state + eff_gate * cand
        return new_memory
