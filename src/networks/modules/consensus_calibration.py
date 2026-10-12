"""Lightweight cross-modal consensus/conflict calibration.

The module is designed to sit between modality-specific patch adapters and the
existing cross-modal attention.  Every newly introduced path is zero-initialized
or zero-gated so an old checkpoint keeps the original behaviour at step zero.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class _MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, out_features),
        )

    def zero_output(self) -> None:
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConsensusConflictCalibration(nn.Module):
    """Calibrate three modality token streams before cross-modal attention.

    Args:
        dim: Token channel dimension.
        bottleneck: Width used by the small calibration/gating MLPs.
        modality_count: Number of streams. The current network uses radar,
            satellite and rainfall, so this defaults to three.

    Inputs are ``[B, T, N, D]`` tensors. Optional ``patch_stats`` contains
    per-modality patch statistics with shape ``[B, T, N, 3]``. Those statistics
    preserve amplitude clues that LayerNorm-heavy token paths can weaken.
    """

    def __init__(self, dim: int, bottleneck: int = 64, modality_count: int = 3) -> None:
        super().__init__()
        if dim <= 0 or bottleneck <= 0 or modality_count < 2:
            raise ValueError("dim/bottleneck must be positive and modality_count must be >= 2")
        self.dim = int(dim)
        self.bottleneck = int(bottleneck)
        self.modality_count = int(modality_count)

        joint_dim = self.modality_count * self.dim
        self.conditional = nn.ModuleList(
            [_MLP(joint_dim, self.bottleneck, 2 * self.dim) for _ in range(self.modality_count)]
        )
        self.condition_scale = nn.Parameter(torch.zeros(self.modality_count))

        self.weight_nets = nn.ModuleList(
            [_MLP(joint_dim, self.bottleneck, 1) for _ in range(self.modality_count)]
        )

        # Gate inputs are cosine similarity, disagreement magnitude, modality
        # weight, plus optional mean/std/max patch statistics for every stream.
        self.gate_features = 3 + 3 * self.modality_count
        self.consensus_gate = _MLP(self.gate_features, self.bottleneck, 1)
        self.reliability_gate = _MLP(self.gate_features, self.bottleneck, 1)
        self.conflict_scale = nn.Parameter(torch.tensor(1.0))
        self.conflict_bias = nn.Parameter(torch.tensor(-2.0))

        self.consensus_adapter = _MLP(self.dim, self.bottleneck, self.dim)
        self.disagreement_adapters = nn.ModuleList(
            [_MLP(self.dim, self.bottleneck, self.dim) for _ in range(self.modality_count)]
        )

        for module in self.conditional:
            module.zero_output()
        for module in self.weight_nets:
            module.zero_output()
        self.consensus_gate.zero_output()
        self.reliability_gate.zero_output()
        self.consensus_adapter.zero_output()
        for module in self.disagreement_adapters:
            module.zero_output()
        self.last_diagnostics: dict[str, torch.Tensor] | None = None

    def _patch_stats(
        self,
        patch_stats: Sequence[torch.Tensor] | None,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if patch_stats is None:
            return reference.new_zeros(*reference.shape[:-1], 3 * self.modality_count)
        if len(patch_stats) != self.modality_count:
            raise ValueError(
                f"expected {self.modality_count} patch-stat tensors, got {len(patch_stats)}"
            )
        stats: list[torch.Tensor] = []
        for value in patch_stats:
            if value.shape[:-1] != reference.shape[:-1] or value.shape[-1] != 3:
                raise ValueError(
                    "patch stats must have shape [B,T,N,3] matching modality tokens, "
                    f"got {tuple(value.shape)} vs {tuple(reference.shape)}"
                )
            stats.append(value)
        return torch.cat(stats, dim=-1)

    def forward(
        self,
        modality_tokens: Sequence[torch.Tensor],
        patch_stats: Sequence[torch.Tensor] | None = None,
    ) -> list[torch.Tensor]:
        if len(modality_tokens) != self.modality_count:
            raise ValueError(
                f"expected {self.modality_count} modality token tensors, got {len(modality_tokens)}"
            )
        reference = modality_tokens[0]
        if reference.ndim != 4:
            raise ValueError(f"modality tokens must be [B,T,N,D], got {tuple(reference.shape)}")
        if any(tuple(value.shape) != tuple(reference.shape) for value in modality_tokens):
            raise ValueError("all modality token tensors must have identical shapes")
        if reference.shape[-1] != self.dim:
            raise ValueError(f"expected token dim {self.dim}, got {reference.shape[-1]}")

        joint = torch.cat(tuple(modality_tokens), dim=-1)
        calibrated: list[torch.Tensor] = []
        for index, value in enumerate(modality_tokens):
            gamma_beta = self.conditional[index](joint)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            delta = gamma * value + beta
            calibrated.append(value + torch.tanh(self.condition_scale[index]) * delta)

        joint_calibrated = torch.cat(tuple(calibrated), dim=-1)
        logits = torch.cat(
            tuple(network(joint_calibrated) for network in self.weight_nets),
            dim=-1,
        )
        weights = logits.softmax(dim=-1)
        consensus = sum(
            weights[..., index : index + 1] * calibrated[index]
            for index in range(self.modality_count)
        )
        stats = self._patch_stats(patch_stats, reference)
        outputs: list[torch.Tensor] = []
        consensus_gates: list[torch.Tensor] = []
        disagreement_gates: list[torch.Tensor] = []
        conflict_scores: list[torch.Tensor] = []
        reliabilities: list[torch.Tensor] = []
        for index, value in enumerate(calibrated):
            disagreement = value - consensus
            magnitude = disagreement.square().mean(dim=-1, keepdim=True).sqrt()
            similarity = F.cosine_similarity(value, consensus, dim=-1, eps=1e-6).unsqueeze(-1)
            gate_input = torch.cat(
                [similarity, magnitude, weights[..., index : index + 1], stats], dim=-1
            )
            consensus_weight = torch.sigmoid(self.consensus_gate(gate_input))
            reliability = torch.sigmoid(self.reliability_gate(gate_input))
            conflict = torch.sigmoid(F.softplus(self.conflict_scale) * magnitude + self.conflict_bias)
            disagreement_weight = conflict * reliability
            consensus_gates.append(consensus_weight)
            disagreement_gates.append(disagreement_weight)
            conflict_scores.append(conflict)
            reliabilities.append(reliability)
            outputs.append(
                value
                + consensus_weight * self.consensus_adapter(consensus)
                + disagreement_weight * self.disagreement_adapters[index](disagreement)
            )
        # Detached diagnostics are intentionally kept out of autograd. They are
        # available to validation/visualization code without changing the model
        # output contract or retaining the training graph.
        self.last_diagnostics = {
            "modality_weights": weights.detach(),
            "consensus": consensus.detach(),
            "conflict_scores": torch.cat(conflict_scores, dim=-1).detach(),
            "reliability": torch.cat(reliabilities, dim=-1).detach(),
            "consensus_gates": torch.cat(consensus_gates, dim=-1).detach(),
            "disagreement_gates": torch.cat(disagreement_gates, dim=-1).detach(),
        }
        return outputs
