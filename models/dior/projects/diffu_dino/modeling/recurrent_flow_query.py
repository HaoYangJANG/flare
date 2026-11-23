"""Recurrent semantic query updates for aligned box-flow trajectories."""

import torch
import torch.nn as nn


class RecurrentFlowQueryUpdater(nn.Module):
    """Carry decoded semantic state between aligned outer flow steps."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        flow_steps: int,
        max_update_scale: float = 0.25,
    ):
        super().__init__()
        if embed_dim < 1 or hidden_dim < 1:
            raise ValueError("query dimensions must be positive")
        if flow_steps < 2:
            raise ValueError("recurrent query updates require at least two flow steps")
        if not 0.0 < max_update_scale <= 1.0:
            raise ValueError("max_update_scale must lie in (0, 1]")

        self.embed_dim = int(embed_dim)
        self.flow_steps = int(flow_steps)
        self.max_update_scale = float(max_update_scale)

        self.query_norm = nn.LayerNorm(self.embed_dim)
        self.residual_projection = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.embed_dim),
        )
        self.gate_projection = nn.Sequential(
            nn.Linear(self.embed_dim + 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        nn.init.zeros_(self.residual_projection[-1].weight)
        nn.init.zeros_(self.residual_projection[-1].bias)

    def forward(
        self,
        base_query,
        decoded_query,
        class_logits,
        current_boxes,
        predicted_boxes,
        step_index: int,
    ):
        if base_query.ndim != 3 or base_query.shape != decoded_query.shape:
            raise ValueError(
                "base and decoded queries must share [batch, query, dim] shape"
            )
        if base_query.shape[-1] != self.embed_dim:
            raise ValueError("query embedding dimension does not match updater")
        if class_logits.ndim != 3 or class_logits.shape[:2] != base_query.shape[:2]:
            raise ValueError("class logits must share query batch dimensions")
        expected_box_shape = (*base_query.shape[:2], 4)
        if current_boxes.shape != expected_box_shape:
            raise ValueError("current boxes must have [batch, query, 4] shape")
        if predicted_boxes.shape != expected_box_shape:
            raise ValueError("predicted boxes must have [batch, query, 4] shape")
        if not 0 <= step_index < self.flow_steps - 1:
            raise ValueError("step_index must identify a non-final flow step")

        normalized_query = self.query_norm(decoded_query)
        confidence = class_logits.detach().sigmoid().amax(dim=-1, keepdim=True)
        predicted_size = predicted_boxes[..., 2:].detach().clamp(0.0, 1.0)
        smallness = 1.0 - predicted_size.prod(dim=-1, keepdim=True).sqrt()
        displacement = (
            predicted_boxes.detach() - current_boxes.detach()
        ).abs().mean(dim=-1, keepdim=True)
        progress = confidence.new_full(
            confidence.shape,
            float(step_index + 1) / float(self.flow_steps),
        )

        gate_input = torch.cat(
            (
                normalized_query,
                confidence,
                smallness,
                displacement,
                progress,
            ),
            dim=-1,
        )
        gate = self.gate_projection(gate_input).sigmoid()
        residual = self.residual_projection(normalized_query).tanh()

        return base_query + self.max_update_scale * gate * residual
