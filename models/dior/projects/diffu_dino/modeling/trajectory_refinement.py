import torch
import torch.nn as nn
from torchvision.ops import roi_align

from detrex.layers import box_cxcywh_to_xyxy


def encode_diffusion_trajectory(box_steps, eps=1e-4, max_motion=2.0):
    """Encode the last two outer diffusion motions relative to final box scale."""
    if isinstance(box_steps, (list, tuple)):
        box_steps = torch.stack(tuple(box_steps), dim=0)
    if box_steps.ndim != 4 or box_steps.shape[0] < 3 or box_steps.shape[-1] != 4:
        raise ValueError("box_steps must have shape [steps, batch, queries, 4] with at least 3 steps")
    previous, penultimate, final = box_steps[-3:]
    size = final[..., 2:].clamp(min=eps)
    scale = torch.cat((size, size), dim=-1)
    first_motion = ((penultimate - previous) / scale).clamp(
        min=-max_motion, max=max_motion
    )
    second_motion = ((final - penultimate) / scale).clamp(
        min=-max_motion, max=max_motion
    )
    return torch.cat((first_motion, second_motion), dim=-1)


def select_best_box_candidate(candidates, quality_logits, improvement_margin=0.0):
    """Select one query-aligned candidate while retaining candidate zero as fallback."""
    if candidates.ndim != 4 or candidates.shape[-1] != 4:
        raise ValueError("candidates must have shape [batch, queries, candidates, 4]")
    if quality_logits.shape != candidates.shape[:-1]:
        raise ValueError("quality logits must align with the candidate dimension")
    best_index = quality_logits.argmax(dim=-1)
    best_quality = quality_logits.sigmoid().gather(
        2, best_index.unsqueeze(-1)
    ).squeeze(-1)
    baseline_quality = quality_logits[..., 0].sigmoid()
    use_refinement = (
        (best_index > 0)
        & (best_quality >= baseline_quality + float(improvement_margin))
    )
    selected_index = torch.where(use_refinement, best_index, torch.zeros_like(best_index))
    box_index = selected_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 4)
    selected_boxes = candidates.gather(2, box_index).squeeze(2)
    selected_quality = quality_logits.gather(2, selected_index.unsqueeze(-1)).squeeze(-1)
    return selected_boxes, selected_quality, selected_index


def normalized_trajectory_uncertainty(
    previous_boxes,
    final_boxes,
    eps=1e-4,
    max_uncertainty=2.0,
):
    """Measure decoder-layer drift relative to each final box's scale."""
    size = final_boxes[..., 2:].clamp(min=eps)
    scale = torch.cat((size, size), dim=-1)
    return ((final_boxes - previous_boxes).abs() / scale).clamp(
        max=max_uncertainty
    )


def relative_box_refinement(
    boxes,
    raw_delta,
    gate_logits,
    max_center_scale=0.10,
    max_size_log_scale=0.15,
    eps=1e-4,
):
    """Apply a bounded residual in box-relative center/log-size coordinates."""
    center = boxes[..., :2]
    size = boxes[..., 2:].clamp(min=eps)
    gate = gate_logits.sigmoid()
    bounded = raw_delta.tanh()

    refined_center = center + gate * max_center_scale * bounded[..., :2] * size
    refined_size = size * torch.exp(
        gate * max_size_log_scale * bounded[..., 2:]
    )
    return torch.cat(
        (
            refined_center.clamp(eps, 1.0 - eps),
            refined_size.clamp(eps, 1.0),
        ),
        dim=-1,
    )


def normalized_boxes_to_feature_rois(
    boxes,
    image_sizes,
    padded_image_size,
    feature_size,
):
    """Convert per-image normalized boxes to padded feature-map coordinates."""
    batch_size, num_boxes = boxes.shape[:2]
    padded_height, padded_width = padded_image_size
    feature_height, feature_width = feature_size
    boxes_xyxy = box_cxcywh_to_xyxy(boxes)

    image_height = image_sizes[:, 0].to(boxes).view(batch_size, 1, 1)
    image_width = image_sizes[:, 1].to(boxes).view(batch_size, 1, 1)
    x_scale = image_width * (float(feature_width) / float(padded_width))
    y_scale = image_height * (float(feature_height) / float(padded_height))
    scale = torch.cat((x_scale, y_scale, x_scale, y_scale), dim=-1)
    boxes_feature = boxes_xyxy * scale

    x1 = boxes_feature[..., 0].clamp(0.0, float(feature_width))
    y1 = boxes_feature[..., 1].clamp(0.0, float(feature_height))
    x2 = boxes_feature[..., 2].clamp(0.0, float(feature_width))
    y2 = boxes_feature[..., 3].clamp(0.0, float(feature_height))
    boxes_feature = torch.stack((x1, y1, x2, y2), dim=-1)

    batch_indices = torch.arange(
        batch_size, device=boxes.device, dtype=boxes.dtype
    ).view(batch_size, 1, 1)
    batch_indices = batch_indices.expand(batch_size, num_boxes, 1)
    return torch.cat((batch_indices, boxes_feature), dim=-1).reshape(-1, 5)


class TrajectoryAwareLocalBoxRefinement(nn.Module):
    """Refine final boxes from local P3 evidence and decoder trajectory drift."""

    def __init__(
        self,
        in_channels=256,
        roi_channels=64,
        hidden_dim=256,
        pooled_size=3,
        max_center_scale=0.10,
        max_size_log_scale=0.15,
        detach_base=True,
    ):
        super().__init__()
        self.pooled_size = pooled_size
        self.max_center_scale = max_center_scale
        self.max_size_log_scale = max_size_log_scale
        self.detach_base = detach_base

        self.feature_projection = nn.Sequential(
            nn.Conv2d(in_channels, roi_channels, kernel_size=1),
            nn.GroupNorm(1, roi_channels),
            nn.ReLU(inplace=True),
        )
        input_dim = roi_channels * pooled_size * pooled_size + 8
        self.refinement_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.output_head = nn.Linear(hidden_dim, 5)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def forward(
        self,
        feature,
        previous_boxes,
        final_boxes,
        image_sizes,
        padded_image_size,
    ):
        if previous_boxes.shape != final_boxes.shape:
            raise ValueError("trajectory boxes must have identical shapes")
        if final_boxes.shape[-1] != 4:
            raise ValueError("trajectory boxes must use cxcywh coordinates")

        if self.detach_base:
            feature = feature.detach()
            previous_boxes = previous_boxes.detach()
            final_boxes = final_boxes.detach()

        uncertainty = normalized_trajectory_uncertainty(
            previous_boxes, final_boxes
        )
        projected_feature = self.feature_projection(feature)
        rois = normalized_boxes_to_feature_rois(
            final_boxes,
            image_sizes,
            padded_image_size,
            projected_feature.shape[-2:],
        )
        pooled = roi_align(
            projected_feature,
            rois,
            output_size=self.pooled_size,
            spatial_scale=1.0,
            sampling_ratio=2,
            aligned=True,
        )
        batch_size, num_boxes = final_boxes.shape[:2]
        pooled = pooled.flatten(1).reshape(batch_size, num_boxes, -1)
        refinement_input = torch.cat(
            (pooled, final_boxes, uncertainty), dim=-1
        )
        prediction = self.output_head(self.refinement_mlp(refinement_input))
        raw_delta = prediction[..., :4]
        gate_logits = prediction[..., 4:]
        refined_boxes = relative_box_refinement(
            final_boxes,
            raw_delta,
            gate_logits,
            max_center_scale=self.max_center_scale,
            max_size_log_scale=self.max_size_log_scale,
        )
        return {
            "pred_boxes_refined": refined_boxes,
            "refinement_gate_logits": gate_logits,
            "trajectory_uncertainty": uncertainty,
        }


class TrajectoryResidualFlowRefinement(nn.Module):
    """Generate two bounded box candidates from a real three-step DDIM trajectory."""

    def __init__(
        self,
        in_channels=256,
        query_dim=256,
        roi_channels=64,
        hidden_dim=256,
        pooled_size=3,
        max_center_scale_per_step=0.025,
        max_size_log_scale_per_step=0.05,
        quality_improvement_margin=0.02,
        detach_base=True,
    ):
        super().__init__()
        self.pooled_size = pooled_size
        self.max_center_scale_per_step = max_center_scale_per_step
        self.max_size_log_scale_per_step = max_size_log_scale_per_step
        self.quality_improvement_margin = quality_improvement_margin
        self.detach_base = detach_base

        self.feature_projection = nn.Sequential(
            nn.Conv2d(in_channels, roi_channels, kernel_size=1),
            nn.GroupNorm(1, roi_channels),
            nn.ReLU(inplace=True),
        )
        self.roi_projection = nn.Linear(
            roi_channels * pooled_size * pooled_size, hidden_dim
        )
        self.query_projection = nn.Linear(query_dim, hidden_dim)
        self.trajectory_projection = nn.Linear(8, hidden_dim)
        self.box_projection = nn.Linear(4, hidden_dim)
        self.step_embedding = nn.Embedding(3, hidden_dim)
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.flow_head = nn.Linear(hidden_dim, 4)
        self.quality_head = nn.Linear(hidden_dim, 3)
        nn.init.zeros_(self.flow_head.weight)
        nn.init.zeros_(self.flow_head.bias)
        nn.init.zeros_(self.quality_head.weight)
        nn.init.zeros_(self.quality_head.bias)

    def _pool(self, projected_feature, boxes, image_sizes, padded_image_size):
        rois = normalized_boxes_to_feature_rois(
            boxes,
            image_sizes,
            padded_image_size,
            projected_feature.shape[-2:],
        )
        pooled = roi_align(
            projected_feature,
            rois,
            output_size=self.pooled_size,
            spatial_scale=1.0,
            sampling_ratio=2,
            aligned=True,
        )
        batch_size, num_boxes = boxes.shape[:2]
        return pooled.flatten(1).reshape(batch_size, num_boxes, -1)

    def _candidate_hidden(
        self,
        projected_feature,
        boxes,
        query_embedding,
        trajectory_embedding,
        image_sizes,
        padded_image_size,
        step_index,
    ):
        pooled = self._pool(
            projected_feature, boxes, image_sizes, padded_image_size
        )
        step = self.step_embedding.weight[step_index].view(1, 1, -1)
        fused = (
            self.roi_projection(pooled)
            + query_embedding
            + trajectory_embedding
            + self.box_projection(boxes)
            + step
        )
        return self.fusion(fused)

    def _advance(self, current_boxes, baseline_boxes, raw_delta):
        bounded = raw_delta.tanh()
        baseline_size = baseline_boxes[..., 2:].clamp(min=1e-4)
        center = current_boxes[..., :2] + (
            self.max_center_scale_per_step
            * bounded[..., :2]
            * baseline_size
        )
        size = current_boxes[..., 2:].clamp(min=1e-4) * torch.exp(
            self.max_size_log_scale_per_step * bounded[..., 2:]
        )
        return torch.cat(
            (center.clamp(1e-4, 1.0 - 1e-4), size.clamp(1e-4, 1.0)),
            dim=-1,
        )

    def forward(
        self,
        feature,
        box_steps,
        query_features,
        image_sizes,
        padded_image_size,
    ):
        if isinstance(box_steps, (list, tuple)):
            box_steps = torch.stack(tuple(box_steps), dim=0)
        if box_steps.ndim != 4 or box_steps.shape[0] < 3:
            raise ValueError("trajectory residual flow requires at least three box steps")
        if box_steps.shape[1:3] != query_features.shape[:2]:
            raise ValueError("trajectory boxes and query features must align")

        if self.detach_base:
            box_steps = box_steps.detach()
            query_features = query_features.detach()

        trajectory = encode_diffusion_trajectory(box_steps)
        baseline = box_steps[-1]
        projected_feature = self.feature_projection(feature)
        query_embedding = self.query_projection(query_features)
        trajectory_embedding = self.trajectory_projection(trajectory)

        baseline_hidden = self._candidate_hidden(
            projected_feature,
            baseline,
            query_embedding,
            trajectory_embedding,
            image_sizes,
            padded_image_size,
            step_index=0,
        )
        quality_logits = [self.quality_head(baseline_hidden)[..., 0]]

        candidates = [baseline]
        current = baseline
        for step_index in (1, 2):
            hidden = self._candidate_hidden(
                projected_feature,
                current,
                query_embedding,
                trajectory_embedding,
                image_sizes,
                padded_image_size,
                step_index=step_index,
            )
            current = self._advance(current, baseline, self.flow_head(hidden))
            candidates.append(current)
            quality_logits.append(self.quality_head(hidden)[..., step_index])

        candidates = torch.stack(candidates, dim=2)
        quality_logits = torch.stack(quality_logits, dim=-1)
        selected, selected_quality, selected_index = select_best_box_candidate(
            candidates,
            quality_logits,
            improvement_margin=self.quality_improvement_margin,
        )
        return {
            "box_candidates": candidates,
            "candidate_quality_logits": quality_logits,
            "pred_boxes_refined": selected,
            "selected_candidate_quality": selected_quality,
            "selected_candidate_index": selected_index,
            "trajectory_descriptor": trajectory,
        }
