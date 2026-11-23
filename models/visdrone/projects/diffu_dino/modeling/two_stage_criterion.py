# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F

from detectron2.utils.events import get_event_storage

## Diffu_dino
# from detrex.modeling.criterion import SetCriterion
from .diffu_criterion import SetCriterion, vari_sigmoid_focal_loss
from .boundary_residual_loss import boundary_residual_loss

from detrex.utils import get_world_size, is_dist_avail_and_initialized
from detrex.layers import box_cxcywh_to_xyxy, generalized_box_iou
from .aux_normalization import global_count_normalizer
import copy


class TwoStageCriterion(SetCriterion):
    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        losses=["class", "boxes"],
        eos_coef=None,
        loss_class_type="focal_loss",
        alpha: float = 0.25,
        gamma: float = 2,
        two_stage_binary_cls=False,

        use_vlb: bool= True,
        use_matcher: bool = False,
        use_encoder_loss: bool = False,
        use_count_loss = False,
        denoise_noobject_bbox: bool = False,
        sigmoid_focal_loss = False,
        contrastive_loss = False,
        use_distributional_box: bool = False,
        reg_max: int = 32,
        reg_scale: float = 4.0,
        reg_up: float = 0.5,
        use_small_nwd_loss: bool = False,
        nwd_constant: float = 16.0,
        small_object_threshold: float = 48.0,
        use_vfl: bool = False,
        use_alpha_scheduler: bool = False,
        quality_score_floor: float = 0.0,
        decoder_final_only: bool = False,
        refinement_iou_lower: float = 0.5,
        refinement_iou_upper: float = 0.9,
        refinement_background_weight: float = 0.05,
        trajectory_residual_iou_lower: float = 0.2,
        trajectory_residual_iou_upper: float = 0.95,
        trajectory_residual_background_weight: float = 0.02,
        quality_target_type: str = "legacy_giou",
        flow_step_rematching: bool = False,
        flow_step_losses=None,
        flow_consistency_quality: str = "legacy_giou",
        flow_consistency_margin: float = 0.0,
        flow_diagnostics_period: int = 0,
        use_distribution_refinement_loss: bool = False,
        distribution_distill_margin: float = 0.02,
        use_score_calibration_loss: bool = False,
        use_boundary_residual_loss: bool = False,
        global_aux_normalization: bool = False,
    ):
        super().__init__(
            num_classes,
            matcher,
            weight_dict,
            losses,
            eos_coef,
            loss_class_type,
            alpha,
            gamma,
            use_distributional_box=use_distributional_box,
            reg_max=reg_max,
            reg_scale=reg_scale,
            reg_up=reg_up,
            use_small_nwd_loss=use_small_nwd_loss,
            nwd_constant=nwd_constant,
            small_object_threshold=small_object_threshold,
            use_vfl=use_vfl,
            use_alpha_scheduler=use_alpha_scheduler,
            quality_score_floor=quality_score_floor,
            use_distribution_refinement_loss=use_distribution_refinement_loss,
            distribution_distill_margin=distribution_distill_margin,
            global_aux_normalization=global_aux_normalization,
        )
        self.two_stage_binary_cls = two_stage_binary_cls


        self.use_vlb = use_vlb
        self.denoise_noobject_bbox = denoise_noobject_bbox
        self.use_matcher = use_matcher
        self.use_encoder_loss = use_encoder_loss
        self.use_count_loss = use_count_loss
        self.sigmoid_focal_loss = sigmoid_focal_loss
        self.contrastive_loss = contrastive_loss
        self.decoder_final_only = decoder_final_only
        if not 0.0 <= refinement_iou_lower < refinement_iou_upper <= 1.0:
            raise ValueError("refinement IoU range must lie within [0, 1]")
        self.refinement_iou_lower = refinement_iou_lower
        self.refinement_iou_upper = refinement_iou_upper
        self.refinement_background_weight = refinement_background_weight
        if not 0.0 <= trajectory_residual_iou_lower < trajectory_residual_iou_upper <= 1.0:
            raise ValueError("trajectory residual IoU range must lie within [0, 1]")
        self.trajectory_residual_iou_lower = trajectory_residual_iou_lower
        self.trajectory_residual_iou_upper = trajectory_residual_iou_upper
        self.trajectory_residual_background_weight = (
            trajectory_residual_background_weight
        )
        if quality_target_type not in {"legacy_giou", "iou"}:
            raise ValueError("quality_target_type must be 'legacy_giou' or 'iou'")
        if flow_consistency_quality not in {"legacy_giou", "iou"}:
            raise ValueError(
                "flow_consistency_quality must be 'legacy_giou' or 'iou'"
            )
        if flow_consistency_margin < 0.0:
            raise ValueError("flow_consistency_margin must be non-negative")
        if flow_diagnostics_period < 0:
            raise ValueError("flow_diagnostics_period must be non-negative")
        self.quality_target_type = quality_target_type
        self.flow_step_rematching = flow_step_rematching
        self.flow_step_losses = (
            None if flow_step_losses is None else tuple(flow_step_losses)
        )
        self.flow_consistency_quality = flow_consistency_quality
        self.flow_consistency_margin = float(flow_consistency_margin)
        self.flow_diagnostics_period = int(flow_diagnostics_period)
        self.use_score_calibration_loss = bool(use_score_calibration_loss)
        self.use_boundary_residual_loss = bool(use_boundary_residual_loss)
        print("sigmoid_focal_loss ",sigmoid_focal_loss)

    def _loss_boundary_evidence_aux(self, outputs, targets, indices, num_boxes, t_weight):
        # Explicit normal-query terminal dispatch keeps encoder/decoder/DN
        # loops and the existing matching identity untouched.
        losses = {}
        if self.use_score_calibration_loss:
            score_output = {"pred_logits": outputs["pred_score_logits"],
                            "pred_boxes": outputs["pred_boxes"]}
            losses["loss_score_calibration"] = self._loss_labels_with_iou_quality(
                score_output, targets, indices, num_boxes, t_weight)["loss_class"]
        if self.use_boundary_residual_loss:
            idx = self._get_src_permutation_idx(indices)
            target_boxes = torch.cat([target["boxes"][target_index]
                                      for target, (_, target_index) in zip(targets, indices)], dim=0)
            losses["loss_boundary_residual"] = boundary_residual_loss(
                outputs["pred_boundary_aux_boxes"][idx],
                outputs["pred_boundary_base_boxes"][idx], target_boxes,
                normalizer=num_boxes if self.global_aux_normalization else None)
        return losses

    @staticmethod
    def _aligned_iou(first_boxes, second_boxes):
        first = box_cxcywh_to_xyxy(first_boxes)
        second = box_cxcywh_to_xyxy(second_boxes)
        top_left = torch.maximum(first[:, :2], second[:, :2])
        bottom_right = torch.minimum(first[:, 2:], second[:, 2:])
        intersection = (bottom_right - top_left).clamp(min=0).prod(dim=-1)
        first_area = (first[:, 2:] - first[:, :2]).clamp(min=0).prod(dim=-1)
        second_area = (second[:, 2:] - second[:, :2]).clamp(min=0).prod(dim=-1)
        return intersection / (first_area + second_area - intersection).clamp(min=1e-6)

    def _matched_quality(self, outputs, targets, indices):
        idx = self._get_src_permutation_idx(indices)
        source_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [
                target["boxes"][target_index]
                for target, (_, target_index) in zip(targets, indices)
            ],
            dim=0,
        )
        return self._aligned_iou(source_boxes, target_boxes).detach().clamp(0.0, 1.0)

    def _loss_labels_with_iou_quality(
        self, outputs, targets, indices, num_boxes, t_weight
    ):
        assert "pred_boxes" in outputs and "pred_logits" in outputs
        idx = self._get_src_permutation_idx(indices)
        source_logits = outputs["pred_logits"]
        target_classes_o = torch.cat(
            [target["labels"][target_index] for target, (_, target_index) in zip(targets, indices)]
        )
        target_classes = torch.full(
            source_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=source_logits.device,
        )
        target_classes[idx] = target_classes_o
        target_onehot = F.one_hot(target_classes, self.num_classes + 1)[..., :-1]
        target_score = torch.zeros_like(target_classes, dtype=source_logits.dtype)
        target_score[idx] = self._matched_quality(outputs, targets, indices).to(
            target_score
        )
        if self.use_alpha_scheduler and t_weight is not None:
            self.alpha = t_weight.view(-1, 1, 1)
        loss = vari_sigmoid_focal_loss(
            source_logits,
            target_onehot,
            target_score,
            num_boxes=num_boxes,
            t_weight=t_weight,
            alpha=self.alpha,
            gamma=self.gamma,
            quality_score_floor=self.quality_score_floor,
        ) * source_logits.shape[1]
        return {"loss_class": loss}

    def loss_labels(self, outputs, targets, indices, num_boxes, t_weight, **kwargs):
        if self.quality_target_type == "iou" and self.use_vfl:
            return self._loss_labels_with_iou_quality(
                outputs, targets, indices, num_boxes, t_weight
            )
        return super().loss_labels(
            outputs, targets, indices, num_boxes, t_weight, **kwargs
        )

    def loss_labels_sigmoid_focal_loss(
        self, outputs, targets, indices, num_boxes, t_weight, **kwargs
    ):
        if self.quality_target_type == "iou":
            return self._loss_labels_with_iou_quality(
                outputs, targets, indices, num_boxes, t_weight
            )
        return super().loss_labels_sigmoid_focal_loss(
            outputs, targets, indices, num_boxes, t_weight, **kwargs
        )

    @staticmethod
    def _stable_matching_indices(
        previous_indices, current_indices, num_queries
    ):
        stable_indices = []
        for (previous_source, previous_target), (
            current_source,
            current_target,
        ) in zip(previous_indices, current_indices):
            previous_target_by_query = torch.full(
                (num_queries,),
                -1,
                dtype=previous_target.dtype,
                device=previous_target.device,
            )
            previous_target_by_query[previous_source] = previous_target
            stable = previous_target_by_query[current_source] == current_target
            stable_indices.append(
                (current_source[stable], current_target[stable])
            )
        return stable_indices

    @staticmethod
    def _matched_target_boxes(targets, indices):
        return torch.cat(
            [
                target["boxes"][target_index]
                for target, (_, target_index) in zip(targets, indices)
            ],
            dim=0,
        )

    @torch.no_grad()
    def _compute_flow_diagnostics(
        self, trajectory, targets, trajectory_indices
    ):
        if len(trajectory) != len(trajectory_indices):
            raise ValueError(
                "flow trajectory outputs and matching indices must align"
            )
        reference = trajectory[-1]["pred_boxes"]
        zero = reference.new_zeros(())
        metrics = {}

        for step_index, (step_output, step_indices) in enumerate(
            zip(trajectory, trajectory_indices)
        ):
            idx = self._get_src_permutation_idx(step_indices)
            target_boxes = self._matched_target_boxes(targets, step_indices)
            if target_boxes.numel() == 0:
                refined_iou = zero
                coarse_iou = zero
            else:
                refined_iou = self._aligned_iou(
                    step_output["pred_boxes"][idx], target_boxes
                ).mean()
                coarse_boxes = step_output.get(
                    "pred_transport_boxes", step_output.get("pred_dist_ref")
                )
                if coarse_boxes is not None:
                    coarse_iou = self._aligned_iou(
                        coarse_boxes[idx], target_boxes
                    ).mean()
                else:
                    coarse_iou = zero
            metrics[f"flow/refined_iou_step_{step_index}"] = refined_iou
            metrics[f"flow/coarse_iou_step_{step_index}"] = coarse_iou

        for step_index in range(len(trajectory) - 1):
            previous_indices = trajectory_indices[step_index]
            current_indices = trajectory_indices[step_index + 1]
            num_queries = trajectory[step_index]["pred_boxes"].shape[1]
            stable_indices = self._stable_matching_indices(
                previous_indices, current_indices, num_queries
            )

            common_count = 0
            switch_count = zero.clone()
            for (previous_source, previous_target), (
                current_source,
                current_target,
            ) in zip(previous_indices, current_indices):
                previous_target_by_query = torch.full(
                    (num_queries,),
                    -1,
                    dtype=previous_target.dtype,
                    device=previous_target.device,
                )
                previous_target_by_query[previous_source] = previous_target
                previous_for_current = previous_target_by_query[current_source]
                common = previous_for_current >= 0
                common_count += int(common.sum().item())
                switch_count += (
                    previous_for_current[common] != current_target[common]
                ).sum().to(switch_count)

            metrics[f"flow/match_switch_rate_{step_index}_{step_index + 1}"] = (
                switch_count / common_count if common_count else zero
            )

            stable_count = sum(len(source) for source, _ in stable_indices)
            if stable_count == 0:
                improvement_rate = zero
                iou_delta = zero
            else:
                idx = self._get_src_permutation_idx(stable_indices)
                target_boxes = self._matched_target_boxes(
                    targets, stable_indices
                )
                previous_quality = self._aligned_iou(
                    trajectory[step_index]["pred_boxes"][idx], target_boxes
                )
                current_quality = self._aligned_iou(
                    trajectory[step_index + 1]["pred_boxes"][idx],
                    target_boxes,
                )
                improvement_rate = (
                    current_quality > previous_quality
                ).float().mean()
                iou_delta = (current_quality - previous_quality).mean()
            metrics[f"flow/iou_improvement_rate_{step_index}_{step_index + 1}"] = (
                improvement_rate
            )
            metrics[f"flow/iou_delta_{step_index}_{step_index + 1}"] = iou_delta

        return metrics

    def _maybe_log_flow_diagnostics(
        self, trajectory, targets, trajectory_indices
    ):
        if self.flow_diagnostics_period <= 0 or len(trajectory) < 2:
            return
        try:
            storage = get_event_storage()
        except (AssertionError, RuntimeError):
            return
        if storage.iter % self.flow_diagnostics_period != 0:
            return
        metrics = self._compute_flow_diagnostics(
            trajectory, targets, trajectory_indices
        )
        metric_names = sorted(metrics)
        metric_values = torch.stack([metrics[name] for name in metric_names])
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(metric_values)
            metric_values /= get_world_size()
        for name, value in zip(metric_names, metric_values):
            storage.put_scalar(name, value.item(), smoothing_hint=True)

    def loss_flow_consistency(
        self,
        outputs,
        targets,
        indices,
        num_boxes,
        trajectory_indices=None,
    ):
        if self.flow_consistency_quality == "legacy_giou":
            return super().loss_flow_consistency(
                outputs, targets, indices, num_boxes
            )

        idx = self._get_src_permutation_idx(indices)
        target_boxes = torch.cat(
            [
                target["boxes"][target_index]
                for target, (_, target_index) in zip(targets, indices)
            ],
            dim=0,
        )
        zero = outputs["pred_boxes"].sum() * 0.0
        # Stable-pair normalization has a collective: empty ranks must reach it.
        global_stable_pairs = (
            self.global_aux_normalization
            and "flow_trajectory_outputs" in outputs
            and trajectory_indices is not None
        )
        if target_boxes.numel() == 0 and not global_stable_pairs:
            return {"loss_flow_consistency": zero}

        def quality(step_output):
            return self._aligned_iou(step_output["pred_boxes"][idx], target_boxes)

        if "flow_trajectory_outputs" in outputs:
            trajectory = [*outputs["flow_trajectory_outputs"], outputs]
            if trajectory_indices is not None:
                if len(trajectory) != len(trajectory_indices):
                    raise ValueError(
                        "flow trajectory outputs and matching indices must align"
                    )
                ranking_losses = []
                stable_count = 0
                for step_index in range(len(trajectory) - 1):
                    stable_indices = self._stable_matching_indices(
                        trajectory_indices[step_index],
                        trajectory_indices[step_index + 1],
                        trajectory[step_index]["pred_boxes"].shape[1],
                    )
                    pair_count = sum(
                        len(source) for source, _ in stable_indices
                    )
                    if pair_count == 0:
                        continue
                    stable_idx = self._get_src_permutation_idx(stable_indices)
                    stable_targets = self._matched_target_boxes(
                        targets, stable_indices
                    )
                    previous_quality = self._aligned_iou(
                        trajectory[step_index]["pred_boxes"][stable_idx],
                        stable_targets,
                    )
                    current_quality = self._aligned_iou(
                        trajectory[step_index + 1]["pred_boxes"][stable_idx],
                        stable_targets,
                    )
                    required = (
                        previous_quality.detach()
                        + self.flow_consistency_margin
                    ).clamp(max=1.0)
                    ranking_losses.append(F.relu(required - current_quality))
                    stable_count += pair_count
                if self.global_aux_normalization:
                    normalizer = global_count_normalizer(stable_count, outputs["pred_boxes"])
                    numerator = torch.cat(ranking_losses).sum() if ranking_losses else zero
                    return {"loss_flow_consistency": numerator / normalizer}
                if not ranking_losses:
                    return {"loss_flow_consistency": zero}
                return {
                    "loss_flow_consistency": torch.cat(
                        ranking_losses
                    ).sum()
                    / max(stable_count, 1)
                }

            ranking_losses = []
            previous_quality = None
            for step_output in trajectory:
                current_quality = quality(step_output)
                if previous_quality is not None:
                    required = (
                        previous_quality.detach() + self.flow_consistency_margin
                    ).clamp(max=1.0)
                    ranking_losses.append(F.relu(required - current_quality))
                previous_quality = current_quality
            if not ranking_losses:
                return {"loss_flow_consistency": zero}
            loss = torch.cat(ranking_losses).sum() / max(
                num_boxes * len(ranking_losses), 1.0
            )
            return {"loss_flow_consistency": loss}

        if "flow_next_pred_boxes" in outputs:
            current_quality = quality(outputs).detach()
            next_output = {"pred_boxes": outputs["flow_next_pred_boxes"]}
            next_quality = quality(next_output)
            required = (current_quality + self.flow_consistency_margin).clamp(max=1.0)
            monotonic = F.relu(required - next_quality)
            endpoint = F.l1_loss(
                next_output["pred_boxes"][idx], target_boxes, reduction="none"
            ).sum(-1)
            return {
                "loss_flow_consistency": (monotonic + endpoint).sum() / num_boxes
            }

        trajectory = [*outputs.get("aux_outputs", []), outputs]
        ranking_losses = []
        previous_quality = None
        for step_output in trajectory:
            current_quality = quality(step_output)
            if previous_quality is not None:
                required = (
                    previous_quality.detach() + self.flow_consistency_margin
                ).clamp(max=1.0)
                ranking_losses.append(F.relu(required - current_quality))
            previous_quality = current_quality
        if not ranking_losses:
            return {"loss_flow_consistency": zero}
        return {
            "loss_flow_consistency": torch.cat(ranking_losses).sum()
            / max(num_boxes * len(ranking_losses), 1.0)
        }

    @staticmethod
    def _aligned_giou(first_boxes, second_boxes):
        first = box_cxcywh_to_xyxy(first_boxes)
        second = box_cxcywh_to_xyxy(second_boxes)
        intersection_top_left = torch.maximum(first[:, :2], second[:, :2])
        intersection_bottom_right = torch.minimum(first[:, 2:], second[:, 2:])
        intersection = (
            intersection_bottom_right - intersection_top_left
        ).clamp(min=0).prod(dim=-1)
        first_area = (first[:, 2:] - first[:, :2]).clamp(min=0).prod(dim=-1)
        second_area = (second[:, 2:] - second[:, :2]).clamp(min=0).prod(dim=-1)
        union = first_area + second_area - intersection
        iou = intersection / union.clamp(min=1e-6)
        enclosing_top_left = torch.minimum(first[:, :2], second[:, :2])
        enclosing_bottom_right = torch.maximum(first[:, 2:], second[:, 2:])
        enclosing_area = (
            enclosing_bottom_right - enclosing_top_left
        ).clamp(min=0).prod(dim=-1)
        return iou - (enclosing_area - union) / enclosing_area.clamp(min=1e-6)

    def loss_baseline_trajectory_residual(
        self, outputs, targets, indices, num_boxes
    ):
        candidates_all = outputs["box_candidates"]
        quality_logits = outputs["candidate_quality_logits"]
        if candidates_all.shape[:-1] != quality_logits.shape:
            raise ValueError("trajectory residual candidates and quality logits must align")
        if candidates_all.shape[2] != 3:
            raise ValueError("trajectory residual branch requires three box candidates")

        idx = self._get_src_permutation_idx(indices)
        candidates = candidates_all[idx]
        target_boxes = torch.cat(
            [
                target["boxes"][target_index]
                for target, (_, target_index) in zip(targets, indices)
            ],
            dim=0,
        )
        quality_target = torch.zeros_like(quality_logits)
        quality_weight = torch.full_like(
            quality_logits, self.trajectory_residual_background_weight
        )
        zero = candidates_all.sum() * 0.0
        if target_boxes.numel() == 0:
            quality_loss = F.binary_cross_entropy_with_logits(
                quality_logits,
                quality_target,
                weight=quality_weight,
                reduction="sum",
            ) / quality_weight.sum().clamp(min=1.0)
            return {
                "loss_trajectory_residual_bbox": zero,
                "loss_trajectory_residual_giou": zero,
                "loss_trajectory_residual_monotonic": zero,
                "loss_trajectory_residual_non_degrade": zero,
                "loss_trajectory_residual_quality": quality_loss,
            }

        base = candidates[:, 0].detach()
        refined = candidates[:, 1:]
        base_iou = self._aligned_iou(base, target_boxes).detach()
        refined_target = target_boxes[:, None, :].expand_as(refined)
        refined_flat = refined.reshape(-1, 4)
        target_flat = refined_target.reshape(-1, 4)
        refined_iou = self._aligned_iou(refined_flat, target_flat).reshape(-1, 2)

        active = (
            (base_iou >= self.trajectory_residual_iou_lower)
            & (base_iou < self.trajectory_residual_iou_upper)
        ).to(refined.dtype)
        active_candidates = active[:, None].expand_as(refined_iou)
        active_denominator = active_candidates.sum().clamp(min=1.0)

        l1 = F.l1_loss(refined, refined_target, reduction="none").sum(dim=-1)
        giou = self._aligned_giou(refined_flat, target_flat).reshape(-1, 2)
        loss_bbox = (l1 * active_candidates).sum() / active_denominator
        loss_giou = ((1.0 - giou) * active_candidates).sum() / active_denominator

        first_iou = refined_iou[:, 0]
        second_iou = refined_iou[:, 1]
        active_count = active.sum().clamp(min=1.0)
        loss_monotonic = (
            F.relu(base_iou - first_iou)
            + F.relu(first_iou.detach() - second_iou)
        )
        loss_monotonic = (loss_monotonic * active).sum() / active_count
        best_refined_iou = refined_iou.max(dim=1).values
        loss_non_degrade = (
            F.relu(base_iou - best_refined_iou) * active
        ).sum() / active_count

        candidate_iou = torch.cat((base_iou[:, None], refined_iou), dim=1)
        quality_target[idx] = candidate_iou.detach().to(quality_target)
        quality_weight[idx] = 1.0
        quality_loss = F.binary_cross_entropy_with_logits(
            quality_logits,
            quality_target,
            weight=quality_weight,
            reduction="sum",
        ) / quality_weight.sum().clamp(min=1.0)
        return {
            "loss_trajectory_residual_bbox": loss_bbox,
            "loss_trajectory_residual_giou": loss_giou,
            "loss_trajectory_residual_monotonic": loss_monotonic,
            "loss_trajectory_residual_non_degrade": loss_non_degrade,
            "loss_trajectory_residual_quality": quality_loss,
        }

    def loss_trajectory_refinement(self, outputs, targets, indices, num_boxes):
        refined_all = outputs["pred_boxes_refined"]
        gate_logits = outputs["refinement_gate_logits"].squeeze(-1)
        idx = self._get_src_permutation_idx(indices)
        refined = refined_all[idx]
        base = outputs["pred_boxes"][idx].detach()
        target_boxes = torch.cat(
            [target["boxes"][target_index] for target, (_, target_index) in zip(targets, indices)],
            dim=0,
        )

        zero = refined_all.sum() * 0.0
        if target_boxes.numel() == 0:
            gate_target = torch.zeros_like(gate_logits)
            return {
                "loss_refinement_bbox": zero,
                "loss_refinement_giou": zero,
                "loss_refinement_non_degrade": zero,
                "loss_refinement_gate": F.binary_cross_entropy_with_logits(
                    gate_logits, gate_target
                ),
            }

        base_iou = self._aligned_iou(base, target_boxes).detach()
        refined_iou = self._aligned_iou(refined, target_boxes)
        active = (
            (base_iou >= self.refinement_iou_lower)
            & (base_iou < self.refinement_iou_upper)
        )
        active_float = active.to(refined.dtype)
        active_count = active_float.sum().clamp(min=1.0)

        l1 = F.l1_loss(refined, target_boxes, reduction="none").sum(dim=-1)
        giou = torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(refined),
                box_cxcywh_to_xyxy(target_boxes),
            )
        )
        loss_bbox = (l1 * active_float).sum() / active_count
        loss_giou = ((1.0 - giou) * active_float).sum() / active_count
        loss_non_degrade = F.relu(base_iou - refined_iou).sum() / max(
            float(num_boxes), 1.0
        )

        gate_target = torch.zeros_like(gate_logits)
        gate_target[idx] = active_float.to(dtype=gate_target.dtype)
        gate_weight = torch.full_like(
            gate_logits, self.refinement_background_weight
        )
        gate_weight[idx] = 1.0
        gate_loss = F.binary_cross_entropy_with_logits(
            gate_logits,
            gate_target,
            weight=gate_weight,
            reduction="sum",
        ) / gate_weight.sum().clamp(min=1.0)
        return {
            "loss_refinement_bbox": loss_bbox,
            "loss_refinement_giou": loss_giou,
            "loss_refinement_non_degrade": loss_non_degrade,
            "loss_refinement_gate": gate_loss,
        }


    def forward(self, outputs, targets, indices=None, t_weight=None,**kwargs):
        # print("f3",indices,t_weight,kwargs)
        outputs_without_aux = {
            k: v
            for k, v in outputs.items()
            if k
            not in {
                "aux_outputs",
                "enc_outputs",
                "flow_trajectory_outputs",
                "baseline_trajectory_residual",
                "distribution_aux_outputs",
            }
        }

        # Retrieve the matching between the outputs of the last layer and the targets
        if self.use_matcher:
            indices = self.matcher(outputs_without_aux, targets) ############### matcher
        # indices_test = self.matcher(outputs_without_aux, targets) ############### matcher

        ## count_loss
        # if self.use_count_loss:
        #     topk_values, topk_indexes = torch.topk(outputs['pred_logits'], 1, dim=-1) # [b, q, 1] , [b, q, 1]) 
        #     topk_indexes[topk_values.squeeze(-1)>-1]= 80 ###### debug ????
        #     topk_indexes = topk_indexes.squeeze(-1)
        #     pred_class_no = []
        #     gt_class_no = []
        #     for i in range(len(targets)):
        #         cur_img_pred = []
        #         cur_img_gt = []
        #         for j in range(0,80):
        #             cur_img_pred.append((topk_indexes[i] == j ).sum())
        #             cur_img_gt.append((targets[i]['labels']== j ).sum())
        #         pred_class_no.append(cur_img_pred)
        #         gt_class_no.append(cur_img_gt)
        #     pred_class_no = torch.tensor(pred_class_no)
        #     gt_class_no = torch.tensor(gt_class_no)

        #     count_loss = {}
        #     count_loss['loss_count'] =  nn.L1Loss()(pred_class_no.float(), gt_class_no.float())*20 ## L1Loss MSELoss HuberLoss


        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        main_indices = indices
        flow_step_outputs = list(outputs.get("flow_trajectory_outputs", []))
        flow_step_indices = []
        for step_output in flow_step_outputs:
            step_indices = main_indices
            if self.flow_step_rematching and self.matcher is not None:
                step_match_output = {
                    "pred_logits": step_output["pred_logits"],
                    "pred_boxes": step_output["pred_boxes"],
                }
                step_indices = self.matcher(step_match_output, targets)
            flow_step_indices.append(step_indices)
        trajectory_indices = [*flow_step_indices, main_indices]
        trajectory_outputs = [*flow_step_outputs, outputs]

        if flow_step_outputs:
            self._maybe_log_flow_diagnostics(
                trajectory_outputs, targets, trajectory_indices
            )

        if self.contrastive_loss:
            # losses.update(self.get_loss("contrastive", outputs, targets, indices, num_boxes, t_weight, **kwargs))
            self.losses.append("contrastive")

        for loss in self.losses:
            if loss == "flow_consistency":
                continue
            # kwargs = {}
            # if 'class_weights' in kwargs:
            #     t_weight = kwargs['class_weights']
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, t_weight, **kwargs))

        losses.update(self._loss_boundary_evidence_aux(outputs, targets, main_indices, num_boxes, t_weight))

        if "flow_consistency" in self.losses:
            losses.update(
                self.loss_flow_consistency(
                    outputs,
                    targets,
                    main_indices,
                    num_boxes,
                    trajectory_indices=(
                        trajectory_indices if flow_step_outputs else None
                    ),
                )
            )

        if "pred_boxes_refined" in outputs:
            losses.update(
                self.loss_trajectory_refinement(
                    outputs, targets, main_indices, num_boxes
                )
            )

        if "baseline_trajectory_residual" in outputs:
            trajectory_output = outputs["baseline_trajectory_residual"]
            trajectory_match_output = {
                "pred_logits": trajectory_output["pred_logits"],
                "pred_boxes": trajectory_output["pred_boxes"],
            }
            residual_indices = (
                self.matcher(trajectory_match_output, targets)
                if self.matcher is not None
                else main_indices
            )
            losses.update(
                self.loss_baseline_trajectory_residual(
                    trajectory_output,
                    targets,
                    residual_indices,
                    num_boxes,
                )
            )

        for step_index, (step_output, step_indices) in enumerate(
            zip(flow_step_outputs, flow_step_indices)
        ):
            auxiliary_losses = self._loss_boundary_evidence_aux(
                step_output, targets, step_indices, num_boxes, t_weight)
            losses.update({key + f"_flow_{step_index}": value
                           for key, value in auxiliary_losses.items()})
            step_losses = (
                self.losses
                if self.flow_step_losses is None
                else self.flow_step_losses
            )
            for loss in step_losses:
                if loss == "flow_consistency":
                    continue
                l_dict = self.get_loss(
                    loss,
                    step_output,
                    targets,
                    step_indices,
                    num_boxes,
                    t_weight,
                    **kwargs,
                )
                l_dict = {
                    key + f"_flow_{step_index}": value
                    for key, value in l_dict.items()
                }
                losses.update(l_dict)



        # if self.use_count_loss:
        #     losses.update(count_loss)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                if self.use_matcher:
                    indices = self.matcher(aux_outputs, targets) ############### matcher
                for loss in self.losses:
                    # Flow consistency is evaluated once for the full decoder trajectory.
                    # It is not a per-layer SetCriterion loss.
                    if loss == "flow_consistency":
                        continue
                    if loss == "transport_boxes":
                        continue
                    if loss == "distribution_refinement":
                        continue
                    if self.decoder_final_only and loss in (
                        "distribution",
                        "small_nwd",
                    ):
                        continue
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, t_weight, **kwargs)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        
        # Compute losses for two-stage deformable-detr
        if self.use_encoder_loss:
            if "enc_outputs" in outputs:
                enc_outputs = outputs["enc_outputs"]
                bin_targets = copy.deepcopy(targets)
                for bt in bin_targets:
                    bt["labels"] = torch.zeros_like(bt["labels"])
                indices = self.matcher(enc_outputs, bin_targets)
                for loss in self.losses:
                    if loss == "flow_consistency":
                        continue
                    if loss in (
                        "distribution",
                        "distribution_refinement",
                        "small_nwd",
                        "transport_boxes",
                        "flow_consistency",
                    ):
                        continue
                    kwargs={}
                    l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, None, **kwargs)
                    l_dict = {k + "_enc": v for k, v in l_dict.items()}
                    losses.update(l_dict)
        
        if self.contrastive_loss:
            self.losses.remove("contrastive")


        return losses
