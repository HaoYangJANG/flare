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

import copy
import math
from numbers import Real
import numpy as np
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from detrex.layers import MLP, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from detrex.utils import inverse_sigmoid

from detectron2.modeling import detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.data.detection_utils import convert_image_to_rgb


from .bbox_embedd import BBoxEmbed , ClassEmbed , TimeStepBlock
from .box_transport import (
    advance_flow_state,
    advance_proposal_flow_state,
    flow_to_diffusion_time,
    progressive_refined_endpoint,
    rectified_interpolate,
    sample_bounded_gaussian_boxes,
    sample_proposal_conditioned_boxes,
    uniform_flow_schedule,
)
from .local_geometry import LocalGeometryRefiner
from .bin_aligned_boundary import BinAlignedBoundaryEvidence
from .refined_score_calibrator import RefinedScoreCalibrator
from .quality_guidance import BoxQualityGuidance
from .flow_spatial_observation import FlowSpatialObservation, prepare_observation_context
from .flow_edge_geometry import FlowEdgeGeometry
from .distributional_box import (
    DistributionIntegral,
    cumulative_distribution_refinement,
    decode_distributional_box,
)
from .flow_output import fuse_query_aligned_flow_outputs
from .recurrent_flow_query import RecurrentFlowQueryUpdater
from .flow_distribution_conditioning import (
    FlowDistributionConditioner,
    build_distribution_transport_condition,
)
from .frequency_detail import (
    AdaptiveFrequencySelectionBlock,
    DetailQueryModulator,
    build_box_foreground_mask,
    foreground_mask_loss,
)
from .trajectory_refinement import (
    TrajectoryAwareLocalBoxRefinement,
    TrajectoryResidualFlowRefinement,
)
from functools import partial
from .ldm.modules.diffusionmodules.util import timestep_embedding , linear
from .ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps

from detectron2.layers import batched_nms


def select_neck_features(features, neck):
    """Keep auxiliary backbone levels out of a neck with fixed input levels."""
    return {feature_name: features[feature_name] for feature_name in neck.in_features}


def replace_first_feature(features, residual):
    """Apply a residual to the highest-resolution neck output without mutation."""
    return (features[0] + residual, *features[1:])


def build_ddim_time_pairs(total_timesteps, sampling_timesteps):
    """Build the exact integer schedule used by the legacy DDIM sampler."""
    if total_timesteps < 1 or sampling_timesteps < 1:
        raise ValueError("DDIM timestep counts must be positive")
    times = torch.linspace(0, total_timesteps - 1, steps=sampling_timesteps + 1)
    times = list(reversed(times.int().tolist()))
    return list(zip(times[:-1], times[1:]))


## Diffu_dino
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    if len(t.shape) > 1:
        t = t.squeeze()
    out = a.gather(-1, t.squeeze())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (
                torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)
####

class DIFFUSION_DINO(nn.Module):
    """Implement DAB-Deformable-DETR in `DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR
    <https://arxiv.org/abs/2203.03605>`_.

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/DINO>`_.

    Args:
        backbone (nn.Module): backbone module
        position_embedding (nn.Module): position embedding module
        neck (nn.Module): neck module to handle the intermediate outputs features
        transformer (nn.Module): transformer module
        embed_dim (int): dimension of embedding
        num_classes (int): Number of total categories.
        num_queries (int): Number of proposal dynamic anchor boxes in Transformer
        criterion (nn.Module): Criterion for calculating the total losses.
        pixel_mean (List[float]): Pixel mean value for image normalization.
            Default: [123.675, 116.280, 103.530].
        pixel_std (List[float]): Pixel std value for image normalization.
            Default: [58.395, 57.120, 57.375].
        aux_loss (bool): Whether to calculate auxiliary loss in criterion. Default: True.
        select_box_nums_for_evaluation (int): the number of topk candidates
            slected at postprocess for evaluation. Default: 300.
        device (str): Training device. Default: "cuda".
    """

    @staticmethod
    def _validate_flow_matching(use_rectified_flow, criterion):
        if use_rectified_flow and not getattr(criterion, "use_matcher", False):
            raise ValueError(
                "rectified flow requires Hungarian set matching because encoder "
                "query order is independent of the randomly permuted flow endpoints"
            )

    @staticmethod
    def _validate_flow_output_mode(flow_output_mode):
        valid_modes = {"ensemble", "final", "query_fusion", "step"}
        if flow_output_mode not in valid_modes:
            raise ValueError(
                f"flow_output_mode must be one of {sorted(valid_modes)}, "
                f"got {flow_output_mode!r}"
            )
        return flow_output_mode

    @staticmethod
    def _validate_cumulative_distribution_refinement(
        enabled,
        use_distributional_box,
        use_rectified_flow,
        distribution_detach_reference,
    ):
        if not enabled:
            return
        if not use_distributional_box:
            raise ValueError(
                "cumulative distribution refinement requires distributional boxes"
            )
        if not use_rectified_flow:
            raise ValueError(
                "cumulative distribution refinement is defined for rectified flow"
            )
        if not distribution_detach_reference:
            raise ValueError(
                "cumulative distribution refinement requires a detached reference"
            )

    @staticmethod
    def _build_cumulative_distribution_output_fields(
        coarse_boxes,
        refined_boxes,
        cumulative_logits,
        fixed_reference,
    ):
        if coarse_boxes.shape != refined_boxes.shape:
            raise ValueError("coarse and refined decoder boxes must align")
        if coarse_boxes.shape[:-1] != cumulative_logits.shape[:-1]:
            raise ValueError("decoder boxes and cumulative corner logits must align")
        if fixed_reference.shape != coarse_boxes.shape[1:]:
            raise ValueError("fixed distribution reference must match one decoder layer")
        auxiliary = [
            {
                "pred_boxes": refined_boxes[layer_index],
                "pred_corners": cumulative_logits[layer_index],
                "pred_dist_ref": fixed_reference,
            }
            for layer_index in range(coarse_boxes.shape[0] - 1)
        ]
        return {
            "pred_boxes": refined_boxes[-1],
            "pred_transport_boxes": coarse_boxes[-1],
            "pred_corners": cumulative_logits[-1],
            "pred_dist_ref": fixed_reference,
            "distribution_aux_outputs": auxiliary,
        }

    @staticmethod
    def _validate_flow_output_step(flow_output_step, flow_steps):
        if (
            isinstance(flow_output_step, bool)
            or not isinstance(flow_output_step, int)
            or not 1 <= flow_output_step <= flow_steps
        ):
            raise ValueError(
                "flow_output_step must be an integer in "
                f"[1, {flow_steps}], got {flow_output_step!r}"
            )
        return flow_output_step

    @staticmethod
    def _validate_flow_transport_mode(flow_transport_mode):
        valid_modes = {"refined", "coarse", "progressive_refined"}
        if flow_transport_mode not in valid_modes:
            raise ValueError(
                f"flow_transport_mode must be one of {sorted(valid_modes)}, "
                f"got {flow_transport_mode!r}"
            )
        return flow_transport_mode

    @staticmethod
    def _validate_flow_auxiliary_source(flow_auxiliary_source):
        valid_sources = {"first", "final"}
        if flow_auxiliary_source not in valid_sources:
            raise ValueError(
                f"flow_auxiliary_source must be one of {sorted(valid_sources)}, "
                f"got {flow_auxiliary_source!r}"
            )
        return flow_auxiliary_source

    @staticmethod
    def _validate_trajectory_refinement_source(trajectory_refinement_source):
        valid_sources = {"decoder_layer", "outer_flow"}
        if trajectory_refinement_source not in valid_sources:
            raise ValueError(
                "trajectory_refinement_source must be one of "
                f"{sorted(valid_sources)}, got {trajectory_refinement_source!r}"
            )
        return trajectory_refinement_source

    @staticmethod
    def _validate_recurrent_flow_query(
        enabled, use_rectified_flow, trajectory_aligned_flow, flow_steps
    ):
        if enabled and not (
            use_rectified_flow and trajectory_aligned_flow and flow_steps >= 2
        ):
            raise ValueError(
                "recurrent flow query requires aligned rectified flow with "
                "at least two flow steps"
            )

    @staticmethod
    def _validate_flow_distribution_conditioning(
        enabled,
        use_recurrent_flow_query,
        use_distributional_box,
        use_rectified_flow,
        trajectory_aligned_flow,
        flow_transport_mode,
        use_cumulative_distribution_refinement,
        use_trajectory_refinement,
    ):
        if enabled and not (
            use_recurrent_flow_query
            and use_distributional_box
            and use_rectified_flow
            and trajectory_aligned_flow
            and flow_transport_mode == "progressive_refined"
            and not use_cumulative_distribution_refinement
            and not use_trajectory_refinement
        ):
            raise ValueError(
                "flow distribution conditioning requires recurrent queries, "
                "distributional boxes, aligned rectified flow, progressive_refined "
                "transport, and no cumulative or trajectory refinement"
            )

    @staticmethod
    def _select_flow_query_state(decoder_state, query_count):
        if decoder_state.ndim != 3:
            raise ValueError("decoder state must have [batch, query, dim] shape")
        if query_count < 1 or query_count > decoder_state.shape[1]:
            raise ValueError("query count must select available decoder queries")
        return decoder_state[:, -query_count:]

    @staticmethod
    def _use_specialized_decoder_head(
        layer_index, num_layers, decoder_final_only
    ):
        return not decoder_final_only or layer_index == num_layers - 1

    def __init__(
        self,
        backbone: nn.Module,
        position_embedding: nn.Module,
        neck: nn.Module,
        transformer: nn.Module,
        embed_dim: int,
        num_classes: int,
        num_queries: int,
        criterion: nn.Module,
        pixel_mean: List[float] = [123.675, 116.280, 103.530],
        pixel_std: List[float] = [58.395, 57.120, 57.375],
        aux_loss: bool = True,
        select_box_nums_for_evaluation: int = 300,
        device="cuda",
        dn_number: int = 100,
        label_noise_ratio: float = 0.2,
        box_noise_scale: float = 1.0,
        input_format: Optional[str] = "RGB",
        vis_period: int = 0,
        old_schedule= True,
        noisy_gt = False,
        v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
        timesteps = 1000,
        sampling_steps = 25,
        beta_schedule = 'linear',
        given_betas=None,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        dino_type = "",
        use_rectified_flow: bool = False,
        flow_steps: int = 3,
        flow_output_mode: str = "ensemble",
        flow_output_step: int = 1,
        trajectory_aligned_flow: bool = False,
        decoder_final_only: bool = False,
        feature_enhance: bool = False,
        query_modulation: bool = False,
        frequency_source: str = "res2",
        frequency_in_channels: int = 256,
        frequency_mask_loss_weight: float = 0.25,
        small_object_threshold: float = 48.0,
        use_distributional_box: bool = False,
        reg_max: int = 32,
        reg_scale: float = 4.0,
        reg_up: float = 0.5,
        distribution_detach_reference: bool = False,
        use_trajectory_refinement: bool = False,
        refinement_roi_channels: int = 64,
        refinement_hidden_dim: int = 256,
        refinement_pooled_size: int = 3,
        refinement_max_center_scale: float = 0.10,
        refinement_max_size_log_scale: float = 0.15,
        refinement_detach_base: bool = True,
        use_baseline_trajectory_residual: bool = False,
        trajectory_frequency_source: str = "res2",
        trajectory_frequency_in_channels: int = 256,
        trajectory_frequency_channels: int = 64,
        trajectory_frequency_mask_loss_weight: float = 0.25,
        trajectory_residual_roi_channels: int = 64,
        trajectory_residual_hidden_dim: int = 256,
        trajectory_residual_pooled_size: int = 3,
        trajectory_residual_max_center_scale_per_step: float = 0.025,
        trajectory_residual_max_size_log_scale_per_step: float = 0.05,
        trajectory_residual_quality_margin: float = 0.02,
        trajectory_residual_score_gain: float = 0.25,
        flow_transport_mode: str = "refined",
        flow_auxiliary_source: str = "first",
        trajectory_refinement_source: str = "decoder_layer",
        flow_fusion_weights=None,
        use_cumulative_distribution_refinement: bool = False,
        use_recurrent_flow_query: bool = False,
        recurrent_query_hidden_dim: int = 256,
        recurrent_query_max_update_scale: float = 0.25,
        use_flow_distribution_conditioning: bool = False,
        use_proposal_conditioned_flow: bool = False,
        proposal_local_queries: int = 720,
        proposal_center_noise_scale: float = 0.25,
        proposal_size_noise_scale: float = 0.25,
        proposal_min_size_px: float = 2.0,
        proposal_flow_transport: str = "log_size",
        detail_readout_mode: str = "legacy",
        use_local_geometry_refinement: bool = False,
        flow_spatial_observation_mode: str = "off",
        flow_edge_geometry_mode: str = "off",
        use_bin_aligned_boundary_evidence: bool = False,
        use_refined_score_calibration: bool = False,
        use_bin_contrast_refinement: bool = False,
        quality_guidance: Optional[dict] = None,
        frequency_ignore_regions: bool = False,
        frequency_global_normalization: bool = False,
    ):
        super().__init__()
        assert dino_type == "diffu_det", "check init file ******************"

        self.flow_edge_geometry_mode = flow_edge_geometry_mode
        if flow_edge_geometry_mode not in ("off", "static", "distribution"):
            raise ValueError("edge geometry mode must be off, static or distribution")
        if flow_edge_geometry_mode != "off" and not (
            flow_spatial_observation_mode == "off"
            and use_rectified_flow and trajectory_aligned_flow and flow_steps == 3
            and isinstance(flow_steps, int) and not isinstance(flow_steps, bool)
            and use_proposal_conditioned_flow and proposal_flow_transport == "absolute"
            and flow_transport_mode == "progressive_refined"
            and use_distributional_box and distribution_detach_reference
            and reg_max == 32 and reg_scale == 4 and decoder_final_only
            and embed_dim == 256 and frequency_in_channels == 256
            and transformer.decoder.num_layers == 6
            and getattr(transformer.decoder, "return_intermediate", False)
            and not any((use_recurrent_flow_query, use_flow_distribution_conditioning,
                         use_cumulative_distribution_refinement, use_trajectory_refinement,
                         use_baseline_trajectory_residual, use_local_geometry_refinement))
        ):
            raise ValueError("edge geometry requires the supported six-layer intermediate 256-channel aligned three-step proposal absolute PDR distribution recipe without spatial observation")

        self.flow_spatial_observation_mode = flow_spatial_observation_mode
        if flow_spatial_observation_mode not in ("off", "static", "distribution"):
            raise ValueError("spatial observation mode must be off, static or distribution")
        if flow_spatial_observation_mode != "off" and not (
            use_rectified_flow and trajectory_aligned_flow and flow_steps == 3
            and isinstance(flow_steps, int) and not isinstance(flow_steps, bool)
            and use_proposal_conditioned_flow and proposal_flow_transport == "absolute"
            and flow_transport_mode == "progressive_refined"
            and use_distributional_box and distribution_detach_reference
            and reg_max == 32 and reg_scale == 4 and decoder_final_only
            and embed_dim == 256 and frequency_in_channels == 256
            and transformer.decoder.num_layers == 6
            and not any((use_recurrent_flow_query, use_flow_distribution_conditioning,
                         use_cumulative_distribution_refinement, use_trajectory_refinement,
                         use_baseline_trajectory_residual, use_local_geometry_refinement))
        ):
            raise ValueError("spatial observation requires the supported six-layer 256-channel aligned three-step proposal absolute PDR distribution recipe")

        self.use_local_geometry_refinement = bool(use_local_geometry_refinement)
        if self.use_local_geometry_refinement and not (
            use_rectified_flow and trajectory_aligned_flow and use_distributional_box
            and distribution_detach_reference and decoder_final_only
            and not use_cumulative_distribution_refinement
            and not use_trajectory_refinement and not use_baseline_trajectory_residual
        ):
            raise ValueError(
                "local geometry requires aligned rectified flow, distributional boxes, "
                "detached references, final-only decoder heads, and no cumulative "
                "or alternate trajectory refinement"
            )

        self.use_bin_aligned_boundary_evidence = bool(use_bin_aligned_boundary_evidence)
        self.use_refined_score_calibration = bool(use_refined_score_calibration)
        self.use_bin_contrast_refinement = bool(use_bin_contrast_refinement)
        if (self.use_refined_score_calibration or self.use_bin_contrast_refinement) and not (
            self.use_bin_aligned_boundary_evidence and flow_steps == 3
            and isinstance(flow_steps, int) and not isinstance(flow_steps, bool)
        ):
            raise ValueError("boundary closed-loop features require the existing three-step Bin path")
        if self.use_bin_aligned_boundary_evidence and not (
            use_rectified_flow and trajectory_aligned_flow and use_distributional_box
            and distribution_detach_reference and decoder_final_only
            and flow_transport_mode == "progressive_refined"
            and flow_spatial_observation_mode == "off" and flow_edge_geometry_mode == "off"
            and not any((use_local_geometry_refinement, use_cumulative_distribution_refinement,
                         use_recurrent_flow_query, use_flow_distribution_conditioning,
                         use_trajectory_refinement, use_baseline_trajectory_residual))
        ):
            raise ValueError("bin-aligned boundary requires aligned distributional progressive refined flow with detached references and final-only decoder heads, without alternate geometry or trajectory branches")

        self.use_proposal_conditioned_flow = bool(use_proposal_conditioned_flow)
        self.proposal_local_queries = proposal_local_queries
        self.proposal_center_noise_scale = proposal_center_noise_scale
        self.proposal_size_noise_scale = proposal_size_noise_scale
        self.proposal_min_size_px = proposal_min_size_px
        self.proposal_flow_transport = proposal_flow_transport
        if self.use_proposal_conditioned_flow:
            if not (
                use_rectified_flow
                and trajectory_aligned_flow
                and isinstance(flow_steps, int)
                and not isinstance(flow_steps, bool)
                and flow_steps >= 2
                and not transformer.learnt_init_query
            ):
                raise ValueError(
                    "proposal-conditioned flow requires aligned rectified flow, "
                    "at least two integer flow steps, and learnt_init_query=False"
                )
            if any((use_recurrent_flow_query, use_flow_distribution_conditioning,
                    use_trajectory_refinement, use_cumulative_distribution_refinement,
                    use_baseline_trajectory_residual)):
                raise ValueError(
                    "proposal-conditioned flow does not support recurrent queries, "
                    "distribution conditioning, trajectory refinement, cumulative "
                    "refinement, or baseline trajectory residuals"
                )
            if (isinstance(proposal_local_queries, bool)
                    or not isinstance(proposal_local_queries, int)
                    or not 1 <= proposal_local_queries <= num_queries):
                raise ValueError("proposal_local_queries must be an integer in [1, num_queries]")
            for name, value in (
                ("proposal_center_noise_scale", proposal_center_noise_scale),
                ("proposal_size_noise_scale", proposal_size_noise_scale),
            ):
                if (isinstance(value, bool) or not isinstance(value, Real)
                        or not math.isfinite(float(value)) or value < 0):
                    raise ValueError(f"{name} must be finite and non-negative")
            if (isinstance(proposal_min_size_px, bool)
                    or not isinstance(proposal_min_size_px, Real)
                    or not math.isfinite(float(proposal_min_size_px))
                    or proposal_min_size_px <= 0):
                raise ValueError("proposal_min_size_px must be finite and positive")
            if proposal_flow_transport not in ("absolute", "log_size"):
                raise ValueError("proposal_flow_transport must be absolute or log_size")

        # define backbone and position embedding module
        self.backbone = backbone
        self.position_embedding = position_embedding

        # define neck module
        self.neck = neck

        # number of dynamic anchor boxes and embedding dimension
        self.num_queries = num_queries
        self.embed_dim = embed_dim

        # define transformer module
        self.transformer = transformer

        # define classification head and box head
        self.class_embed = nn.Linear(embed_dim, num_classes)
        self.bbox_embed = MLP(embed_dim, embed_dim, 4, 3)
        self.num_classes = num_classes

        self.use_rectified_flow = use_rectified_flow
        self._validate_flow_matching(self.use_rectified_flow, criterion)
        self.flow_steps = flow_steps
        if self.use_rectified_flow and self.flow_steps < 1:
            raise ValueError("flow_steps must be positive")
        self.flow_output_mode = self._validate_flow_output_mode(flow_output_mode)
        self.flow_fusion_weights = (
            None
            if flow_fusion_weights is None
            else tuple(float(weight) for weight in flow_fusion_weights)
        )
        if self.flow_fusion_weights is not None:
            if len(self.flow_fusion_weights) != self.flow_steps:
                raise ValueError(
                    "flow fusion weights must match the configured flow steps"
                )
            if any(
                not math.isfinite(weight) or weight <= 0
                for weight in self.flow_fusion_weights
            ):
                raise ValueError("flow fusion weights must be finite and positive")
        self.flow_output_step = self._validate_flow_output_step(
            flow_output_step, self.flow_steps
        )
        self.trajectory_aligned_flow = trajectory_aligned_flow
        self.use_recurrent_flow_query = bool(use_recurrent_flow_query)
        self._validate_recurrent_flow_query(
            self.use_recurrent_flow_query,
            self.use_rectified_flow,
            self.trajectory_aligned_flow,
            self.flow_steps,
        )
        self.flow_transport_mode = self._validate_flow_transport_mode(
            flow_transport_mode
        )
        self.flow_auxiliary_source = self._validate_flow_auxiliary_source(
            flow_auxiliary_source
        )
        self.trajectory_refinement_source = (
            self._validate_trajectory_refinement_source(
                trajectory_refinement_source
            )
        )
        if self.flow_transport_mode in {"coarse", "progressive_refined"} and not use_distributional_box:
            raise ValueError(
                f"{self.flow_transport_mode} flow transport requires "
                "distributional box regression"
            )
        if self.flow_transport_mode == "progressive_refined" and not (
            self.use_rectified_flow and self.trajectory_aligned_flow
        ):
            raise ValueError(
                "progressive refined transport requires aligned rectified flow"
            )
        if self.flow_auxiliary_source == "final" and not (
            self.use_rectified_flow and self.trajectory_aligned_flow
        ):
            raise ValueError(
                "final flow auxiliary supervision requires aligned rectified flow"
            )
        self.decoder_final_only = decoder_final_only
        self.use_trajectory_refinement = use_trajectory_refinement
        if self.use_trajectory_refinement:
            if not self.use_rectified_flow:
                raise ValueError(
                    "trajectory refinement requires rectified flow"
                )
            if not self.trajectory_aligned_flow:
                raise ValueError(
                    "trajectory refinement requires aligned flow training"
                )
            if self.flow_output_mode != "final":
                raise ValueError(
                    "trajectory refinement requires final flow output"
                )
            if (
                self.trajectory_refinement_source == "outer_flow"
                and self.flow_steps < 2
            ):
                raise ValueError(
                    "outer-flow trajectory refinement requires at least two flow steps"
                )
            self.trajectory_refiner = TrajectoryAwareLocalBoxRefinement(
                in_channels=embed_dim,
                roi_channels=refinement_roi_channels,
                hidden_dim=refinement_hidden_dim,
                pooled_size=refinement_pooled_size,
                max_center_scale=refinement_max_center_scale,
                max_size_log_scale=refinement_max_size_log_scale,
                detach_base=refinement_detach_base,
            )

        self.use_baseline_trajectory_residual = use_baseline_trajectory_residual
        self.trajectory_frequency_source = trajectory_frequency_source
        self.trajectory_frequency_mask_loss_weight = (
            trajectory_frequency_mask_loss_weight
        )
        self.trajectory_residual_quality_margin = trajectory_residual_quality_margin
        self.trajectory_residual_score_gain = trajectory_residual_score_gain
        if self.use_baseline_trajectory_residual:
            if self.use_rectified_flow:
                raise ValueError(
                    "baseline trajectory residual is defined only for DDIM diffusion"
                )
            if sampling_steps != 3:
                raise ValueError(
                    "baseline trajectory residual requires sampling_steps=3 so training "
                    "and inference use the same outer trajectory"
                )
            if not 0.0 <= trajectory_residual_quality_margin < 1.0:
                raise ValueError(
                    "trajectory residual quality margin must lie in [0, 1)"
                )
            if trajectory_residual_max_center_scale_per_step <= 0:
                raise ValueError(
                    "trajectory residual center scale must be positive"
                )
            if trajectory_residual_max_size_log_scale_per_step <= 0:
                raise ValueError(
                    "trajectory residual size scale must be positive"
                )
            if trajectory_residual_score_gain < 0:
                raise ValueError("trajectory residual score gain must be non-negative")
            self.trajectory_frequency_selector = AdaptiveFrequencySelectionBlock(
                in_channels=trajectory_frequency_in_channels,
                embed_dim=trajectory_frequency_channels,
            )
            self.trajectory_residual_refiner = TrajectoryResidualFlowRefinement(
                in_channels=trajectory_frequency_channels,
                query_dim=embed_dim,
                roi_channels=trajectory_residual_roi_channels,
                hidden_dim=trajectory_residual_hidden_dim,
                pooled_size=trajectory_residual_pooled_size,
                max_center_scale_per_step=(
                    trajectory_residual_max_center_scale_per_step
                ),
                max_size_log_scale_per_step=(
                    trajectory_residual_max_size_log_scale_per_step
                ),
                quality_improvement_margin=trajectory_residual_quality_margin,
                detach_base=True,
            )

        self.feature_enhance = feature_enhance
        self.query_modulation = query_modulation
        self.frequency_source = frequency_source
        self.frequency_mask_loss_weight = frequency_mask_loss_weight
        self.frequency_ignore_regions = frequency_ignore_regions
        if frequency_global_normalization and not frequency_ignore_regions:
            raise ValueError("frequency_global_normalization requires frequency_ignore_regions=True")
        self.frequency_global_normalization = frequency_global_normalization
        self.small_object_threshold = small_object_threshold
        self.detail_readout_mode = detail_readout_mode
        if detail_readout_mode not in ("legacy", "robust_local"):
            raise ValueError("detail_readout_mode must be legacy or robust_local")
        if detail_readout_mode == "robust_local" and not query_modulation:
            raise ValueError("robust_local detail readout requires query_modulation=True")
        if self.feature_enhance or self.query_modulation:
            self.frequency_selector = AdaptiveFrequencySelectionBlock(
                in_channels=frequency_in_channels,
                embed_dim=embed_dim,
            )
        if self.feature_enhance:
            self.frequency_fusion = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
            nn.init.zeros_(self.frequency_fusion.weight)
            nn.init.zeros_(self.frequency_fusion.bias)
        if self.query_modulation:
            self.detail_query_modulator = DetailQueryModulator(
                embed_dim=embed_dim,
                threshold_px=small_object_threshold,
                readout_mode=detail_readout_mode,
            )

        self.use_distributional_box = use_distributional_box
        self.distribution_detach_reference = bool(distribution_detach_reference)
        self.use_cumulative_distribution_refinement = bool(
            use_cumulative_distribution_refinement
        )
        self.use_flow_distribution_conditioning = bool(
            use_flow_distribution_conditioning
        )
        self._validate_flow_distribution_conditioning(
            self.use_flow_distribution_conditioning,
            self.use_recurrent_flow_query,
            self.use_distributional_box,
            self.use_rectified_flow,
            self.trajectory_aligned_flow,
            self.flow_transport_mode,
            self.use_cumulative_distribution_refinement,
            self.use_trajectory_refinement,
        )
        self._validate_cumulative_distribution_refinement(
            self.use_cumulative_distribution_refinement,
            self.use_distributional_box,
            self.use_rectified_flow,
            self.distribution_detach_reference,
        )
        if self.distribution_detach_reference and not self.use_distributional_box:
            raise ValueError(
                "detached distribution references require distributional box regression"
            )
        self.reg_scale = reg_scale

        # where to calculate auxiliary loss in criterion
        self.aux_loss = aux_loss
        self.criterion = criterion

        # denoising
        self.label_enc = nn.Embedding(num_classes +1 , embed_dim) ##### +1  for Diffu_Dino
        self.dn_number = dn_number
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # normalizer for input raw images
        self.device = device
        pixel_mean = torch.Tensor(pixel_mean).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(pixel_std).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        # initialize weights
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for _, neck_layer in self.neck.named_modules():
            if isinstance(neck_layer, nn.Conv2d):
                nn.init.xavier_uniform_(neck_layer.weight, gain=1)
                nn.init.constant_(neck_layer.bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = transformer.decoder.num_layers + 1
        self.class_embed = nn.ModuleList([copy.deepcopy(self.class_embed) for i in range(num_pred)])
        self.bbox_embed = nn.ModuleList([copy.deepcopy(self.bbox_embed) for i in range(num_pred)])
        if self.use_distributional_box:
            self.distribution_integral = DistributionIntegral(reg_max, reg_up, reg_scale)
            self.corner_embed = nn.ModuleList(
                [
                    MLP(embed_dim, embed_dim, 4 * (reg_max + 1), 3)
                    for _ in range(transformer.decoder.num_layers)
                ]
            )
            for corner_head in self.corner_embed:
                nn.init.zeros_(corner_head.layers[-1].weight)
                nn.init.zeros_(corner_head.layers[-1].bias)
        nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)

        # two-stage
        self.transformer.decoder.class_embed = self.class_embed
        self.transformer.decoder.bbox_embed = self.bbox_embed

        # hack implementation for two-stage
        for bbox_embed_layer in self.bbox_embed:
            nn.init.constant_(bbox_embed_layer.layers[-1].bias.data[2:], 0.0)

        # set topk boxes selected for inference
        self.select_box_nums_for_evaluation = select_box_nums_for_evaluation

        # the period for visualizing training samples
        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"


        ## Diffu_dino
        self.v_posterior = v_posterior
        self.num_timesteps = timesteps
        self.sampling_timesteps = sampling_steps
        self.ddim_sampling_eta = 0.0
        
        self.parameterization = "x0"
        # self.register_schedule_old(timesteps)
        if old_schedule:
            print("*************** schedule old")
            self.register_schedule_old(timesteps)
        else:
            print("*************** schedule ldm")
            self.register_schedule_ldm(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)
        # print("******************************* old")

        self.parameterization = "x0"
        # self.register_schedule_ldm(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
        #                        linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)
        # self.make_schedule()
        self.noisy_gt = noisy_gt
        print("$$$$$$$$$$$$$$$ diffu det noise")
        self.scale = 2

        # self.drop_gt = drop_gt
        self.box_renewal = True
        self.use_ensemble = True
        self.use_nms = True

        # Build the isolated experiment module after all existing parameters so
        # enabling it cannot perturb their seeded initialization.
        if self.use_recurrent_flow_query:
            self.recurrent_flow_query_updater = RecurrentFlowQueryUpdater(
                embed_dim=embed_dim,
                hidden_dim=recurrent_query_hidden_dim,
                flow_steps=self.flow_steps,
                max_update_scale=recurrent_query_max_update_scale,
            )
        if self.use_flow_distribution_conditioning:
            with torch.random.fork_rng(devices=[]):
                self.flow_distribution_conditioner = FlowDistributionConditioner(
                    embed_dim
                )
        if self.use_local_geometry_refinement:
            with torch.random.fork_rng(devices=[]):
                self.local_geometry_refiner = LocalGeometryRefiner(
                    frequency_in_channels, embed_dim, reg_max
                )
        if self.flow_spatial_observation_mode != "off":
            with torch.random.fork_rng(devices=[]):
                self.transformer.decoder.flow_spatial_observation = FlowSpatialObservation(
                    frequency_in_channels, embed_dim, self.flow_spatial_observation_mode
                )
        if self.flow_edge_geometry_mode != "off":
            with torch.random.fork_rng(devices=[]):
                self.transformer.decoder.flow_edge_geometry = FlowEdgeGeometry(
                    frequency_in_channels, embed_dim, self.flow_edge_geometry_mode
                )
        if self.use_bin_aligned_boundary_evidence:
            with torch.random.fork_rng(devices=[]):
                self.bin_aligned_boundary_evidence = BinAlignedBoundaryEvidence(
                    frequency_in_channels, embed_dim,
                    use_contrast_refinement=self.use_bin_contrast_refinement,
                )
        if self.use_refined_score_calibration:
            with torch.random.fork_rng(devices=[]):
                self.refined_score_calibrator = RefinedScoreCalibrator(embed_dim, num_classes)

        self.quality_guidance = None
        if quality_guidance is not None:
            if not (self.use_rectified_flow and self.trajectory_aligned_flow):
                raise ValueError("quality guidance requires the aligned full-flow path")
            unsupported = [name for name in (
                "use_cumulative_distribution_refinement", "use_recurrent_flow_query",
                "use_flow_distribution_conditioning", "use_trajectory_refinement",
                "use_local_geometry_refinement", "use_baseline_trajectory_residual",
            ) if getattr(self, name, False)]
            unsupported += [name for name in (
                "flow_spatial_observation_mode", "flow_edge_geometry_mode",
            ) if getattr(self, name, "off") != "off"]
            if unsupported:
                raise ValueError("quality guidance has no validated full-flow integration with: " + ", ".join(unsupported))
            if quality_guidance.get("score_power", 0.) != 0.:
                raise ValueError("full-flow guidance retains the original refined score calibrator; score_power must be zero")
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(int(quality_guidance.get("seed", 42)))
                self.quality_guidance = BoxQualityGuidance(embed_dim, num_classes, **quality_guidance)

    @torch.no_grad()
    def _quality_context(self, feature, image_sizes, padded_size, boundary_evidence_feature=None):
        field = getattr(self, "quality_guidance", None)
        if field is None or not field.is_ready() or field.guidance_scale() <= 0:
            return None
        if feature is None:
            raise ValueError("full-flow quality guidance requires the detached res2 feature")
        return dict(projected=field.project(feature, image_sizes, padded_size),
                    image_sizes=image_sizes, padded_size=padded_size,
                    boundary_evidence_feature=boundary_evidence_feature)

    def _quality_output(self, output, context, step_index, for_inference=False):
        """Guide the refined endpoint, then use the unchanged progressive mixer.

        Raw outputs retain original detector/boundary/calibration supervision.
        At inference the original calibrator sees the final corrected geometry.
        """
        boxes = output["pred_boxes"]
        scores = output.get("pred_score_logits", output["pred_logits"])
        if context is None:
            return boxes, scores, self._flow_transport_endpoint(output, step_index=step_index)
        guided, _ = self.quality_guidance.guide(
            context["projected"], output["_quality_query"], output["pred_logits"], boxes,
            output["_quality_progress"], context["image_sizes"], context["padded_size"])
        transport = self._flow_transport_endpoint({**output, "pred_boxes": guided}, step_index=step_index)
        if for_inference and getattr(self, "use_refined_score_calibration", False) and guided is not boxes:
            scores = self.refined_score_calibrator(
                output["_quality_query"], output["pred_logits"], output["pred_corners"],
                output["pred_dist_ref"], output.get("pred_boundary_base_boxes", boxes), guided,
                context["boundary_evidence_feature"], context["image_sizes"], context["padded_size"],
                self.distribution_integral.project, output["_quality_progress"],
            ).to(output["pred_logits"].dtype)
        return guided, scores, transport

    def _quality_losses(self, outputs, feature, targets, image_sizes, padded_size):
        losses = {}
        for output in outputs:
            with torch.no_grad():
                matched = self.criterion.matcher(
                    {key: output[key].detach() for key in ("pred_logits", "pred_boxes")}, targets)
            current = self.quality_guidance.training_losses(
                feature, output["_quality_query"], output["pred_logits"], output["pred_boxes"],
                output["_quality_progress"], image_sizes, padded_size, targets, matched)
            for name, value in current.items():
                losses[name] = losses.get(name, 0.) + value / len(outputs)
        return losses



    def register_schedule_old(self,time_steps):
        betas = cosine_beta_schedule(time_steps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # snr = alphas_cumprod / (1 - alphas_cumprod)
        # maybe_clipped_snr = snr.clone().clamp(max = 2)
        # self.register_buffer('loss_weight', maybe_clipped_snr,persistent=False)

        lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('loss_weight', lvlb_weights, persistent=False)

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    def register_schedule_ldm(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('loss_weight', lvlb_weights, persistent=False)
        assert not torch.isnan(self.loss_weight).all()

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    # def apply_box_noise(self,boxes: torch.Tensor,t):
    #     """
    #     Args:
    #         boxes (torch.Tensor): Bounding boxes in format ``(x_c, y_c, w, h)`` with
    #             shape ``(num_boxes, 4)``
    #         box_noise_scale (float): Scaling factor for box noising. Default: 0.4.
    #     """
    #     # print(boxes)
    #     # t = t = torch.full((1,), t).long() 
    #     # print(boxes.shape)
    #     # diff = torch.zeros_like(boxes)
    #     # diff[:, :2] = boxes[:, :2] 
    #     # diff[:, 2:] = boxes[:, 2:]
    #     # boxes_noise = (torch.randn_like(boxes)+2)/4#torch.mul(((torch.randn_like(boxes)+1.5)/3), diff)
    #     boxes_noise = torch.randn_like(boxes)
    #     boxes = inverse_sigmoid(boxes)
    #     # print("boxes_noise",boxes_noise)
    #     # # boxes_noise =
    #     # print(diff)
    #     # print('diff' , diff)
    #     # print('boxes_noise' , boxes_noise)
    #     boxes = self.q_sample(x_start = boxes , t = t, noise = boxes_noise)
    #     boxes = boxes.sigmoid()
    #     # boxes = boxes.clamp(min=0.0, max=1.0)
    #     return boxes , boxes_noise.sigmoid()

    def apply_box_noise(self,boxes: torch.Tensor,t):
        """
        Args:
            boxes (torch.Tensor): Bounding boxes in format ``(x_c, y_c, w, h)`` with
                shape ``(num_boxes, 4)``
            box_noise_scale (float): Scaling factor for box noising. Default: 0.4.
        """
        noise = torch.randn_like(boxes)
        
        gt_boxes = (boxes * 2. - 1.) * self.scale
        # x_start = torch.repeat_interleave(gt_boxes, repeat_tensor, dim=0)

        # noise sample
        x = self.q_sample(x_start=gt_boxes, t=t, noise=noise)

        x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x = ((x / self.scale) + 1) / 2.

        diff_boxes = x
        return diff_boxes, noise

    def forward(self, batched_inputs):
        """Forward function of `DINO` which excepts a list of dict as inputs.

        Args:
            batched_inputs (List[dict]): A list of instance dict, and each instance dict must consists of:
                - dict["image"] (torch.Tensor): The unnormalized image tensor.
                - dict["height"] (int): The original image height.
                - dict["width"] (int): The original image width.
                - dict["instance"] (detectron2.structures.Instances):
                    Image meta informations and ground truth boxes and labels during training.
                    Please refer to
                    https://detectron2.readthedocs.io/en/latest/modules/structures.html#detectron2.structures.Instances
                    for the basic usage of Instances.

        Returns:
            dict: Returns a dict with the following elements:
                - dict["pred_logits"]: the classification logits for all queries (anchor boxes in DAB-DETR).
                            with shape ``[batch_size, num_queries, num_classes]``
                - dict["pred_boxes"]: The normalized boxes coordinates for all queries in format
                    ``(x, y, w, h)``. These values are normalized in [0, 1] relative to the size of
                    each individual image (disregarding possible padding). See PostProcess for information
                    on how to retrieve the unnormalized bounding box.
                - dict["aux_outputs"]: Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        images = self.preprocess_image(batched_inputs)

        if self.training:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_ones(batch_size, H, W)
            for img_id in range(batch_size):
                img_h, img_w = batched_inputs[img_id]["instances"].image_size
                img_masks[img_id, :img_h, :img_w] = 0
        else:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_ones(batch_size, H, W)
            for img_id in range(batch_size):
                img_h, img_w = images.image_sizes[img_id]
                if self.use_rectified_flow:
                    img_masks[img_id, :img_h, :img_w] = 0
                else:
                    img_masks[img_id, :(img_h-1), :(img_w-1)] = 0
        # else:
        #     batch_size, _, H, W = images.tensor.shape
        #     img_masks = images.tensor.new_zeros(batch_size, H, W)

        # original features
        features = self.backbone(images.tensor)  # output feature dict
        quality_feature_options = (
            {"quality_feature": features["res2"]} if self.quality_guidance is not None else {}
        )

        observation_feature = None
        observation_options = {}
        if getattr(self, "flow_spatial_observation_mode", "off") != "off":
            observation_feature = self.transformer.decoder.flow_spatial_observation.project_feature(features["res2"])
            observation_options["observation_feature"] = observation_feature

        edge_geometry_feature = None
        edge_geometry_options = {}
        if getattr(self, "flow_edge_geometry_mode", "off") != "off":
            edge_geometry_feature = self.transformer.decoder.flow_edge_geometry.project_feature(features["res2"])
            edge_geometry_options["edge_geometry_feature"] = edge_geometry_feature

        local_geometry_feature = None
        local_geometry_options = {}
        if getattr(self, "use_local_geometry_refinement", False):
            local_geometry_feature = self.local_geometry_refiner.project_feature(features["res2"])
            local_geometry_options["local_geometry_feature"] = local_geometry_feature

        boundary_evidence_options = {}
        if getattr(self, "use_bin_aligned_boundary_evidence", False):
            boundary_evidence_options["boundary_evidence_feature"] = (
                self.bin_aligned_boundary_evidence.project_feature(features["res2"])
            )

        detail_feature = None
        foreground_logits = None
        if self.feature_enhance or self.query_modulation:
            if self.frequency_source not in features:
                raise KeyError(
                    f"frequency source '{self.frequency_source}' is unavailable; "
                    "add it to backbone.out_features"
                )
            detail_feature, foreground_logits, _ = self.frequency_selector(
                features[self.frequency_source]
            )

        trajectory_detail_feature = None
        trajectory_foreground_logits = None
        if self.use_baseline_trajectory_residual:
            if self.trajectory_frequency_source not in features:
                raise KeyError(
                    f"trajectory frequency source '{self.trajectory_frequency_source}' "
                    "is unavailable; add it to backbone.out_features"
                )
            (
                trajectory_detail_feature,
                trajectory_foreground_logits,
                _,
            ) = self.trajectory_frequency_selector(
                features[self.trajectory_frequency_source].detach()
            )

        # project backbone features to the reuired dimension of transformer
        # we use multi-scale features in DINO
        multi_level_feats = self.neck(select_neck_features(features, self.neck))
        if self.feature_enhance:
            detail_for_memory = F.interpolate(
                detail_feature,
                size=multi_level_feats[0].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            multi_level_feats = replace_first_feature(
                multi_level_feats, self.frequency_fusion(detail_for_memory)
            )
        multi_level_masks = []
        multi_level_position_embeddings = []
        for feat in multi_level_feats:
            multi_level_masks.append(
                F.interpolate(img_masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0)
            )
            multi_level_position_embeddings.append(self.position_embedding(multi_level_masks[-1]))

        # denoising preprocessing
        # prepare label query embedding
        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            ## Diffu_dino
            # targets = self.prepare_targets(gt_instances)
            targets, new_targets_diffusion= self.prepare_targets(gt_instances)

            input_query_label, input_query_bbox, attn_mask, dn_meta = self.prepare_for_cdn(
                targets,
                dn_number=self.dn_number,
                label_noise_ratio=self.label_noise_ratio,
                box_noise_scale=self.box_noise_scale,
                num_queries=self.num_queries,
                num_classes=self.num_classes,
                hidden_dim=self.embed_dim,
                label_enc=self.label_enc,
            )
            if self.use_rectified_flow:
                noisy_queries, init_query_points, queries, t, indices, num_gts = (
                    self.process_flow_targets(targets, new_targets_diffusion)
                )
            else:
                noisy_queries, init_query_points , queries, t , indices, num_gts = self.process_targets(targets, new_targets_diffusion)
        else:
            # gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            # targets = self.prepare_targets(gt_instances)
            # input_query_label, input_query_bbox, attn_mask, dn_meta = self.prepare_for_cdn(
            #     targets,
            #     dn_number=self.dn_number,
            #     label_noise_ratio=self.label_noise_ratio,
            #     box_noise_scale=self.box_noise_scale,
            #     num_queries=self.num_queries,
            #     num_classes=self.num_classes,
            #     hidden_dim=self.embed_dim,
            #     label_enc=self.label_enc,
            # )
            # input_query_label, input_query_bbox, attn_mask, dn_meta = None, None, None, None
            if self.use_rectified_flow:
                return self.flow_sample(
                    batched_inputs,
                    multi_level_feats,
                    multi_level_masks,
                    multi_level_position_embeddings,
                    images,
                    detail_feature,
                    foreground_logits,
                    **local_geometry_options,
                    **observation_options,
                    **edge_geometry_options,
                    **boundary_evidence_options,
                    **quality_feature_options,
                )
            if self.use_baseline_trajectory_residual:
                return self.ddim_sample_with_trajectory_residual(
                    batched_inputs,
                    multi_level_feats,
                    multi_level_masks,
                    multi_level_position_embeddings,
                    images,
                    trajectory_detail_feature,
                )
            return self.ddim_sample(batched_inputs, multi_level_feats, multi_level_masks, multi_level_position_embeddings,images)

        
        query_embeds = (input_query_label, input_query_bbox)
        query_embeds_diffusion = (noisy_queries, init_query_points)

        # feed into transformer
        source_options = {}
        if observation_feature is not None:
            source_options["observation_context"] = prepare_observation_context(
                observation_feature, images.image_sizes, images.tensor.shape[-2:], None, t
            )
        if edge_geometry_feature is not None:
            source_options["edge_geometry_context"] = prepare_observation_context(
                edge_geometry_feature, images.image_sizes, images.tensor.shape[-2:], None, t
            )
        if getattr(self, "use_proposal_conditioned_flow", False):
            source_options["source_initializer"] = partial(
                self._initialize_proposal_flow,
                image_sizes=torch.as_tensor(
                    images.image_sizes, device=self.device, dtype=torch.float32
                ),
            )
        transformer_outputs = self.transformer(
            multi_level_feats,
            multi_level_masks,
            multi_level_position_embeddings,
            query_embeds,
            attn_masks=[attn_mask, None],
            query_embeds_diffusion = query_embeds_diffusion,
            time_steps=(
                flow_to_diffusion_time(t, self.num_timesteps)
                if self.use_rectified_flow
                else t
            ),
            return_context=(
                self.use_rectified_flow or self.use_baseline_trajectory_residual
            ),
            **source_options,
        )
        geometry_residuals = None
        if self.use_rectified_flow or self.use_baseline_trajectory_residual:
            (
                inter_states,
                init_reference,
                inter_references,
                enc_state,
                enc_reference,
                flow_context,
            ) = transformer_outputs
            geometry_residuals = flow_context.get("geometry_residuals")
            if getattr(self, "use_proposal_conditioned_flow", False):
                init_query_points = flow_context["flow_initial_state"]
        else:
            (
                inter_states,
                init_reference,
                inter_references,
                enc_state,
                enc_reference,
            ) = transformer_outputs
        # hack implementation for distributed training
        inter_states[0] += self.label_enc.weight[0, 0] * 0.0

        # Calculate output coordinates and classes.
        outputs_classes = []
        outputs_coords = []
        outputs_coarse_coords = []
        outputs_corners = []
        outputs_distribution_refs = []
        cumulative_corner_deltas = []
        cumulative_fixed_reference = None
        final_corner_logits = None
        final_distribution_reference = None
        boundary_output_fields = {}
        image_sizes_tensor = torch.as_tensor(images.image_sizes, device=self.device, dtype=torch.float32)
        for lvl in range(inter_states.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            layer_state = inter_states[lvl]
            use_specialized_head = self._use_specialized_decoder_head(
                lvl, inter_states.shape[0], self.decoder_final_only
            )
            if self.query_modulation and use_specialized_head:
                layer_state = self.detail_query_modulator(
                    layer_state,
                    detail_feature,
                    foreground_logits.sigmoid(),
                    reference.sigmoid(),
                    image_sizes_tensor,
                    images.tensor.shape[-2:],
                )
            outputs_class = self.class_embed[lvl](layer_state)
            geometry_state = layer_state
            if geometry_residuals is not None:
                geometry_state = layer_state + geometry_residuals[lvl].to(layer_state.dtype)
            tmp = self.bbox_embed[lvl](geometry_state)
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            if self.use_cumulative_distribution_refinement:
                outputs_coarse_coords.append(outputs_coord)
                cumulative_corner_deltas.append(
                    self.corner_embed[lvl](layer_state)
                )
            elif self.use_distributional_box and use_specialized_head:
                distribution_reference = outputs_coord
                corner_logits = self.corner_embed[lvl](geometry_state)
                if getattr(self, "use_local_geometry_refinement", False):
                    corner_logits = self._refine_local_geometry_logits(
                        corner_logits, layer_state, local_geometry_feature,
                        distribution_reference, image_sizes_tensor, images.tensor.shape[-2:],
                    )
                if getattr(self, "use_bin_aligned_boundary_evidence", False):
                    boundary_readout = self._refine_bin_boundary_logits(
                        corner_logits, layer_state,
                        boundary_evidence_options["boundary_evidence_feature"],
                        distribution_reference, image_sizes_tensor, images.tensor.shape[-2:], t,
                        return_details=self.use_bin_contrast_refinement,
                    )
                    if self.use_bin_contrast_refinement:
                        corner_logits, boundary_details = boundary_readout
                        boundary_output_fields.update({key: boundary_details[key] for key in (
                            "pred_boundary_base_boxes", "pred_boundary_aux_boxes")})
                    else:
                        corner_logits = boundary_readout
                outputs_coord = self._sanitize_boxes(
                    decode_distributional_box(
                        distribution_reference,
                        corner_logits,
                        self.distribution_integral,
                        reg_scale=self.reg_scale,
                        detach_reference=self.distribution_detach_reference,
                    )
                )
                if self.use_refined_score_calibration:
                    boundary_output_fields["pred_score_logits"] = self.refined_score_calibrator(
                        layer_state, outputs_class, corner_logits, distribution_reference,
                        boundary_output_fields.get("pred_boundary_base_boxes", outputs_coord),
                        outputs_coord, boundary_evidence_options["boundary_evidence_feature"],
                        image_sizes_tensor, images.tensor.shape[-2:], self.distribution_integral.project, t,
                    )
                if self.decoder_final_only:
                    final_corner_logits = corner_logits
                    final_distribution_reference = distribution_reference
                else:
                    outputs_corners.append(corner_logits)
                    outputs_distribution_refs.append(distribution_reference)
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        # tensor shape: [num_decoder_layers, bs, num_query, num_classes]
        outputs_coord = torch.stack(outputs_coords)
        if self.use_cumulative_distribution_refinement:
            outputs_coarse_coords = torch.stack(outputs_coarse_coords)
            cumulative_corner_deltas = torch.stack(cumulative_corner_deltas)
            (
                outputs_coord,
                outputs_corners,
                cumulative_fixed_reference,
            ) = cumulative_distribution_refinement(
                outputs_coarse_coords,
                cumulative_corner_deltas,
                self.distribution_integral,
                reg_scale=self.reg_scale,
            )
            outputs_coord = self._sanitize_boxes(outputs_coord)
            outputs_distribution_refs = cumulative_fixed_reference.unsqueeze(0).expand_as(
                outputs_coarse_coords
            )
        elif self.use_distributional_box and not self.decoder_final_only:
            outputs_corners = torch.stack(outputs_corners)
            outputs_distribution_refs = torch.stack(outputs_distribution_refs)
        # tensor shape: [num_decoder_layers, bs, num_query, 4]

        # denoising postprocessing
        if dn_meta is not None:
            if self.use_distributional_box and dn_meta["single_padding"] > 0:
                padding_size = dn_meta["single_padding"] * dn_meta["dn_num"]
                boundary_output_fields = {key: value[:, padding_size:]
                                          for key, value in boundary_output_fields.items()}
                if self.use_cumulative_distribution_refinement:
                    outputs_coarse_coords = outputs_coarse_coords[:, :, padding_size:]
                    outputs_corners = outputs_corners[:, :, padding_size:]
                    outputs_distribution_refs = outputs_distribution_refs[
                        :, :, padding_size:
                    ]
                    cumulative_fixed_reference = outputs_distribution_refs[0]
                elif self.decoder_final_only:
                    final_corner_logits = final_corner_logits[:, padding_size:]
                    final_distribution_reference = final_distribution_reference[:, padding_size:]
                else:
                    outputs_corners = outputs_corners[:, :, padding_size:]
                    outputs_distribution_refs = outputs_distribution_refs[:, :, padding_size:]
            outputs_class, outputs_coord = self.dn_post_process(
                outputs_class, outputs_coord, dn_meta
            )

        if self.use_rectified_flow:
            lvlb_class_weights = None
            t_weight = None
        else:
            lvlb_class_weights = extract(self.loss_weight, t, [batch_size])
            t_weight = extract(self.loss_weight, t, [batch_size , self.num_queries])
            t_weight = [t_weight[i].expand(len(indices[i][0])) for i in range(batch_size)]
            t_weight = torch.cat(t_weight)

        # prepare for loss computation
        first_flow_output = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
            **boundary_output_fields,
        }
        quality_options = {}
        if self.quality_guidance is not None:
            first_flow_output["_quality_query"] = self._select_flow_query_state(
                layer_state, outputs_class[-1].shape[1])
            first_flow_output["_quality_progress"] = t.float()
            quality_options["quality_context"] = self._quality_context(
                features["res2"], image_sizes_tensor, images.tensor.shape[-2:],
                boundary_evidence_options.get("boundary_evidence_feature"))
        if self.use_recurrent_flow_query:
            first_flow_output["flow_query_state"] = self._select_flow_query_state(
                layer_state, outputs_class[-1].shape[1]
            )
        if (
            self.use_trajectory_refinement
            and self.trajectory_refinement_source == "decoder_layer"
        ):
            previous_index = max(outputs_coord.shape[0] - 2, 0)
            first_flow_output["pred_boxes_prev"] = outputs_coord[previous_index]
        if self.use_distributional_box:
            if self.use_cumulative_distribution_refinement:
                first_flow_output.update(
                    self._build_cumulative_distribution_output_fields(
                        outputs_coarse_coords,
                        outputs_coord,
                        outputs_corners,
                        cumulative_fixed_reference,
                    )
                )
            elif self.decoder_final_only:
                first_flow_output["pred_corners"] = final_corner_logits
                first_flow_output["pred_dist_ref"] = final_distribution_reference
            else:
                first_flow_output["pred_corners"] = outputs_corners[-1]
                first_flow_output["pred_dist_ref"] = outputs_distribution_refs[-1]
        output = dict(first_flow_output)
        use_first_flow_auxiliary = (
            not self.use_rectified_flow
            or not self.trajectory_aligned_flow
            or self.flow_auxiliary_source == "first"
            or self.flow_steps == 1
        )
        if self.aux_loss and use_first_flow_auxiliary:
            output["aux_outputs"] = self._set_aux_loss(
                outputs_class,
                outputs_coord,
                (
                    outputs_corners
                    if self.use_distributional_box and not self.decoder_final_only
                    else None
                ),
                (
                    outputs_distribution_refs
                    if self.use_distributional_box and not self.decoder_final_only
                    else None
                ),
            )
        if self.use_rectified_flow:
            if self.trajectory_aligned_flow:
                trajectory_outputs = self.predict_aligned_flow_trajectory(
                    flow_context,
                    init_query_points,
                    first_flow_output,
                    detail_feature,
                    foreground_logits,
                    image_sizes_tensor,
                    images.tensor.shape[-2:],
                    **local_geometry_options,
                    **observation_options,
                    **edge_geometry_options,
                    **boundary_evidence_options,
                    **quality_options,
                )
                if self.use_trajectory_refinement:
                    trajectory_outputs[-1] = self._apply_trajectory_refinement(
                        trajectory_outputs[-1],
                        multi_level_feats[0],
                        image_sizes_tensor,
                        images.tensor.shape[-2:],
                    )
                output.update(trajectory_outputs[-1])
                output["flow_trajectory_outputs"] = trajectory_outputs[:-1]
            else:
                output["flow_next_pred_boxes"] = self.predict_next_flow_endpoint(
                    flow_context,
                    init_query_points,
                    self._flow_transport_endpoint(first_flow_output),
                    t,
                    detail_feature,
                    foreground_logits,
                    image_sizes_tensor,
                    images.tensor.shape[-2:],
                )

        if self.use_baseline_trajectory_residual:
            trajectory_rng_devices = []
            if enc_reference.is_cuda:
                trajectory_rng_devices = [
                    enc_reference.device.index
                    if enc_reference.device.index is not None
                    else torch.cuda.current_device()
                ]
            with torch.random.fork_rng(devices=trajectory_rng_devices):
                ddim_trajectory = self._collect_ddim_trajectory(
                    flow_context, enc_reference
                )
            if len(ddim_trajectory["box_steps"]) < 3:
                raise RuntimeError(
                    "trajectory residual training requires three DDIM outer steps"
                )
            trajectory_refinement = self.trajectory_residual_refiner(
                trajectory_detail_feature,
                ddim_trajectory["box_steps"],
                ddim_trajectory["query_features"],
                image_sizes=image_sizes_tensor,
                padded_image_size=images.tensor.shape[-2:],
            )
            output["baseline_trajectory_residual"] = {
                "pred_logits": ddim_trajectory["class_steps"][-1],
                "pred_boxes": ddim_trajectory["box_steps"][-1],
                **trajectory_refinement,
            }

        # prepare two stage output
        interm_coord = enc_reference
        interm_class = self.transformer.decoder.class_embed[-1](enc_state)
        output["enc_outputs"] = {"pred_logits": interm_class, "pred_boxes": interm_coord}

        if self.training:
            # visualize training samples
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    box_cls = output["pred_logits"]
                    box_pred = output["pred_boxes"]
                    results = self.inference(box_cls, box_pred, images.image_sizes)
                    self.visualize_training(batched_inputs, results)
            
            # compute loss
            loss_dict = self.criterion(output, targets, dn_meta, indices,t_weight, class_weights=lvlb_class_weights)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            if self.quality_guidance is not None:
                loss_dict.update(self._quality_losses(
                    trajectory_outputs, features["res2"], targets,
                    image_sizes_tensor, images.tensor.shape[-2:]))
            if foreground_logits is not None and self.frequency_mask_loss_weight > 0:
                frequency_targets = (
                    self._prepare_frequency_targets(targets, batched_inputs)
                    if self.frequency_ignore_regions else targets
                )
                foreground_target = build_box_foreground_mask(
                    frequency_targets,
                    foreground_logits.shape[-2:],
                    foreground_logits.device,
                    images.tensor.shape[-2:],
                    return_valid_mask=self.frequency_ignore_regions,
                )
                foreground_valid = None
                if self.frequency_ignore_regions:
                    foreground_target, foreground_valid = foreground_target
                foreground_loss = foreground_mask_loss(
                    foreground_logits, foreground_target, valid_mask=foreground_valid,
                    global_normalization=self.frequency_global_normalization,
                )
                loss_dict["loss_frequency_mask"] = (
                    foreground_loss * self.frequency_mask_loss_weight
                )
            if (
                trajectory_foreground_logits is not None
                and self.trajectory_frequency_mask_loss_weight > 0
            ):
                trajectory_frequency_targets = (
                    self._prepare_frequency_targets(targets, batched_inputs)
                    if self.frequency_ignore_regions else targets
                )
                trajectory_foreground_target = build_box_foreground_mask(
                    trajectory_frequency_targets,
                    trajectory_foreground_logits.shape[-2:],
                    trajectory_foreground_logits.device,
                    images.tensor.shape[-2:],
                    return_valid_mask=self.frequency_ignore_regions,
                )
                trajectory_foreground_valid = None
                if self.frequency_ignore_regions:
                    trajectory_foreground_target, trajectory_foreground_valid = trajectory_foreground_target
                loss_dict["loss_trajectory_frequency_mask"] = (
                    foreground_mask_loss(
                        trajectory_foreground_logits,
                        trajectory_foreground_target,
                        valid_mask=trajectory_foreground_valid,
                        global_normalization=self.frequency_global_normalization,
                    )
                    * self.trajectory_frequency_mask_loss_weight
                )
            return loss_dict
        else:
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            results = self.inference(box_cls, box_pred, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def visualize_training(self, batched_inputs, results):
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_box = 20

        for input, results_per_image in zip(batched_inputs, results):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=results_per_image.pred_boxes[:max_vis_box].tensor.detach().cpu().numpy()
            )
            pred_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, pred_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted boxes"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch


    @torch.jit.unused
    def _set_aux_loss(
        self,
        outputs_class,
        outputs_coord,
        outputs_corners=None,
        outputs_distribution_refs=None,
    ):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        auxiliary = []
        for index, (class_output, box_output) in enumerate(
            zip(outputs_class[:-1], outputs_coord[:-1])
        ):
            item = {"pred_logits": class_output, "pred_boxes": box_output}
            if outputs_corners is not None:
                item["pred_corners"] = outputs_corners[index]
                item["pred_dist_ref"] = outputs_distribution_refs[index]
            auxiliary.append(item)
        return auxiliary

    def prepare_for_cdn(
        self,
        targets,
        dn_number,
        label_noise_ratio,
        box_noise_scale,
        num_queries,
        num_classes,
        hidden_dim,
        label_enc,
    ):
        """
        A major difference of DINO from DN-DETR is that the author process pattern embedding pattern embedding
            in its detector
        forward function and use learnable tgt embedding, so we change this function a little bit.
        :param dn_args: targets, dn_number, label_noise_ratio, box_noise_scale
        :param training: if it is training or inference
        :param num_queries: number of queires
        :param num_classes: number of classes
        :param hidden_dim: transformer hidden dim
        :param label_enc: encode labels in dn
        :return:
        """
        if dn_number <= 0:
            return None, None, None, None
            # positive and negative dn queries
        dn_number = dn_number * 2
        known = [(torch.ones_like(t["labels"])).to(self.device) for t in targets]
        # known = [(torch.ones_like(t["labels"])) for t in targets]

        batch_size = len(known)
        known_num = [sum(k) for k in known]
        if int(max(known_num)) == 0:
            return None, None, None, None

        dn_number = dn_number // (int(max(known_num) * 2))

        if dn_number == 0:
            dn_number = 1
        unmask_bbox = unmask_label = torch.cat(known)
        labels = torch.cat([t["labels"] for t in targets])
        boxes = torch.cat([t["boxes"] for t in targets])
        batch_idx = torch.cat(
            [torch.full_like(t["labels"].long(), i) for i, t in enumerate(targets)]
        )

        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)

        known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)
        known_labels = labels.repeat(2 * dn_number, 1).view(-1)
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)
        known_bboxs = boxes.repeat(2 * dn_number, 1)
        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        if label_noise_ratio > 0:
            p = torch.rand_like(known_labels_expaned.float())
            chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(
                -1
            )  # half of bbox prob
            new_label = torch.randint_like(
                chosen_indice, 0, num_classes
            )  # randomly put a new one here
            known_labels_expaned.scatter_(0, chosen_indice, new_label)
        single_padding = int(max(known_num))

        pad_size = int(single_padding * 2 * dn_number)
        positive_idx = (
            torch.tensor(range(len(boxes))).long().to(self.device).unsqueeze(0).repeat(dn_number, 1)
        )
        # positive_idx = (
        #     torch.tensor(range(len(boxes))).long().unsqueeze(0).repeat(dn_number, 1)
        # )
        positive_idx += (torch.tensor(range(dn_number)) * len(boxes) * 2).long().to(self.device).unsqueeze(1)
        # positive_idx += (torch.tensor(range(dn_number)) * len(boxes) * 2).long().unsqueeze(1)

        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(boxes)
        if box_noise_scale > 0:
            known_bbox_ = torch.zeros_like(known_bboxs)
            known_bbox_[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2
            known_bbox_[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2

            diff = torch.zeros_like(known_bboxs)
            diff[:, :2] = known_bboxs[:, 2:] / 2
            diff[:, 2:] = known_bboxs[:, 2:] / 2

            rand_sign = (
                torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
            )
            rand_part = torch.rand_like(known_bboxs)
            rand_part[negative_idx] += 1.0
            rand_part *= rand_sign
            known_bbox_ = known_bbox_ + torch.mul(rand_part, diff).to(self.device) * box_noise_scale
            # known_bbox_ = known_bbox_ + torch.mul(rand_part, diff) * box_noise_scale
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
            known_bbox_expand[:, :2] = (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
            known_bbox_expand[:, 2:] = known_bbox_[:, 2:] - known_bbox_[:, :2]

        m = known_labels_expaned.long().to(self.device)
        # m = known_labels_expaned.long()

        input_label_embed = label_enc(m)
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)

        padding_label = torch.zeros(pad_size, hidden_dim).to(self.device)
        padding_bbox = torch.zeros(pad_size, 4).to(self.device)
        # padding_label = torch.zeros(pad_size, hidden_dim)
        # padding_bbox = torch.zeros(pad_size, 4)

        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

        map_known_indice = torch.tensor([]).to(self.device)
        # map_known_indice = torch.tensor([])

        if len(known_num):
            map_known_indice = torch.cat(
                [torch.tensor(range(num)) for num in known_num]
            )  # [1,2, 1,2,3]
            map_known_indice = torch.cat(
                [map_known_indice + single_padding * i for i in range(2 * dn_number)]
            ).long()
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

        tgt_size = pad_size + num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to(self.device) < 0
        # attn_mask = torch.ones(tgt_size, tgt_size) < 0

        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(dn_number):
            if i == 0:
                attn_mask[
                    single_padding * 2 * i : single_padding * 2 * (i + 1),
                    single_padding * 2 * (i + 1) : pad_size,
                ] = True
            if i == dn_number - 1:
                attn_mask[
                    single_padding * 2 * i : single_padding * 2 * (i + 1), : single_padding * i * 2
                ] = True
            else:
                attn_mask[
                    single_padding * 2 * i : single_padding * 2 * (i + 1),
                    single_padding * 2 * (i + 1) : pad_size,
                ] = True
                attn_mask[
                    single_padding * 2 * i : single_padding * 2 * (i + 1), : single_padding * 2 * i
                ] = True

        dn_meta = {
            "single_padding": single_padding * 2,
            "dn_num": dn_number,
        }

        return input_query_label, input_query_bbox, attn_mask, dn_meta

    def dn_post_process(self, outputs_class, outputs_coord, dn_metas):
        if dn_metas and dn_metas["single_padding"] > 0:
            padding_size = dn_metas["single_padding"] * dn_metas["dn_num"]
            output_known_class = outputs_class[:, :, :padding_size, :]
            output_known_coord = outputs_coord[:, :, :padding_size, :]
            outputs_class = outputs_class[:, :, padding_size:, :]
            outputs_coord = outputs_coord[:, :, padding_size:, :]

            out = {"pred_logits": output_known_class[-1], "pred_boxes": output_known_coord[-1]}
            if self.aux_loss:
                out["aux_outputs"] = self._set_aux_loss(output_known_class, output_known_coord)
            dn_metas["output_known_lbs_bboxes"] = out
        return outputs_class, outputs_coord

    def preprocess_image(self, batched_inputs):
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images

    def inference(self, box_cls, box_pred, image_sizes, final = True):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []
        # print(box_cls)
        # box_cls.shape: 1, 300, 80
        # box_pred.shape: 1, 300, 4
        prob = box_cls.sigmoid()
        candidate_count = prob.shape[1] * prob.shape[2]
        topk = min(self.select_box_nums_for_evaluation, candidate_count)
        topk_values, topk_indexes = torch.topk(
            prob.view(box_cls.shape[0], -1), topk, dim=1
        )
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, box_cls.shape[2], rounding_mode="floor")
        labels = topk_indexes % box_cls.shape[2]

        boxes = torch.gather(box_pred, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        # scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)
        # boxes = box_pred

        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(
            zip(scores, labels, boxes, image_sizes)
        ):

            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))
            if not final:
                return result.pred_boxes.tensor, scores_per_image, labels_per_image
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image


            results.append(result)
        return results

    def _ensemble_flow_outputs(self, class_steps, box_steps, image_sizes):
        if not class_steps or len(class_steps) != len(box_steps):
            raise ValueError("flow ensemble requires aligned class and box steps")
        batch_size = class_steps[0].shape[0]
        if batch_size != len(image_sizes):
            raise ValueError("flow ensemble batch size does not match image_sizes")
        step_weights = getattr(self, "flow_fusion_weights", None)
        if step_weights is None:
            step_weights = (1.0,) * len(class_steps)

        results = []
        for batch_index, image_size in enumerate(image_sizes):
            step_boxes = []
            step_scores = []
            step_labels = []
            for step_index, (class_output, box_output) in enumerate(
                zip(class_steps, box_steps)
            ):
                boxes, scores, labels = self.inference(
                    class_output[batch_index : batch_index + 1],
                    box_output[batch_index : batch_index + 1],
                    [image_size],
                    final=False,
                )
                scores = scores * step_weights[step_index]
                step_boxes.append(boxes)
                step_scores.append(scores)
                step_labels.append(labels)

            boxes = torch.cat(step_boxes, dim=0)
            scores = torch.cat(step_scores, dim=0)
            labels = torch.cat(step_labels, dim=0)
            if self.use_nms:
                keep = batched_nms(boxes, scores, labels, 0.8)
            else:
                keep = torch.argsort(scores, descending=True)
            keep = keep[: self.select_box_nums_for_evaluation]

            result = Instances(image_size)
            result.pred_boxes = Boxes(boxes[keep])
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores[keep]
            result.pred_classes = labels[keep]
            results.append(result)
        return results

    def _build_flow_results(
        self,
        class_steps,
        box_steps,
        final_class,
        final_endpoint,
        image_sizes,
    ):
        if self.flow_output_mode == "query_fusion":
            fused_class, fused_boxes = fuse_query_aligned_flow_outputs(
                class_steps,
                box_steps,
                self.flow_fusion_weights,
            )
            return self.inference(fused_class, fused_boxes, image_sizes)
        if self.flow_output_mode == "ensemble" and len(class_steps) > 1:
            return self._ensemble_flow_outputs(class_steps, box_steps, image_sizes)
        if self.flow_output_mode == "step":
            if len(class_steps) != len(box_steps):
                raise ValueError("flow step output requires aligned class and box steps")
            step_index = self.flow_output_step - 1
            if step_index >= len(class_steps):
                raise ValueError("requested flow output step was not collected")
            return self.inference(
                class_steps[step_index], box_steps[step_index], image_sizes
            )
        return self.inference(final_class, final_endpoint, image_sizes)

    @staticmethod
    def _sanitize_boxes(boxes, eps=1e-4):
        center = boxes[..., :2].clamp(eps, 1.0 - eps)
        size = boxes[..., 2:].clamp(eps, 1.0)
        return torch.cat((center, size), dim=-1)

    def _apply_trajectory_refinement(
        self,
        output,
        local_feature,
        image_sizes_tensor,
        padded_image_size,
    ):
        if not self.use_trajectory_refinement:
            return output
        if "pred_boxes_prev" not in output:
            raise KeyError("trajectory refinement requires previous trajectory boxes")
        refinement = self.trajectory_refiner(
            local_feature,
            output["pred_boxes_prev"],
            output["pred_boxes"],
            image_sizes=image_sizes_tensor,
            padded_image_size=padded_image_size,
        )
        return {**output, **refinement}

    def _flow_transport_endpoint(self, output, step_index=None):
        transport_boxes = output.get(
            "pred_transport_boxes", output.get("pred_dist_ref")
        )
        if self.flow_transport_mode == "coarse":
            if transport_boxes is None:
                raise KeyError(
                    "coarse flow transport requires a transport endpoint"
                )
            return transport_boxes
        if self.flow_transport_mode == "progressive_refined":
            if transport_boxes is None:
                raise KeyError(
                    "progressive refined transport requires a transport endpoint"
                )
            if step_index is None:
                raise ValueError(
                    "progressive refined transport requires an outer step index"
                )
            return progressive_refined_endpoint(
                transport_boxes,
                output["pred_boxes"],
                step_index,
                self.flow_steps,
            )
        return output["pred_boxes"]

    def _build_flow_auxiliary_outputs(
        self,
        inter_states,
        inter_references,
        initial_reference,
        detail_feature,
        foreground_logits,
        image_sizes_tensor,
        padded_image_size,
        geometry_residuals=None,
    ):
        auxiliary_outputs = []
        num_layers = inter_states.shape[0]
        for layer_index in range(num_layers - 1):
            reference = (
                initial_reference
                if layer_index == 0
                else inter_references[layer_index - 1]
            )
            layer_state = inter_states[layer_index]
            use_specialized_head = self._use_specialized_decoder_head(
                layer_index, num_layers, self.decoder_final_only
            )
            if self.query_modulation and use_specialized_head:
                layer_state = self.detail_query_modulator(
                    layer_state,
                    detail_feature,
                    foreground_logits.sigmoid(),
                    reference,
                    image_sizes_tensor,
                    padded_image_size,
                )
            logits = self.class_embed[layer_index](layer_state)
            geometry_state = layer_state
            if geometry_residuals is not None:
                geometry_state = layer_state + geometry_residuals[layer_index].to(layer_state.dtype)
            coarse_boxes = (
                self.bbox_embed[layer_index](geometry_state)
                + inverse_sigmoid(reference)
            ).sigmoid()
            layer_output = {
                "pred_logits": logits,
                "pred_boxes": coarse_boxes,
            }
            if self.use_distributional_box and use_specialized_head:
                corner_logits = self.corner_embed[layer_index](geometry_state)
                layer_output["pred_boxes"] = self._sanitize_boxes(
                    decode_distributional_box(
                        coarse_boxes,
                        corner_logits,
                        self.distribution_integral,
                        reg_scale=self.reg_scale,
                        detach_reference=self.distribution_detach_reference,
                    )
                )
                layer_output["pred_corners"] = corner_logits
                layer_output["pred_dist_ref"] = coarse_boxes
            auxiliary_outputs.append(layer_output)
        return auxiliary_outputs

    def _build_cumulative_flow_output(
        self,
        inter_states,
        inter_references,
        initial_reference,
        detail_feature,
        foreground_logits,
        image_sizes_tensor,
        padded_image_size,
        return_aux_outputs,
    ):
        class_outputs = []
        coarse_outputs = []
        corner_deltas = []
        num_layers = inter_states.shape[0]
        for layer_index in range(num_layers):
            reference = (
                initial_reference
                if layer_index == 0
                else inter_references[layer_index - 1]
            )
            layer_state = inter_states[layer_index]
            use_specialized_head = self._use_specialized_decoder_head(
                layer_index, num_layers, self.decoder_final_only
            )
            if self.query_modulation and use_specialized_head:
                layer_state = self.detail_query_modulator(
                    layer_state,
                    detail_feature,
                    foreground_logits.sigmoid(),
                    reference,
                    image_sizes_tensor,
                    padded_image_size,
                )
            class_outputs.append(self.class_embed[layer_index](layer_state))
            box_delta = self.bbox_embed[layer_index](layer_state)
            reference_unactivated = inverse_sigmoid(reference)
            if reference.shape[-1] == 4:
                box_delta = box_delta + reference_unactivated
            else:
                if reference.shape[-1] != 2:
                    raise ValueError(
                        "flow references must contain two or four coordinates"
                    )
                box_delta[..., :2] = (
                    box_delta[..., :2] + reference_unactivated
                )
            coarse_outputs.append(box_delta.sigmoid())
            corner_deltas.append(self.corner_embed[layer_index](layer_state))

        class_outputs = torch.stack(class_outputs)
        coarse_outputs = torch.stack(coarse_outputs)
        corner_deltas = torch.stack(corner_deltas)
        refined_outputs, cumulative_logits, fixed_reference = (
            cumulative_distribution_refinement(
                coarse_outputs,
                corner_deltas,
                self.distribution_integral,
                reg_scale=self.reg_scale,
            )
        )
        refined_outputs = self._sanitize_boxes(refined_outputs)
        output = {
            "pred_logits": class_outputs[-1],
            **self._build_cumulative_distribution_output_fields(
                coarse_outputs,
                refined_outputs,
                cumulative_logits,
                fixed_reference,
            ),
            "pred_boxes_prev": refined_outputs[max(num_layers - 2, 0)],
        }
        if getattr(self, "use_recurrent_flow_query", False):
            output["flow_query_state"] = layer_state
        if return_aux_outputs:
            output["aux_outputs"] = [
                {
                    "pred_logits": class_outputs[layer_index],
                    "pred_boxes": refined_outputs[layer_index],
                }
                for layer_index in range(num_layers - 1)
            ]
        return output

    def _update_flow_query_state(
        self, base_query, step_output, current_boxes, step_index
    ):
        if not getattr(self, "use_recurrent_flow_query", False):
            return base_query
        if "flow_query_state" not in step_output:
            raise KeyError("recurrent flow output requires flow_query_state")
        decoded_query = step_output["flow_query_state"]
        updated_query = self.recurrent_flow_query_updater(
            base_query,
            decoded_query,
            step_output["pred_logits"],
            current_boxes,
            step_output["pred_boxes"],
            step_index=step_index,
        )
        if not getattr(self, "use_flow_distribution_conditioning", False):
            return updated_query
        if "pred_corners" not in step_output:
            raise KeyError("flow distribution conditioning requires pred_corners")
        endpoint_boxes = self._flow_transport_endpoint(
            step_output, step_index=step_index
        )
        condition = build_distribution_transport_condition(
            step_output["pred_corners"],
            self.distribution_integral.project,
            current_boxes,
            endpoint_boxes,
            step_index,
            self.flow_steps,
        ).to(dtype=updated_query.dtype)
        return updated_query + self.flow_distribution_conditioner(condition)

    def _refine_local_geometry_logits(
        self, logits, query, projected, coarse, image_sizes, padded_image_size,
    ):
        if not getattr(self, "use_local_geometry_refinement", False):
            return logits
        if projected is None:
            raise ValueError("local geometry requires the projected res2 feature")
        return logits + self.local_geometry_refiner(
            query, projected, coarse, image_sizes, padded_image_size
        )

    def _refine_bin_boundary_logits(
        self, logits, query, feature, coarse, image_sizes, padded_image_size, time,
        *, return_details=False,
    ):
        if not getattr(self, "use_bin_aligned_boundary_evidence", False):
            return logits
        if feature is None:
            raise ValueError("bin-aligned boundary requires projected res2 evidence")
        if not getattr(self, "use_bin_contrast_refinement", False):
            delta = self.bin_aligned_boundary_evidence(
                query, feature, coarse, image_sizes, padded_image_size,
                self.distribution_integral.project, self.reg_scale, time,
            )
            return logits + delta.to(logits.dtype)
        details = self.bin_aligned_boundary_evidence(
            query, feature, coarse, image_sizes, padded_image_size,
            self.distribution_integral.project, self.reg_scale, time,
            base_logits=logits, return_details=True,
        )
        refined_logits = logits + details["delta"].to(logits.dtype)
        if not return_details:
            return refined_logits
        # Both auxiliary boxes share this one frozen FP32 baseline. The primary
        # path above keeps its original cast/add/decode precision.
        with torch.autocast(device_type=logits.device.type, enabled=False):
            reference = coarse.detach().float()
            legacy_logits = (logits.detach().float() + details["legacy_delta"].detach()).detach()
            auxiliary_logits = legacy_logits + details["aux_increment"]
            if any(not torch.isfinite(value).all() for value in (reference, legacy_logits, auxiliary_logits)):
                raise ValueError("boundary auxiliary decode inputs must be finite before sanitizing")
            base = decode_distributional_box(reference, legacy_logits, self.distribution_integral,
                                            reg_scale=self.reg_scale, detach_reference=True)
            auxiliary = decode_distributional_box(reference, auxiliary_logits, self.distribution_integral,
                                                 reg_scale=self.reg_scale, detach_reference=True)
            if not torch.isfinite(base).all() or not torch.isfinite(auxiliary).all():
                raise ValueError("boundary auxiliary boxes must be finite before sanitizing")
            base_boxes, aux_boxes = self._sanitize_boxes(base), self._sanitize_boxes(auxiliary)
            diagnostics = {
                "base_sanitized": (base_boxes != base).any(-1).detach(),
                "aux_sanitized": (aux_boxes != auxiliary).any(-1).detach(),
                "base_nonpositive_size": (base[..., 2:] <= 0).any(-1).detach(),
                "aux_nonpositive_size": (auxiliary[..., 2:] <= 0).any(-1).detach(),
            }
        return refined_logits, {
            "pred_boundary_base_boxes": base_boxes,
            "pred_boundary_aux_boxes": aux_boxes,
            "boundary_decode_diagnostics": diagnostics,
        }

    def _predict_flow_endpoint(
        self,
        query,
        memory,
        mask_flatten,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        state,
        time,
        detail_feature,
        foreground_logits,
        image_sizes_tensor,
        padded_image_size,
        return_output=False,
        return_aux_outputs=False,
        local_geometry_feature=None,
        observation_feature=None,
        previous_output=None,
        edge_geometry_feature=None,
        boundary_evidence_feature=None,
    ):
        if return_aux_outputs and not return_output:
            raise ValueError("flow auxiliary outputs require return_output=True")
        time_cond = self.transformer.embed_time(
            flow_to_diffusion_time(time, self.num_timesteps)
        )
        observation_options = {}
        if getattr(self, "flow_spatial_observation_mode", "off") != "off":
            observation_options["observation_context"] = prepare_observation_context(
                observation_feature, image_sizes_tensor, padded_image_size, state, time,
                previous_output, self.distribution_integral.project, self.reg_scale,
            )
        edge_geometry_options = {}
        if getattr(self, "flow_edge_geometry_mode", "off") != "off":
            edge_geometry_options["edge_geometry_context"] = prepare_observation_context(
                edge_geometry_feature, image_sizes_tensor, padded_image_size, state, time,
                previous_output, self.distribution_integral.project, self.reg_scale,
            )
        decoder_outputs = self.transformer.decoder(
            query=query,
            key=memory,
            value=memory,
            query_pos=None,
            key_padding_mask=mask_flatten,
            reference_points=state,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            t=time_cond,
            **observation_options,
            **edge_geometry_options,
        )
        geometry_residuals = None
        if edge_geometry_options:
            inter_states, inter_references, geometry_residuals = decoder_outputs
        else:
            inter_states, inter_references = decoder_outputs
        if self.use_cumulative_distribution_refinement:
            output = self._build_cumulative_flow_output(
                inter_states,
                inter_references,
                state,
                detail_feature,
                foreground_logits,
                image_sizes_tensor,
                padded_image_size,
                return_aux_outputs,
            )
            if return_output:
                return output
            return output["pred_boxes"], output["pred_logits"]
        layer_index = inter_states.shape[0] - 1
        previous_layer_index = max(layer_index - 1, 0)
        previous_reference = (
            state
            if previous_layer_index == 0
            else inter_references[previous_layer_index - 1]
        )
        previous_geometry_state = inter_states[previous_layer_index]
        if geometry_residuals is not None:
            previous_geometry_state = previous_geometry_state + geometry_residuals[
                previous_layer_index
            ].to(previous_geometry_state.dtype)
        previous_endpoint = (
            self.bbox_embed[previous_layer_index](
                previous_geometry_state
            )
            + inverse_sigmoid(previous_reference)
        ).sigmoid()
        reference = state if layer_index == 0 else inter_references[layer_index - 1]
        layer_state = inter_states[layer_index]
        if self.query_modulation:
            layer_state = self.detail_query_modulator(
                layer_state,
                detail_feature,
                foreground_logits.sigmoid(),
                reference,
                image_sizes_tensor,
                padded_image_size,
            )
        outputs_class = self.class_embed[layer_index](layer_state)
        geometry_state = layer_state
        if geometry_residuals is not None:
            geometry_state = layer_state + geometry_residuals[layer_index].to(layer_state.dtype)
        coarse_endpoint = (
            self.bbox_embed[layer_index](geometry_state) + inverse_sigmoid(reference)
        ).sigmoid()
        boundary_output_fields = {}
        if self.use_distributional_box:
            corner_logits = self.corner_embed[layer_index](geometry_state)
            if getattr(self, "use_local_geometry_refinement", False):
                corner_logits = self._refine_local_geometry_logits(
                    corner_logits, layer_state, local_geometry_feature,
                    coarse_endpoint, image_sizes_tensor, padded_image_size,
                )
            if getattr(self, "use_bin_aligned_boundary_evidence", False):
                boundary_readout = self._refine_bin_boundary_logits(
                    corner_logits, layer_state, boundary_evidence_feature,
                    coarse_endpoint, image_sizes_tensor, padded_image_size, time,
                    return_details=getattr(self, "use_bin_contrast_refinement", False),
                )
                if getattr(self, "use_bin_contrast_refinement", False):
                    corner_logits, boundary_details = boundary_readout
                    boundary_output_fields.update({key: boundary_details[key] for key in (
                        "pred_boundary_base_boxes", "pred_boundary_aux_boxes")})
                else:
                    corner_logits = boundary_readout
            endpoint = decode_distributional_box(
                coarse_endpoint,
                corner_logits,
                self.distribution_integral,
                reg_scale=self.reg_scale,
                detach_reference=self.distribution_detach_reference,
            )
            endpoint = self._sanitize_boxes(endpoint)
            if getattr(self, "use_refined_score_calibration", False):
                boundary_output_fields["pred_score_logits"] = self.refined_score_calibrator(
                    layer_state, outputs_class, corner_logits, coarse_endpoint,
                    boundary_output_fields.get("pred_boundary_base_boxes", endpoint), endpoint,
                    boundary_evidence_feature, image_sizes_tensor, padded_image_size,
                    self.distribution_integral.project, time,
                )
        else:
            endpoint = coarse_endpoint
        if return_output:
            output = {
                "pred_logits": outputs_class,
                "pred_boxes": endpoint,
                "pred_boxes_prev": previous_endpoint,
                **boundary_output_fields,
            }
            if getattr(self, "quality_guidance", None) is not None:
                output["_quality_query"] = layer_state
                output["_quality_progress"] = time.float()
            if getattr(self, "use_recurrent_flow_query", False):
                output["flow_query_state"] = layer_state
            if self.use_distributional_box:
                output["pred_corners"] = corner_logits
                output["pred_dist_ref"] = coarse_endpoint
            if return_aux_outputs:
                output["aux_outputs"] = self._build_flow_auxiliary_outputs(
                    inter_states,
                    inter_references,
                    state,
                    detail_feature,
                    foreground_logits,
                    image_sizes_tensor,
                    padded_image_size,
                    geometry_residuals=geometry_residuals,
                )
            return output
        selection_logits = boundary_output_fields.get("pred_score_logits", outputs_class)
        return endpoint, selection_logits.to(outputs_class.dtype)

    def predict_next_flow_endpoint(
        self,
        context,
        current_state,
        current_endpoint,
        time,
        detail_feature,
        foreground_logits,
        image_sizes_tensor,
        padded_image_size,
    ):
        next_time = time + 0.5 * (1.0 - time)
        next_state = (
            current_state + 0.5 * (current_endpoint.detach() - current_state)
        ).clamp(0.0, 1.0)
        endpoint, _ = self._predict_flow_endpoint(
            context["query"],
            context["memory"],
            context["mask_flatten"],
            context["spatial_shapes"],
            context["level_start_index"],
            context["valid_ratios"],
            next_state,
            next_time,
            detail_feature,
            foreground_logits,
            image_sizes_tensor,
            padded_image_size,
        )
        return endpoint

    def _initialize_proposal_flow(self, proposals, image_sizes):
        return sample_proposal_conditioned_boxes(
            proposals, image_sizes,
            local_queries=self.proposal_local_queries,
            center_noise_scale=self.proposal_center_noise_scale,
            size_noise_scale=self.proposal_size_noise_scale,
            min_size_px=self.proposal_min_size_px,
            bound=self.scale,
        )

    def _advance_configured_flow_state(self, state, endpoint, step_index, frame=None):
        if not getattr(self, "use_proposal_conditioned_flow", False):
            return advance_flow_state(state, endpoint, step_index, self.flow_steps)
        if frame is None:
            raise ValueError("proposal-conditioned flow requires its fixed proposal frame")
        return advance_proposal_flow_state(
            state, endpoint, frame, self.proposal_local_queries,
            step_index, self.flow_steps, transport=self.proposal_flow_transport,
        )

    def predict_aligned_flow_trajectory(
        self,
        context,
        initial_state,
        first_output,
        detail_feature,
        foreground_logits,
        image_sizes_tensor,
        padded_image_size,
        local_geometry_feature=None,
        observation_feature=None,
        edge_geometry_feature=None,
        boundary_evidence_feature=None,
        quality_context=None,
    ):
        local_geometry_options = (
            {"local_geometry_feature": local_geometry_feature}
            if getattr(self, "use_local_geometry_refinement", False) else {}
        )
        boundary_evidence_options = (
            {"boundary_evidence_feature": boundary_evidence_feature}
            if getattr(self, "use_bin_aligned_boundary_evidence", False) else {}
        )
        trajectory = [first_output]
        state = initial_state
        base_query = context["query"]
        query_state = self._update_flow_query_state(
            base_query, first_output, state, step_index=0
        )
        batch_size = state.shape[0]
        for step_index in range(1, self.flow_steps):
            transport_endpoint = self._flow_transport_endpoint(
                trajectory[-1], step_index=step_index - 1)
            if quality_context is not None:
                _, _, transport_endpoint = self._quality_output(
                    trajectory[-1], quality_context, step_index=step_index - 1)
            state = self._advance_configured_flow_state(
                state,
                transport_endpoint.detach(),
                step_index=step_index - 1,
                frame=context.get("flow_proposal_frame"),
            ).detach()
            time = torch.full(
                (batch_size,),
                float(step_index) / float(self.flow_steps),
                device=state.device,
                dtype=state.dtype,
            )
            observation_options = {}
            if getattr(self, "flow_spatial_observation_mode", "off") != "off":
                observation_options = dict(observation_feature=observation_feature,
                                           previous_output=trajectory[-1])
            edge_geometry_options = {}
            if getattr(self, "flow_edge_geometry_mode", "off") != "off":
                edge_geometry_options = dict(edge_geometry_feature=edge_geometry_feature,
                                             previous_output=trajectory[-1])
            step_output = self._predict_flow_endpoint(
                query_state,
                context["memory"],
                context["mask_flatten"],
                context["spatial_shapes"],
                context["level_start_index"],
                context["valid_ratios"],
                state,
                time,
                detail_feature,
                foreground_logits,
                image_sizes_tensor,
                padded_image_size,
                return_output=True,
                return_aux_outputs=(
                    self.aux_loss
                    and self.flow_auxiliary_source == "final"
                    and step_index == self.flow_steps - 1
                ),
                **local_geometry_options,
                **observation_options,
                **edge_geometry_options,
                **boundary_evidence_options,
            )
            trajectory.append(step_output)
            if step_index < self.flow_steps - 1:
                query_state = self._update_flow_query_state(
                    base_query, step_output, state, step_index=step_index
                )
        if (
            self.use_trajectory_refinement
            and self.trajectory_refinement_source == "outer_flow"
        ):
            trajectory[-1]["pred_boxes_prev"] = trajectory[-2][
                "pred_boxes"
            ].detach()
        return trajectory

    def process_flow_targets(self, targets, padded_targets):
        """Build rectified-flow states; losses are assigned by Hungarian set matching."""
        noisy_queries = []
        initial_boxes = []
        endpoint_queries = []
        times = []
        indices = []
        num_gts = []
        for target, padded in zip(targets, padded_targets):
            num_gt = len(target["boxes"])
            endpoint = padded["boxes"]
            labels = self.label_enc(padded["labels"].long())
            permutation = torch.randperm(endpoint.shape[0], device=endpoint.device)
            endpoint = endpoint[permutation]
            labels = labels[permutation]
            source = sample_bounded_gaussian_boxes(
                endpoint.shape,
                endpoint.device,
                endpoint.dtype,
                bound=self.scale,
            )
            if getattr(self, "trajectory_aligned_flow", False):
                time = torch.zeros(1, device=endpoint.device, dtype=endpoint.dtype)
                state = source
            else:
                time = torch.rand(1, device=endpoint.device, dtype=endpoint.dtype)
                state = rectified_interpolate(
                    source.unsqueeze(0), endpoint.unsqueeze(0), time
                ).squeeze(0)

            source_indices = torch.nonzero(permutation < num_gt, as_tuple=False).squeeze(1)
            target_indices = permutation[source_indices]
            indices.append((source_indices, target_indices))
            noisy_queries.append(torch.cat((labels, state), dim=-1).unsqueeze(0))
            initial_boxes.append(state.unsqueeze(0))
            endpoint_queries.append(torch.cat((labels, endpoint), dim=-1).unsqueeze(0))
            times.append(time)
            num_gts.append(num_gt)
        return (
            torch.cat(noisy_queries, dim=0),
            torch.cat(initial_boxes, dim=0),
            torch.cat(endpoint_queries, dim=0),
            torch.cat(times, dim=0),
            indices,
            num_gts,
        )

    @torch.no_grad()
    def flow_sample(
        self,
        batched_inputs,
        multi_level_feats,
        multi_level_masks,
        multi_level_position_embeddings,
        images,
        detail_feature=None,
        foreground_logits=None,
        local_geometry_feature=None,
        observation_feature=None,
        edge_geometry_feature=None,
        boundary_evidence_feature=None,
        quality_feature=None,
    ):
        local_geometry_options = (
            {"local_geometry_feature": local_geometry_feature}
            if getattr(self, "use_local_geometry_refinement", False) else {}
        )
        boundary_evidence_options = (
            {"boundary_evidence_feature": boundary_evidence_feature}
            if getattr(self, "use_bin_aligned_boundary_evidence", False) else {}
        )
        batch_size = len(batched_inputs)
        empty_query = [None, None]
        initial_time = torch.zeros(batch_size, device=self.device)
        (
            memory,
            mask_flatten,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            proposals,
            target_unact,
        ) = self.transformer(
            multi_level_feats,
            multi_level_masks,
            multi_level_position_embeddings,
            empty_query,
            attn_masks=None,
            time_steps=initial_time,
            query_embeds_diffusion=[None, None],
            is_train=False,
        )
        if self.transformer.learnt_init_query:
            if self.transformer.tgt_embed_new:
                base_query = self.transformer.tgt_embed.weight[None].repeat(
                    batch_size, self.num_queries, 1
                )
            else:
                base_query = self.transformer.tgt_embed.weight[None].repeat(batch_size, 1, 1)
        else:
            base_query = target_unact
        query_state = base_query

        source_frame = None
        if getattr(self, "use_proposal_conditioned_flow", False):
            state, source_frame = self._initialize_proposal_flow(
                proposals,
                torch.as_tensor(images.image_sizes, device=memory.device, dtype=torch.float32),
            )
        else:
            state = sample_bounded_gaussian_boxes(
                (batch_size, self.num_queries, 4),
                device=memory.device,
                dtype=memory.dtype,
                bound=self.scale,
            )
        image_sizes_tensor = torch.as_tensor(
            images.image_sizes, device=memory.device,
            dtype=(torch.float32 if (getattr(self, "use_local_geometry_refinement", False)
                                    or getattr(self, "use_bin_aligned_boundary_evidence", False))
                   else memory.dtype),
        )
        outputs_class = None
        ensemble_classes = []
        ensemble_boxes = []
        schedule = uniform_flow_schedule(
            self.flow_steps, device=memory.device, dtype=memory.dtype
        )
        outer_endpoints = []
        quality_context = self._quality_context(
            quality_feature, images.image_sizes, images.tensor.shape[-2:], boundary_evidence_feature)
        needs_step_output = (
            self.use_trajectory_refinement
            or self.flow_transport_mode in {"coarse", "progressive_refined"}
            or getattr(self, "use_recurrent_flow_query", False)
            or getattr(self, "flow_spatial_observation_mode", "off") != "off"
            or getattr(self, "flow_edge_geometry_mode", "off") != "off"
            or getattr(self, "quality_guidance", None) is not None
        )
        previous_output = None
        for step, time_value in enumerate(schedule):
            time = time_value.expand(batch_size)
            observation_options = {}
            if getattr(self, "flow_spatial_observation_mode", "off") != "off":
                observation_options = dict(observation_feature=observation_feature,
                                           previous_output=previous_output)
            edge_geometry_options = {}
            if getattr(self, "flow_edge_geometry_mode", "off") != "off":
                edge_geometry_options = dict(edge_geometry_feature=edge_geometry_feature,
                                             previous_output=previous_output)
            if needs_step_output:
                step_output = self._predict_flow_endpoint(
                    query_state,
                    memory,
                    mask_flatten,
                    spatial_shapes,
                    level_start_index,
                    valid_ratios,
                    state,
                    time,
                    detail_feature,
                    foreground_logits,
                    image_sizes_tensor,
                    images.tensor.shape[-2:],
                    return_output=True,
                    **local_geometry_options,
                    **observation_options,
                    **edge_geometry_options,
                    **boundary_evidence_options,
                )
                previous_output = step_output
                endpoint = step_output["pred_boxes"]
                outputs_class = step_output.get("pred_score_logits", step_output["pred_logits"])
                outputs_class = outputs_class.to(step_output["pred_logits"].dtype)
                transport_endpoint = self._flow_transport_endpoint(
                    step_output, step_index=step
                )
                if quality_context is not None:
                    endpoint, outputs_class, transport_endpoint = self._quality_output(
                        step_output, quality_context, step_index=step, for_inference=True)
            else:
                endpoint, outputs_class = self._predict_flow_endpoint(
                    query_state,
                    memory,
                    mask_flatten,
                    spatial_shapes,
                    level_start_index,
                    valid_ratios,
                    state,
                    time,
                    detail_feature,
                    foreground_logits,
                    image_sizes_tensor,
                    images.tensor.shape[-2:],
                    **local_geometry_options,
                    **boundary_evidence_options,
                )
                transport_endpoint = endpoint
            outer_endpoints.append(endpoint)
            if self.flow_output_mode in {"ensemble", "query_fusion", "step"}:
                ensemble_classes.append(outputs_class)
                ensemble_boxes.append(endpoint)
            if (
                getattr(self, "use_recurrent_flow_query", False)
                and step < self.flow_steps - 1
            ):
                query_state = self._update_flow_query_state(
                    base_query, step_output, state, step_index=step
                )
            state = self._advance_configured_flow_state(
                state,
                transport_endpoint,
                step_index=step,
                frame=source_frame,
            )

        final_endpoint = endpoint
        if self.use_trajectory_refinement:
            if self.trajectory_refinement_source == "outer_flow":
                step_output = dict(step_output)
                step_output["pred_boxes_prev"] = outer_endpoints[-2]
            step_output = self._apply_trajectory_refinement(
                step_output,
                multi_level_feats[0],
                image_sizes_tensor,
                images.tensor.shape[-2:],
            )
            final_endpoint = step_output["pred_boxes_refined"]

        results = self._build_flow_results(
            ensemble_classes,
            ensemble_boxes,
            final_class=outputs_class,
            final_endpoint=final_endpoint,
            image_sizes=images.image_sizes,
        )
        processed_results = []
        for result, input_per_image, image_size in zip(
            results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            processed_results.append(
                {"instances": detector_postprocess(result, height, width)}
            )
        return processed_results

    # def prepare_targets(self, targets):
    #     new_targets = []
    #     for targets_per_image in targets:
    #         h, w = targets_per_image.image_size
    #         image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
    #         gt_classes = targets_per_image.gt_classes
    #         gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
    #         gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
    #         new_targets.append({"labels": gt_classes, "boxes": gt_boxes})
    #     return new_targets

    ## Diffu_dino
    def _prepare_frequency_targets(self, targets, batched_inputs):
        """Training-only ignore metadata; keep criterion targets unchanged."""
        if len(targets) != len(batched_inputs):
            raise ValueError("frequency targets and inputs must have the same batch length")
        frequency_targets = []
        for target, inputs in zip(targets, batched_inputs):
            if "ignore_boxes" not in inputs:
                raise ValueError("frequency_ignore_regions requires mapper preserve_ignore_boxes=True")
            ignored = inputs["ignore_boxes"].detach().to(device=self.device, dtype=torch.float32)
            if ignored.ndim != 2 or ignored.shape[1] != 4 or not torch.isfinite(ignored).all():
                raise ValueError("mapper ignore_boxes must be finite XYXY_ABS with shape (N, 4)")
            height, width = target["image_size"]
            size_xyxy = torch.stack((width, height, width, height)).to(ignored)
            normalized = box_xyxy_to_cxcywh(ignored / size_xyxy)
            frequency_targets.append({**target, "ignore_boxes": normalized})
        return frequency_targets

    def prepare_targets(self, targets):
        new_targets = []
        num_gts = []
        new_targets_diffusion = []
        for targets_per_image in targets:
            no_object_bbox , no_object_label = torch.tensor([0.5,0.5,0.5,0.5],device=self.device), torch.tensor([self.num_classes],device= self.device)
            # no_object_bbox , no_object_label = torch.tensor([1,1,1,1],device=self.device), torch.tensor([80],device= self.device)

            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            num_gt = len(gt_boxes)
            no_object_label = no_object_label.repeat(self.num_queries - num_gt, 1) # 295 , 256
            no_object_bbox = no_object_bbox.repeat(self.num_queries - num_gt , 1) # 295 , 4

            ####### noisy ground truth
            if self.noisy_gt:
                no_object_bbox = torch.randn(no_object_bbox.shape, device=self.device)
                no_object_bbox = no_object_bbox.sigmoid()
            ######

            gt_boxes_diffusion = torch.cat([gt_boxes , no_object_bbox])
            gt_classes_diffusion = torch.cat([gt_classes , no_object_label.squeeze()])
            # noised_gtboxes, noise , t  = self.prepare_for_diffusion(gt_boxes)
            new_targets_diffusion.append({"labels": gt_classes_diffusion, "boxes": gt_boxes_diffusion})
            new_targets.append(
                {
                    "labels": gt_classes,
                    "boxes": gt_boxes,
                    "image_size": torch.as_tensor(
                        [h, w], dtype=torch.float32, device=self.device
                    ),
                }
            )
            # num_gts.append(num_gt)
        return new_targets , new_targets_diffusion



    def _predict_ddim_step(self, context, normalized_reference, time):
        """Predict one endpoint from one outer DDIM state."""
        batch_size = normalized_reference.shape[0]
        timestep = torch.full(
            (batch_size,), time, device=normalized_reference.device, dtype=torch.long
        )
        time_cond = timestep_embedding(
            timestep.float(), self.transformer.embed_dim, repeat_only=False
        )
        time_cond = self.transformer.time_embed(time_cond)
        inter_states, inter_references = self.transformer.decoder(
            query=context["query"],
            key=context["memory"],
            value=context["memory"],
            query_pos=None,
            key_padding_mask=context["mask_flatten"],
            reference_points=normalized_reference,
            spatial_shapes=context["spatial_shapes"],
            level_start_index=context["level_start_index"],
            valid_ratios=context["valid_ratios"],
            t=time_cond,
        )
        layer_index = inter_states.shape[0] - 1
        reference = (
            normalized_reference
            if layer_index == 0
            else inter_references[layer_index - 1]
        )
        query_features = inter_states[layer_index]
        pred_logits = self.class_embed[layer_index](query_features)
        box_delta = self.bbox_embed[layer_index](query_features)
        if reference.shape[-1] == 4:
            box_delta = box_delta + inverse_sigmoid(reference)
        else:
            if reference.shape[-1] != 2:
                raise ValueError("DDIM reference points must have two or four coordinates")
            box_delta[..., :2] = box_delta[..., :2] + inverse_sigmoid(reference)
        return {
            "pred_logits": pred_logits,
            "pred_boxes": box_delta.sigmoid(),
            "query_features": query_features,
        }

    def _advance_ddim_state(
        self,
        latent_reference,
        normalized_reference,
        step_output,
        time,
        time_next,
    ):
        """Apply the legacy DDIM update and optional box renewal exactly once."""
        pred_boxes = step_output["pred_boxes"]
        pred_logits = step_output["pred_logits"]
        x_start = (pred_boxes * 2.0 - 1.0) * self.scale
        x_start = x_start.clamp(min=-self.scale, max=self.scale)
        timestep = torch.full(
            (pred_boxes.shape[0],),
            time,
            device=pred_boxes.device,
            dtype=torch.long,
        )
        pred_noise = self.predict_noise_from_start(latent_reference, timestep, x_start)

        alpha = self.alphas_cumprod[time]
        alpha_next = self.alphas_cumprod[time_next]
        eta = self.ddim_sampling_eta
        sigma = eta * (
            (1.0 - alpha / alpha_next)
            * (1.0 - alpha_next)
            / (1.0 - alpha)
        ).sqrt()
        coefficient = (1.0 - alpha_next - sigma ** 2).sqrt()
        latent_reference = (
            pred_boxes * alpha_next.sqrt() + coefficient * pred_noise
        )
        normalized_reference = latent_reference.clamp(
            min=-self.scale, max=self.scale
        )
        normalized_reference = ((normalized_reference / self.scale) + 1.0) / 2.0
        normalized_reference = normalized_reference.float()

        if self.box_renewal:
            threshold = time_next / 100.0
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(f"invalid DDIM renewal threshold {threshold}")
            confidence = pred_logits.sigmoid().amax(dim=-1)
            keep = confidence > threshold
            random_latent = torch.randn_like(normalized_reference)
            random_latent[keep] = latent_reference[keep].float()
            latent_reference = random_latent
            random_normalized = random_latent.clamp(
                min=-self.scale, max=self.scale
            )
            random_normalized = ((random_normalized / self.scale) + 1.0) / 2.0
            random_normalized[keep] = normalized_reference[keep]
            normalized_reference = random_normalized
        return latent_reference, normalized_reference

    @torch.no_grad()
    def _collect_ddim_trajectory(self, context, initial_reference):
        """Collect query-aligned predictions from real outer DDIM iterations."""
        batch_size = initial_reference.shape[0]
        initial_time = torch.full(
            (batch_size,),
            self.num_timesteps - 1,
            device=initial_reference.device,
            dtype=torch.long,
        )
        _, latent_reference = self.apply_box_noise(
            initial_reference.detach(), t=initial_time
        )
        normalized_reference = latent_reference.clamp(
            min=-self.scale, max=self.scale
        )
        normalized_reference = ((normalized_reference / self.scale) + 1.0) / 2.0
        normalized_reference = normalized_reference.float()

        class_steps = []
        box_steps = []
        query_steps = []
        time_pairs = build_ddim_time_pairs(
            self.num_timesteps, self.sampling_timesteps
        )
        for time, time_next in time_pairs:
            step_output = self._predict_ddim_step(
                context, normalized_reference, time
            )
            class_steps.append(step_output["pred_logits"])
            box_steps.append(step_output["pred_boxes"])
            query_steps.append(step_output["query_features"])
            latent_reference, normalized_reference = self._advance_ddim_state(
                latent_reference,
                normalized_reference,
                step_output,
                time,
                time_next,
            )
        return {
            "class_steps": class_steps,
            "box_steps": box_steps,
            "query_steps": query_steps,
            "query_features": query_steps[-1],
        }

    def _quality_selected_refined_detections(
        self, class_logits, refinement, batch_index
    ):
        selected_index = refinement["selected_candidate_index"][batch_index]
        quality_logits = refinement["candidate_quality_logits"][batch_index]
        selected_quality = quality_logits.gather(
            1, selected_index.unsqueeze(-1)
        ).squeeze(-1).sigmoid()
        baseline_quality = quality_logits[:, 0].sigmoid()
        quality_improvement = (selected_quality - baseline_quality).clamp(min=0.0)
        active = (
            (selected_index > 0)
            & (
                quality_improvement
                >= float(self.trajectory_residual_quality_margin)
            )
        )
        if not torch.any(active):
            return (
                class_logits.new_zeros((0, 4)),
                class_logits.new_zeros((0,)),
                torch.zeros((0,), device=class_logits.device, dtype=torch.long),
            )

        class_probability = class_logits[batch_index].sigmoid()
        score_gain = 1.0 + (
            float(self.trajectory_residual_score_gain)
            * quality_improvement[:, None]
        )
        refined_scores = class_probability * score_gain
        refined_scores = refined_scores.masked_fill(~active[:, None], -1.0)
        topk = min(
            self.select_box_nums_for_evaluation, refined_scores.numel()
        )
        scores, flat_indices = torch.topk(refined_scores.flatten(), topk)
        valid = scores >= 0.0
        scores = scores[valid].clamp(max=1.0)
        flat_indices = flat_indices[valid]
        query_indices = torch.div(
            flat_indices, self.num_classes, rounding_mode="floor"
        )
        labels = flat_indices % self.num_classes
        boxes = refinement["pred_boxes_refined"][batch_index][query_indices]
        return box_cxcywh_to_xyxy(boxes), scores, labels

    @torch.no_grad()
    def ddim_sample_with_trajectory_residual(
        self,
        batched_inputs,
        multi_level_feats,
        multi_level_masks,
        multi_level_position_embeddings,
        images,
        trajectory_detail_feature,
    ):
        """Run baseline DDIM and add only quality-approved local refinements."""
        batch_size = len(batched_inputs)
        initial_time = torch.zeros(
            (batch_size,), device=self.device, dtype=torch.long
        )
        empty_query = [None, None]
        (
            memory,
            mask_flatten,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            initial_reference,
            target_unact,
        ) = self.transformer(
            multi_level_feats,
            multi_level_masks,
            multi_level_position_embeddings,
            empty_query,
            attn_masks=None,
            time_steps=initial_time,
            query_embeds_diffusion=[None, None],
            is_train=False,
        )
        if self.transformer.learnt_init_query:
            if self.transformer.tgt_embed_new:
                query = self.transformer.tgt_embed.weight[None].repeat(
                    batch_size, self.num_queries, 1
                )
            else:
                query = self.transformer.tgt_embed.weight[None].repeat(
                    batch_size, 1, 1
                )
        else:
            query = target_unact
        context = {
            "memory": memory,
            "mask_flatten": mask_flatten,
            "spatial_shapes": spatial_shapes,
            "level_start_index": level_start_index,
            "valid_ratios": valid_ratios,
            "query": query,
        }
        trajectory = self._collect_ddim_trajectory(context, initial_reference)
        if len(trajectory["box_steps"]) < 3:
            raise RuntimeError("trajectory residual inference requires three DDIM steps")
        image_sizes_tensor = torch.as_tensor(
            images.image_sizes, device=memory.device, dtype=memory.dtype
        )
        refinement = self.trajectory_residual_refiner(
            trajectory_detail_feature,
            trajectory["box_steps"],
            trajectory["query_features"],
            image_sizes=image_sizes_tensor,
            padded_image_size=images.tensor.shape[-2:],
        )

        results = []
        for batch_index, image_size in enumerate(images.image_sizes):
            ensemble_boxes = []
            ensemble_scores = []
            ensemble_labels = []
            for class_output, box_output in zip(
                trajectory["class_steps"], trajectory["box_steps"]
            ):
                boxes, scores, labels = self.inference(
                    class_output[batch_index : batch_index + 1],
                    box_output[batch_index : batch_index + 1],
                    [image_size],
                    final=False,
                )
                ensemble_boxes.append(boxes)
                ensemble_scores.append(scores)
                ensemble_labels.append(labels)

            refined_boxes, refined_scores, refined_labels = (
                self._quality_selected_refined_detections(
                    trajectory["class_steps"][-1], refinement, batch_index
                )
            )
            if refined_boxes.numel() > 0:
                ensemble_boxes.append(refined_boxes)
                ensemble_scores.append(refined_scores)
                ensemble_labels.append(refined_labels)

            boxes = torch.cat(ensemble_boxes, dim=0)
            scores = torch.cat(ensemble_scores, dim=0)
            labels = torch.cat(ensemble_labels, dim=0)
            if self.use_nms:
                keep = batched_nms(boxes, scores, labels, 0.8)
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]
            sorted_indices = torch.argsort(scores, descending=True)
            selected_indices = sorted_indices[
                : self.select_box_nums_for_evaluation
            ]
            result = Instances(image_size)
            result.pred_boxes = Boxes(boxes[selected_indices])
            result.pred_boxes.scale(
                scale_x=image_size[1], scale_y=image_size[0]
            )
            result.scores = scores[selected_indices]
            result.pred_classes = labels[selected_indices]
            results.append(result)

        processed_results = []
        for result, input_per_image, image_size in zip(
            results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            processed_results.append(
                {"instances": detector_postprocess(result, height, width)}
            )
        return processed_results

    @torch.no_grad()
    def ddim_sample(self, batched_inputs, multi_level_feats,multi_level_masks,
                    multi_level_position_embeddings,images,
                    clip_denoised=True, do_postprocess=True):
        N = len(batched_inputs)
        # shape = (batch, self.num_proposals, 4)
        total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(0, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        t = torch.zeros((N,), device=self.device, dtype=torch.long)
        query = [None,  None]
        # prepare for loss computation
        (memory,
        mask_flatten ,
        spatial_shapes ,
        level_start_index,
        valid_ratios, 
        init_reference,
        target_unact) = self.transformer(multi_level_feats, multi_level_masks, multi_level_position_embeddings,query, attn_masks=None ,time_steps=t,query_embeds_diffusion= [None,None], is_train =False)
        # query = torch.randn((N,self.num_queries,self.embed_dim), device=self.device)
        if self.transformer.learnt_init_query:
            if self.transformer.tgt_embed_new:
                query = self.transformer.tgt_embed.weight[None].repeat(N, self.num_queries, 1)
            else:
                query = self.transformer.tgt_embed.weight[None].repeat(N, 1, 1)
        else:
            query = target_unact

        # query = torch.zeros_like(query)
        # query  = torch.randn((N, self.num_queries, self.embed_dim), device=self.device, dtype=torch.float)
        t = torch.full((N,),self.num_timesteps - 1, device=self.device, dtype=torch.long)
        _ , reference_points = self.apply_box_noise(init_reference , t = t)
        # reference_points = query[:,:,-4:]
        # init_reference_points = reference_points.float()
        
        noise = torch.clamp(reference_points, min=-1 * self.scale, max=self.scale)
        noise = ((noise / self.scale) + 1) / 2
        # reference_points = query[:,:,-4:]
        init_reference_points = noise.float()
        ensemble_score, ensemble_label, ensemble_coord = [], [], []

        for time, time_next in time_pairs:
            # index = self.num_timesteps - time - 1

            t = torch.full((N,),time, device=self.device, dtype=torch.long)
            time_cond = timestep_embedding(t.float() , self.transformer.embed_dim, repeat_only=False)
            time_cond = self.transformer.time_embed(time_cond)
            # self_cond = x_start if self.self_condition else None
            (inter_states,
            inter_references)= self.transformer.decoder(
                query = query,  # bs, num_queries, embed_dims
                key=memory,  # bs, num_tokens, embed_dims
                value=memory,  # bs, num_tokens, embed_dims
                query_pos=None,
                key_padding_mask=mask_flatten,  # bs, num_tokens
                reference_points=init_reference_points,  # num_queries, 4
                spatial_shapes=spatial_shapes,  # nlvl, 2
                level_start_index=level_start_index,  # nlvl
                valid_ratios=valid_ratios,  # bs, nlvl, 2
                t = time_cond,
            )
            outputs_classes = []
            outputs_coords = []

            for lvl in range(inter_states.shape[0]):
                # current_inter_states = inter_states[lvl]
                # class_state , ref_points_state = current_inter_states.chunk(2, dim = -1)
                if lvl == 0:
                    reference = init_reference_points
                else:
                    reference = inter_references[lvl - 1]
                outputs_class = self.class_embed[lvl](inter_states[lvl])
                tmp = self.bbox_embed[lvl](inter_states[lvl])
                # assert reference.shape[-1] == 4
                if reference.shape[-1] == 4:
                    tmp += inverse_sigmoid(reference)
                    # tmp = tmp
                outputs_coord = tmp.sigmoid()
                outputs_classes.append(outputs_class)
                outputs_coords.append(outputs_coord)
        
            outputs_class = torch.stack(outputs_classes)
            outputs_coord = torch.stack(outputs_coords)

            # select parameters corresponding to the currently considered timestep
            # bbox_start = inverse_sigmoid(outputs_coord[-1])
            # bbox_start = bbox_start.clamp(min = -4 , max = 4)
            bbox_start = outputs_coord[-1]
            x_start = (bbox_start * 2 - 1.) * self.scale
            x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)

            # init_reference_points = inverse_sigmoid(init_reference_points)
            
            # bbox_start = bbox_start.clamp(min = -3 ,max = 3)
            # init_reference_points =init_reference_points.clamp(min = -3 ,max = 3)

            # query_start = inter_states[-1]
            # init_reference_points = init_reference_points
            # bbox_start = outputs_coord[-1]

            # pred_noise_query = self.predict_noise_from_start(query_start, t , query)
            pred_noise_bbox = self.predict_noise_from_start(reference_points, t , x_start)

            # pred_noise_query = self.predict_noise_from_start(query, t , query_start)

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            # query = query_start * alpha_next.sqrt() + \
            #       c * pred_noise_query 
        
            reference_points = bbox_start * alpha_next.sqrt() + \
                  c * pred_noise_bbox 
            noise = torch.clamp(reference_points, min=-1 * self.scale, max=self.scale)
            noise = ((noise / self.scale) + 1) / 2
            # reference_points = query[:,:,-4:]
            init_reference_points = noise.float()
            if self.box_renewal:  # filter
                score_per_image, box_per_image = outputs_class[-1], outputs_coord[-1]
                threshold = 0.5 #0.5

                ##adaptive threshold
                threshold = time_next/100 ##v1
                # threshold = (time+time_next)/200 #v2
                assert threshold>=0 and threshold<=1, f"{threshold} fix threshold"
                # print(threshold)

                score_per_image = score_per_image.sigmoid()
                value, _ = torch.max(score_per_image, -1, keepdim=False)
                keep_idx = value > threshold 
                num_remain = torch.sum(keep_idx)
                new_ref = torch.randn_like(init_reference_points , device = init_reference_points.device)
                reference = new_ref
                reference[keep_idx] = reference_points[keep_idx].float()
                reference_points = reference
                new_ref = torch.clamp(new_ref, min=-1 * self.scale, max=self.scale)
                new_ref = ((new_ref / self.scale) + 1) / 2
                # pred_logits =  torch.cat((pred_logits , outputs_class[-1][keep_idx]), dim=1)
                # pred_boxes =  torch.cat((pred_boxes , outputs_coord[-1][keep_idx] ), dim=1)
                new_ref[keep_idx] = init_reference_points[keep_idx] 
                init_reference_points = new_ref
            if self.use_ensemble and self.sampling_timesteps > 1:
                box_pred_per_image, scores_per_image, labels_per_image = self.inference(outputs_class[-1],
                                                                                        outputs_coord[-1],
                                                                                        images.image_sizes, final = False)
                ensemble_score.append(scores_per_image)
                ensemble_label.append(labels_per_image)
                ensemble_coord.append(box_pred_per_image)

        if self.use_ensemble and self.sampling_timesteps > 1:
            box_pred_per_image = torch.cat(ensemble_coord, dim=0)
            scores_per_image = torch.cat(ensemble_score, dim=0)
            labels_per_image = torch.cat(ensemble_label, dim=0)

            ## no ensemble but with nms
            # box_pred_per_image = ensemble_coord[-1]
            # scores_per_image = ensemble_score[-1]
            # labels_per_image = ensemble_label[-1]
            
            if self.use_nms:
                keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.8) #0.5
                box_pred_per_image = box_pred_per_image[keep]
                scores_per_image = scores_per_image[keep]
                labels_per_image = labels_per_image[keep]
            # reference_points = query[:,:,-4:]
            # init_reference_points = reference_points.clamp(min = -3 , max = 3)
            # init_reference_points = reference_points.float().sigmoid()
            # init_reference_points = reference_points.float()

            # t1 = torch.full((N,),time_next, device=self.device, dtype=torch.long)

            # # _ , reference_points = self.apply_box_noise(init_reference , t = t)

            # reference_points = self.q_sample(x_start = bbox_start , t = t1, noise = None)
            
            # # reference_points = query[:,:,-4:]
            # init_reference_points = reference_points.float().sigmoid()
            # s = torch.randperm(init_reference_points.size(1))
            # # Shuffle the tensor
            # init_reference_points = init_reference_points[:,s,:]

            # query = query.float()
        # outputs_class = self.class_embed[-1](query)
        # outputs_coord = self.bbox_embed[-1](query)
            sorted_indices = torch.argsort(scores_per_image, descending=True)
            top_k = self.select_box_nums_for_evaluation
            selected_indices = sorted_indices[:top_k]
            # Extract top results
            box_pred_per_image = box_pred_per_image[selected_indices]
            scores_per_image = scores_per_image[selected_indices]
            labels_per_image = labels_per_image[selected_indices]
            image_size = images.image_sizes[0]
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_pred_per_image)
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results = [result]
        else:
            output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            results = self.inference(box_cls, box_pred, images.image_sizes)
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results
    
    def prepare_for_diffusion(self, gt_boxes, labels, num_gt, is_train):
        t = torch.randint(0,self.num_timesteps,(1,), device=self.device).long() 
        # no_object_label = self.label_enc(torch.tensor(self.num_classes, dtype=torch.long , device=self.device))
        # no_object_bbox = torch.tensor([0,0,0,0], device = self.device) 
        # num_gt = len(gt_boxes) # number of objects
        # no_object_label = no_object_label.repeat(self.num_queries - num_gt, 1) # 295 , 256
        # no_object_bbox = no_object_bbox.repeat(self.num_queries - num_gt , 1) # 295 , 4

        # labels = torch.cat([labels, no_object_label] ,dim = 0) # 900 , 256
        # gt_boxes = torch.cat([gt_boxes, no_object_bbox] , dim = 0) # 900 , 4

        queries = torch.cat([labels , gt_boxes] , dim = 1)

        # noise_bbox = torch.randn(gt_boxes.shape, device=self.device) 
        noise_labels = torch.randn(labels.shape, device=self.device)

        # gt_boxes = (gt_boxes * 2. - 1.) * self.box_noise_scale
        # init_anchors = self.prior_anchors.weight
        # noise sample 
        diff_boxes , noise = self.apply_box_noise(gt_boxes , t)
        # diff_boxes = diff_boxes[..., :2]
        # diff_boxes = self.q_sample(x_start=gt_boxes, t=t, noise=None)
        # diff_boxes = torch.clamp(diff_boxes, min= -1 * self.box_noise_scale, max=self.box_noise_scale)
        # diff_boxes = ((diff_boxes / self.box_noise_scale) + 1) / 2.
        # labels = (labels * 2. - 1.) * self.label_noise_ratio
        diff_labels = self.q_sample(x_start = labels , t = t, noise = noise_labels)
        # labels = torch.clamp(labels,min = -1 * self.label_noise_ratio , max = self.label_noise_ratio)
        # diff_labels = ((labels / self.label_noise_ratio) + 1) / 2.
        noisy_queries = torch.cat([diff_labels , diff_boxes] , dim = 1)

        shuffled_indices = torch.randperm(diff_labels.size(0))
        # Shuffle the tensor
        diff_labels = diff_labels[shuffled_indices]
        diff_boxes = diff_boxes[shuffled_indices]
        shuffled_indices = torch.argsort(shuffled_indices)
        # Create tuples pairing old indices with new shuffled indices
        indices = shuffled_indices[:num_gt] , torch.arange(diff_labels.size(0))[:num_gt]
        # indices = torch.arange(diff_labels.size(0))[:num_gt] , torch.arange(diff_labels.size(0))[:num_gt]
        # indices = None
        return noisy_queries, diff_boxes, queries, t, indices


    def process_targets(self, targets, new_targets_diffusion, is_train = True):
        init_query_bboxes , noisy_queries, original_queries, ts, indices, num_gts = [], [], [] , [], [], []
        for  target, target_diffusion in zip(targets, new_targets_diffusion):
            # bbox = [bbox for bbox in target['boxes']]
            # labels = [bbox for bbox in target['labels']]
            bbox , labels = target_diffusion['boxes'] , target_diffusion['labels']
            num_gt = len(target['boxes'])
            labels = self.label_enc(labels.squeeze())   # number of object , 256
            noisy_query, diff_boxes, queries, t, indice = self.prepare_for_diffusion(bbox, labels, num_gt, is_train=is_train)
            init_query_bboxes.append(diff_boxes.unsqueeze(0))
            noisy_queries.append(noisy_query.unsqueeze(0))
            original_queries.append(queries.unsqueeze(0))
            num_gts.append(num_gt)
            ts.append(t)
            indices.append(indice)
        return (torch.cat(noisy_queries,dim = 0), torch.cat(init_query_bboxes,dim=0)
                ,torch.cat(original_queries,dim = 0),torch.cat(ts,dim=0)
                ,indices, num_gts)
