"""Validate resolved repair recipes before output setup or model construction.

Legacy entry points have no marker and retain their historical behavior.
This guard protects the independent repair experiments from silent overrides.
"""
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any


OUTPUT_ROOT = PurePosixPath("/root/diffudetr_runs/voc2007_repaired_20261105")
FULL_FLAGS = (
    "use_rectified_flow", "trajectory_aligned_flow", "feature_enhance",
    "query_modulation", "use_distributional_box", "use_proposal_conditioned_flow",
    "use_bin_aligned_boundary_evidence", "use_bin_contrast_refinement",
    "use_refined_score_calibration",
)
EXTRA_FLAGS = (
    "use_trajectory_refinement", "use_baseline_trajectory_residual",
    "use_cumulative_distribution_refinement", "use_recurrent_flow_query",
    "use_flow_distribution_conditioning", "use_local_geometry_refinement",
)


def _target_name(config: Mapping[str, Any]) -> str:
    value = config.get("_target_", "")
    if isinstance(value, str):
        return value.rsplit(".", 1)[-1]
    return getattr(value, "__name__", "")


def _fixed_1200(resize: Mapping[str, Any]) -> bool:
    short = resize.get("short_edge_length")
    return (_target_name(resize) == "ResizeShortestEdge"
            and (short == 1200 or short == [1200] or short == (1200,))
            and resize.get("max_size") == 1200)


def validate_repair_protocol(cfg: Mapping[str, Any]) -> None:
    """Raise on recipe drift; do not modify configuration or create files."""
    marker = cfg.get("train", {}).get("repair_protocol")
    if marker is None:
        return
    if marker.get("version") != 1 or marker.get("arm") not in ("full", "all_off"):
        raise ValueError("unsupported repair protocol version/arm")
    full = marker["arm"] == "full"
    single_gpu_accum2 = "execution" in marker
    if single_gpu_accum2 and (marker["execution"] != "single_gpu_accum2" or not full):
        raise ValueError("unsupported repair execution; single_gpu_accum2 is FULL-only")
    try:
        train, model, data = cfg["train"], cfg["model"], cfg["dataloader"]
        mapper = data["train"]["mapper"]
        augmentations = mapper["augmentation"]
        evaluation = data["test"]["mapper"]["augmentation"]
        output = PurePosixPath(train["output_dir"])
        if ".." in output.parts or OUTPUT_ROOT not in output.parents:
            raise ValueError("repair output must be a child of the independent output root")
        arm_root = OUTPUT_ROOT / marker["arm"]
        if output != arm_root and arm_root not in output.parents:
            raise ValueError("repair output must belong to its own arm")
        if PurePosixPath(data["evaluator"]["output_dir"]) != output / "inference":
            raise ValueError("evaluator output must belong to the independent run")
        if any(bool(model.get(name, False)) != full for name in FULL_FLAGS):
            raise ValueError("FULL/all-off module flags differ from the repair protocol")
        if (model.get("detail_readout_mode", "legacy") != "legacy"
                or model.get("small_object_threshold", 48.) != 48.
                or model.get("quality_guidance") is not None):
            raise ValueError("repair retains legacy readout/gate and disables quality guidance")
        if (any(model.get(name, False) for name in EXTRA_FLAGS)
                or model.get("flow_spatial_observation_mode", "off") != "off"
                or model.get("flow_edge_geometry_mode", "off") != "off"):
            raise ValueError("extra refinement modules are outside the legacy repair")
        if (len(augmentations) != 2 or _target_name(augmentations[0]) != "RandomFlip"
                or mapper.get("augmentation_with_crop") is not None):
            raise ValueError("repair training requires flip plus resize, without crop")
        resize = augmentations[1]
        if full:
            if (_target_name(resize) != "IsolatedMultiScaleResize"
                    or tuple(resize.get("short_edge_lengths", ())) != (800, 1000, 1200, 1200)
                    or resize.get("max_size") != 1200):
                raise ValueError("FULL multiscale training was disabled or changed")
            if (model["flow_steps"] != 3
                    or model["flow_transport_mode"] != "progressive_refined"
                    or model["flow_output_mode"] != "ensemble"
                    or model["proposal_flow_transport"] != "absolute"
                    or tuple(model["flow_fusion_weights"]) != (.82, .91, 1.)):
                raise ValueError("FULL flow/transport policy changed")
        elif not _fixed_1200(resize):
            raise ValueError("all-off training must retain fixed1200 resize")
        if len(evaluation) != 1 or not _fixed_1200(evaluation[0]):
            raise ValueError("repair evaluation must retain fixed1200 resize")
        if bool(model["criterion"].get("global_aux_normalization", False)) != full:
            raise ValueError("auxiliary normalization repair flag differs")
        if not model["criterion"]["matcher"].get("force_fp32_cost", False):
            raise ValueError("both repair arms require matcher low-precision protection")
        if bool(model.get("frequency_ignore_regions", False)) != full:
            raise ValueError("frequency ignore-region repair flag differs")
        if bool(model.get("frequency_global_normalization", False)) != full:
            raise ValueError("foreground global normalization repair flag differs")
        if bool(mapper.get("preserve_ignore_boxes", False)) != full:
            raise ValueError("mapper ignore-box repair flag differs")
        expected_batch, expected_accumulation = (2, 2) if single_gpu_accum2 else (4, 1)
        if (data["train"]["total_batch_size"] != expected_batch
                or train["grad_accumulation_steps"] != expected_accumulation):
            raise ValueError(
                f"repair global batch/accumulation must remain {expected_batch}/{expected_accumulation}"
            )
        if (cfg["optimizer"]["lr"] != 1e-4 or train["seed"] != 42
                or not train["amp"]["enabled"] or train.get("init_from_ema", False)
                or not train["model_ema"]["enabled"] or train["model_ema"]["decay"] != .999
                or not train["model_ema"].get("use_ema_weights_for_eval_only", False)):
            raise ValueError("repair learning rate, seed, AMP or EMA recipe changed")
        lr = cfg["lr_multiplier"]
        if (train["max_iter"] != 25000 or lr["warmup_length"] != 0.
                or list(lr["scheduler"]["values"]) != [1.]
                or list(lr["scheduler"]["milestones"]) != []
                or lr["scheduler"]["num_updates"] != 100000):
            raise ValueError("repair budget and inherited constant learning-rate schedule changed")
        if (train["eval_period"] != 2500 or train["checkpointer"]["period"] != 2500
                or train["eval_start_iter"] != 5000):
            raise ValueError("repair evaluation/save cadence must remain 2500 from5000")
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"incomplete resolved repair configuration: {error}") from error
