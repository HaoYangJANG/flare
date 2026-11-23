"""Learning-rate groups for SODA-D fine-tuning from a COCO detector."""


NEW_OR_RESET_PARAMETER_KEYWORDS = (
    "class_embed",
    "label_enc",
    "frequency_selector",
    "frequency_fusion",
    "detail_query_modulator",
    "corner_embed",
    "flow_time_embed",
    "trajectory_refiner",
    "trajectory_residual_refiner",
)


def coco_finetune_lr_factor(module_name: str) -> float:
    """Keep loaded detector weights stable while training new branches faster."""
    if any(keyword in module_name for keyword in NEW_OR_RESET_PARAMETER_KEYWORDS):
        return 1.0
    if "backbone" in module_name:
        return 0.01
    return 0.1
