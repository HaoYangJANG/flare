"""Count normalization for auxiliary losses followed by DDP gradient averaging."""

import torch


@torch.no_grad()
def global_count_normalizer(local_count: int, reference: torch.Tensor) -> float:
    """Return max(global count, 1) / world size; every rank must call this.

    Unlike clamping the per-rank average to one, this preserves a true global
    mean when there is only one stable match across multiple ranks. The count
    tensor follows the prediction device so both Gloo and NCCL are supported.
    """
    if local_count < 0:
        raise ValueError("local_count must be non-negative")
    count = torch.tensor(float(local_count), dtype=torch.float32, device=reference.device)
    world_size = 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(count)
        world_size = torch.distributed.get_world_size()
    return count.clamp(min=1.0).item() / world_size
