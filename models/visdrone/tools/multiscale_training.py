import os

import numpy as np
import torch
from detectron2.data import transforms as T
from PIL import Image


class IsolatedMultiScaleResize(T.Augmentation):
    """Resize the shortest edge using a worker-local, private RNG stream."""

    _SALT = 20260921

    def __init__(
        self,
        short_edge_lengths=(800, 1000, 1200, 1200),
        max_size=1200,
        interp=Image.BILINEAR,
    ):
        short_edge_lengths = tuple(short_edge_lengths)
        if not short_edge_lengths:
            raise ValueError("short_edge_lengths must not be empty")
        if any(isinstance(size, bool) or not isinstance(size, int) for size in short_edge_lengths):
            raise TypeError("short_edge_lengths must contain only integers")
        if any(size <= 0 for size in short_edge_lengths):
            raise ValueError("short_edge_lengths must contain only positive values")
        if isinstance(max_size, bool) or not isinstance(max_size, int):
            raise TypeError("max_size must be an integer")
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        super().__init__()
        self._init(locals())
        self._rng = None
        self._rng_identity = None

    def _private_rng(self):
        identity = (os.getpid(), torch.initial_seed())
        if identity != self._rng_identity:
            numeric_seed = (identity[1] + self._SALT) % (1 << 64)
            self._rng = np.random.default_rng(numeric_seed)
            self._rng_identity = identity
        return self._rng

    def get_transform(self, image):
        height, width = image.shape[:2]
        size = self._private_rng().choice(self.short_edge_lengths)
        new_height, new_width = T.ResizeShortestEdge.get_output_shape(
            height, width, int(size), self.max_size
        )
        return T.ResizeTransform(
            height, width, new_height, new_width, self.interp
        )
