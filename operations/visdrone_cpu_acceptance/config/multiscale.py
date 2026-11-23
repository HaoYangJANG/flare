"""Aspect-preserving VisDrone training resize with a private scale RNG."""
from detectron2.data import transforms as T
from PIL import Image
from tools.multiscale_training import IsolatedMultiScaleResize


TRAIN_LONG_EDGES = (1080, 1200, 1320, 1200)


class VisDroneLongEdgeResize(IsolatedMultiScaleResize):
    """Sample the output long edge directly, including for wide aerial frames."""

    def __init__(self, long_edge_lengths=TRAIN_LONG_EDGES, interp=Image.BILINEAR):
        lengths = tuple(long_edge_lengths)
        super().__init__(short_edge_lengths=lengths, max_size=max(lengths, default=1), interp=interp)
        self.long_edge_lengths = lengths

    def get_transform(self, image):
        height, width = image.shape[:2]
        edge = int(self._private_rng().choice(self.long_edge_lengths))
        longest = max(height, width)
        # Round half up, the convention used by Detectron2 resize transforms.
        new_h = max(1, (2 * height * edge + longest) // (2 * longest))
        new_w = max(1, (2 * width * edge + longest) // (2 * longest))
        return T.ResizeTransform(height, width, new_h, new_w, self.interp)
