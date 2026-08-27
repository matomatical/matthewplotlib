"""Shared instruments for the plot tests: colormaps that record what they
were given, and a probe for which cells a plot drew."""

import numpy as np


class RecordingVectorColormap:
    """A vector colormap that retains the field it received."""

    def __init__(self):
        self.input = None

    def __call__(self, values):
        self.input = np.array(values, copy=True)
        return np.zeros((*np.shape(values)[:2], 3), dtype=np.uint8)


class RecordingColormap:
    """A greyscale colormap that retains the scalar grid it received."""

    def __init__(self):
        self.input = None

    def __call__(self, values):
        self.input = np.array(values, copy=True)
        return np.repeat(
            (255 * values[..., np.newaxis]).astype(np.uint8),
            3,
            axis=-1,
        )


def drawn_cells(plot):
    """Which character cells of a plot have anything in them."""
    return plot.chars.isnonblank()
