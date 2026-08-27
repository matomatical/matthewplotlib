"""
Nonlinear colour scales.

Two Gaussian sources, one a hundred times fainter than the other, drawn
twice. A linear colour scale spends every colour on the bright summit, losing
the tails to black and the faint source with them. A logarithmic one gives
each order of magnitude the same share of the colormap, follows the tails out
to the corners of the window, and picks the faint source cleanly off them.
The bar beside each picture covers the same numbers either way: the spacing
moves the values between the ends, not the ends themselves.

By Claude Fable 5.
"""

import tyro
import numpy as np
import matthewplotlib as mp


XRANGE = (-4.0, 4.0)
YRANGE = (-4.0, 4.0)

# the corners of the window reach exp(-16), a hair above 1e-7, so the log
# scale runs from there up to the bright summit: seven decades
VRANGES = {
    " linear ": (0.0, 1.0),
    " logarithmic ": mp.logscale(1e-7, 1.0),
}

WIDTH = 26      # cells of picture across
HEIGHT = 13     # cells down, matching: a cell is two pixels tall


def peaks(xy: np.ndarray) -> np.ndarray:
    """A unit Gaussian at the origin, and one a hundred times fainter and
    much narrower out towards a corner."""
    bright = np.exp(-(xy[:, 0] ** 2 + xy[:, 1] ** 2) / 2)
    faint = np.exp(-((xy[:, 0] - 3.2) ** 2 + (xy[:, 1] + 3.2) ** 2) / 0.25)
    return bright + 0.01 * faint


def panel(title: str, vrange: tuple[float, float] | mp.scale) -> mp.plot:
    """The sources drawn on one scale, with a bar labelled over its
    interval."""
    picture = mp.function2(
        peaks,
        xrange=XRANGE,
        yrange=YRANGE,
        width=WIDTH,
        height=HEIGHT,
        vrange=vrange,
        colormap=mp.viridis,
    )
    return mp.hstack(
        mp.axes(picture, title=title, xfmt="{x:g}", yfmt="{y:g}"),
        # the blank row drops the bar past the picture's top rule, so that
        # its two ends line up with the first and last rows of the picture
        mp.vstack(
            mp.blank(height=1, width=1),
            mp.axes(
                mp.colorbar(picture, colormap=mp.viridis, length=HEIGHT),
                east="label",
                yfmt="{y:g}",
            ),
        ),
    )


def main(save: str | None = None):
    """Nonlinear colour scales."""
    plot = mp.hstack(
        *(panel(title, vrange) for title, vrange in VRANGES.items()),
    )
    print(plot)
    if save:
        plot.saveimg(save)


if __name__ == "__main__":
    tyro.cli(main)
