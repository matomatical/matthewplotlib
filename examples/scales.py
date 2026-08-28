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


def main(
    height: int = 13,
    save: str | None = None,
):
    """Linear versus nonlinear colour scales."""
    plot = mp.hstack(
        panel(
            "linear",
            (0, 1),  # a plain pair: the bounds, spaced linearly
            height=height,
        ),
        panel(
            "logarithmic",
            mp.logscale(1e-7, 1.0),  # a scale: the bounds and their spacing
            height=height,
        ),
    )
    print(plot)
    if save:
        plot.saveimg(save)


def panel(
    title: str,
    vrange: tuple[float, float] | mp.scale,
    height: int,
) -> mp.plot:
    """The sources drawn on one scale, with a bar labelled over its
    interval."""
    picture = mp.function2(
        peaks,
        xrange=(-4.0, 4.0),
        yrange=(-4.0, 4.0),
        width=2*height,
        height=height,
        vrange=vrange,
        colormap=mp.viridis,
    )
    return mp.hstack(
        mp.axes(picture, title=f" {title} ", xfmt="{x:g}", yfmt="{y:g}"),
        # centring the bar in the axes' extra rows sits it level with the
        # picture, whatever the height
        mp.center(
            mp.axes(
                mp.colorbar(picture, colormap=mp.viridis, length=height),
                east="label",
                yfmt="{y:g}",
            ),
            height=height + 2,
        ),
    )


def peaks(xy: np.ndarray) -> np.ndarray:
    """A unit Gaussian at the origin, and one a hundred times fainter and
    much narrower out towards a corner."""
    bright = np.exp(-(xy[:, 0] ** 2 + xy[:, 1] ** 2) / 2)
    faint = np.exp(-((xy[:, 0] - 3.2) ** 2 + (xy[:, 1] + 3.2) ** 2) / 0.25)
    return bright + 0.01 * faint


if __name__ == "__main__":
    tyro.cli(main)
