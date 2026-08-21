"""
Every way of drawing an axis, around one two-slit interference pattern.

Two sources eight millimetres apart, radiating at a six millimetre wavelength,
make a field of hyperbolic fringes. The same field is drawn over and over,
changing only its axes: what each side is, which sides carry the scale, how
heavy the rules are, and how a quantity with a single coordinate is labelled.

By Claude Opus 5.
"""

import tyro
import numpy as np
import matthewplotlib as mp


WAVELENGTH = 6.0            # mm
SEPARATION = 8.0            # mm, between the two sources
XRANGE = (-12.0, 12.0)      # mm
YRANGE = (-6.0, 6.0)        # mm
ZRANGE = (-2.0, 2.0)        # amplitude, in units of one source's

COLUMN = 20                 # every cell of the gallery, padded out to this

# the captions are coloured rather than left to the terminal's foreground, so
# that they survive being exported to an image, where an uncoloured character
# comes out transparent
HEADING = (0.55, 0.55, 0.60)
CAPTION = (0.40, 0.40, 0.45)


def interference(xy):
    """Amplitude where two circular waves meet, at each point of the field."""
    k = 2 * np.pi / WAVELENGTH
    left = np.hypot(xy[:, 0] + SEPARATION / 2, xy[:, 1])
    right = np.hypot(xy[:, 0] - SEPARATION / 2, xy[:, 1])
    return np.sin(k * left) + np.sin(k * right)


def field(width: int = 14, height: int = 4) -> mp.function2:
    return mp.function2(
        interference,
        xrange=XRANGE,
        yrange=YRANGE,
        width=width,
        height=height,
        vrange=ZRANGE,
        colormap=mp.divblues,
    )


def scale(direction: mp.Direction, length: int = 14) -> mp.colorbar:
    """A gradient standing for the amplitude, carrying one coordinate."""
    return mp.colorbar(
        ZRANGE,
        colormap=mp.divblues,
        direction=direction,
        length=length,
    )


def panel(caption: str, plot: mp.plot) -> mp.plot:
    """One cell of the gallery: a caption over a plot, padded out to a common
    width so that the columns line up. Stacking takes care of the heights."""
    body = mp.vstack(mp.text(caption, fgcolor=CAPTION), plot)
    if body.width >= COLUMN:
        return body
    return mp.hstack(body, mp.blank(height=1, width=COLUMN - body.width))


def gallery(heading: str, *panels: mp.plot) -> mp.plot:
    return mp.vstack(mp.text(heading, fgcolor=HEADING), mp.hstack(*panels))


def main(save: str | None = None):
    """Every way of drawing an axis, around one interference pattern."""
    ticks = {"xfmt": "{x:.0f}", "yfmt": "{y:.0f}"}

    # one side at a time: the other three are ruled, so only the north moves,
    # and it moves right under the caption where the difference shows
    modes = gallery(
        "each side is one of four things, here the northern one:",
        *[
            panel(f'north="{mode}"', mp.axes(
                field(),
                north=mode, east="rule", south="rule", west="rule",
                **ticks,
            ))
            for mode in ("crop", "pad", "rule", "label")
        ],
    )

    # which sides carry the scale, and what is inferred when none is named
    sides = gallery(
        "which sides carry the scale:",
        panel("inferred", mp.axes(field(), **ticks)),
        panel('north="label"', mp.axes(field(), north="label", **ticks)),
        panel("all four", mp.axes(
            field(), north="label", east="label", south="label", west="label",
            **ticks,
        )),
        panel("none of them", mp.axes(
            field(), north="crop", east="crop", south="crop", west="crop",
            **ticks,
        )),
    )

    # the rules themselves, at each of the four weights
    weights = gallery(
        "the weight of the rules:",
        *[
            panel(name.lower(), mp.axes(field(), style=style, **ticks))
            for name, style in (
                ("LIGHT", mp.LineStyle.LIGHT),
                ("HEAVY", mp.LineStyle.HEAVY),
                ("ROUND", mp.LineStyle.ROUND),
                ("DOUBLE", mp.LineStyle.DOUBLE),
            )
        ],
    )

    # a gradient carries one coordinate, so only the sides facing it can be
    # labelled, and the rest are dropped rather than boxing a strip
    scales = gallery(
        "one coordinate, so one labelled side:",
        panel("inferred", mp.axes(scale("up", length=4), yfmt="{y:.1f}")),
        panel('east="label"', mp.axes(
            scale("up", length=4), east="label", yfmt="{y:.1f}",
        )),
        panel("inferred", mp.axes(scale("right"), xfmt="{x:.1f}")),
        panel('north="label"', mp.axes(
            scale("right"), north="label", xfmt="{x:.1f}",
        )),
    )

    # and the reason for all of it: a map with its scale beside it
    pattern = field(width=44, height=8)
    together = mp.vstack(
        mp.text("and what it is all for:", fgcolor=HEADING),
        mp.hstack(
            mp.axes(
                pattern,
                title=" two-slit interference ",
                xlabel="x (mm)",
                ylabel="y (mm)",
                **ticks,
            ),
            mp.blank(height=1, width=2),
            mp.axes(
                mp.colorbar(pattern, colormap=mp.divblues, length=5),
                east="label",
                ylabel="amp",
                yfmt="{y:+.1f}",
            ),
        ),
    )

    plot = mp.vstack(modes, sides, weights, scales, together)
    print(plot)
    if save:
        plot.saveimg(save)


if __name__ == "__main__":
    tyro.cli(main)
