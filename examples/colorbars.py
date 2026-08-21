"""
Colour scales, and the bars that stand for them.

An island's elevation as a heatmap, with a bar beside it over the same
interval, then the same scale in each of the four directions and at four
thicknesses.

By Claude Opus 5.
"""

import tyro
import numpy as np
import matthewplotlib as mp


XRANGE = (-15.0, 15.0)      # km east of the island's centre
YRANGE = (-9.0, 9.0)        # km north
ZRANGE = (-400.0, 900.0)    # m relative to sea level

# each hill: where its summit is, how far it spreads, and how high it reaches
HILLS = (
    (-4.0, 1.5, 7.0, 900.0),
    (5.0, -2.0, 5.0, 700.0),
    (8.0, 4.0, 3.0, 500.0),
)

MAP = 63                    # cells of map across, which with its scale is 76
# a cell is twice as tall as it is wide and holds two pixels, so this many
# rows keeps the hills from smearing sideways across a map this wide
HEIGHT = 12                 # cells of map down, and so cells of bar
GAP = 2                     # cells between the map and its scale

# the captions are coloured rather than left to the terminal's foreground, so
# that they survive being exported to an image, where an uncoloured character
# comes out transparent
CAPTION = (0.40, 0.40, 0.45)

WHOLE = {"yfmt": "{y:.0f}", "xfmt": "{x:.0f}"}


def elevation(width: int, height: int) -> np.ndarray:
    """The height of the land above sea level, over the whole map."""
    X, Y = mp.window(
        xrange=XRANGE,
        yrange=YRANGE,
        width=width,
        height=height,
    ).pixel_centres()
    land = np.zeros(X.shape)
    for east, north, spread, summit in HILLS:
        land += summit * np.exp(-((X - east)**2 + (Y - north)**2) / spread**2)
    return ZRANGE[0] + land


def island(width: int = MAP, height: int = HEIGHT) -> mp.heatmap:
    """The map itself, carrying both its coordinates and its interval."""
    return mp.heatmap(
        elevation(width=width, height=height),
        colormap=mp.viridis,
        vrange=ZRANGE,
        xrange=XRANGE,
        yrange=YRANGE,
    )


def scale(direction: mp.Direction, length: int, thickness: int = 1) -> mp.plot:
    """A bar over the elevation's interval, labelled along the one side that
    faces the single coordinate it carries."""
    return mp.axes(
        mp.colorbar(
            ZRANGE,
            colormap=mp.viridis,
            direction=direction,
            length=length,
            thickness=thickness,
        ),
        **WHOLE,
    )


def panel(caption: str, plot: mp.plot, column: int) -> mp.plot:
    """One cell of a gallery: a caption over a plot, padded out to a common
    width so that the columns line up. Stacking takes care of the heights."""
    body = mp.vstack(mp.text(caption, fgcolor=CAPTION), plot)
    if body.width >= column:
        return body
    return mp.hstack(body, mp.blank(height=1, width=column - body.width))


def main(save: str | None = None):
    """Colour scales, and the bars that stand for them."""
    # the bar borrows the map's interval, so the numbers on the two cannot
    # drift apart, and stands as tall as the map it describes
    land = island()
    together = mp.hstack(
        mp.axes(
            land,
            title=" elevation ",
            xlabel="km east",
            ylabel="km north",
            **WHOLE,
        ),
        mp.blank(height=1, width=GAP),
        # the blank row drops the bar past the map's top rule, so that its two
        # ends line up with the first and last rows of the map itself
        mp.vstack(
            mp.blank(height=1, width=1),
            mp.axes(
                mp.colorbar(
                    land,
                    colormap=mp.viridis,
                    length=HEIGHT,
                    thickness=2,
                ),
                east="label",
                ylabel="m",
                **WHOLE,
            ),
        ),
    )

    # the galleries below fill the same width, whatever the map came out at
    column = together.width // 4
    directions = mp.hstack(*[
        panel(
            f'"{direction}"',
            scale(direction, length=5 if direction in ("up", "down") else 12),
            column,
        )
        for direction in ("up", "down", "right", "left")
    ])
    thicknesses = mp.hstack(*[
        panel(
            f"thickness={thickness}",
            scale("up", length=5, thickness=thickness),
            column,
        )
        for thickness in (1, 2, 4, 8)
    ])

    plot = mp.vstack(together, directions, thicknesses)
    print(plot)
    if save:
        plot.saveimg(save)


if __name__ == "__main__":
    tyro.cli(main)
