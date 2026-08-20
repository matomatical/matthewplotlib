"""
Colour scales, and the bars that stand for them.

An island's elevation as a heatmap, described by a colorbar: the same scale
in each of the four directions, then beside the map it belongs to, then read
off whatever other plot happens to carry one.

By Claude Opus 5.
"""

import datetime

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

COLUMN = 19                 # every cell of a gallery, padded out to this

# the captions are coloured rather than left to the terminal's foreground, so
# that they survive being exported to an image, where an uncoloured character
# comes out transparent
HEADING = (0.55, 0.55, 0.60)
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


def island(width: int = 42, height: int = 8) -> mp.heatmap:
    """The map itself, carrying both its coordinates and its colour scale."""
    return mp.heatmap(
        elevation(width=width, height=height),
        colormap=mp.viridis,
        vrange=ZRANGE,
        xrange=XRANGE,
        yrange=YRANGE,
    )


def landfalls(count: int = 3000) -> tuple[np.ndarray, np.ndarray]:
    """Where a scattering of gulls came down, in the map's own coordinates."""
    rng = np.random.default_rng(7)
    east = rng.normal(-2.0, 6.0, count)
    north = rng.normal(1.0, 3.0, count)
    return east, north


def rainfall(days: int = 98) -> dict[datetime.date, float]:
    """Millimetres of rain on each day of one season, wettest in the middle."""
    rng = np.random.default_rng(20)
    season = np.sin(np.linspace(0, np.pi, days)) ** 2
    daily = np.clip(24 * season + rng.normal(0.0, 3.0, days), 0.0, None)
    start = datetime.date(2025, 3, 1)
    return {
        start + datetime.timedelta(days=day): float(mm)
        for day, mm in enumerate(daily)
    }


def panel(caption: str, plot: mp.plot) -> mp.plot:
    """One cell of a gallery: a caption over a plot, padded out to a common
    width so that the columns line up. Stacking takes care of the heights."""
    body = mp.vstack(mp.text(caption, fgcolor=CAPTION), plot)
    if body.width >= COLUMN:
        return body
    return mp.hstack(body, mp.blank(height=1, width=COLUMN - body.width))


def gallery(heading: str, *panels: mp.plot) -> mp.plot:
    return mp.vstack(mp.text(heading, fgcolor=HEADING), mp.hstack(*panels))


def main(save: str | None = None):
    """Colour scales, and the bars that stand for them."""

    # each bar is labelled along whichever side faces the one coordinate it
    # carries, and left alone on the other three
    directions = gallery(
        "the scale runs whichever way you point it:",
        *[
            panel(f'"{direction}"', mp.axes(
                mp.colorbar(
                    ZRANGE,
                    colormap=mp.viridis,
                    direction=direction,
                    length=5 if direction in ("up", "down") else 12,
                ),
                **WHOLE,
            ))
            for direction in ("up", "down", "right", "left")
        ],
    )

    # a bar as thick as it is long is a swatch of the whole scale
    thicknesses = gallery(
        "and it is as thick as you ask for:",
        *[
            panel(f"thickness={thickness}", mp.axes(
                mp.colorbar(
                    ZRANGE,
                    colormap=mp.viridis,
                    length=5,
                    thickness=thickness,
                ),
                **WHOLE,
            ))
            for thickness in (1, 2, 4, 8)
        ],
    )

    # the reason for all of it: the bar takes the interval and the colormap
    # off the map, so the two cannot disagree about what a colour means
    land = island()
    together = mp.vstack(
        mp.text("beside the map it describes:", fgcolor=HEADING),
        mp.hstack(
            mp.axes(
                land,
                title=" elevation ",
                xlabel="km east",
                ylabel="km north",
                **WHOLE,
            ),
            mp.blank(height=1, width=2),
            mp.axes(
                mp.colorbar(land, length=5),
                east="label",
                ylabel="m",
                **WHOLE,
            ),
        ),
    )

    # neither of these was told its interval. The histogram settled on its
    # own fullest bin and the strip of weeks on its wettest day, and the bar
    # reports whatever each of them arrived at
    east, north = landfalls()
    gulls = mp.histogram2(
        east,
        north,
        width=20,
        height=5,
        xrange=XRANGE,
        yrange=YRANGE,
        colormap=mp.magma,
    )
    rain = mp.weeks(rainfall(), colormap=mp.blues)
    elsewhere = mp.vstack(
        mp.text("or off any plot that worked out its own:", fgcolor=HEADING),
        mp.hstack(
            mp.axes(gulls, title=" gulls ", **WHOLE),
            mp.blank(height=1, width=1),
            mp.axes(
                mp.colorbar(gulls, length=5),
                east="label",
                **WHOLE,
            ),
            mp.blank(height=1, width=2),
            rain,
            mp.blank(height=1, width=1),
            # the blank rows drop the bar past the captions, level with the
            # days themselves
            mp.vstack(
                mp.blank(height=2, width=1),
                mp.axes(
                    mp.colorbar(rain, length=7),
                    east="label",
                    ylabel="mm",
                    **WHOLE,
                ),
            ),
        ),
    )

    plot = mp.vstack(directions, thicknesses, together, elsewhere)
    print(plot)
    if save:
        plot.saveimg(save)


if __name__ == "__main__":
    tyro.cli(main)
