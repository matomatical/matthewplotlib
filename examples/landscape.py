"""
A wireframe landscape scrolling under a banded sun, drawn entirely out of
straight lines in space.

Everything on screen is one call to `mp.line3`: the mesh of the terrain, and
the sun behind it. There is no surface and no shading anywhere -- the terrain
reads as a landscape only because a grid drawn in perspective does, and the sun
reads as a sun only because a stack of horizontal chords does. Both are wires,
handed to the same projection, and the near ones come out long while the far
ones converge.

How a mesh becomes one line: every wire of the grid is a separate stroke, and a
non-finite point ends a stroke, so all thirty-odd wires travel as one series
with gaps punched between them. The camera sees the whole soup of segments at
once.

Depth does the rest of the work. Colour runs from magenta at the camera's feet
to deep blue at the horizon, which is the only cue the wires ever get that they
are far away, and the sun is banded rather than filled because horizontal
chords of a circle, spaced further apart towards the bottom, are how that sun
has always been drawn.

The scroll loops seamlessly: the terrain is a sum of sines whose periods all
divide the distance travelled over one loop, so after a whole loop the
landscape is exactly the landscape it started as.
"""

import tyro
import numpy as np

import matthewplotlib as mp


# The mesh: how many wires across and back, and how far apart they are laid.
ACROSS = 25
BACK = 10
SPACING = 1.0

# A cell is twice as tall as it is wide and braille splits it two dots across
# by four down, so the view has half as much resolution to spend up the screen
# as across it. Crosswise wires are laid every few rows to match, or they crowd
# into an unreadable band as they approach the horizon.
CROSSWISE = 2

# How far in front of the camera the mesh begins, and how high the camera flies
# over the ground plane.
AHEAD = 1.2
ALTITUDE = 1.15
TILT = 0.15

# The terrain: sines with these periods, in units of distance, each with this
# amplitude. Every period divides SCROLL, so the whole field repeats over it.
SCROLL = 12.0
RIDGES = ((SCROLL / 2, 0.5), (SCROLL / 3, 0.3), (SCROLL / 6, 0.15))

# The ridges are gentle down the middle and steepen towards the sides, where
# the ground climbs as well, which is what makes this read as a valley to fly
# along rather than a rumpled sheet.
VALLEY_RISE = 8

# The sun: how big, how high, how far off, and how its bands are spaced. The
# gaps grow towards the bottom of the disc.
SUN_RADIUS = 11.0
SUN_HEIGHT = 6.5
SUN_DISTANCE = 45.0
SUN_BANDS = 9

# The palette: near ground, far ground, and the two ends of the sun.
NEAR = np.array([255, 64, 180])
FAR = np.array([18, 8, 48])
SUN_TOP = np.array([255, 240, 120])
SUN_BOTTOM = np.array([255, 40, 130])


def main(
    num_frames: int = 48,
    fps: int = 20,
    width: int = 76,
    height: int = 22,
    thickness: float = 1.0,
    sun: bool = True,
    loop: bool = True,
    save: str | None = None,
):
    """A vaporwave landscape, scrolling towards you for ever.

    The animation is periodic in `num_frames`, so it loops seamlessly however
    many frames you ask for. Pass `--no-sun` for the terrain alone, or a larger
    `--thickness` for a heavier wire.
    """
    # one whole period of the terrain per loop, so it comes back to itself
    travelled = SCROLL * np.arange(num_frames) / num_frames

    frames = []
    for distance in travelled:
        wires, colors = terrain(distance)
        if sun:
            sun_wires, sun_colors = banded_sun()
            wires = np.concatenate([wires, sun_wires])
            colors = np.concatenate([colors, sun_colors])
        frames.append(mp.line3(
            (wires[:, 0], wires[:, 1], wires[:, 2], colors),
            # the camera flies along the valley, tipped just far enough
            # towards the ground to put the horizon a third of the way up
            camera_position=np.array([0.0, ALTITUDE, 0.0]),
            camera_target=np.array([0.0, ALTITUDE - TILT, -1.0]),
            vertical_fov_degrees=52,
            width=width,
            height=height,
            thickness=thickness,
        ))

    animation = mp.tstack(*frames, fps=fps)
    animation.play(loop=loop)

    if save:
        animation.savegif(save, bgcolor="black")


def ground(xs: np.ndarray, zs: np.ndarray) -> np.ndarray:
    """The height of the terrain under a set of points on the ground plane.

    A sum of sines in the direction of travel, so the field is periodic and the
    scroll can loop, times a profile across the valley, so the middle of the
    view stays low and the sides climb.
    """
    ridges = sum(
        amplitude * np.sin(2 * np.pi * zs / period)
        for period, amplitude in RIDGES
    )
    # a second, slower undulation across the valley, so the ridges are not
    # straight walls running to the horizon
    width = ACROSS * SPACING
    ridges = ridges * (0.7 + 0.3 * np.cos(2 * np.pi * xs / (width / 3)))
    # ...and both the ridges and the ground itself climb towards the sides
    across = np.abs(xs) / (width / 2)
    return (
        ridges * (0.35 + 0.65 * across ** 2)
        + VALLEY_RISE * across ** 3
    )


def terrain(distance: float) -> tuple[np.ndarray, np.ndarray]:
    """The mesh, as one broken line, with a colour for every point.

    Both the grid and the ground under it move with `distance`. The wires walk
    towards the camera and wrap after one cell, so that the grid appears to
    travel without ever running out of wires, and the terrain is sampled a full
    `distance` further along, so what passes underneath is new ground.

    A cell here means a cell of the *drawn* grid, `CROSSWISE * SPACING`, and not
    of the grid the terrain is sampled on. Wrapping every `SPACING` instead
    steps the crosswise wires by half of their own spacing, which lands them
    exactly between where they were: they appear to swap places every few
    frames rather than to travel.
    """
    xs = np.linspace(-1, 1, ACROSS) * (ACROSS - 1) / 2 * SPACING
    walked = distance % (CROSSWISE * SPACING)
    zs = -(AHEAD + np.arange(BACK) * SPACING - walked)
    grid_x, grid_z = np.meshgrid(xs, zs)
    grid_y = ground(grid_x, grid_z - distance)

    # the wires: every row of the grid, then every column of it
    lengthwise = np.stack([grid_x, grid_y, grid_z], axis=-1)
    rows = lengthwise[::CROSSWISE]
    columns = lengthwise.transpose(1, 0, 2)

    # depth decides colour, so the mesh fades into the horizon
    depth = np.clip((-grid_z - AHEAD) / (BACK * SPACING), 0, 1)
    tint = NEAR + (FAR - NEAR) * depth[..., np.newaxis] ** 0.7

    return (
        np.concatenate([
            break_strokes(rows.reshape(-1, 3), ACROSS),
            break_strokes(columns.reshape(-1, 3), BACK),
        ]),
        np.concatenate([
            break_strokes(tint[::CROSSWISE].reshape(-1, 3), ACROSS, fill=0.0),
            break_strokes(
                tint.transpose(1, 0, 2).reshape(-1, 3), BACK, fill=0.0,
            ),
        ]),
    )


def banded_sun() -> tuple[np.ndarray, np.ndarray]:
    """The sun, as horizontal chords of a circle standing far off the camera.

    The bands crowd together at the top of the disc and open up towards the
    bottom, which is the whole of the effect: the gaps are the sun.
    """
    # band positions eased towards the top of the disc, so the gaps below grow
    fraction = np.linspace(0, 1, SUN_BANDS) ** 1.55
    ys = SUN_HEIGHT + SUN_RADIUS * (1 - 2 * fraction)
    heights = ys - SUN_HEIGHT
    half_widths = np.sqrt(np.maximum(SUN_RADIUS ** 2 - heights ** 2, 0))

    lefts = np.stack([
        -half_widths,
        ys,
        np.full(SUN_BANDS, -SUN_DISTANCE),
    ], axis=-1)
    rights = lefts * np.array([-1.0, 1.0, 1.0])

    # each band is a stroke of its own, so gaps go between them
    gaps = np.full((SUN_BANDS, 3), np.nan)
    points = np.stack([lefts, rights, gaps], axis=1).reshape(-1, 3)

    tint = SUN_BOTTOM + (SUN_TOP - SUN_BOTTOM) * (1 - fraction[:, np.newaxis])
    colors = np.repeat(tint, 3, axis=0)

    return points, colors


def break_strokes(
    values: np.ndarray,
    stride: int,
    fill: float = np.nan,
) -> np.ndarray:
    """Punch a row of `fill` into a flat array every `stride` rows of it.

    A row of the mesh is a stroke and the next row is another: without a gap
    between them, the end of one would be joined to the start of the next by a
    wire straight back across the terrain. Colours are broken the same way, to
    stay lined up with the points they belong to, but with a value that is
    merely ignored rather than one that means "gap".
    """
    columns = values.shape[-1]
    strokes = values.reshape(-1, stride, columns)
    gaps = np.full((len(strokes), 1, columns), fill)
    return np.concatenate([strokes, gaps], axis=1).reshape(-1, columns)


if __name__ == "__main__":
    tyro.cli(main)
