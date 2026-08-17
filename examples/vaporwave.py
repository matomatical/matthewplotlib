"""
A wireframe landscape scrolling under a banded sun. The backdrop -- sky,
ground, sun -- is an image of half-blocks. The terrain is one `mp.line3` over
the top of it, a series per wire.

By Matthew Farrugia-Roberts and Claude Opus 5.
"""

import tyro
import numpy as np

import matthewplotlib as mp


# The mesh: how many wires across and back, and how far apart they are laid.
ACROSS = 25
BACK = 24
SPACING = 1.0

# How far short of the mesh's far end the fog closes in, in rows, so that a
# wire arrives already the colour of the ground rather than appearing; and how
# sharply it closes in, above one to keep the near grid bright.
FOG_MARGIN = 4
FOG_POWER = 2

# Crosswise wires every few rows: there is half as much resolution up the
# screen as across it, and at every row they crowd into a band.
CROSSWISE = 2

# Where the mesh begins in front of the camera, how high the camera flies, how
# far it tips down, and how much it sees.
AHEAD = 1.2
ALTITUDE = 1.15
TILT = 0.15
FOV_DEGREES = 52

# The terrain: sines with these periods, in units of distance, each with this
# amplitude. Every period divides SCROLL, so the whole field repeats over it.
SCROLL = 12.0
RIDGES = ((SCROLL / 2, 0.5), (SCROLL / 3, 0.3), (SCROLL / 6, 0.15))

# The ridges are gentle down the middle and steepen towards the sides, where
# the ground climbs too: a valley to fly along rather than a rumpled sheet.
VALLEY_RISE = 5

# The sun, in fractions of the frame: how big, how far its centre stands above
# the horizon, how many divisions the slits are cut on, how much of a division
# the slit takes, and how much the divisions widen towards the bottom (below
# one; at one they would be even).
SUN_RADIUS = 0.19
SUN_ABOVE = 0.4
SUN_BANDS = 7
SUN_SLIT = 0.45
SUN_EASE = 0.55

# The palette. The backdrop is opaque, so these are the only colours on screen.
SKY_HIGH = np.array([48, 14, 72])       # dark purple, at the top of the frame
SKY_LOW = np.array([214, 42, 138])      # magenta, down at the horizon
GROUND = np.array([9, 16, 58])          # dark blue, under it
WIRE = np.array([104, 246, 255])        # cyan, at the camera's feet
SUN_HIGH = np.array([255, 238, 126])    # yellow, at the top of the disc
SUN_LOW = np.array([255, 112, 44])      # orange, at the bottom


def main(
    num_frames: int = 48,
    fps: int = 20,
    width: int = 80,
    height: int = 24,
    thickness: float = 1.0,
    loop: bool = True,
    save: str | None = None,
):
    """A vaporwave landscape, scrolling forever."""
    # one whole period of the terrain per loop, so it comes back to itself
    travelled = SCROLL * np.arange(num_frames) / num_frames

    wires = [
        mp.line3(
            *terrain(distance),
            camera_position=np.array([0.0, ALTITUDE, 0.0]),
            camera_target=np.array([0.0, ALTITUDE - TILT, -1.0]),
            vertical_fov_degrees=FOV_DEGREES,
            width=width,
            height=height,
            thickness=thickness,
        )
        for distance in travelled
    ]

    # one backdrop for the whole loop, so that nothing behind the wires moves
    behind = mp.image(backdrop(wires, width=width, height=height))
    frames = [mp.dstack(behind, frame) for frame in wires]

    animation = mp.tstack(*frames, fps=fps)
    animation.play(loop=loop)

    if save:
        # a braille dot is four pixels square in the font, so keeping every
        # fourth pixel writes the animation at its true resolution: one pixel
        # per dot
        animation.savegif(save, downscale=4)


def horizon(height: int) -> int:
    """Which row of the backdrop's pixels the horizon falls on."""
    # the camera is tipped down by TILT, so the horizon is that far above the
    # middle of the frame. Rounded to a cell boundary: a cell straddling it
    # would lose its top pixel to any wire drawn in the same cell
    up = TILT / np.tan(np.radians(FOV_DEGREES) / 2)
    return 2 * round(((1 - up) * height - 0.5) / 2)


def skyline(wires: list[mp.plot], height: int) -> np.ndarray: # int[width]
    """The pixel row at which the ground begins, column by column.

    Read off the wires themselves, as the topmost cell each column of them ever
    reaches, so that the ground meets the terrain wherever the terrain happens
    to be rather than at a straight horizon the mountains rise above.

    Every frame at once, and not each in turn: the terrain walks, so its
    silhouette moves by a cell as it goes, and a boundary following it would
    shimmer along its length and bite pieces out of the sun. The highest the
    terrain ever reaches costs a cell of sky and holds still.
    """
    drawn = [frame.chars.isnonblank() for frame in wires]
    reached = np.stack(drawn).any(axis=0)
    return 2 * np.where(
        reached.any(axis=0),
        np.argmax(reached, axis=0),
        horizon(height) // 2,
    )


def backdrop(
    wires: list[mp.plot],
    width: int,
    height: int,
) -> np.ndarray: # uint8[2*height, width, 3]
    """The sky and the sun behind the terrain, and the ground under it."""
    rows = 2 * height
    ground_from = skyline(wires, height)

    # the sky's colour goes by height above the horizon, not above the terrain
    into_sky = np.clip(np.arange(rows) / max(horizon(height), 1), 0, 1)
    sky = SKY_HIGH + (SKY_LOW - SKY_HIGH) * into_sky[:, np.newaxis]
    is_sky = np.arange(rows)[:, np.newaxis] < ground_from[np.newaxis, :]
    pixels = np.where(is_sky[..., np.newaxis], sky[:, np.newaxis, :], GROUND)

    lit, tint = sun(rows=rows, width=width)
    # the terrain stands in front of the sun, so it takes what it covers
    lit &= is_sky
    pixels[lit] = tint[lit]
    return pixels.astype(np.uint8)


def sun(rows: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Which pixels the sun lights, and what colour they are."""
    radius = SUN_RADIUS * rows
    down, across = np.meshgrid(
        np.arange(rows) - (horizon(rows // 2) - SUN_ABOVE * radius),
        np.arange(width) - (width - 1) / 2,
        indexing="ij",
    )

    # slits on divisions down the disc, warped to widen towards the bottom. At
    # the top a division is thinner than a pixel, so the sun is solid there
    into_disc = np.clip((down + radius) / (2 * radius), 0, 1)
    division = into_disc ** SUN_EASE * SUN_BANDS
    # ...and the gradient spans the part of the disc that is above the horizon
    into_view = np.clip(into_disc / ((1 + SUN_ABOVE) / 2), 0, 1)

    lit = (
        (down ** 2 + across ** 2 <= radius ** 2)
        & (division % 1 < 1 - SUN_SLIT)
    )
    tint = SUN_HIGH + (SUN_LOW - SUN_HIGH) * into_view[..., np.newaxis]
    return lit, tint


def ground(xs: np.ndarray, zs: np.ndarray) -> np.ndarray:
    """The height of the terrain under points on the ground plane."""
    ridges = sum(
        amplitude * np.sin(2 * np.pi * zs / period)
        for period, amplitude in RIDGES
    )
    # a slower undulation across the valley, so the ridges are not straight
    # walls running to the horizon, and a climb towards either side
    width = ACROSS * SPACING
    ridges = ridges * (0.7 + 0.3 * np.cos(2 * np.pi * xs / (width / 3)))
    across = np.abs(xs) / (width / 2)
    return ridges * (0.35 + 0.65 * across ** 2) + VALLEY_RISE * across ** 3


def terrain(distance: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """The mesh, as one series per wire, each with a colour for every point."""
    xs = np.linspace(-1, 1, ACROSS) * (ACROSS - 1) / 2 * SPACING
    # wrapping after a cell of the *drawn* grid; every SPACING instead steps
    # the crosswise wires half their spacing and they swap rather than travel
    walked = distance % (CROSSWISE * SPACING)
    zs = -(AHEAD + np.arange(BACK) * SPACING - walked)
    grid_x, grid_z = np.meshgrid(xs, zs)
    grid_y = ground(grid_x, grid_z - distance)
    mesh = np.stack([grid_x, grid_y, grid_z], axis=-1)

    # depth runs the colour out at the ground's own colour, which is how the
    # fog takes a wire: by making it the same as what is behind it
    fog = (BACK - FOG_MARGIN) * SPACING
    depth = np.clip((-grid_z - AHEAD) / fog, 0, 1)[..., np.newaxis]
    tint = (WIRE + (GROUND - WIRE) * depth ** FOG_POWER).astype(np.uint8)

    return [
        *zip(mesh[::CROSSWISE], tint[::CROSSWISE]),
        *zip(mesh.transpose(1, 0, 2), tint.transpose(1, 0, 2)),
    ]


if __name__ == "__main__":
    tyro.cli(main)
