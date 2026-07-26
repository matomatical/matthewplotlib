"""
The Amiga Boing Ball, 1984: the demo that proved a machine could animate.

A red and white checkered sphere spinning about a tilted axis, bouncing across a
purple grid. There is no asset and no video decoder here -- every frame is
arithmetic on a coordinate grid, so the whole animation is one array with a time
axis, which is exactly what `mp.animation` takes.

That makes this the example for animations as *values*: the frames are computed
up front, furnished with a border, and only then played or saved. Compare
`examples/teapot.py`, which cannot precompute anything because it is a loop.
"""

import tyro
import numpy as np

import matthewplotlib as mp


# The ball, as the Amiga drew it: sixteen segments of longitude, eight bands of
# latitude, alternating red and white, spinning about an axis tilted towards the
# viewer.
SEGMENTS = 16
BANDS = 8
TILT_DEGREES = 17.0

RED = np.array([222, 42, 48])
WHITE = np.array([243, 243, 243])
GRID = np.array([116, 48, 148])
BACKDROP = np.array([58, 18, 78])

# Light from over the viewer's left shoulder. Fixed in screen space, not on the
# ball, so the sheen stays put while the checks rotate underneath it.
LIGHT = np.array([-0.45, -0.62, 0.65])


def main(
    num_frames: int = 48,
    fps: int = 20,
    width: int = 64,
    height: int = 24,
    bounces: int = 2,
    spins: float = 1.0,
    loop: bool = True,
    save: str | None = None,
):
    """The Amiga Boing Ball, bouncing.

    The animation is periodic in `num_frames`, so it loops seamlessly however
    many frames you ask for.
    """
    animation = mp.animation(
        boing(num_frames, height, width, bounces=bounces, spins=spins),
        fps=fps,
    ).map(lambda frame: mp.border(frame, title=" boing "))

    animation.play(loop=loop)

    if save:
        animation.savegif(save)


def boing(
    num_frames: int,
    height: int,
    width: int,
    bounces: int = 2,
    spins: float = 1.0,
) -> np.ndarray:            # uint8[num_frames, 2 * height, width, 3]
    """Render the whole animation at once, as one array of RGB frames.

    Vectorised over time as well as space: every array below carries a leading
    frame axis, so there is no loop anywhere and the result goes straight into
    `mp.animation`.

    Half-block rendering puts two image rows in every character row, so the image
    is `2 * height` rows tall and its pixels come out roughly square.
    """
    rows, cols = 2 * height, width

    # Pixel centres, with x widened by the aspect ratio so that a circle on the
    # screen is a circle rather than an ellipse.
    aspect = cols / rows
    y = (np.arange(rows) + 0.5) / rows * 2 - 1
    x = ((np.arange(cols) + 0.5) / cols * 2 - 1) * aspect
    x, y = np.meshgrid(x, y)                        # [rows, cols]

    # One phase per frame, on [0, 1). Everything below is periodic in it, which
    # is what makes the animation loop.
    phase = (np.arange(num_frames) / num_frames)[:, None, None]

    # Across and back once per cycle, on a triangle wave.
    reach = aspect - 0.72
    cx = reach * (4 * np.abs(phase - 0.5) - 1)
    # Bouncing off the floor: the height of a ball under gravity, over and over.
    floor = 0.30
    cy = floor - 0.92 * np.abs(np.sin(np.pi * bounces * phase))

    # Squashed on impact and stretched sideways to match, so the ball keeps its
    # area. Impact is where the bounce height is zero.
    squash = 0.18 * np.maximum(0, 1 - 7 * np.abs(np.sin(np.pi * bounces * phase)))
    radius = 0.62
    rx, ry = radius * (1 + squash / 2), radius * (1 - squash)

    # The ball as a sphere seen head on: at each pixel, where on the unit sphere
    # the surface faces. Outside the disc there is no surface, and `nz` is
    # clamped rather than masked so the arithmetic stays uniform.
    nx = (x - cx) / rx
    ny = (y - cy) / ry
    off_ball = nx * nx + ny * ny
    nz = np.sqrt(np.clip(1 - off_ball, 0, None))
    on_ball = off_ball <= 1

    # The spin axis, and a frame around it to measure longitude from. Latitude
    # comes from the angle to the axis, so the bands stay put while the segments
    # sweep past -- the whole trick of the original.
    tilt = np.radians(TILT_DEGREES)
    axis = np.array([np.sin(tilt), -np.cos(tilt), 0.0])
    east = np.array([0.0, 0.0, 1.0])
    north = np.cross(axis, east)

    latitude = np.arccos(np.clip(
        nx * axis[0] + ny * axis[1] + nz * axis[2], -1, 1,
    )) / np.pi
    longitude = np.arctan2(
        nx * north[0] + ny * north[1] + nz * north[2],
        nx * east[0] + ny * east[1] + nz * east[2],
    ) / (2 * np.pi)
    checker = (
        np.floor((longitude - spins * phase) * SEGMENTS).astype(int)
        + np.floor(latitude * BANDS).astype(int)
    ) % 2 == 0
    ball = np.where(checker[..., None], RED, WHITE)

    # Diffuse light plus a tight highlight, in screen space.
    light = LIGHT / np.linalg.norm(LIGHT)
    facing = np.clip(
        nx * light[0] + ny * light[1] + nz * light[2], 0, None,
    )[..., None]
    ball = ball * (0.34 + 0.66 * facing) + 255 * 0.45 * facing**26

    # The backdrop: a grid of squares, brightened along the lines. Ruled in whole
    # pixels rather than in the coordinates above, because at this resolution a
    # cell is only a few pixels wide -- a line placed as a fraction of a cell
    # falls between pixel centres and is sampled away almost everywhere.
    cell = max(4, rows // 6)
    down = np.arange(rows) % cell == 0
    across = np.arange(cols) % cell == 0
    line = down[:, None] | across[None, :]
    backdrop = np.where(line[..., None], GRID, BACKDROP)
    backdrop = np.broadcast_to(backdrop, (num_frames, rows, cols, 3)).copy()

    # A soft shadow on it, down and to the right, flattened as if cast on a
    # floor and fading as the ball climbs away from it.
    shadow = (
        ((x - cx - 0.13) / (rx * 1.02)) ** 2 + ((y - cy - 0.16) / (ry * 0.62)) ** 2
    )
    darkness = 0.55 * np.clip(1 - shadow, 0, 1) ** 0.6 * np.clip(
        1 - (floor - cy) * 0.55, 0.25, 1,
    )
    backdrop = backdrop * (1 - darkness[..., None])

    frame = np.where(on_ball[..., None], ball, backdrop)
    return np.clip(frame, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    tyro.cli(main)
