"""
The Amiga Boing Ball, 1984, animated the way the Amiga animated it.

The ball is not redrawn each frame. It is drawn once, as a grid of *colour
indices*, and it appears to spin because the palette those indices point into is
rewritten between frames -- colour cycling, which on the Amiga cost a handful of
register writes rather than a screenful of pixels. The bounce is the other half of
the trick: the ball is a rigid sprite moved to whole-pixel positions, as the
blitter moved it, so nothing about the ball's own pixels changes as it travels.

Both halves are lookups, which makes the whole animation two indexing
expressions:

* compositing, `sprite[r, c]` gathered at per-frame offsets, giving one index
  image per frame;
* colouring, `palettes[t, index]`, giving the frames themselves.

No shading and no gradients, because there were none: the ball reads as a sphere
purely from the way a checkerboard distorts across one. And because every frame is
a lookup into one precomputed sprite, this is the example for animations as
*values* -- the whole thing is an array with a time axis, which is what
`mp.animation` takes. Compare `examples/teapot.py`, which cannot precompute
anything because it is a loop.
"""

import tyro
import numpy as np

import matthewplotlib as mp


# The ball as the Amiga drew it: sixteen segments of longitude, eight bands of
# latitude, alternating red and white, spinning about an axis tilted towards the
# viewer.
SEGMENTS = 16
BANDS = 8
TILT_DEGREES = 17.0

# Longitude is indexed more finely than it is coloured, and it has to be. Roll the
# palette by exactly one segment and every check lands where its neighbour was --
# which for a two-colour checkerboard is indistinguishable from swapping red and
# white, so the ball would flicker instead of turning. Indexing four cells to a
# segment buys quarter-segment steps, and the ball turns.
CELLS_PER_SEGMENT = 4
CELLS = SEGMENTS * CELLS_PER_SEGMENT

# The palette. Everything the animation can show is one of these.
BACKDROP, GRID, SHADOW = 0, 1, 2
BALL = 3                        # ..3 + 2 * CELLS, two parities per cell
PALETTE = {
    BACKDROP: (58, 18, 78),
    GRID: (116, 48, 148),
    SHADOW: (38, 11, 52),
}
RED = (222, 42, 48)
WHITE = (243, 243, 243)


def main(
    num_frames: int = 48,
    fps: int = 20,
    width: int = 64,
    height: int = 24,
    bounces: int = 2,
    spins: int = 1,
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
    )

    animation.play(loop=loop)

    if save:
        # Every character cell is eight pixels wide and holds two rows of
        # half-block, so each pixel of the animation is rendered as an eight by
        # eight square. Keeping every eighth pixel undoes exactly that, and the
        # gif comes out at the animation's true resolution.
        animation.savegif(save, downscale=8)


def boing(
    num_frames: int,
    height: int,
    width: int,
    bounces: int = 2,
    spins: int = 1,
) -> np.ndarray:            # uint8[num_frames, 2 * height, width, 3]
    """Render the animation: composite the indices, then colour them.

    Half-block rendering puts two image rows in every character row, so the image
    is `2 * height` rows tall and its pixels come out roughly square.
    """
    rows, cols = 2 * height, width
    ball, ball_mask = ball_sprite(diameter=min(rows, cols) * 5 // 8)
    reach = ball.shape[0]

    # Where the ball is, in whole pixels, once per frame. One phase per frame on
    # [0, 1), and everything below is periodic in it, which is what makes the
    # animation loop.
    phase = np.arange(num_frames) / num_frames
    floor_top = rows - reach - 1
    # Across and back once per cycle, on a triangle wave: wall, wall, wall.
    left = np.round((cols - reach) * np.abs(2 * phase - 1)).astype(int)
    # The height of a ball under gravity, over and over.
    height_above = np.abs(np.sin(np.pi * bounces * phase))
    top = np.round(floor_top * (1 - height_above)).astype(int)

    # The backdrop, then the shadow lying on the floor, then the ball over both.
    index = np.broadcast_to(backdrop(rows, cols), (num_frames, rows, cols)).copy()
    shadow, shadow_mask = shadow_sprite(reach)
    floor = np.full(num_frames, rows - shadow.shape[0])
    index = composite(index, shadow, shadow_mask, top=floor, left=left)
    index = composite(index, ball, ball_mask, top=top, left=left)

    # Colour cycling: one palette per frame, the ball's entries rolled round by a
    # whole number of longitude cells. This is the only thing that turns the ball.
    roll = np.round(spins * CELLS * phase).astype(int) % CELLS
    palettes = np.stack([palette(r) for r in roll])         # [frames, entries, 3]
    return palettes[np.arange(num_frames)[:, None, None], index]


def ball_sprite(diameter: int) -> tuple[np.ndarray, np.ndarray]:
    """The ball, once: a colour index per pixel, and which pixels are the ball.

    The indices carry longitude and latitude, not colour. What colour a given
    longitude comes out is the palette's business, and changing its mind is what
    makes the ball spin.
    """
    span = np.linspace(-1, 1, diameter)
    u, v = np.meshgrid(span, span)
    off_ball = u * u + v * v
    on_ball = off_ball <= 1
    w = np.sqrt(np.clip(1 - off_ball, 0, None))

    # The spin axis, and a frame around it to measure longitude from. Latitude is
    # the angle to the axis, so the bands stay put while the segments sweep past.
    tilt = np.radians(TILT_DEGREES)
    axis = np.array([np.sin(tilt), -np.cos(tilt), 0.0])
    east = np.array([0.0, 0.0, 1.0])
    north = np.cross(axis, east)
    normal = np.stack([u, v, w], axis=-1)

    latitude = np.arccos(np.clip(normal @ axis, -1, 1)) / np.pi
    longitude = np.arctan2(normal @ north, normal @ east) / (2 * np.pi) + 0.5
    cell = np.clip((longitude * CELLS).astype(int), 0, CELLS - 1)
    parity = (latitude * BANDS).astype(int) % 2
    return BALL + 2 * cell + parity, on_ball


def shadow_sprite(reach: int) -> tuple[np.ndarray, np.ndarray]:
    """A flattened ellipse for the ball to sit on, so the bounce has a floor.

    Only as tall as the ellipse needs, so it can be stamped flush against the
    bottom of the frame and stay there while the ball climbs away from it.
    """
    depth = max(3, reach // 4)
    v, u = np.meshgrid(
        np.linspace(-1, 1, depth), np.linspace(-1, 1, reach), indexing="ij"
    )
    inside = (u / 0.86) ** 2 + v**2 <= 1
    return np.full(inside.shape, SHADOW), inside


def backdrop(rows: int, cols: int) -> np.ndarray:
    """A grid of squares, ruled in whole pixels.

    Whole pixels rather than a fraction of a cell: a cell here is only a few
    pixels across, so a line placed as a fraction of one falls between pixel
    centres and is sampled away almost everywhere.
    """
    cell = max(4, rows // 6)
    down = np.arange(rows) % cell == 0
    across = np.arange(cols) % cell == 0
    return np.where(down[:, None] | across[None, :], GRID, BACKDROP)


def composite(
    index: np.ndarray,      # int[frames, rows, cols]
    sprite: np.ndarray,     # int[reach, reach]
    mask: np.ndarray,       # bool[reach, reach]
    top: np.ndarray,        # int[frames]
    left: np.ndarray,       # int[frames]
) -> np.ndarray:
    """Stamp a sprite onto every frame at its own whole-pixel offset.

    A gather rather than a loop over frames: read the sprite at the offset each
    frame asks for, and keep the reading wherever the sprite covers that pixel.
    Whole-pixel offsets are what make this exact -- the ball is never resampled,
    just as the blitter never resampled it.
    """
    frames, rows, cols = index.shape
    depth, reach = sprite.shape
    row = np.arange(rows)[None, :, None] - top[:, None, None]
    col = np.arange(cols)[None, None, :] - left[:, None, None]
    covered = (row >= 0) & (row < depth) & (col >= 0) & (col < reach)
    row, col = np.clip(row, 0, depth - 1), np.clip(col, 0, reach - 1)
    return np.where(covered & mask[row, col], sprite[row, col], index)


def palette(roll: int) -> np.ndarray:       # uint8[entries, 3]
    """The palette with the ball's longitude cells rotated round by `roll`.

    A cell's colour is the colour of the segment it belonged to `roll` cells ago,
    so rewriting this and nothing else turns the ball.
    """
    entries = np.zeros((BALL + 2 * CELLS, 3), dtype=np.uint8)
    for i, rgb in PALETTE.items():
        entries[i] = rgb
    cell = np.arange(CELLS)
    segment = ((cell - roll) % CELLS) // CELLS_PER_SEGMENT
    for parity in (0, 1):
        checked = (segment + parity) % 2 == 0
        entries[BALL + 2 * cell + parity] = np.where(
            checked[:, None], np.array(RED), np.array(WHITE)
        )
    return entries


if __name__ == "__main__":
    tyro.cli(main)
