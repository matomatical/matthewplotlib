"""
A rose of rays, turning, with the pen swelling from one dot to six and back.

Between them the rays ask for everything a line rasteriser has to do:
twenty-four directions, so every octant and every slope; three lengths, so the
same pen has to draw a stub and a full radius; and a thickness that changes
under the whole figure at once.

Three things worth watching:

* the short rays and the long rays keep the same weight of line, because a
  stroke's thickness belongs to the pen and not to how far it travels;
* as the pen thickens the rays meet in a disc rather than a knot, because a
  thick stroke is its segment widened by a disc, and a bundle of them widens
  into the union of those discs;
* the colour of a ray is not one colour: each is dim at the hub and saturated
  at the tip, interpolated along the segment as it is drawn.

The figure returns to itself exactly. Ray lengths repeat every six rays, which
is a quarter of the way round, so a quarter turn over the loop puts every ray
back where an identical ray was and the last frame is the first frame again.
"""

import tyro
import numpy as np

import matthewplotlib as mp


# How far the rays reach, as a pattern repeating around the rose: one long ray
# to a quarter turn, with mediums and shorts between.
REACH = np.array([0.95, 0.45, 0.62, 0.45, 0.62, 0.45])
RAYS = 24

# The rays start off the centre, so the hub stays a ring rather than filling in
# as the pen thickens.
HUB = 0.18


def main(
    num_frames: int = 40,
    fps: int = 20,
    width: int = 48,
    height: int = 24,
    thinnest: float = 1.0,
    thickest: float = 6.0,
    caption: bool = True,
    loop: bool = True,
    save: str | None = None,
):
    """A turning rose of rays, drawn with a pen of changing thickness.

    The animation is periodic in `num_frames`, so it loops seamlessly however
    many frames you ask for. Pass `--no-caption` for the rose on its own.
    """
    phase = np.arange(num_frames) / num_frames

    # a quarter turn per loop, which is one period of the pattern of lengths
    rotations = phase * 2 * np.pi / 4
    # and one swell of the pen per loop, likewise back where it started
    thicknesses = thinnest + (thickest - thinnest) * (
        0.5 - 0.5 * np.cos(2 * np.pi * phase)
    )

    frames = []
    for rotation, thickness in zip(rotations, thicknesses):
        xs, ys, colors = rose(rotation)
        frame = mp.line(
            (xs, ys, colors),
            # the plot is `width` cells across and `height` cells down, and a
            # cell is twice as tall as it is wide, so this is the range that
            # makes the rose round rather than an ellipse
            xrange=(-width / (2 * height), width / (2 * height)),
            yrange=(-1.0, 1.0),
            width=width,
            height=height,
            thickness=thickness,
        )
        if caption:
            label = mp.text(
                f"thickness {thickness:.1f} dots",
                fgcolor=(0.5, 0.5, 0.5),
            )
            frame = frame / mp.center(label, width=frame.width)
        frames.append(frame)

    animation = mp.tstack(*frames, fps=fps)
    animation.play(loop=loop)

    if save:
        animation.savegif(save, bgcolor="black")


def rose(rotation: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The rays of the rose at a given rotation, as one broken line.

    Each ray is a stroke of its own, which is what the gaps are for: a
    non-finite point ends a stroke, so a single series can carry all
    twenty-four of them without the tip of one being joined to the hub of the
    next.
    """
    angles = np.linspace(0, 2 * np.pi, RAYS, endpoint=False) + rotation
    reach = np.tile(REACH, RAYS // len(REACH))

    directions = np.stack([np.cos(angles), np.sin(angles)])
    hubs = HUB * directions
    tips = reach * directions
    gaps = np.full((2, RAYS), np.nan)
    # hub, tip, gap, hub, tip, gap, ...: the gap after each ray is what keeps
    # the next ray from being joined onto it
    xs, ys = np.stack([hubs, tips, gaps], axis=2).reshape(2, -1)

    # each ray runs from a dim version of its hue at the hub to the full hue at
    # the tip, and the segment is drawn as the gradient between the two
    hues = mp.rainbow(angles % (2 * np.pi) / (2 * np.pi))
    colors = np.stack([hues // 3, hues, hues], axis=1).reshape(-1, 3)

    return xs, ys, colors


if __name__ == "__main__":
    tyro.cli(main)
