"""
A rose of rays, turning, with the pen swelling from one dot to six and back.

Twenty-four directions and three lengths, so one pen has to draw a stub and a
full radius, and each ray runs from a dim hub to a saturated tip along the way.
Ray lengths repeat every quarter turn, which is how far the rose turns, so the
loop closes exactly.

By Claude Opus 5.
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
    """A turning rose of rays, drawn with a pen of changing thickness."""
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
    """The rays at a given rotation, as one line broken between each."""
    angles = np.linspace(0, 2 * np.pi, RAYS, endpoint=False) + rotation
    reach = np.tile(REACH, RAYS // len(REACH))

    directions = np.stack([np.cos(angles), np.sin(angles)])
    hubs = HUB * directions
    tips = reach * directions
    gaps = np.full((2, RAYS), np.nan)
    # hub, tip, gap, hub, tip, gap, ...: the gap after each ray is what keeps
    # the next ray from being joined onto it
    xs, ys = np.stack([hubs, tips, gaps], axis=2).reshape(2, -1)

    # dim at the hub, full hue at the tip, as a gradient along the segment
    hues = mp.rainbow(angles % (2 * np.pi) / (2 * np.pi))
    colors = np.stack([hues // 3, hues, hues], axis=1).reshape(-1, 3)

    return xs, ys, colors


if __name__ == "__main__":
    tyro.cli(main)
