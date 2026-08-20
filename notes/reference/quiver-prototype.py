"""
Throwaway prototype behind the measurements in `notes/quiver-plots.md`.

Draws unit-length arrows on a lattice over a Taylor-Green vortex field, at four
combinations of shaft length, arrowhead length and lattice pitch, so that the
resolution at which an arrowhead stops reading can be seen rather than guessed.

Kept as the evidence for the numbers in the note, not as a design: it works in
dot coordinates by declaring the window to be the dot grid, builds one `line`
series per segment, and hard-codes the field.

    python notes/reference/quiver-prototype.py out.png

By Claude Opus 5, 2026-08-20.
"""

import sys

import numpy as np

import matthewplotlib as mp


WIDTH, HEIGHT = 44, 12                          # cells
DOT_WIDTH, DOT_HEIGHT = 2 * WIDTH, 4 * HEIGHT   # braille dots

# shaft length, arrowhead length, lattice columns, lattice rows: all lengths in
# dots, so that they can be read against the 2-by-4 dots of a character cell
TRIALS = (
    ("5 dots, no head", 5, 0, 20, 12),
    ("8 dots, no head", 8, 0, 14, 8),
    ("8 dots, 3 dot head", 8, 3, 14, 8),
    ("12 dots, 4 dot head", 12, 4, 10, 6),
)

HEAD_ANGLE = np.deg2rad(150)


def main(path: str):
    panels = [
        mp.border(quiver(shaft, head, cols, rows), title=f" {label} ")
        for label, shaft, head, cols, rows in TRIALS
    ]
    plot = mp.vstack(*panels)
    print(plot)
    plot.saveimg(path, bgcolor="black")


def taylor_green(p: np.ndarray) -> np.ndarray:
    x, y = p[:, 0], p[:, 1]
    return np.stack([np.sin(x) * np.cos(y), -np.cos(x) * np.sin(y)], axis=-1)


def quiver(shaft: float, head: float, cols: int, rows: int) -> mp.plot:
    """Unit arrows on a cols-by-rows lattice, with lengths in dots."""
    # lattice centres, in dots
    xs = (np.arange(cols) + 0.5) * DOT_WIDTH / cols
    ys = (np.arange(rows) + 0.5) * DOT_HEIGHT / rows
    X, Y = np.meshgrid(xs, ys)
    centres = np.stack([X.ravel(), Y.ravel()], axis=-1)

    # the field over [-pi, pi] squared, sampled at those same centres
    sample = np.stack([
        centres[:, 0] / DOT_WIDTH * 2 * np.pi - np.pi,
        centres[:, 1] / DOT_HEIGHT * 2 * np.pi - np.pi,
    ], axis=-1)
    v = taylor_green(sample)
    unit = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)

    # the shaft, then one segment for each barb of the head
    tails = centres - unit * shaft / 2
    tips = centres + unit * shaft / 2
    segments = [(tails, tips)]
    for sign in (+1, -1):
        angle = HEAD_ANGLE * sign
        barb = np.stack([
            unit[:, 0] * np.cos(angle) - unit[:, 1] * np.sin(angle),
            unit[:, 0] * np.sin(angle) + unit[:, 1] * np.cos(angle),
        ], axis=-1)
        if head:
            segments.append((tips, tips + barb * head))

    series = [
        (np.stack([start, end]), (1.0, 1.0, 1.0))
        for starts, ends in segments
        for start, end in zip(starts, ends)
    ]
    return mp.line(
        series[0],
        *series[1:],
        xrange=(0, DOT_WIDTH),
        yrange=(0, DOT_HEIGHT),
        width=WIDTH,
        height=HEIGHT,
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "quiver-prototype.png")
