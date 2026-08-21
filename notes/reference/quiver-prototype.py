"""
Throwaway prototype behind the measurements in the `quiver-plots` note.

Two studies over the same Taylor-Green vortex field, in a 44 by 12 cell plot,
which is 88 by 48 braille dots. All lengths below are in dots, so that they can
be read against the 2 by 4 dots of a character cell.

* `DENSITIES` varies the shaft length, the arrowhead length and the lattice
  pitch, to find the resolution at which an arrowhead stops reading.
* `senses` draws the same lattice five ways, to see what can carry the sense of
  an arrow when the geometry cannot: colour, a brightness gradient, stroke
  weight, or a head after all.

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

WHITE = (1.0, 1.0, 1.0)
HEAD_ANGLE = np.deg2rad(150)

# shaft length, arrowhead length, lattice columns, lattice rows
DENSITIES = (
    ("5 dots, no head", 5, 0, 20, 12),
    ("8 dots, no head", 8, 0, 14, 8),
    ("8 dots, 3 dot head", 8, 3, 14, 8),
    ("12 dots, 4 dot head", 12, 4, 10, 6),
)

# the lattice the sense study runs at: 8 dot shafts, 14 by 8 arrows
SENSE_LATTICE = (8, 14, 8)


def main(path: str):
    plates = [
        mp.border(arrows(cols, rows, shaft, head=head), title=f" {label} ")
        for label, shaft, head, cols, rows in DENSITIES
    ]
    shaft, cols, rows = SENSE_LATTICE
    plates += [
        mp.border(draw(cols, rows, shaft), title=f" {label} ")
        for label, draw in senses()
    ]
    plot = mp.vstack(*plates)
    print(plot)
    plot.saveimg(path, bgcolor="black")


def senses():
    """The ways of carrying an arrow's sense, each as a name and a drawer."""
    return (
        ("plain stroke", arrows),
        ("coloured by direction", coloured),
        ("dark tail, bright tip", gradient),
        ("thick tail, thin tip", tapered),
        ("arrowhead at this density", lambda c, r, s: arrows(c, r, s, head=3)),
    )


def taylor_green(p: np.ndarray) -> np.ndarray:
    x, y = p[:, 0], p[:, 1]
    return np.stack([np.sin(x) * np.cos(y), -np.cos(x) * np.sin(y)], axis=-1)


def lattice(cols: int, rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Arrow centres in dots, and the unit field vector at each of them."""
    xs = (np.arange(cols) + 0.5) * DOT_WIDTH / cols
    ys = (np.arange(rows) + 0.5) * DOT_HEIGHT / rows
    X, Y = np.meshgrid(xs, ys)
    centres = np.stack([X.ravel(), Y.ravel()], axis=-1)
    sample = np.stack([
        centres[:, 0] / DOT_WIDTH * 2 * np.pi - np.pi,
        centres[:, 1] / DOT_HEIGHT * 2 * np.pi - np.pi,
    ], axis=-1)
    v = taylor_green(sample)
    unit = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
    return centres, unit


def draw(series: list, thickness: float = 1.0) -> mp.plot:
    """Segments given in dot coordinates, by declaring the window to be them."""
    return mp.line(
        series[0],
        *series[1:],
        xrange=(0, DOT_WIDTH),
        yrange=(0, DOT_HEIGHT),
        width=WIDTH,
        height=HEIGHT,
        thickness=thickness,
    )


def arrows(cols: int, rows: int, shaft: float, head: float = 0) -> mp.plot:
    """Plain white strokes, with an optional two-barb arrowhead."""
    centres, unit = lattice(cols, rows)
    series = []
    for centre, u in zip(centres, unit):
        tail, tip = centre - u * shaft / 2, centre + u * shaft / 2
        series.append((np.stack([tail, tip]), WHITE))
        if head:
            for sign in (+1, -1):
                angle = HEAD_ANGLE * sign
                barb = np.array([
                    u[0] * np.cos(angle) - u[1] * np.sin(angle),
                    u[0] * np.sin(angle) + u[1] * np.cos(angle),
                ])
                series.append((np.stack([tip, tip + barb * head]), WHITE))
    return draw(series)


def coloured(cols: int, rows: int, shaft: float) -> mp.plot:
    """Sense from hue, straight out of the chroma colormap."""
    centres, unit = lattice(cols, rows)
    return draw([
        (np.stack([c - u * shaft / 2, c + u * shaft / 2]), color)
        for c, u, color in zip(centres, unit, mp.chroma(unit))
    ])


def gradient(cols: int, rows: int, shaft: float) -> mp.plot:
    """Sense from brightness: `line` interpolates colour along a segment."""
    centres, unit = lattice(cols, rows)
    ramp = np.array([[40, 40, 40], [255, 255, 255]], dtype=np.uint8)
    return draw([
        (np.stack([c - u * shaft / 2, c + u * shaft / 2]), ramp)
        for c, u in zip(centres, unit)
    ])


def tapered(cols: int, rows: int, shaft: float) -> mp.plot:
    """Sense from stroke weight: two layers at different thicknesses."""
    centres, unit = lattice(cols, rows)
    base = draw(
        [(np.stack([c - u * shaft / 2, c]), WHITE) for c, u in zip(centres, unit)],
        thickness=2.6,
    )
    tip = draw(
        [(np.stack([c, c + u * shaft / 2]), WHITE) for c, u in zip(centres, unit)],
        thickness=1.0,
    )
    return base @ tip


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "quiver-prototype.png")
