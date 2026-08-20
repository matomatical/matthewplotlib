"""
Phase portraits of six planar vector fields, as colour fields.

Each panel colours every pixel by the vector the field takes there: hue for the
direction, brightness for the magnitude. Where a flow's velocity falls to zero
it goes black, so the fixed points show up as dark spots, and the way the hue
turns around one says which kind it is---a full turn of the colour wheel around
a centre or a spiral, half a turn the other way around a saddle.

Unlike a field of arrows this draws a vector in every cell, so nothing is lost
between the glyphs and the structure survives all the way down to a terminal
that is only a few dozen columns wide.

By Claude Opus 5.
"""

from typing import Callable, NamedTuple

import tyro
import numpy as np

import matthewplotlib as mp


class Field(NamedTuple):
    """A planar vector field, and how to look at it.

    `midpoint` is the magnitude to draw at around half brightness; see
    `softened` for what happens to the rest.
    """
    name: str
    field: Callable[[np.ndarray], np.ndarray]
    extent: float
    midpoint: float


def softened(
    field: Callable[[np.ndarray], np.ndarray],
    midpoint: float,
) -> Callable[[np.ndarray], np.ndarray]:
    """The same field with its directions kept and its magnitudes squashed.

    `vfunction2` scales magnitudes linearly, which is the honest default but
    the wrong one for a field whose magnitude covers orders of magnitude: the
    dipole below is a thousand times faster beside a charge than it is out at
    the corners, and on a linear scale everything but the two charges comes out
    black. Passing the magnitudes through a `tanh` first spends the brightness
    evenly instead, at the cost of no longer being able to read a magnitude
    off a brightness.

    The result has magnitudes in [0, 1), so the plots below leave `vrange` at
    the unit interval rather than letting it find the largest.
    """
    def softened_field(p: np.ndarray) -> np.ndarray:
        v = field(p)
        magnitude = np.linalg.norm(v, axis=-1, keepdims=True)
        direction = v / np.maximum(magnitude, 1e-12)
        return direction * np.tanh(magnitude / midpoint)

    return softened_field


def main(
    width: int = 22,
    height: int = 11,
    save: str | None = None,
):
    """Six planar vector fields, side by side."""
    panels = [
        mp.axes(
            mp.vfunction2(
                softened(entry.field, entry.midpoint),
                xrange=(-entry.extent, entry.extent),
                yrange=(-entry.extent, entry.extent),
                width=width,
                height=height,
                vrange=(0.0, 1.0),
            ),
            title=entry.name,
            xfmt="{x:+.0f}",
            yfmt="{y:+.0f}",
        )
        for entry in FIELDS
    ]
    grid = mp.wrap(*panels, cols=3)
    plot = grid / mp.center(
        mp.text(
            "hue = direction  ·  brightness = magnitude",
            fgcolor=(0.55, 0.55, 0.55),
        ),
        width=grid.width,
    )
    print(plot)

    if save:
        plot.saveimg(save)


def saddle(p: np.ndarray) -> np.ndarray:
    """x' = x, y' = -y. Two straight-line flows, one out and one in."""
    x, y = p[:, 0], p[:, 1]
    return np.stack([x, -y], axis=-1)


def centre(p: np.ndarray) -> np.ndarray:
    """x' = -y, y' = x. Rigid rotation: every orbit is a closed circle."""
    x, y = p[:, 0], p[:, 1]
    return np.stack([-y, x], axis=-1)


def spiral(p: np.ndarray) -> np.ndarray:
    """A rotation with a little damping, so the orbits fall inwards."""
    x, y = p[:, 0], p[:, 1]
    return np.stack([-0.35 * x - y, x - 0.35 * y], axis=-1)


def vortex_lattice(p: np.ndarray) -> np.ndarray:
    """Taylor-Green: a tiling of vortices, each turning against its neighbours.

    Between every four vortices is a saddle, so one window holds both kinds of
    fixed point several times over and the alternation is easy to see.
    """
    x, y = p[:, 0], p[:, 1]
    return np.stack([np.sin(x) * np.cos(y), -np.cos(x) * np.sin(y)], axis=-1)


def pendulum(p: np.ndarray) -> np.ndarray:
    """A damped pendulum: angle against angular velocity.

    The saddles at odd multiples of pi are the upright position, and the sinks
    between them are the hanging one.
    """
    theta, omega = p[:, 0], p[:, 1]
    return np.stack([omega, -np.sin(theta) - 0.25 * omega], axis=-1)


def dipole(p: np.ndarray) -> np.ndarray:
    """The field of a positive and a negative charge, two units apart.

    The distance is held away from zero so that the two singularities stay
    finite, which keeps them on the brightness scale with the rest of the
    field instead of taking it over.
    """
    total = np.zeros_like(p)
    for charge, position in ((1.0, [-1.0, 0.0]), (-1.0, [1.0, 0.0])):
        delta = p - np.array(position)
        distance = np.linalg.norm(delta, axis=-1, keepdims=True)
        total += charge * delta / np.maximum(distance, 0.4) ** 3
    return total


FIELDS = (
    Field("saddle", saddle, extent=3.0, midpoint=1.5),
    Field("centre", centre, extent=3.0, midpoint=1.5),
    Field("stable spiral", spiral, extent=3.0, midpoint=1.5),
    Field("vortex lattice", vortex_lattice, extent=np.pi, midpoint=0.5),
    Field("damped pendulum", pendulum, extent=6.0, midpoint=2.0),
    Field("dipole", dipole, extent=3.0, midpoint=0.3),
)


if __name__ == "__main__":
    tyro.cli(main)
