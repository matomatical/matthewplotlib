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

from typing import Callable

import tyro
import numpy as np

import matthewplotlib as mp


def main(
    width: int = 22,
    height: int = 11,
    save: str | None = None,
):
    """Six planar vector fields, side by side."""
    # name, field, how far the window reaches, and the magnitude to draw at
    # around half brightness
    fields = [
        ("saddle", saddle, 3.0, 1.5),
        ("centre", centre, 3.0, 1.5),
        ("stable spiral", spiral, 3.0, 1.5),
        ("vortex lattice", vortex_lattice, np.pi, 0.5),
        ("damped pendulum", pendulum, 6.0, 2.0),
        ("dipole", dipole, 3.0, 0.3),
    ]

    panels = [
        mp.axes(
            mp.vfunction2(
                softened(field, midpoint),
                xrange=(-extent, extent),
                yrange=(-extent, extent),
                width=width,
                height=height,
                vrange=(0.0, 1.0),
            ),
            title=name,
            xfmt="{x:+.0f}",
            yfmt="{y:+.0f}",
        )
        for name, field, extent, midpoint in fields
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


def softened(
    field: Callable[[np.ndarray], np.ndarray],
    midpoint: float,
) -> Callable[[np.ndarray], np.ndarray]:
    """The same field with its directions kept and its magnitudes squashed.

    `vrange` scales magnitudes linearly, which spends nearly all of the
    brightness on the fastest corner of a field like the dipole. A `tanh`
    spreads it evenly instead, at the cost of no longer being able to read a
    magnitude off a brightness. The result lands in the unit disc, so the plots
    above leave `vrange` there.
    """
    def softened_field(p: np.ndarray) -> np.ndarray:
        v = field(p)
        magnitude = np.linalg.norm(v, axis=-1, keepdims=True)
        direction = v / np.maximum(magnitude, 1e-12)
        return direction * np.tanh(magnitude / midpoint)

    return softened_field


if __name__ == "__main__":
    tyro.cli(main)
