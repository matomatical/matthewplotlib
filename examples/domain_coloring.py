"""
Domain colourings of six complex functions.

A complex function has a two-dimensional input and a two-dimensional output, so
its graph wants four dimensions and there is nowhere to draw it. Domain
colouring gives up the output axes and paints the value onto the input plane
instead: hue for the phase, lightness for the modulus.

That is enough to read a surprising amount off the picture.

* A zero of the function is black and a pole is white.
* The colour wheel turns once around a simple zero, and once the other way
  around a simple pole. Counting the turns gives the order, so a double zero
  passes through red twice on the way around it.
* Each dark ring is a doubling of the modulus. Rings crowd together where the
  function is growing quickly and spread out where it is flat.
* A jump in the colour that is not a zero or a pole is a branch cut, where the
  function has been forced to pick one of several values.

By Claude Opus 5.
"""

import tyro
import numpy as np

import matthewplotlib as mp


def main(
    width: int = 22,
    height: int = 11,
    save: str | None = None,
):
    """Six complex functions, side by side."""
    # name, function, and how far the window reaches along both axes
    functions = [
        ("z^3", cube, 2.0),
        ("1/z^2", double_pole, 2.0),
        ("a rational function", rational, 3.0),
        ("sin z", sine, 4.0),
        ("exp(1/z)", essential, 1.0),
        ("log z", logarithm, 2.0),
    ]

    panels = [
        mp.axes(
            mp.cfunction2(
                function,
                xrange=(-extent, extent),
                yrange=(-extent, extent),
                width=width,
                height=height,
            ),
            title=name,
            xfmt="{x:+.0f}",
            yfmt="{y:+.0f}i",
        )
        for name, function, extent in functions
    ]
    grid = mp.wrap(*panels, cols=3)
    plot = grid / mp.center(
        mp.text(
            "hue = phase  ·  black = zero  ·  white = pole  "
            "·  one ring per doubling",
            fgcolor=(0.55, 0.55, 0.55),
        ),
        width=grid.width,
    )
    print(plot)

    if save:
        plot.saveimg(save)


def cube(z: np.ndarray) -> np.ndarray:
    """A zero of order three: the wheel turns three times around the origin."""
    return z ** 3


def double_pole(z: np.ndarray) -> np.ndarray:
    """A pole of order two, and the wheel turning backwards around it."""
    return 1 / z ** 2


def rational(z: np.ndarray) -> np.ndarray:
    """Simple zeros at 1 and -1, a double zero at 2+i, two simple poles."""
    return (z ** 2 - 1) * (z - 2 - 1j) ** 2 / (z ** 2 + 2 + 2j)


def sine(z: np.ndarray) -> np.ndarray:
    """Zeros along the real axis, and exponential growth away from it.

    The rings pile up towards the top and bottom of the window because the
    modulus is doubling every few tenths of an imaginary unit.
    """
    return np.sin(z)


def essential(z: np.ndarray) -> np.ndarray:
    """An essential singularity: every value, infinitely often, near zero.

    Approached from the right the function runs off to infinity and from the
    left it falls to zero, and in between the rings and the colour wheel wind
    up tighter than any resolution can follow.
    """
    return np.exp(1 / z)


def logarithm(z: np.ndarray) -> np.ndarray:
    """A branch cut along the negative reals, where the phase jumps by a turn.

    There is also a zero at 1, where the logarithm vanishes, and a pole at the
    origin where its modulus runs away.
    """
    return np.log(z)


if __name__ == "__main__":
    tyro.cli(main)
