"""
How values are measured against an interval: the interval a scale covers,
and how far along it each value lies. Shared by every plot that turns
values into colours or lengths.
"""
from __future__ import annotations

import numpy as np

from numpy.typing import NDArray
from matthewplotlib.data import number


def _value_range(
    vrange: tuple[number, number] | None,
    values: NDArray,
    what: str,
    from_zero: bool = False,
    allow_flat: bool = True,
) -> tuple[float, float]:
    """
    The interval of values a scale covers.

    Given, it is taken as it stands. Omitted, it is inferred from the finite
    values there are: from the lowest of them to the highest, so that the scale
    spans the data, or from zero to the highest if `from_zero`, for a scale
    whose bottom end is a baseline rather than the smallest value measured.
    With no finite values at all it falls back to the unit interval.

    An interval covering nothing is an error where the caller wrote it, since
    there is no reading of it to act on. Inferred, over values that are all the
    same, it is returned as it stands unless `allow_flat` is false, for a plot
    that draws positions along the interval and so has nowhere to put them.
    `what` names the caller in either error.
    """
    if vrange is not None:
        vmin, vmax = float(vrange[0]), float(vrange[1])
        if vmin == vmax:
            raise ValueError(f"{what} vrange covers no interval: {vrange!r}")
        return (vmin, vmax)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (0.0, 1.0)
    vmin = 0.0 if from_zero else float(finite.min())
    vmax = float(finite.max())
    if vmin == vmax and not allow_flat:
        raise ValueError(
            f"every value in the {what} sits at {vmin}; give a vrange "
            "spanning an interval to plot them in"
        )
    return (vmin, vmax)


def _normalise(
    values: NDArray,
    vrange: tuple[number, number],
) -> NDArray: # float[...]
    """
    How far along an interval each value lies, saturating at its ends.

    Runs 0.0 to 1.0 from the first limit to the second, so that an interval
    given descending turns the scale around. Where the interval covers
    nothing---which only an inferred one does, over values that are all the
    same---everything comes out at the bottom, and so does any value that is
    not a number.
    """
    vmin, vmax = vrange
    if vmin == vmax:
        return np.zeros(values.shape, dtype=float)
    scaled = np.clip((values - vmin) / (vmax - vmin), 0., 1.)
    return np.where(np.isnan(scaled), 0., scaled)
