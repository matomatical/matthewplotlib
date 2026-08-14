"""
Specifying the data that goes into a plot.

Plot constructors are deliberately permissive about how data arrives: a single
array of points, a pair of coordinate sequences, or an axis object standing in
for one of the coordinates, each optionally paired with colors. This module
defines what is accepted and normalises it before plotting.

Types:

* `number`: A scalar, Python or NumPy.
* `Series` and `Series3`: The accepted shapes for 2d and 3d point data. See
  these aliases for the full list of forms.

Special series:

* `axis`, and its subclasses `xaxis`, `yaxis` and `zaxis`: Stand-ins for a
  coordinate that runs over a range, so that a series can be given as one
  sequence of values against an axis rather than as two sequences.

Parsers:

* `parse_series`, `parse_series3`, and their `parse_multiple_*` variants: Turn
  any accepted form into arrays of points and colors.
* `parse_range`: Fill in missing axis limits from the data.

For turning 3d data into positions on a camera's film, see
`matthewplotlib.camera`.
"""

from __future__ import annotations

import dataclasses
from typing import Sequence, cast

import numpy as np
from numpy.typing import NDArray, ArrayLike

from matthewplotlib.colors import ColorSpec, parse_colors


# # # 
# Types


type number = int | float | np.integer | np.floating


type Series = (
    NDArray                                     # number[n,2]
    | tuple[NDArray, ColorSpec]                 # number[n,2], colors
    | tuple[ArrayLike, ArrayLike]               # number[n]^2
    | tuple[ArrayLike, ArrayLike, ColorSpec]    # number[n]^2, colors
    | axis                                      # axis
    | tuple[axis, ColorSpec]                    # axis, colors
)


type Series3 = (
    NDArray                                             # number[n,3]
    | tuple[NDArray, ColorSpec]                         # number[n,3], colors
    | tuple[ArrayLike, ArrayLike, ArrayLike]            # number[n]^3
    | tuple[ArrayLike, ArrayLike, ArrayLike, ColorSpec] # number[n]^3, colors
    | axis                                              # axis
    | tuple[axis, ColorSpec]                            # axis, uint8[n,rgb]
)


# # # 
# Parsers


def parse_range(
    data: NDArray,
    range: tuple[number | None, number | None] | None,
) -> tuple[number, number]:
    """
    Fill in missing axis limits from the data.

    Limits come from the data's extremes, ignoring non-finite values, since
    those mark gaps in it rather than describing how far it reaches. Data that
    reaches no distance at all, a constant series or an empty one, is given a
    range around itself to be drawn in the middle of.
    """
    if range is None:
        range = (None, None)
    lo, hi = range
    if lo is None or hi is None:
        finite = data[np.isfinite(data)]
        if not finite.size:
            finite = np.zeros(1)
        lo = finite.min() if lo is None else lo
        hi = finite.max() if hi is None else hi
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    return lo, hi


def parse_series(
    series: Series, # Series<n>
) -> tuple[
    NDArray,        # number[n]
    NDArray,        # number[n]
    NDArray,        # uint8[n,3]
]:
    match series:
        case axis() as a:
            xs = a.xs
            ys = a.ys
            cs = parse_colors(None, a.n)
        case (axis() as a, cs_):
            xs = a.xs
            ys = a.ys
            cs = parse_colors(cast(ColorSpec, cs_), a.n)
        case np.ndarray(shape=(n, 2)) as a:
            xs = a[:, 0]
            ys = a[:, 1]
            cs = parse_colors(None, n)
        case (np.ndarray(shape=(n, 2)) as a, cs_):
            xs = a[:, 0]
            ys = a[:, 1]
            cs = parse_colors(cast(ColorSpec, cs_), n)
        case (xs_, ys_):
            xs = np.asarray(xs_)
            ys = np.asarray(ys_)
            n, = xs.shape
            cs = parse_colors(None, n)
        case (xs_, ys_, cs_):
            xs = np.asarray(xs_)
            ys = np.asarray(ys_)
            n, = xs.shape
            cs = parse_colors(cast(ColorSpec, cs_), n)
        case _:
            raise TypeError(f"Invalid Series {series!r}")
    return xs, ys, cs

            
def parse_multiple_series(
    *seriess: Series,
) -> tuple[
    NDArray,    # number[N]
    NDArray,    # number[N]
    NDArray,    # uint8[N,3]
]:
    xss, yss, css = zip(*map(parse_series, seriess))
    return (
        np.concatenate(xss),
        np.concatenate(yss),
        np.concatenate(css),
    )

    
def parse_segments(
    *seriess: Series,
) -> tuple[
    NDArray,    # number[m, 2]
    NDArray,    # number[m, 2]
    NDArray,    # uint8[m, 3]
    NDArray,    # uint8[m, 3]
]:
    """
    Turn series into the segments joining their consecutive points, with the
    colors at each end of each.

    Where `parse_multiple_series` pools every series into one cloud of points,
    this pairs the points up within each series first: a series is one stroke
    of the pen, and the last point of one is never joined to the first point of
    the next.
    """
    return _pair_up([parse_series(s) for s in seriess], dimensions=2)


def parse_series3(
    series: Series3, # Series3<n>
) -> tuple[
    NDArray,        # number[n]
    NDArray,        # number[n]
    NDArray,        # number[n]
    NDArray,        # uint8[n,3]
]:
    match series:
        case axis() as a:
            xs = a.xs
            ys = a.ys
            zs = a.zs
            cs = parse_colors(None, a.n)
        case (axis() as a, cs_):
            xs = a.xs
            ys = a.ys
            zs = a.zs
            cs = parse_colors(cast(ColorSpec, cs_), a.n)
        case np.ndarray(shape=(n, 3)) as a:
            xs = a[:, 0]
            ys = a[:, 1]
            zs = a[:, 2]
            cs = parse_colors(None, n)
        case (np.ndarray(shape=(n, 3)) as a, cs_):
            xs = a[:, 0]
            ys = a[:, 1]
            zs = a[:, 2]
            cs = parse_colors(cast(ColorSpec, cs_), n)
        case (xs_, ys_, zs_):
            xs = np.asarray(xs_)
            ys = np.asarray(ys_)
            zs = np.asarray(zs_)
            n, = xs.shape
            cs = parse_colors(None, n)
        case (xs_, ys_, zs_, cs_):
            xs = np.asarray(xs_)
            ys = np.asarray(ys_)
            zs = np.asarray(zs_)
            n, = xs.shape
            cs = parse_colors(cast(ColorSpec, cs_), n)
        case _:
            raise TypeError(f"Invalid Series3 {series!r}")
    return xs, ys, zs, cs


def parse_multiple_series3(
    *seriess: Series3,
) -> tuple[
    NDArray,    # number[N]
    NDArray,    # number[N]
    NDArray,    # number[N]
    NDArray,    # uint8[N,3]
]:
    xss, yss, zss, css = zip(*map(parse_series3, seriess))
    return (
        np.concatenate(xss),
        np.concatenate(yss),
        np.concatenate(zss),
        np.concatenate(css),
    )

    
def parse_segments3(
    *seriess: Series3,
) -> tuple[
    NDArray,    # number[m, 3]
    NDArray,    # number[m, 3]
    NDArray,    # uint8[m, 3]
    NDArray,    # uint8[m, 3]
]:
    """
    Turn 3d series into the segments joining their consecutive points, with the
    colors at each end of each. See `parse_segments`.
    """
    return _pair_up([parse_series3(s) for s in seriess], dimensions=3)


def _pair_up(
    polylines: Sequence[tuple[NDArray, ...]],
    dimensions: int,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """The consecutive pairs of each parsed series, pooled after pairing."""
    starts = []
    ends = []
    start_colors = []
    end_colors = []
    for polyline in polylines:
        *coordinates, colors = polyline
        points = np.stack(coordinates, axis=1)
        starts.append(points[:-1])
        ends.append(points[1:])
        start_colors.append(colors[:-1])
        end_colors.append(colors[1:])
    empty_points = np.zeros((0, dimensions))
    empty_colors = np.zeros((0, 3), dtype=np.uint8)
    return (
        np.concatenate([*starts, empty_points]),
        np.concatenate([*ends, empty_points]),
        np.concatenate([*start_colors, empty_colors]),
        np.concatenate([*end_colors, empty_colors]),
    )


# # # 
# Special series


@dataclasses.dataclass(frozen=True)
class axis:
    a: number = 0.
    b: number = 1.
    n: int = 10
    
    @property
    def xs(self) -> NDArray:
        return np.zeros(self.n)

    @property
    def ys(self) -> NDArray:
        return np.zeros(self.n)
    
    @property
    def zs(self) -> NDArray:
        return np.zeros(self.n)


class xaxis(axis):
    @property
    def xs(self) -> NDArray:
        return np.linspace(self.a, self.b, self.n)


class yaxis(axis):
    @property
    def ys(self) -> NDArray:
        return np.linspace(self.a, self.b, self.n)


class zaxis(axis):
    @property
    def zs(self) -> NDArray:
        return np.linspace(self.a, self.b, self.n)
