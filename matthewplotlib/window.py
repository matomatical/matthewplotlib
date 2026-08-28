"""
The window a plot draws in.

A 2d plot covers an interval of data on each axis and draws it into a
rectangle of character cells. This module names that mapping, so that the
plots can share it and the furnishings around them can read it.

* `window`: The scale on each axis, the rectangle, and the conversions
  between data coordinates and the sub-cell grids the drawing routines work
  in, in both directions: where a point of the data lands, and what
  coordinate a given part of the grid stands for.

For the mapping from 3d data onto a plane, see `matthewplotlib.camera`.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import einops

from typing import cast
from numpy.typing import NDArray

from matthewplotlib.data import number
from matthewplotlib.scales import scale


# # #
# THE WINDOW


@dataclasses.dataclass(frozen=True)
class window:
    """
    The interval of data a plot covers on each axis, and the rectangle of
    character cells it covers them with.

    Inputs:

    * xrange : optional (number, number) | scale.
        The data coordinates at the left and the right edges of the
        rectangle. None if the horizontal axis carries no coordinate.
    * yrange : optional (number, number) | scale.
        The data coordinates at the bottom and the top edges of the
        rectangle. None if the vertical axis carries no coordinate.
    * width : int.
        The width of the rectangle, in character cells.
    * height : int.
        The height of the rectangle, in character cells.

    A range runs from the value at the low end of the screen axis to the value
    at the high end: `xrange` from the left edge to the right, `yrange` from
    the bottom edge to the top. Giving one descending inverts that axis, which
    mirrors the picture and changes nothing else.

    A range given as a `scale` spaces coordinates nonlinearly between the
    same two edges: the interval's ends still sit at the edges, and the scale
    decides where the values between them land. A plain pair is kept as a
    linear `scale`, so both spellings compare equal and either unpacks as
    `lo, hi = w.xrange`. Either way the scale must have both ends: a window
    covers a definite interval.

    Plots divide the rectangle in one of two ways, and a range means the same
    thing for both---the interval the plot covers---but the two place the
    limits differently within the outermost cell.

    * Plots that draw points and lines into a grid of braille dots put the
      limits at the centres of the outermost dots, so that a point at the
      extreme of its data is drawn rather than landing on a boundary.
    * Plots that tile the rectangle with coloured squares put the limits at the
      outer edges of the outermost squares, so that every square stands for an
      equal area.

    The difference is under half a cell, and invisible to labels drawn at the
    edges of the rectangle.
    """
    xrange: tuple[number, number] | scale | None
    yrange: tuple[number, number] | scale | None
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ValueError(
                "window must be at least one cell in each direction, not "
                f"{self.width}x{self.height}"
            )
        for name, given in (("xrange", self.xrange), ("yrange", self.yrange)):
            if given is None:
                continue
            axis = given if isinstance(given, scale) else scale(*given)
            if axis.lo is None or axis.hi is None:
                raise ValueError(
                    f"{name} has a missing endpoint: {axis!r}; a window "
                    "covers a definite interval, so give both ends"
                )
            if axis.lo == axis.hi:
                raise ValueError(f"{name} covers no interval: {given!r}")
            object.__setattr__(self, name, axis)

    def __repr__(self) -> str:
        coordinates = []
        for name, axis in (("x", self.xrange), ("y", self.yrange)):
            if axis is None:
                continue
            axis = cast(scale, axis)
            spacing = "" if type(axis) is scale else type(axis).__name__
            coordinates.append(
                f"{name}={spacing}[{axis.lo:.2f},{axis.hi:.2f}]"
            )
        extent = f"{self.width}x{self.height} cells"
        return f"window({', '.join([*coordinates, extent])})"

    def dots(self, points: NDArray) -> NDArray:
        """
        Points in the data's coordinates, as coordinates in the plot's dots.

        The dot grid has two columns and four rows of dots per character cell.
        A dot coordinate is a position in that grid, so that its integer part
        selects a dot and its fractional part places the point within that
        dot. The data's limits land on the centres of the outermost dots, and
        an axis's scale decides where the values between them land.

        A point with a coordinate its axis's scale is not defined over---a
        negative value on a `logscale` axis---gets a dot coordinate that is
        not a number, marking a point that cannot be placed; the drawing
        routines leave such points out, as they do points beyond the limits.

        Inputs:

        * points : float[n, 2].
            The points, as (x, y) pairs in data coordinates.

        Returns:

        * dots : float[n, 2].
            The same points, as (row, column) pairs in dot coordinates, with
            row zero at the top of the window.
        """
        if self.xrange is None or self.yrange is None:
            raise ValueError(f"{self!r} has no coordinates to place points in")
        xaxis = cast(scale, self.xrange)
        yaxis = cast(scale, self.yrange)
        txlo, txhi = xaxis._transformed_ends()
        tylo, tyhi = yaxis._transformed_ends()
        with np.errstate(divide="ignore", invalid="ignore"):
            tx = np.asarray(xaxis.transform(points[:, 0]), dtype=float)
            ty = np.asarray(yaxis.transform(points[:, 1]), dtype=float)
        cols = (tx - txlo) / (txhi - txlo) * (2 * self.width - 1)
        rows = (tyhi - ty) / (tyhi - tylo) * (4 * self.height - 1)
        return np.stack([rows + 0.5, cols + 0.5], axis=1)

    def pixel_edges(self) -> tuple[NDArray, NDArray]:
        """
        The data coordinates of the boundaries between the plot's pixels.

        The pixel grid has one column and two rows of pixels per character
        cell, and the pixels tile the window's intervals exactly. The
        boundaries are evenly spaced along each axis's scale, so on a
        nonlinear axis every pixel covers an equal share of the scale rather
        than of the interval; the outermost boundaries are the interval's own
        ends exactly.

        Returns:

        * xedges : float[width + 1].
            The boundaries from the left edge of the window to the right.
        * yedges : float[2 * height + 1].
            The boundaries from the bottom edge of the window to the top.

        Both run in screen order, and so descend numerically wherever the
        corresponding range does.
        """
        if self.xrange is None or self.yrange is None:
            raise ValueError(f"{self!r} has no coordinates to lay out a grid")
        return (
            _edges(cast(scale, self.xrange), num=self.width + 1),
            _edges(cast(scale, self.yrange), num=2 * self.height + 1),
        )

    def pixel_centres(self) -> tuple[NDArray, NDArray]:
        """
        The data coordinates at the centre of each of the plot's pixels.

        Each pixel stands for the square of the plane it covers, so the
        coordinate that represents it is the one at its centre---the centre
        of the square on the screen, which on a nonlinear axis is the
        midpoint along the scale rather than the midpoint of the values.

        Returns:

        * X : float[2 * height, width].
        * Y : float[2 * height, width].
            The x and the y coordinate of each pixel's centre, with row zero
            at the top of the window.
        """
        if self.xrange is None or self.yrange is None:
            raise ValueError(f"{self!r} has no coordinates to lay out a grid")
        xs = _centres(cast(scale, self.xrange), num=self.width)
        ys = _centres(cast(scale, self.yrange), num=2 * self.height)
        return np.meshgrid(xs, ys[::-1])

    def sample_points(self, endpoints: bool = False) -> NDArray:
        """
        The (x, y) coordinate that each of the plot's pixels stands for, as a
        list of points to hand to a function of the plane.

        Inputs:

        * endpoints : bool (default: False).
            By default the pixels tile the window's ranges exactly and each one
            is represented by its own centre. If true, the coordinates are
            instead spread from one end of each range to the other, so that the
            four corner pixels stand for the four corner combinations of the
            ranges and the pixels reach half a pixel beyond them.

        Either way the coordinates are evenly spaced along each axis's scale,
        so a nonlinear axis is sampled with its spacing: a `logscale` axis at
        log-spaced points.

        Returns:

        * points : float[2 * height * width, 2].
            The coordinates as (x, y) pairs, the top row of pixels first, so
            that reshaping the values back to `[2 * height, width]` puts them
            where they came from.
        """
        if self.xrange is None or self.yrange is None:
            raise ValueError(f"{self!r} has no coordinates to sample")
        if endpoints:
            X, Y = np.meshgrid(
                _edges(cast(scale, self.xrange), num=self.width),
                _edges(cast(scale, self.yrange), num=2 * self.height)[::-1],
            ) # float[h, w] (x2)
        else:
            X, Y = self.pixel_centres()
        return einops.rearrange(np.dstack((X, Y)), 'h w xy -> (h w) xy')


# # #
# SPACING POINTS ALONG ONE AXIS


def _edges(axis: scale, num: int) -> NDArray:
    """
    `num` data coordinates evenly spaced along the axis's scale, the ends of
    its interval at the outside exactly rather than through the round trip of
    the transform and its inverse.
    """
    tlo, thi = axis._transformed_ends()
    edges = np.asarray(
        axis._checked_inverse(np.linspace(tlo, thi, num=num)),
        dtype=float,
    )
    lo, hi = axis.interval
    edges[0] = lo
    if num >= 2:
        edges[-1] = hi
    return edges


def _centres(axis: scale, num: int) -> NDArray:
    """
    The data coordinates at the midpoints between `num + 1` boundaries evenly
    spaced along the axis's scale: where each of `num` pixels tiling the axis
    has its centre.
    """
    tlo, thi = axis._transformed_ends()
    tedges = np.linspace(tlo, thi, num=num + 1)
    return np.asarray(
        axis._checked_inverse((tedges[:-1] + tedges[1:]) / 2),
        dtype=float,
    )
