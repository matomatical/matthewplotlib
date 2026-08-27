"""
Plots that draw their data with braille dots: scatter and line plots, their
three-dimensional versions, and hilbert curves.

* `scatter`
* `scatter3`
* `line`
* `line3`
* `hilbert`
"""
from __future__ import annotations

import numpy as np
import hilbert as _hilbert

from numpy.typing import ArrayLike
from matthewplotlib.colors import ColorLike
from matthewplotlib.data import (
    number,
    Series,
    Series3,
    parse_range,
    parse_segments,
    parse_segments3,
    parse_multiple_series,
    parse_multiple_series3,
)
from matthewplotlib.camera import (
    project3,
    project3_segments,
)
from matthewplotlib.window import window
from matthewplotlib.core import (
    unicode_braille_array,
    unicode_braille_points,
    unicode_braille_segments,
)
from matthewplotlib.plots.base import plot


class scatter(plot):
    """
    Render a scatterplot using a grid of braille unicode characters.

    Each character cell in the plot corresponds to a 2x4 grid of sub-pixels,
    represented by braille dots.

    Inputs:

    * series : Series.
         X Y data, for example a tuple (xs, ys) or triple (xs, ys, cs) where
         cs is a ColorLike or a list of RGB triples. See documentation for more
         examples.
    * *etc.
        Further series.
    * xrange : optional (number, number).
        The x-axis limits `(xmin, xmax)`. If not provided, the limits are
        inferred from the min and max x-values in the data.
    * yrange : optional (number, number).
        The y-axis limits `(ymin, ymax)`. If not provided, the limits are
        inferred from the min and max y-values in the data.
    * width : int (default: 30).
        The width of the plot in characters. The effective pixel width will be
        2 * width.
    * height : int (default: 10).
        The height of the plot in rows. The effective pixel height will be 4 *
        height.
    """
    def __init__(
        self,
        series: Series,
        *etc: Series,
        xrange: tuple[number | None, number | None] | None = None,
        yrange: tuple[number | None, number | None] | None = None,
        width: int = 30,
        height: int = 10,
    ):
        # parse inputs into standard format
        xs, ys, cs = parse_multiple_series(series, *etc)
        n, = xs.shape
        w = window(
            xrange=parse_range(xs, xrange),
            yrange=parse_range(ys, yrange),
            width=width,
            height=height,
        )

        points = np.stack([xs, ys], axis=1)
        super().__init__(unicode_braille_points(
            points=w.dots(points),
            height=4 * height,
            width=2 * width,
            colors=cs,
        ))
        self.window = w
        self.num_points = n

    def __repr__(self):
        return f"scatter(<{self.num_points} points>, {self.window!r})"


class scatter3(plot):
    """
    Scatter plot representing a 3d point cloud.

    * series : Series3.
         X Y Z data, for example a triple (xs, ys, zs) or quad (xs, ys, zs, cs)
         where cs is a ColorLike or a list of RGB triples. See documentation
         for more examples.
    * *etc.: Series3
        Further series.
    * camera_position: float[3] (default: [0. 0. 2.]).
        The position at which the camera is placed.
    * camera_target: float[3] (default: [0. 0. 0.]).
        The position towards which the camera is facing. Should be distinct
        from camera position. The default is that the camera is facing towards
        the origin.
    * scene_up: float[3] (default: [0. 1. 0.]).
        The unit vector designating the 'up' direction for the scene. The
        default is the positive Y direction. Should not have the same direction
        as camera_target - camera_position.
    * vertical_fov_degrees: float (default 90).
        Vertical field of view. Points within a vertical cone of this angle are
        projected into the viewing area. The horizontal field of view is then
        determined based on the aspect ratio.
    * aspect_ratio: optional float.
        Aspect ratio for the set of points, as a fraction (W:H represented as
        W/H). If not provided, uses W=width, H=2*height, which is uniform given
        the resolution of the plot.
    * width : int.
        The number of character columns in the plot.
    * height : int.
        The number of character rows in the plot.

    Projected coordinates are not the data's own, so this is not a `scatter`
    and `axes` does not take it: there is nothing meaningful to label an axis
    with.

    TODO:

    * Maybe allow configurable xyz ranges with clipping prior to projection?
    """
    def __init__(
        self,
        series: Series3,
        *etc: Series3,
        camera_position: np.ndarray = np.array([0., 0., 2.]),   # float[3]
        camera_target: np.ndarray = np.zeros(3),                # float[3]
        scene_up: np.ndarray = np.array([0.,1.,0.]),            # float[3]
        vertical_fov_degrees: float = 90.0,
        aspect_ratio: float | None = None,
        width: int = 30,
        height: int = 15,
    ):
        # parse inputs into standard format
        xs, ys, zs, cs = parse_multiple_series3(series, *etc)

        xy, valid = project3(
            xyz=np.c_[xs, ys, zs],
            camera_position=camera_position,
            camera_target=camera_target,
            scene_up=scene_up,
            fov_degrees=vertical_fov_degrees,
        )
        if aspect_ratio is None:
            aspect_ratio = width / (2*height)
        # the film's coordinates, which are not the data's, so they are not
        # kept: a projected point cloud has no axes to be labelled with
        film = window(
            xrange=(-aspect_ratio, aspect_ratio),
            yrange=(-1., 1.),
            width=width,
            height=height,
        )

        super().__init__(unicode_braille_points(
            points=film.dots(xy[valid]),
            height=4 * height,
            width=2 * width,
            colors=cs[valid],
        ))
        self.num_points = int(valid.sum())

    def __repr__(self):
        return (
            f"scatter3(height={self.height}, width={self.width}, "
            f"data=<{self.num_points} points in front of the camera>)"
        )


class line(plot):
    """
    Render a line plot by connecting a sequence of points, using a grid of
    braille unicode characters.

    Each character cell in the plot corresponds to a 2x4 grid of sub-pixels,
    represented by braille dots.

    Inputs:

    * series : Series.
         X Y data, for example a tuple (xs, ys) or triple (xs, ys, cs) where
         cs is a ColorLike or a list of RGB triples. See documentation for more
         examples.
    * *etc.
        Further series. Each is a separate line: the end of one is not joined
        to the start of the next.
    * xrange : optional (number, number).
        The x-axis limits `(xmin, xmax)`. If not provided, the limits are
        inferred from the min and max x-values in the data.
    * yrange : optional (number, number).
        The y-axis limits `(ymin, ymax)`. If not provided, the limits are
        inferred from the min and max y-values in the data.
    * width : int (default: 30).
        The width of the plot in characters. The effective pixel width will be
        2 * width.
    * height : int (default: 10).
        The height of the plot in rows. The effective pixel height will be 4 *
        height.
    * thickness : float (default: 1.0).
        How wide to draw the line, in dots. Corners between segments are
        filled and the ends are rounded.

    A point with a non-finite coordinate breaks the line, so that one series
    can be drawn as several disconnected strokes. Colors are interpolated
    along each segment, so a series with a color per point comes out as a
    gradient.
    """
    def __init__(
        self,
        series: Series,
        *etc: Series,
        xrange: tuple[number | None, number | None] | None = None,
        yrange: tuple[number | None, number | None] | None = None,
        width: int = 30,
        height: int = 10,
        thickness: float = 1.0,
    ):
        # the segments joining each series' own points, pooled after pairing
        starts, ends, start_colors, end_colors = parse_segments(series, *etc)
        points = np.concatenate([starts, ends])
        w = window(
            xrange=parse_range(points[:, 0], xrange),
            yrange=parse_range(points[:, 1], yrange),
            width=width,
            height=height,
        )

        super().__init__(unicode_braille_segments(
            starts=w.dots(starts),
            ends=w.dots(ends),
            height=4 * height,
            width=2 * width,
            start_colors=start_colors,
            end_colors=end_colors,
            thickness=thickness,
        ))
        self.window = w
        self.num_segments = len(starts)
        self.num_strokes = 1 + len(etc)
        self.thickness = thickness

    def __repr__(self):
        return (
            f"line(<{self.num_segments} segments, {self.num_strokes} "
            f"strokes>, thickness={self.thickness}, {self.window!r})"
        )


class line3(plot):
    """
    Render a wireframe by connecting a sequence of 3d points, seen from a
    camera.

    Inputs:

    * series : Series3.
         X Y Z data, for example a triple (xs, ys, zs) or quad (xs, ys, zs, cs)
         where cs is a ColorLike or a list of RGB triples. See documentation
         for more examples.
    * *etc.: Series3
        Further series. Each is a separate line: the end of one is not joined
        to the start of the next.
    * camera_position: float[3] (default: [0. 0. 2.]).
        The position at which the camera is placed.
    * camera_target: float[3] (default: [0. 0. 0.]).
        The position towards which the camera is facing. Should be distinct
        from camera position. The default is that the camera is facing towards
        the origin.
    * scene_up: float[3] (default: [0. 1. 0.]).
        The unit vector designating the 'up' direction for the scene. The
        default is the positive Y direction. Should not have the same direction
        as camera_target - camera_position.
    * vertical_fov_degrees: float (default 90).
        Vertical field of view. Points within a vertical cone of this angle are
        projected into the viewing area. The horizontal field of view is then
        determined based on the aspect ratio.
    * aspect_ratio: optional float.
        Aspect ratio for the scene, as a fraction (W:H represented as W/H). If
        not provided, uses W=width, H=2*height, which is uniform given the
        resolution of the plot.
    * width : int.
        The number of character columns in the plot.
    * height : int.
        The number of character rows in the plot.
    * thickness : float (default: 1.0).
        How wide to draw the line, in dots. Corners between segments are
        filled and the ends are rounded.

    A point with a non-finite coordinate breaks the line, which is how a mesh
    of several separate wires is drawn in one call. A segment reaching behind
    the camera is cut off in front of it, and one entirely behind the camera is
    not drawn.
    """
    def __init__(
        self,
        series: Series3,
        *etc: Series3,
        camera_position: np.ndarray = np.array([0., 0., 2.]),   # float[3]
        camera_target: np.ndarray = np.zeros(3),                # float[3]
        scene_up: np.ndarray = np.array([0.,1.,0.]),            # float[3]
        vertical_fov_degrees: float = 90.0,
        aspect_ratio: float | None = None,
        width: int = 30,
        height: int = 15,
        thickness: float = 1.0,
    ):
        starts, ends, start_colors, end_colors = parse_segments3(series, *etc)
        xy_starts, xy_ends, drawn = project3_segments(
            starts=starts,
            ends=ends,
            camera_position=camera_position,
            camera_target=camera_target,
            scene_up=scene_up,
            fov_degrees=vertical_fov_degrees,
        )
        if aspect_ratio is None:
            aspect_ratio = width / (2*height)

        # the film's coordinates, which are not the data's, so they are not
        # kept: projected segments have no axes to be labelled with
        film = window(
            xrange=(-aspect_ratio, aspect_ratio),
            yrange=(-1., 1.),
            width=width,
            height=height,
        )
        super().__init__(unicode_braille_segments(
            starts=film.dots(xy_starts),
            ends=film.dots(xy_ends),
            height=4 * height,
            width=2 * width,
            start_colors=start_colors[drawn],
            end_colors=end_colors[drawn],
            thickness=thickness,
        ))
        self.num_segments = int(drawn.sum())
        self.thickness = thickness

    def __repr__(self):
        return (
            f"line3(height={self.height}, width={self.width}, "
            f"thickness={self.thickness}, "
            f"data=<{self.num_segments} segments drawn>)"
        )


class hilbert(plot):
    """
    Visualize a 1D boolean array along a 2D Hilbert curve.

    Maps a 1D sequence of data points to a 2D grid using a space-filling
    Hilbert curve, which helps preserve locality. The curve is rendered using
    braille unicode characters for increased resolution.

    Inputs:

    * data : bool[N].
        A 1D array of booleans. The length `N` determines the order of the
        Hilbert curve required to fit all points. True values are rendered as
        dots, and False values are rendered as blank spaces.
    * color : optional ColorLike.
        The foreground color used for dots (points along the curve where `data`
        is `True`). Defaults to the terminal's default foreground color.
    """
    def __init__(
        self,
        data: ArrayLike, # bool[N]
        color: ColorLike | None = None,
    ):
        # preprocess and compute grid shape
        data = np.asarray(data)
        N, = data.shape
        n = max(2, ((N-1).bit_length() + 1) // 2)

        # compute dot array
        curve: np.ndarray = _hilbert.decode(
            hilberts=np.arange(N),
            num_dims=2,
            num_bits=n,
        )
        lit_curve = curve[data]

        # make empty dot matrix
        dots = np.zeros((2**n,2**n), dtype=bool)
        dots[lit_curve[:,0], lit_curve[:,1]] = True
        # transform to have origin at bottom left
        dots = dots.T
        dots = dots[::-1]
        
        # render data grid as a grid of braille characters
        chars = unicode_braille_array(
            dots=dots,
            fgcolor=color,
        )
        super().__init__(chars)
        self.num_points = len(curve)
        self.all_points = N
        self.n = n

    def __repr__(self):
        return (
            f"hilbert(height={self.height}, width={self.width}, "
            f"data=<{self.num_points} points out of {self.all_points} "
            f"on a {2**self.n} x {2**self.n} grid>)"
        )
