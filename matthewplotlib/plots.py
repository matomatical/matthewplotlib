"""
A collection of building blocks for plotting. There are lots of options---take a
look through this module. They are roughly grouped as follows.

Base class:

* `plot`: Every plot object inherits from this one. See this class for methods,
  properties, and shortcut operators available with every plot object.

Data plots:

* `scatter`
* `function`
* `scatter3`
* `line`
* `line3`
* `image`
* `heatmap`
* `function2`
* `histogram2`
* `progress`
* `bars`
* `histogram`
* `columns`
* `vistogram`
* `hilbert`
* `calendar`
* `weeks`

Furnishing plots:

* `text`
* `border`
* `axes`
* `colorbar`

Arrangement plots:

* `blank`
* `hstack`
* `vstack`
* `dstack`
* `wrap`
* `center`

The third stacking operation, `tstack`, arranges plots in time rather than
across the screen, and lives with the rest of the animation machinery in
`matthewplotlib.animations`.
"""
from __future__ import annotations

import calendar as _calendar
import datetime
import enum
import shutil
import numpy as np
import einops
import hilbert as _hilbert

from PIL import Image

from typing import Callable, Literal, NamedTuple, Self, cast
from numpy.typing import ArrayLike, NDArray
from matthewplotlib.colormaps import ColorMap
from matthewplotlib.colors import ColorLike, parse_color, parse_colors
from matthewplotlib.data import (
    number,
    Series,
    Series3,
    DateLike,
    DateSeries,
    parse_date,
    parse_date_series,
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
    CharArray,
    ords,
    _validate_text,
    BoxStyle,
    LineStyle,
    unicode_box,
    unicode_frame,
    unicode_braille_array,
    unicode_bar,
    unicode_col,
    unicode_image,
    unicode_braille_points,
    unicode_braille_segments,
)




# # # 
# BASE PLOT CLASS WITH SHORTCUTS


class plot:
    """
    Abstract base class for all plot objects.

    A plot is essentially a 2D grid of coloured characters. This class provides
    the core functionality for rendering and composing plots. It is not
    typically instantiated directly, but it's useful to know its properties and
    methods.
    """

    window: window | None = None
    """
    The interval of data the plot covers on each axis, and the rectangle of
    character cells it covers them with. None for a plot with no coordinates.
    """

    def __init__(self, chars: CharArray):
        self.chars = chars


    @property
    def height(self) -> int:
        """
        Number of character rows in the plot.
        """
        return self.chars.height


    @property
    def width(self) -> int:
        """
        Number of character columns in the plot.
        """
        return self.chars.width


    def renderstr(self) -> str:
        """
        Convert the plot into a string for printing to the terminal.

        Note: plot.renderstr() is equivalent to str(plot).
        """
        return self.chars.to_ansi_str()


    def clearstr(self: Self) -> str:
        """
        Convert the plot into a string that, if printed immediately after
        plot.renderstr(), will clear that plot from the terminal.

        Like every string in this library, the result is shaped for a plain
        `print`: it erases the plot and then steps one row above it, so the
        newline `print` appends leaves the cursor where the plot began, ready
        to be overdrawn. Printing it with `end=""` instead leaves the cursor a
        row high, and in an animation loop the plot then climbs the screen one
        row per frame.

        Requires a spare row above the plot. If the plot begins on the
        terminal's first row there is nowhere to step, and the redraw lands one
        row lower.

        Erases the plot's own rows and nothing else, so anything on the screen
        below the plot survives. That costs a few bytes per row rather than a
        single erase-to-end-of-screen, which is nothing beside the redraw that
        follows.

        Rows, though, and not columns: each row is erased margin to margin, so
        anything sitting to the right of the plot goes with it. A differential
        redraw (`plot - prev`) is careful about that boundary where this is not.
        See `notes/erase-granularity.md`.
        """
        H = self.height
        if H == 0:
            return "\x1b[1A"    # nothing to erase; just absorb print's newline
        erase = "\x1b[2K" + "\x1b[B\x1b[2K" * (H - 1)
        return f"\x1b[{H}A{erase}\x1b[{H}A"


    def renderimg(
        self,
        upscale: int = 1,
        downscale: int = 1,
        bgcolor: ColorLike | None = None,
    ) -> np.ndarray: # uint8[scale_factor * 16H, scale_factor * 8W, 4]
        """
        Convert the plot into an RGBA array for rendering with Pillow.
        """
        # render
        image = self.chars.to_rgba_array(bgcolor=bgcolor)
        # upscale
        if upscale > 1:
            image = einops.repeat(
                image,
                'H W rgba -> (H scale1) (W scale2) rgba',
                scale1=upscale,
                scale2=upscale,
            )
        # downscale
        if downscale > 1:
            image = image[::downscale, ::downscale]
        return image


    def saveimg(
        self,
        filename: str,
        upscale: int = 1,
        downscale: int = 1,
        bgcolor: ColorLike | None = None,
    ):
        """
        Render the plot as an RGBA image and save it as a PNG file at the path
        `filename`.
        """
        image_data = self.renderimg(
            bgcolor=bgcolor,
            upscale=upscale,
            downscale=downscale,
        )
        image = Image.fromarray(image_data)
        image.save(filename)


    def __str__(self) -> str:
        """
        Shortcut for the string for printing the plot.
        """
        return self.renderstr()


    def __neg__(self: Self) -> str:
        """
        Shortcut for the string for clearing the plot.
        """
        return self.clearstr()


    def updatestr(self: Self, prev: plot | None) -> str:
        """
        Convert the plot into a string that, if printed when the cursor is just
        below `prev` (i.e. immediately after printing `prev`), updates the
        terminal to show this plot instead -- repainting only the cells that
        differ from `prev`, and leaving the cursor just below this plot.

        This is the fast path for animation: redrawing a whole frame re-emits
        every cell, while this re-emits only what changed, which can be far
        fewer bytes over a slow connection.

        Inputs:

        * prev : plot | None.
          The plot currently on screen. Pass None for the first frame of an
          animation, when the screen is still empty, and the whole plot is
          rendered. A plot of a different size is fine: the overlapping region is
          still diffed, while the rows and columns only one of them covers are
          painted or erased as needed.

        As everywhere in this library the result is shaped for a plain `print`.
        See `CharArray.to_ansi_diff_str` for the precise cursor contract.
        """
        if prev is None or prev.height == 0:
            return self.renderstr()     # nothing on screen to diff against
        if self.height == 0:
            return prev.clearstr()      # nothing left to show
        return self.chars.to_ansi_diff_str(prev.chars)


    def __sub__(self: Self, other: plot | None) -> str:
        """
        Operator shortcut for a differential redraw: the string that updates the
        terminal from `other` to `self` in place.

        Subtracting None means there is nothing on screen yet, so the whole plot
        is drawn. That makes every frame of an animation the same statement:

        ```python
        prev = None
        for frame in frames:
            print(frame - prev)
            prev = frame
        ```

        Compare `-plot` (clear) and `str(plot)` (full redraw). See `updatestr`.
        """
        return self.updatestr(other)


    def __add__(self: Self, other: plot) -> hstack:
        """
        Operator shortcut for horizontal stack.
        
        ```
        plot1 + plot2 ==> hstack(plot1, plot2) ==> plot1 plot2
        ```

        When combining with vertical stacking, note that `/` binds before `+`,
        but `|` binds after:
        ```
        plot1 / plot2 + plot3 / plot4
        ==> hstack(vstack(plot1, plot2), vstack(plot3, plot4))
        ==> plot1 plot3
            plot2 plot4
        
        plot1 + plot2 | plot3 + plot4
        ==> vstack(hstack(plot1, plot2), hstack(plot3, plot4))
        ==> plot1 plot3
            plot2 plot4
        ```
        """
        return hstack(self, other)


    def __truediv__(self: Self, other: plot) -> vstack:
        """
        High-precedence operator shortcut for vertical stack.
        
        ```
        plot1 / plot2 ==> vstack(plot1, plot2) ==> plot1
                                                   plot2
        ```

        When combining with horizontal stacking, note that `/` binds before
        `+`:
        ```
        plot1 / plot2 + plot3 / plot4
        ==> plot1 plot3
            plot2 plot4
        ```

        For a version that binds after `+`, see `|`.
        """
        return vstack(self, other)


    def __or__(self: Self, other: plot) -> vstack:
        """
        Low-precedence operator shortcut for vertical stack.
        
        ```
        plot1 | plot2 ==> vstack(plot1, plot2) ==> plot1
                                                   plot2
        ```

        When combining with horizontal stacking, note that `|` binds after `+`:
        ```
        plot1 + plot2 | plot3 + plot4
        ==> plot1 plot3
            plot2 plot4
        ```

        For a version that binds before `+`, see `/`.
        """
        return vstack(self, other)


    def __matmul__(self: Self, other: plot) -> dstack:
        """
        Operator shortcut for depth stack.

        ```
        plot1_ @ plot_2 ==> dstack(plot1_, plot2_) => plot12
        (where _ is a blank character)
        ```

        Note that the precedence of `@` competes with `/`, so use parentheses
        or pair with `|`.
        """
        return dstack(self, other)
    

# # # 
# DATA PLOTTING CLASSES


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


class image(plot):
    """
    Render a small image or 2d array using a grid of unicode half-block
    characters.

    Represents an image by mapping pairs of vertically adjacent pixels to the
    foreground and background colors of a single character cell (this
    effectively doubles the vertical resolution in the terminal).

    Inputs:

    * im : float[h,w,3] | int[h,w,3] | float[h,w] | int[h,w].
        The image data. Without a colormap, an array-like matching any of the
        following formats:
        * `float[h,w,3]`: A 2D array of RGB triples of floats in range [0,1].
        * `int[h,w,3]`: A 2D array of RGB triples of ints in range [0,255].
        * `float[h,w]`: A 2D array of scalars in the range [0,1], treated as
          greyscale (uniform colorisation).
        * `int[h,w]`: A 2D array of ints in the range [0,255], treated as
          greyscale (uniform colorisation).

        With a colormap, the input may instead be any array accepted by that
        function, provided the colormap returns an RGB image of shape [h,w,3].
          
    * colormap : optional ColorMap.
        Applied to the input before its colour shape is validated. The
        colormaps provided by this library map (batches of) scalars to
        (batches of) RGB triples, such as:
        * continuous colormaps like `viridis : float[...] -> uint8[...,3]`, and
        * discrete colormaps like `pico8 : int[...] -> uint8[...,3]`.

        A custom colormap may consume any array data but must return an RGB
        image of shape [h,w,3].

    * xrange : optional (number, number).
        The data coordinates at the left and the right edges of the image. By
        default the image carries no horizontal coordinate, and so cannot be
        given an axis or overlaid on another plot.
    * yrange : optional (number, number).
        The data coordinates at the bottom and the top edges of the image. By
        default the image carries no vertical coordinate.

    Since each character cell holds two pixels, an image with an odd number of
    pixel rows leaves the bottom half of its last row blank. Such an image
    cannot be given coordinates, since its rectangle would claim half a cell
    more than the picture covers.

    A grid of values on any other scale is a `heatmap`, which normalises them
    onto this one and keeps the interval it used, so that the picture can be
    given a colorbar.
    """
    def __init__(
        self,
        im: ArrayLike, # float[h,w] | float[h,w,rgb] | int[h,w] | int[h,w,rgb]
        colormap: ColorMap | None = None,
        xrange: tuple[number, number] | None = None,
        yrange: tuple[number, number] | None = None,
    ):
        # preprocessing: all inputs become uint8[h, w, rgb]
        arr = parse_colors(
            im,
            shape=("h", "w"),
            colormap=colormap,
        )
        if arr.shape[0] % 2 and (xrange is not None or yrange is not None):
            raise ValueError(
                f"an image of {arr.shape[0]} pixel rows cannot be given "
                "coordinates, since it half-fills its last character row"
            )

        # construct the plot
        chars = unicode_image(arr)

        # form a plot object
        super().__init__(chars)
        self.window = window(
            xrange=xrange,
            yrange=yrange,
            width=chars.width,
            height=chars.height,
        )
        self.colormap = colormap
        self.vrange: tuple[number, number] | None = None

    def __repr__(self):
        return f"image({self.window!r})"


def _value_range(
    vrange: tuple[number, number] | None,
    values: NDArray,
    what: str,
) -> tuple[number, number]:
    """
    The interval of values a colour scale covers.

    Given, it is taken as it stands, and one covering no interval is an error.
    Omitted, it runs from the lowest to the highest finite value there is, so
    that the colours span the data, and falls back to the unit interval where
    there are no finite values at all. `what` names the caller in the error.
    """
    if vrange is not None:
        vmin, vmax = vrange
        if vmin == vmax:
            raise ValueError(f"{what} vrange covers no interval: {vrange!r}")
        return (vmin, vmax)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (0.0, 1.0)
    return (finite.min(), finite.max())


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


class heatmap(image):
    """
    Render a grid of values, colouring each by where it falls in an interval.

    The values are normalised onto the range 0.0 to 1.0 and handed to a
    colormap, so that the caller does not scale them by hand and the colours
    mean the same thing from one plot to the next. The interval and the
    colormap are kept, so that the plot can be given a `colorbar`.

    Inputs:

    * values : number[h, w].
        The value at each pixel, the first row at the top.
    * colormap : optional ColorMap.
        Maps each normalised value onto its colour. By default the values come
        out as shades of grey, black at the bottom of the interval and white
        at the top.
    * vrange : optional (number, number).
        The interval of values the colormap covers. Values outside it saturate
        at the nearest end. By default the interval runs from the lowest to the
        highest value in the grid, so that the colours span the data.

        Given descending, the scale turns around: `vrange=(1, 0)` colours the
        low values the way the high values would have been coloured.
    * xrange : optional (number, number).
        The data coordinates at the left and the right edges of the grid. By
        default the heatmap carries no horizontal coordinate, and so cannot be
        given an axis or overlaid on another plot.
    * yrange : optional (number, number).
        The data coordinates at the bottom and the top edges of the grid. By
        default the heatmap carries no vertical coordinate.

    Since each character cell holds two pixels, a grid with an odd number of
    rows leaves the bottom half of its last row blank, and cannot be given
    coordinates.

    Where every value is the same, they all come out at the bottom of the
    colormap, since there is no interval for the colours to span. An explicit
    `vrange` covering no interval is an error rather than a guess.

    A value that is not a number is left out of an inferred interval, and comes
    out at the bottom of the colormap wherever it appears. Infinities saturate
    at the ends like any other value beyond the interval.

    A grid of colours, of palette indices, or of values already scaled onto
    the range 0.0 to 1.0 is an `image` rather than a heatmap: those need no
    interval, and carry no colour scale.
    """
    def __init__(
        self,
        values: ArrayLike, # number[h, w]
        colormap: ColorMap | None = None,
        vrange: tuple[number, number] | None = None,
        xrange: tuple[number, number] | None = None,
        yrange: tuple[number, number] | None = None,
    ):
        grid = np.asarray(values, dtype=float)
        if grid.ndim != 2:
            raise ValueError(
                "heatmap needs a 2d grid of values, not an array of shape "
                f"{grid.shape}"
            )
        vrange = _value_range(vrange, grid, "heatmap")

        super().__init__(
            im=_normalise(grid, vrange),
            colormap=colormap,
            xrange=xrange,
            yrange=yrange,
        )
        self.vrange = vrange

    def __repr__(self):
        vmin, vmax = self.vrange
        return f"heatmap({self.window!r}, vrange=[{vmin:.2f},{vmax:.2f}])"


class function2(heatmap):
    """
    Heatmap representing the image of a 2d function over a square.

    Inputs:

    * F : float[batch, 2] -> number[batch].
        The (vectorised) function to plot. The input should be a batch of
        (x, y) vectors. The output should be a batch of scalars f(x, y).
    * xrange : (float, float).
        Lower and upper bounds on the x values to pass into the function.
    * yrange : (float, float).
        Lower and upper bounds on the y values to pass into the function.
    * width : int.
        The number of character columns in the plot. This will also become the
        number of grid squares along the x axis.
    * height : int.
        The number of character rows in the plot. This will also be half of the
        number of grid squares, since the result is an image plot with two
        half-character-pixels per row.
    * vrange : optional (float, float).
        Expected lower and upper bounds on the f(x, y) values. Used for
        determining the bounds of the colour scale. By default, the minimum and
        maximum output over the grid are used. Values outside these bounds
        saturate at the nearest end of the colour scale.
    * colormap : optional colormap (e.g. mp.viridis).
        By default, the output will be in greyscale, with black corresponding
        to vrange[0] and white corresponding to vrange[1]. You can choose a
        different colormap (e.g. mp.reds, mp.viridis, etc.) here.
    * endpoints : bool (default: False).
        By default, the grid squares tile the ranges exactly and each one shows
        the value of the function at its own centre.

        If true, the function is instead sampled at points spread from one end
        of each range to the other, so that the four corner squares show the
        four corner combinations of xrange and yrange. The squares then reach
        half a square beyond the ranges, which the axes still report as the
        limits.
    """
    def __init__(
        self,
        F: Callable[[np.ndarray], np.ndarray],
        xrange: tuple[float, float],
        yrange: tuple[float, float],
        width: int,
        height: int,
        vrange: tuple[float, float] | None = None,
        colormap: ColorMap | None = None,
        endpoints: bool = False,
    ):
        # the coordinates each grid square stands for, top row first
        w = window(xrange=xrange, yrange=yrange, width=width, height=height)
        if endpoints:
            X, Y = np.meshgrid(
                np.linspace(*xrange, num=width),
                np.linspace(*yrange, num=2*height)[::-1],
            ) # float[h, w] (x2)
        else:
            X, Y = w.pixel_centres()
        XY = einops.rearrange(np.dstack((X, Y)), 'h w xy -> (h w) xy')

        # sample the function
        Z = F(XY)

        # create the heatmap itself
        zgrid = einops.rearrange(Z, '(h w) -> h w', h=2*height, w=width)
        super().__init__(
            values=zgrid,
            colormap=colormap,
            vrange=vrange,
            xrange=xrange,
            yrange=yrange,
        )
        self.name = getattr(F, '__name__', '?')

    def __repr__(self):
        return f"function2(f={self.name}, {self.window!r})"


class histogram2(heatmap):
    """
    Heatmap representing the density of a collection of 2d points.

    Inputs:

    * x : number[n].
        X coordinates of 2d points to bin and count.
    * y : number[n].
        Y coordinates of 2d points to bin and count.
    * width : int (default 24).
        Specifies the width of the plot in characters. This is also the number
        of bins in the x direction.
    * height : int (default 12).
        Specifies the height of the plot in characters. This is also half the
        number of bins in the y direction.
    * xrange : optional (number, number).
        The x-axis limits `(xmin, xmax)`. If not provided, the limits are
        inferred from the min and max x-values in the data.
    * yrange : optional (number, number).
        The y-axis limits `(ymin, ymax)`. If not provided, the limits are
        inferred from the min and max y-values in the data.
    * weights : optional number[n].
        If provided, each 2d point in data contributes this amount to the count
        for its bin (rather than the default 1). See np.histogram2d's weights
        argument for details.
    * density : bool (default False).
        If true, normalise bin counts so that they sum to 1,0. See
        np.histogram2d's density argument for details.
    * max_count : optional number.
        If provided, cell colours are scaled so that only bars matching or
        exceeding this count max out the colour. Otherwise, the colours are
        scaled so that the bin with the highest count has the colour maxed out.
        An explicitly supplied value must be positive. If every bin has a count
        of zero, all cells remain at the bottom of the colour scale.
    * colormap : optional colormap (e.g. mp.viridis).
        By default, the output will be in greyscale, with black corresponding
        to zero density and white corresponding to max_count. You can choose a
        different colormap (e.g. mp.reds, mp.viridis, etc.) here.
    """
    def __init__(
        self,
        x: ArrayLike, # number[n]
        y: ArrayLike, # number[n]
        width: int = 24,
        height: int = 12,
        xrange: tuple[number, number] | None = None,
        yrange: tuple[number, number] | None = None,
        weights = None, # see np.histogram2d
        density = False, # see np.histogram2d
        max_count: None | number = None,
        colormap: ColorMap | None = None,
    ):
        # prepare data
        x = np.asarray(x)
        y = np.asarray(y)
        
        # determine data bounds
        xmin, ymin = x.min(), y.min()
        xmax, ymax = x.max(), y.max()
        if xrange is None:
            xrange = (xmin, xmax)
        else:
            xmin, xmax = xrange
        if yrange is None:
            yrange = (ymin, ymax)
        else:
            ymin, ymax = yrange
        
        # bin data. the squares tile the window, and numpy needs their edges
        # ascending, so bin in that order and turn the counts back around
        # afterwards wherever the range runs the other way
        w = window(
            xrange=xrange,
            yrange=yrange,
            width=width,
            height=height,
        )
        xedges, yedges = w.pixel_edges()
        xflip = xedges[0] > xedges[-1]
        yflip = yedges[0] > yedges[-1]
        hist, xbins, ybins = np.histogram2d(
            x=x,
            y=y,
            bins=(
                xedges[::-1] if xflip else xedges,
                yedges[::-1] if yflip else yedges,
            ),
            weights=weights,
            density=density,
        )

        # reorient the counts to match the window
        hist = hist.T[::-1]     # row zero is the top of the window
        if yflip:
            hist = hist[::-1]
        if xflip:
            hist = hist[:, ::-1]

        # the counts the colormap covers, from an empty bin at the bottom to
        # the fullest one at the top, or to a ceiling the caller sets. A
        # histogram with nothing in it has no fullest bin to find, so the unit
        # interval stands in and every bin sits at the bottom regardless.
        if max_count is None:
            max_count = hist.max() if hist.max() > 0 else 1
        elif max_count <= 0:
            raise ValueError(f"max_count must be positive, not {max_count!r}")

        # construct the heatmap
        super().__init__(
            values=hist,
            colormap=colormap,
            vrange=(0, max_count),
            xrange=xrange,
            yrange=yrange,
        )
        self.xbins = xedges
        self.ybins = yedges
        self.num_points = len(x)
        self.max_count = max_count
        
    def __repr__(self):
        return f"histogram2(<{self.num_points} points>, {self.window!r})"


class progress(plot):
    """
    A single-line progress bar.

    Construct a progress bar with a percentage label. The bar is rendered using
    Unicode block element characters to show fractional progress with finer
    granularity.

    Inputs:

    * progress : float.
        The progress to display, as a float between 0.0 and 1.0. Values outside
        this range will be clipped.
    * width : int (default: 40).
        The total width of the progress bar plot in character columns,
        including the label and brackets.
    * height: int (default: 1).
        The height of the progress bar in character rows.
    * color : optional ColorLike.
        The color of the filled portion of the progress bar. Defaults to the
        terminal's default foreground color.
    """
    def __init__(
        self,
        progress: float,
        width: int = 40,
        height: int = 1,
        color: ColorLike | None = None,
    ):
        progress = np.clip(progress, 0., 1.)

        # construct label
        label = f"{progress:4.0%}"
        
        # construct bar
        raw_chars = unicode_bar(
            proportion=progress,
            width=width - 2 - len(label),
            height=height,
            fgcolor=color,
            bgcolor=None,
        )

        # add boundaries
        all_chars = raw_chars.pad(
            left=len(label)+1,
            right=1,
        )
        all_chars.codes[0, :len(label)] = ords(label)
        all_chars.codes[:, len(label)] = ord("[")
        all_chars.codes[:, -1] = ord("]")

        # put it together
        super().__init__(all_chars)
        self.progress = progress

    def __repr__(self):
        return f"progress({self.progress:%})"


class bars(plot):
    """
    A multi-line bar chart.

    Transform a list of values into horizontal bars with width indicating the
    values. The bars are rendered using Unicode block element characters for
    finer granularity.

    Inputs:

    * values : float[n].
        An array of non-negative values to display.
    * width : int (default: 30).
        The total width of full bars.
    * bar_height: int (default: 1).
        The number of rows comprising each bar.
    * bar_spacing: int (default: 0).
        The number of rows between each bar.
    * vrange : optional (float, float).
        The interval of values the bars measure: a bar at the first value or
        below has zero width and one at the second value or above occupies the
        whole width. By default the interval runs from zero to the largest
        value, so that the largest bar or bars fill the width.
    * color : optional ColorLike.
        The color of the filled portion of the bars. Defaults to the terminal's
        default foreground color.
    * colors : optional ColorLike[n].
        The colours of the filled portion of each bar. Should be an array or
        list of the same length as `values`.

    TODO:

    * Make it possible to draw bars to the left for values below 0.
    * Make it possible to align all bars to the right rather than left.
    """
    def __init__(
        self,
        values: ArrayLike, # numeric[n]
        width: int = 30,
        bar_height: int = 1,
        bar_spacing: int = 0,
        vrange: tuple[number, number] | None = None,
        color: ColorLike | None = None,
        colors: list[ColorLike | None] | None = None,
    ):
        # standardise inputs
        values = np.asarray(values)
        vmin, vmax = (0.0, values.max()) if vrange is None else vrange
        num_bars = len(values)

        # compute the bar widths
        norm_values = (values - vmin) / (vmax - vmin + 1e-15)

        # determine the colors for each bar
        if colors is None:
            colors = [color for _ in range(len(values))]
        
        # construct the bars
        bars_chars = [
            unicode_bar(
                proportion=v,
                width=width,
                height=bar_height,
                fgcolor=colors[i],
                bgcolor=None,
            ).pad(
                below=bar_spacing * (i!=num_bars-1),
            )
            for i, v in enumerate(norm_values)
        ]
        all_chars = CharArray.map(
            lambda xs: np.concatenate(xs, axis=0),
            bars_chars,
        )
        super().__init__(chars=all_chars)
        self.vrange = (vmin, vmax)
        self.num_bars = num_bars

    def __repr__(self):
        vmin, vmax = self.vrange
        return (
            f"bars(height={self.height}, width={self.width}, "
            f"values=<{self.num_bars} bars on "
            f"[{vmin:.2f},{vmax:.2f}]>)"
        )


class histogram(bars):
    """
    A histogram bar chart.

    Transform a sequence of values into horizontal bars representing the
    density in different bins. The bars are rendered using Unicode block
    element characters for finer granularity.

    Inputs:

    * data : number[n].
        An array of values to count.
    * xrange : optional (float, float).
        If provided, bins range over this interval, and values outside the
        range are discarded. Same as np.histogram's range argument.
    * bins : int (default: 10).
        Used to determine number of bins. Bins are evenly spaced as if this
        number if provided to np.histogram's bins argument.
    * weights : optional number[n].
        If provided, each element in data contributes this amount to the count
        for its bin (rather than the default 1). See np.histogram's weights
        argument for details.
    * density : bool (default False).
        If true, normalise bin counts so that they sum to 1,0. See
        np.histogram's density argument for details.
    * max_count : optional number.
        If provided, the bars are scaled so that only bars matching or
        exceeding this count are full. Otherwise, the bars are scaled so that
        the bin with the highest count has a full bar.
    * width : int (default: 22).
        The total width of full bars.
    * color : optional ColorLike.
        The color of the filled portion of the bars. Defaults to the terminal's
        default foreground color.
    """
    def __init__(
        self,
        data: ArrayLike, # number[n]
        bins: int = 10,
        xrange: tuple[float, float] | None = None,
        weights: ArrayLike | None = None, # optional number[n]
        density: bool = False,
        max_count: number | None = None,
        width: int = 22,
        color: ColorLike | None = None,
    ):
        # prepare data
        data = np.asarray(data)
        weights_ = None if weights is None else np.asarray(weights)
        
        # bin data
        hist, bins_ = np.histogram(
            a=data,
            bins=bins,
            range=xrange,
            weights=weights_,
            density=cast(Literal[True, False], density),
        )

        # build bar chart
        if max_count is None:
            max_count = hist.max()
        super().__init__(
            values=hist,
            width=width,
            bar_height=1,
            bar_spacing=0,
            vrange=(0, max_count),
            color=color,
        )
        self.bins = bins_

    def __repr__(self):
        return (
            f"histogram(height={self.height}, width={self.width}, "
            f"bins=<{len(self.bins)-1} on "
            f"[{self.bins[0]:.2f},{self.bins[-1]:.2f}]>)"
        )


class columns(plot):
    """
    A column chart.

    Transform a list of values into vertical columns with height indicating the
    values. The columns are rendered using Unicode block element characters for
    finer granularity.

    Inputs:

    * values : number[n].
        An array of non-negative values to display.
    * height : int (default: 10).
        The total width of full columns.
    * column_width: int (default 1).
    * column_spacing: int (default 0).
    * vrange : optional (number, number).
        The interval of values the columns measure: a column at the first value
        or below has zero height and one at the second value or above occupies
        the whole height. By default the interval runs from zero to the largest
        value, so that the tallest column or columns fill the height.
    * color : optional ColorLike.
        The color of the filled portion of the columns. Defaults to the
        terminal's default foreground color.
    * colors : optional ColorLike[n].
        The colours of the filled portion of each column. Should be an array or
        list of the same length as `values`.

    TODO:

    * Make it possible to draw columns downward for values below 0.
    * Make it possible to align all columns to the top rather than bottom.
    """
    def __init__(
        self,
        values: ArrayLike, # number[n], actually int[n] will also work
        height: int = 10,
        column_width: int = 1,
        column_spacing: int = 0,
        vrange: tuple[number, number] | None = None,
        color: ColorLike | None = None,
        colors: list[ColorLike | None] | None = None,
    ):
        # standardise inputs
        values = np.asarray(values)
        vmin, vmax = (0.0, values.max()) if vrange is None else vrange
        num_cols = len(values)

        # compute the column heights
        norm_values = (values - vmin) / (vmax - vmin + 1e-15)

        # determine the colours
        if colors is None:
            colors = [color for _ in range(len(values))]
        
        # construct the columns
        cols_chars = [
            unicode_col(
                proportion=v,
                height=height,
                width=column_width,
                fgcolor=colors[i],
                bgcolor=None,
            ).pad(
                right=column_spacing * (i!=num_cols-1),
            )
            for i, v in enumerate(norm_values)
        ]
        all_chars = CharArray.map(
            lambda xs: np.concatenate(xs, axis=1),
            cols_chars,
        )
        super().__init__(chars=all_chars)
        self.vrange = (vmin, vmax)
        self.num_cols = num_cols

    def __repr__(self):
        vmin, vmax = self.vrange
        return (
            f"columns(height={self.height}, width={self.width}, "
            f"values=<{self.num_cols} columns on "
            f"[{vmin:.2f},{vmax:.2f}]>)"
        )


class vistogram(columns):
    """
    A histogram column chart ("vertical histogram", referring to the direction
    of the bars rather than the bins).

    Transform a sequence of values into columns representing the density in
    different bins. The columns are rendered using Unicode block element
    characters for finer granularity.

    Inputs:

    * data : number[n].
        An array of values to count.
    * xrange : optional (float, float).
        If provided, bins range over this interval, and values outside the
        range are discarded. Same as np.histogram's range argument.
    * bins : int (default: 10).
        Used to determine number of bins. Bins are evenly spaced as if this
        number if provided to np.histogram's bins argument.
    * weights : optional number[n].
        If provided, each element in data contributes this amount to the count
        for its bin (rather than the default 1). See np.histogram's weights
        argument for details.
    * density : bool (default False).
        If true, normalise bin counts so that they sum to 1,0. See
        np.histogram's density argument for details.
    * max_count : optional number.
        If provided, the bars are scaled so that only bars matching or
        exceeding this count are full. Otherwise, the bars are scaled so that
        the bin with the highest count has a full bar.
    * height : int (default: 22).
        The total height of full bars.
    * color : optional ColorLike.
        The color of the filled portion of the bars. Defaults to the terminal's
        default foreground color.
    """
    def __init__(
        self,
        data: ArrayLike, # number[n]
        bins: int = 10,
        xrange: tuple[float, float] | None = None,
        weights: ArrayLike | None = None, # optional number[n]
        density: bool = False,
        max_count: None | number = None,
        height: int = 10,
        color: ColorLike | None = None,
    ):
        # prepare data
        data = np.asarray(data)
        weights_ = None if weights is None else np.asarray(weights)
        
        # bin data
        hist, bins_ = np.histogram(
            a=data,
            bins=bins,
            range=xrange,
            weights=weights_,
            density=cast(Literal[True, False], density),
        )

        # build column chart
        if max_count is None:
            max_count = hist.max()
        super().__init__(
            values=hist,
            height=height,
            column_width=1,
            column_spacing=0,
            vrange=(0, max_count),
            color=color,
        )
        self.bins = bins_

    def __repr__(self):
        return (
            f"vistogram(height={self.height}, width={self.width}, "
            f"bins=<{len(self.bins)-1} on "
            f"[{self.bins[0]:.2f},{self.bins[-1]:.2f}]>)"
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
            f"on a {2**self.n} x {2**self.n} grid>"
        )


_WEEKDAY_INITIALS = "MTWtFSs"
"""
The initials of the weekdays from Monday, distinguished by case where two of
them claim the same letter: `t` for Thursday and `s` for Sunday.
"""

# The glyphs a day is drawn with. The first cell of each day is notched in its
# top-left corner, which is what separates a day from the ones above and to the
# left of it, since adjacent days would otherwise merge into one block of
# colour.
_DAY_BODY = ord("█")
_DAY_CORNER = ord("▟")


class _ColoredDays(NamedTuple):
    """The days a dated plot draws, and the scale their colours came from."""
    colors: dict[datetime.date, NDArray]    # uint8[3] per day
    first: datetime.date
    last: datetime.date
    vrange: tuple[number, number]


def _color_days(
    data: DateSeries,
    vrange: tuple[number, number] | None,
    colormap: ColorMap | None,
    daterange: tuple[DateLike, DateLike] | None,
    what: str,
) -> _ColoredDays:
    """
    Work out which days a plot of dated values draws, and in what colour.

    A day is drawn when the data gives it a finite value and it falls inside
    the range of days asked for. `what` names the caller in the errors, which
    it raises for data with no days in it at all and for a range that ends
    before it starts.
    """
    dates, values = parse_date_series(data)
    if not dates:
        raise ValueError(f"{what} needs at least one dated value")

    # determine the range of days to draw
    if daterange is None:
        first, last = dates[0], dates[-1]
    else:
        first = parse_date(daterange[0])
        last = parse_date(daterange[1])
    if last < first:
        raise ValueError(f"daterange ends ({last}) before it starts ({first})")

    # keep the days that get a colour: the ones in range with a value
    drawn = [
        (date, value)
        for date, value in zip(dates, values)
        if first <= date <= last and np.isfinite(value)
    ]

    # determine the value scale over those days, and colour them by it
    levels = np.array([value for _, value in drawn], dtype=float)
    vrange = _value_range(vrange, levels, what)
    rgb = parse_colors(
        _normalise(levels, vrange),
        n=len(levels),
        colormap=colormap,
    )
    return _ColoredDays(
        colors={date: color for (date, _), color in zip(drawn, rgb)},
        first=first,
        last=last,
        vrange=vrange,
    )


def _paint_day(
    chars: CharArray,
    row: int,
    column: int,
    day_width: int,
    color: NDArray,       # uint8[3]
    bgcolor: NDArray | None,
) -> None:
    """
    Fill one day's cells, notching the first so that it reads as its own day
    rather than merging into its neighbours.
    """
    cells = slice(column, column + day_width)
    chars.codes[row, cells] = _DAY_BODY
    chars.codes[row, column] = _DAY_CORNER
    chars.fg[row, cells] = True
    chars.fg_rgb[row, cells] = color
    if bgcolor is not None:
        chars.bg[row, cells] = True
        chars.bg_rgb[row, cells] = bgcolor


_WEEKDAY_GUTTER = 2
"""
The width of the gutter a strip of weeks heads its rows in: the weekday's
initial, and a blank column separating it from the days.
"""


def _write_caption(
    chars: CharArray,
    row: int,
    column: int,
    caption: str,
    width: int,
) -> None:
    """
    Write a caption at a column of a row, unless it would run off the end of
    the row or into the caption already there, which needs a blank column
    between them.
    """
    if column + len(caption) > width:
        return
    occupied = chars.codes[row, max(0, column - 1):column + len(caption)]
    if (occupied != ord(" ")).any():
        return
    chars.codes[row, column:column + len(caption)] = ords(caption)


def _month_caption(month: int, year: int, width: int) -> str:
    """
    Name a month and its year within a block `width` characters wide.

    The name goes on the left and the year on the right of the widest spelling
    that fits, abbreviating the month and then the year as the width runs out,
    and giving up on the year entirely before the month. Empty if not even the
    abbreviated month fits.
    """
    spellings = [
        (_calendar.month_name[month], f"{year:4d}"),
        (_calendar.month_abbr[month], f"{year:4d}"),
        (_calendar.month_abbr[month], f"{year % 100:02d}"),
        (_calendar.month_abbr[month], ""),
    ]
    # Wider months are captioned in the width the longest month name needs, so
    # that the years line up down a column of them however wide the days are.
    longest = max(len(name) for name in _calendar.month_name[1:])
    limit = min(width, longest + len(" 2025"))
    for name, year_ in spellings:
        if len(name) + len(year_) + bool(year_) <= limit:
            gap = limit - len(name) - len(year_)
            return name + " " * gap + year_
    return ""


class calendar(plot):
    """
    Calendar heatmap of values observed on dates.

    Draws a block per month, a row per week and a column per weekday, colouring
    each day by its value, and wraps the months into a grid. A year of daily
    data becomes a wall calendar.

    Inputs:

    * data : DateSeries.
        The dated values to colour: a mapping from dates to values, a pair of
        sequences of dates and values, or one date and the values on the days
        running from it. See `DateSeries` for the full list of forms.
    * vrange : optional (number, number).
        The interval of values the colormap covers: values at the first limit
        or below come out at the bottom of the colormap and values at the
        second or above come out at the top. By default the interval runs from
        the lowest to the highest value among the days drawn, so that the
        colours span the data.
    * colormap : optional ColorMap.
        Maps each day's value, normalised to the range 0.0 to 1.0, onto its
        colour. By default the days are shades of grey, black for the bottom of
        the range and white for the top.
    * daterange : optional (date, date).
        The first and last day to draw, each spelled any way a `DateLike` can
        be. Days outside the range are left blank even where the data has
        values for them. If omitted, the range spans the dates in the data.
    * cols : optional int (default 4).
        The number of months in each row of the grid. If None, as many as fit
        the width of the terminal.
    * first_weekday : int (default 0).
        The weekday to start each week on, from 0 for Monday to 6 for Sunday,
        numbered as in Python's `calendar` module.
    * day_width : int (default 2).
        The number of character cells each day is drawn in. Two is close to
        square in a terminal.
    * month_spacing : int (default 1).
        The gap to leave between month blocks, counted in days so that it stays
        square whatever `day_width` is.
    * month_labels : bool (default True).
        Whether to caption each month with its name and year, abbreviating the
        caption to fit if the days are narrow.
    * weekday_labels : bool (default True).
        Whether to head each month's columns with the initials of the weekdays.
    * bgcolor : optional ColorLike.
        The color to show through the notch in the corner of each day. Defaults
        to a transparent background, showing the terminal's own.

    A date the data says nothing about, or gives a value that is not finite, is
    left blank, so that a day with no value stays distinct from a day whose
    value is zero.
    """
    def __init__(
        self,
        data: DateSeries,
        vrange: tuple[number, number] | None = None,
        colormap: ColorMap | None = None,
        daterange: tuple[DateLike, DateLike] | None = None,
        cols: int | None = 4,
        first_weekday: int = 0,
        day_width: int = 2,
        month_spacing: int = 1,
        month_labels: bool = True,
        weekday_labels: bool = True,
        bgcolor: ColorLike | None = None,
    ):
        # standardise inputs
        if day_width < 1:
            raise ValueError(f"day_width must be positive, not {day_width}")
        if month_spacing < 0:
            raise ValueError(
                f"month_spacing must not be negative, not {month_spacing}"
            )
        if not 0 <= first_weekday <= 6:
            raise ValueError(
                f"first_weekday must be a weekday from 0 to 6, not "
                f"{first_weekday}"
            )
        dated = _color_days(
            data=data,
            vrange=vrange,
            colormap=colormap,
            daterange=daterange,
            what="calendar",
        )
        colors = dated.colors
        first, last = dated.first, dated.last
        bg = parse_color(bgcolor)

        # determine the months to draw
        months = []
        year, month = first.year, first.month
        while (year, month) <= (last.year, last.month):
            months.append((year, month))
            year, month = (year, month + 1) if month < 12 else (year + 1, 1)

        # draw each month
        weeks_of = _calendar.Calendar(first_weekday).monthdayscalendar
        width = 7 * day_width
        month_plots = []
        for year, month in months:
            weeks = weeks_of(year, month)
            captions = []
            if month_labels:
                captions.append(_month_caption(month, year, width))
            if weekday_labels:
                captions.append("".join(
                    _WEEKDAY_INITIALS[(first_weekday + i) % 7].ljust(day_width)
                    for i in range(7)
                ))
            chars = CharArray.from_size(
                height=len(captions) + len(weeks),
                width=width,
            )
            for row, caption in enumerate(captions):
                chars.codes[row, :len(caption)] = ords(caption)
            for week, days in enumerate(weeks):
                for weekday, day in enumerate(days):
                    if day == 0:
                        continue
                    date = datetime.date(year, month, day)
                    if date not in colors:
                        continue
                    _paint_day(
                        chars=chars,
                        row=len(captions) + week,
                        column=weekday * day_width,
                        day_width=day_width,
                        color=colors[date],
                        bgcolor=bg,
                    )
            month_plots.append(plot(chars.pad(
                below=month_spacing,
                right=month_spacing * day_width,
            )))

        # arrange the months into a grid, less the spacing off its far edges
        grid = wrap(*month_plots, cols=cols).chars
        # A grid is as many columns wide as it was asked for, whether or not
        # there are months to fill them, so the empty ones come back off. How
        # many there were is read back from the grid rather than worked out
        # again, since with no `cols` it was the terminal's width that decided.
        cell_width = width + month_spacing * day_width
        filled = min(grid.width // cell_width, len(months))
        rows = grid.height - month_spacing
        columns = filled * cell_width - month_spacing * day_width
        super().__init__(CharArray(
            codes=grid.codes[:rows, :columns],
            fg=grid.fg[:rows, :columns],
            fg_rgb=grid.fg_rgb[:rows, :columns],
            bg=grid.bg[:rows, :columns],
            bg_rgb=grid.bg_rgb[:rows, :columns],
        ))
        self.vrange = dated.vrange
        self.colormap = colormap
        self.daterange = (first, last)
        self.num_days = len(colors)

    def __repr__(self):
        first, last = self.daterange
        vmin, vmax = self.vrange
        return (
            f"calendar(height={self.height}, width={self.width}, "
            f"data=<{self.num_days} days from {first} to {last} on "
            f"[{vmin:.2f},{vmax:.2f}]>)"
        )


class weeks(plot):
    """
    Calendar heatmap of values observed on dates, as an unbroken strip.

    Draws a column per week and a row per weekday, colouring each day by its
    value, running without a break from the first day drawn to the last. A
    year of daily data becomes a band seven rows deep, captioned with the
    months along the top.

    Inputs:

    * data : DateSeries.
        The dated values to colour: a mapping from dates to values, a pair of
        sequences of dates and values, or one date and the values on the days
        running from it. See `DateSeries` for the full list of forms.
    * vrange : optional (number, number).
        The interval of values the colormap covers: values at the first limit
        or below come out at the bottom of the colormap and values at the
        second or above come out at the top. By default the interval runs from
        the lowest to the highest value among the days drawn, so that the
        colours span the data.
    * colormap : optional ColorMap.
        Maps each day's value, normalised to the range 0.0 to 1.0, onto its
        colour. By default the days are shades of grey, black for the bottom of
        the range and white for the top.
    * daterange : optional (date, date).
        The first and last day to draw, each spelled any way a `DateLike` can
        be. Days outside the range are left blank even where the data has
        values for them. If omitted, the range spans the dates in the data.
    * width : optional int.
        The most characters the strip may occupy. A strip with more weeks than
        fit continues on further bands below, each captioned again, with a
        blank row between them. If omitted the strip runs its whole length on
        one band, however wide that is.
    * first_weekday : int (default 0).
        The weekday to put in the top row, from 0 for Monday to 6 for Sunday,
        numbered as in Python's `calendar` module.
    * day_width : int (default 2).
        The number of character cells each day is drawn in. Two is close to
        square in a terminal.
    * year_labels : bool (default True).
        Whether to caption the first month drawn of each year with the year,
        in a row above the months.
    * month_labels : bool (default True).
        Whether to caption the week each month begins in with the month's
        abbreviated name. A caption that would collide with the one before it
        is dropped, so narrow days give fewer of them.
    * weekday_labels : bool (default True).
        Whether to head each row with the initial of its weekday, in a gutter
        two characters wide to the left of the strip.
    * bgcolor : optional ColorLike.
        The color to show through the notch in the corner of each day. Defaults
        to a transparent background, showing the terminal's own.

    A date the data says nothing about, or gives a value that is not finite, is
    left blank, so that a day with no value stays distinct from a day whose
    value is zero.
    """
    def __init__(
        self,
        data: DateSeries,
        vrange: tuple[number, number] | None = None,
        colormap: ColorMap | None = None,
        daterange: tuple[DateLike, DateLike] | None = None,
        width: int | None = None,
        first_weekday: int = 0,
        day_width: int = 2,
        year_labels: bool = True,
        month_labels: bool = True,
        weekday_labels: bool = True,
        bgcolor: ColorLike | None = None,
    ):
        # standardise inputs
        if day_width < 1:
            raise ValueError(f"day_width must be positive, not {day_width}")
        if not 0 <= first_weekday <= 6:
            raise ValueError(
                f"first_weekday must be a weekday from 0 to 6, not "
                f"{first_weekday}"
            )
        gutter = _WEEKDAY_GUTTER if weekday_labels else 0
        if width is not None and width < gutter + day_width:
            raise ValueError(
                f"width must leave room for the gutter and a week, "
                f"{gutter + day_width} characters, not {width}"
            )
        dated = _color_days(
            data=data,
            vrange=vrange,
            colormap=colormap,
            daterange=daterange,
            what="weeks",
        )
        colors = dated.colors
        first, last = dated.first, dated.last
        bg = parse_color(bgcolor)

        # the strip starts at the top of the week the first day falls in, so
        # that a weekday always keeps to its own row
        start = first - datetime.timedelta(
            days=(first.weekday() - first_weekday) % 7
        )
        num_weeks = (last - start).days // 7 + 1

        # break the weeks into bands narrow enough to fit
        if width is None:
            band_weeks = num_weeks
        else:
            band_weeks = (width - gutter) // day_width
        bands = [
            (week, min(week + band_weeks, num_weeks))
            for week in range(0, num_weeks, band_weeks)
        ]

        # caption the week each month begins in
        captions = []
        year, month = first.year, first.month
        while (year, month) <= (last.year, last.month):
            begins = max(datetime.date(year, month, 1), first)
            week = (begins - start).days // 7
            captions.append((week, _calendar.month_abbr[month], year))
            year, month = (year, month + 1) if month < 12 else (year + 1, 1)

        # draw each band
        band_width = gutter + band_weeks * day_width
        header = int(year_labels) + int(month_labels)
        band_chars = []
        for band, (from_week, to_week) in enumerate(bands):
            chars = CharArray.from_size(height=header + 7, width=band_width)

            # the weekday initials, down the gutter
            if weekday_labels:
                for row in range(7):
                    initial = _WEEKDAY_INITIALS[(first_weekday + row) % 7]
                    chars.codes[header + row, 0] = ord(initial)

            # the captions, each in the band its week landed in. A band names
            # a year over the first of its months in it, so that a band can be
            # read on its own rather than by looking back at the one above.
            named = set()
            for week, month_caption, year in captions:
                if not from_week <= week < to_week:
                    continue
                column = gutter + (week - from_week) * day_width
                if year_labels and year not in named:
                    named.add(year)
                    _write_caption(
                        chars, 0, column, f"{year:4d}", band_width,
                    )
                if month_labels:
                    _write_caption(
                        chars,
                        int(year_labels),
                        column,
                        month_caption,
                        band_width,
                    )

            # the days
            for week in range(from_week, to_week):
                for weekday in range(7):
                    date = start + datetime.timedelta(days=7 * week + weekday)
                    if date not in colors:
                        continue
                    _paint_day(
                        chars=chars,
                        row=header + weekday,
                        column=gutter + (week - from_week) * day_width,
                        day_width=day_width,
                        color=colors[date],
                        bgcolor=bg,
                    )

            # a blank row between the bands, but not after the last
            band_chars.append(chars.pad(below=band != len(bands) - 1))

        super().__init__(CharArray.map(
            lambda arrays: np.concatenate(arrays, axis=0),
            band_chars,
        ))
        self.vrange = dated.vrange
        self.colormap = colormap
        self.daterange = (first, last)
        self.num_days = len(colors)
        self.num_weeks = num_weeks

    def __repr__(self):
        first, last = self.daterange
        vmin, vmax = self.vrange
        return (
            f"weeks(height={self.height}, width={self.width}, "
            f"data=<{self.num_days} days over {self.num_weeks} weeks from "
            f"{first} to {last} on [{vmin:.2f},{vmax:.2f}]>)"
        )


# # # 
# FURNISHING CLASSES


class text(plot):
    """
    A plot object containing one or more lines of text.

    This class wraps a string in the plot interface, allowing it to be
    composed with other plot objects. It handles multi-line strings by
    splitting them at newline characters.

    Inputs:

    * text : str.
        The text to be displayed. Newline characters will create separate lines
        in the plot.
    * fgcolor : optional ColorLike.
        The foreground color of the text. Defaults to the terminal's default
        foreground color.
    * bgcolor : optional ColorLike.
        The background color for the text. Defaults to a transparent
        background.

    Carriage returns and newlines separate lines. Other C0 and C1 control
    characters are rejected, including the escapes used for raw ANSI
    formatting: styling has to be part of the plot so that composition and
    rendering know its size.
    
    TODO:

    * Allow alignment and resizing.
    * Account for non-printable and wide characters.
    """
    def __init__(
        self,
        text: str,
        fgcolor: ColorLike | None = None,
        bgcolor: ColorLike | None = None,
    ):
        _validate_text(text, allow_line_breaks=True)
        lines = text.splitlines()
        height = len(lines)
        width = max(len(line) for line in lines)
        
        # blank canvas
        chars = CharArray.from_size(
            height=height,
            width=width,
            fgcolor=fgcolor,
            bgcolor=bgcolor,
        )

        # paint the text
        for i, line in enumerate(lines):
            chars.codes[i, :len(line)] = ords(line)
        
        # initialise
        super().__init__(chars=chars)
        if height > 1 or width > 8:
            self.preview = lines[0][:5] + "..."
        else:
            self.preview = lines[0][:8]

    def __repr__(self):
        return (
            f"text(height={self.height}, width={self.width}, "
            f"text={self.preview!r})"
        )


class border(plot):
    """
    Add a border around a plot using box-drawing characters.

    Inputs:

    * plot : plot.
        The plot object to be enclosed by the border.
    * title: str.
        An optional title for the box. Placed centrally along the top row of
        the box. Truncated to fit.
    * style : BoxStyle (default: BoxStyle.ROUND).
        The style of the border. Predefined styles are available in `BoxStyle`.
    * color : optional ColorLike.
        The color of the border characters. Defaults to the terminal's
        default foreground color.
    """
    def __init__(
        self,
        plot: plot,
        title: str = "",
        style: BoxStyle = BoxStyle.ROUND,
        color: ColorLike | None = None,
    ):
        chars = unicode_box(
            chars=plot.chars,
            title=title,
            style=style,
            fgcolor=color,
        )
        super().__init__(
            chars,
        )
        self.style = style[2]
        self.plot = plot
    
    def __repr__(self):
        return f"border(style={self.style!r}, plot={self.plot!r})"


type Side = Literal["crop", "pad", "rule", "label"]
"""
What is drawn along one side of an `axes`, in increasing order of what it
costs and what it shows.

* `"crop"`: nothing at all, taking no space.
* `"pad"`: one blank cell, holding the space open.
* `"rule"`: one cell, holding a line.
* `"label"`: the line, ticks at its two ends, and a row or column outside it
  carrying the limits of the coordinate and the name of the axis.
"""


def _infer_side_pair(
    present: bool,
    primary: Side | None,
    secondary: Side | None,
    frame: Side,
) -> tuple[Side, Side]:
    """
    Fill in the modes for the two sides facing each other across one axis.

    The axis is labelled on its primary side---south for x, west for y---
    unless it is labelled on the other side instead, or the coordinate is
    missing, or the caller asked for something else. Anything not labelled
    falls back to the frame's own mode.
    """
    if not present:
        return (primary or frame, secondary or frame)
    labelled = "label" in (primary, secondary)
    return (primary or (frame if labelled else "label"), secondary or frame)


def _end_labels(
    lo: str,
    hi: str,
    span: tuple[int, int],
    room: int,
) -> tuple[tuple[str, int], tuple[str, int]]:
    """
    Where the two limits of one axis go along a row or column of labels.

    They sit at the ends of the span the ticks mark if they fit there, spread
    across everything available if they do not, and are replaced by hashes if
    even that is too narrow---so that a plot is never widened to fit its
    labels, and never shows a number that has had digits taken off it.

    Inputs:

    * lo, hi : str.
        The labels for the low and the high end of the axis.
    * span : (int, int).
        Where the ticks are: the first cell and the number of cells between
        them inclusive.
    * room : int.
        The total number of cells available, the span included.

    Returns:

    * lo, hi : (str, int).
        Each label and the offset it starts at.
    """
    start, width = span
    if len(lo) + len(hi) <= width:
        return (lo, start), (hi, start + width - len(hi))
    if len(lo) + len(hi) <= room:
        return (lo, 0), (hi, room - len(hi))
    half = room // 2
    lo = lo if len(lo) <= half else "#" * half
    hi = hi if len(hi) <= room - half else "#" * (room - half)
    return (lo, 0), (hi, room - len(hi))


class axes(plot):
    """
    Rule and label the sides of a plot that carries coordinates.

    Each of the four sides is drawn independently, so that a plot can be given
    a full frame with labels below and to its left, a single labelled rule
    along one side, or anything in between. The characters where the rules
    meet, and the ticks that reach out towards the labels, follow from which
    sides are drawn.

    Inputs:

    * plot : plot.
        The plot to draw the axes around. Must carry a window.
    * north, east, south, west : optional Side.
        What to draw along each side: `"crop"`, `"pad"`, `"rule"` or
        `"label"`. A side may only be labelled if the plot carries the
        matching coordinate: north and south need an x range, east and west a
        y range.

        Left unspecified, each axis the plot carries is labelled once---below
        it and to its left---and the remaining sides are ruled if the plot
        carries both coordinates, or dropped if it carries only one, so that a
        colorbar is labelled along one side and left alone on the others.
        Asking for a label on one side of an axis rules the opposite side
        rather than labelling it twice.
    * title: optional str.
        Placed centrally along the top. Written into the north side if that
        side is blank or ruled, and given a row of its own above everything
        otherwise. Truncated to fit.
    * xlabel: optional str.
        The name of the x axis, written along each labelled horizontal side,
        between the limits and truncated to fit between them.
    * ylabel: optional str.
        The name of the y axis, written vertically along each labelled
        vertical side. Truncated to fit.
    * xfmt: str (default "{x:.1f}").
        Format string for x labels. Should have one keyword argument with the
        key 'x'.
    * yfmt: str (default "{y:.1f}").
        Format string for y labels. Should have one keyword argument with the
        key 'y'.
    * ypad: int (default 1).
        How many columns between a vertical axis and its name.
    * style : LineStyle (default: LineStyle.LIGHT).
        The weight of the rules.
    * color : optional ColorLike.
        The color of the rules and the labels. Defaults to 50% gray. Set to
        `None` to use the foreground color.

    A limit that will not fit in the space its side has is replaced by hashes,
    as a spreadsheet does, rather than shortened into a different number or
    allowed to widen the plot.
    """
    def __init__(
        self,
        plot: plot,
        north: Side | None = None,
        east: Side | None = None,
        south: Side | None = None,
        west: Side | None = None,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        xfmt: str = "{x:.1f}",
        yfmt: str = "{y:.1f}",
        ypad: int = 1,
        style: LineStyle = LineStyle.LIGHT,
        color: ColorLike | None = (0.5, 0.5, 0.5),
    ):
        w = plot.window
        if w is None or (w.xrange is None and w.yrange is None):
            raise ValueError(
                f"{type(plot).__name__} has no coordinates to draw axes for; "
                "for an unlabelled frame, use border"
            )
        frame: Side = (
            "rule" if w.xrange is not None and w.yrange is not None else "crop"
        )
        south, north = _infer_side_pair(w.xrange is not None, south, north, frame)
        west, east = _infer_side_pair(w.yrange is not None, west, east, frame)
        for name, side, range in (
            ("north", north, w.xrange), ("south", south, w.xrange),
            ("east", east, w.yrange), ("west", west, w.yrange),
        ):
            if side == "label" and range is None:
                raise ValueError(
                    f"cannot label the {name} side of a plot with no "
                    f"{'x' if name in ('north', 'south') else 'y'} coordinate"
                )

        # the numbers each labelled side carries
        xlo, xhi = (
            (xfmt.format(x=w.xrange[0]), xfmt.format(x=w.xrange[1]))
            if w.xrange is not None else ("", "")
        )
        ylo, yhi = (
            (yfmt.format(y=w.yrange[0]), yfmt.format(y=w.yrange[1]))
            if w.yrange is not None else ("", "")
        )
        gutter = max(len(ylo), len(yhi))
        if ylabel:
            gutter = max(gutter, ypad + 1)
        left_gutter = gutter if west == "label" else 0
        right_gutter = gutter if east == "label" else 0
        title_row = bool(title) and north in ("crop", "label")
        above = int(title_row) + int(north == "label")

        # rule the sides, and reserve room outside them for the labels
        sides = (north, east, south, west)
        cells = (
            north != "crop", east != "crop", south != "crop", west != "crop",
        )
        rules = (
            north in ("rule", "label"), east in ("rule", "label"),
            south in ("rule", "label"), west in ("rule", "label"),
        )
        ticks = (
            north == "label", east == "label",
            south == "label", west == "label",
        )
        chars = unicode_frame(
            chars=plot.chars,
            style=style,
            cells=cells,
            rules=rules,
            ticks=ticks,
            title="" if title_row else title,
            fgcolor=color,
        ).pad(
            above=above,
            below=int(south == "label"),
            left=left_gutter,
            right=right_gutter,
            fgcolor=color,
        )

        # where the ticks are: the ends of each rule, in the padded array
        first_row = above + cells[0] - rules[0]
        last_row = above + cells[0] + plot.height + rules[2] - 1
        first_col = left_gutter + cells[3] - rules[3]
        last_col = left_gutter + cells[3] + plot.width + rules[1] - 1

        # the limits of the vertical axis, in the gutters beside its ticks,
        # hashed out where the two ticks land on one row
        crowded = first_row == last_row
        yhi_, ylo_ = ("#" * gutter, "") if crowded else (yhi, ylo)
        if west == "label":
            chars.codes[first_row, left_gutter-len(yhi_):left_gutter] = ords(yhi_)
            chars.codes[last_row, left_gutter-len(ylo_):left_gutter] = ords(ylo_)
        if east == "label":
            edge = chars.width - right_gutter
            chars.codes[first_row, edge:edge+len(yhi_)] = ords(yhi_)
            chars.codes[last_row, edge:edge+len(ylo_)] = ords(ylo_)

        # and its name, down whatever the limits leave between them
        room = last_row - first_row - 1
        if ylabel and room > 0:
            name = ylabel[:room].center(room)
            rows = slice(first_row + 1, last_row)
            if west == "label":
                chars.codes[rows, left_gutter-1-ypad] = ords(name)
            if east == "label":
                chars.codes[rows, chars.width-right_gutter+ypad] = ords(name)

        # the limits of the horizontal axis, in the rows outside its ticks
        span = (first_col, last_col - first_col + 1)
        for side, row in ((north, int(title_row)), (south, chars.height - 1)):
            if side != "label":
                continue
            (lo, lo_col), (hi, hi_col) = _end_labels(xlo, xhi, span, chars.width)
            chars.codes[row, lo_col:lo_col+len(lo)] = ords(lo)
            chars.codes[row, hi_col:hi_col+len(hi)] = ords(hi)
            gap = hi_col - lo_col - len(lo)
            if xlabel and gap > 0:
                name = xlabel[:gap].center(gap)
                chars.codes[row, lo_col+len(lo):hi_col] = ords(name)

        # the title, when there was no side to write it into
        if title_row:
            name = title[:chars.width]
            start = chars.width // 2 - len(name) // 2
            chars.codes[0, start:start+len(name)] = ords(name)

        super().__init__(chars)
        self.sides = sides
        self.plot = plot

    def __repr__(self):
        n, e, s, w = self.sides
        return f"axes(n={n}, e={e}, s={s}, w={w}, plot={self.plot!r})"


type Direction = Literal["up", "down", "left", "right"]
"""
Which way along the screen a `colorbar` runs: the axis it lies along, and the
end of that axis its interval finishes at.

* `"up"`, `"down"`: a vertical bar, two pixels per cell.
* `"left"`, `"right"`: a horizontal bar, one pixel per cell.
"""


class colorbar(heatmap):
    """
    A gradient standing for the mapping from values onto colours.

    The bar is a strip one coordinate wide, so `axes` labels the one side that
    means anything and leaves the other three alone, and a `border` boxes it
    without claiming it has two dimensions.

    Inputs:

    * source : plot | (number, number).
        The colour scale to draw. A plot that carries one---a `heatmap`, or
        anything built on one, or a `calendar` or `weeks`---contributes both
        its interval and its colormap, so that the bar cannot disagree with
        the picture it describes. An interval of values on its own works too,
        for a scale assembled by hand.
    * colormap : optional ColorMap.
        Maps each position along the bar onto its colour, overriding the one a
        plot brought with it. By default the bar runs black to white.
    * direction : Direction (default: "up").
        Which way along the screen the values increase, naming both the axis
        the bar runs along and the sense it runs in. The first limit of the
        interval sits at the end the direction points away from, and the
        second at the end it points towards.
    * length : int (default: 12).
        How many character cells the bar covers along the scale.
    * thickness : int (default: 1).
        How many character cells the bar covers across the scale.

    A vertical bar has twice the gradient resolution of a horizontal one of
    the same length, since a character cell holds two half-block pixels
    vertically and one horizontally.

    ```
    heat = mp.heatmap(values, colormap=mp.viridis)
    mp.axes(heat, title="field") + mp.axes(mp.colorbar(heat), east="label")
    ```
    """
    def __init__(
        self,
        source: plot | tuple[number, number],
        colormap: ColorMap | None = None,
        direction: Direction = "up",
        length: int = 12,
        thickness: int = 1,
    ):
        if isinstance(source, plot):
            vrange = getattr(source, "vrange", None)
            if vrange is None or not hasattr(source, "colormap"):
                raise ValueError(
                    f"{type(source).__name__} carries no colour scale for a "
                    "colorbar to draw; pass an interval of values instead"
                )
            if colormap is None:
                colormap = source.colormap
        else:
            vrange = source
        if direction not in ("up", "down", "left", "right"):
            raise ValueError(
                f"direction must be up, down, left or right, not {direction!r}"
            )
        if length < 1 or thickness < 1:
            raise ValueError(
                "a colorbar must be at least one cell in each direction, not "
                f"{length} by {thickness}"
            )

        # the ramp, in screen order: the top row or the leftmost column first.
        # A vertical cell holds two pixels and a horizontal one holds one, so
        # the length counts cells and the pixels follow from the direction.
        first, second = vrange
        vertical = direction in ("up", "down")
        steps = 2 * length if vertical else length
        ramp = (
            np.linspace(second, first, num=steps)
            if direction in ("up", "left")
            else np.linspace(first, second, num=steps)
        )
        values = (
            ramp[:, None].repeat(thickness, axis=1)
            if vertical
            else ramp[None, :].repeat(2 * thickness, axis=0)
        )

        # the coordinate, from the low end of the screen axis to the high end
        span = (
            (second, first)
            if direction in ("down", "left")
            else (first, second)
        )
        super().__init__(
            values=values,
            colormap=colormap,
            vrange=vrange,
            xrange=None if vertical else span,
            yrange=span if vertical else None,
        )
        self.direction = direction

    def __repr__(self):
        return f"colorbar({self.direction}, {self.window!r})"


# # #
# ARRANGEMENT CLASSES


class blank(plot):
    """
    Creates a rectangular plot composed entirely of blank space.

    Useful for adding padding or aligning items in a complex layout.

    Inputs:

    * height : optional int.
      The height of the blank area in character rows. Default 1.
    * width : optional int.
      The width of the blank area in character columns. Default 1.
    """
    def __init__(
        self,
        height: int = 1,
        width: int = 1,
    ):
        chars = CharArray.from_size(height=height, width=width)
        super().__init__(chars)

    def __repr__(self):
        return f"blank(height={self.height}, width={self.width})"


class hstack(plot):
    """
    Horizontally arrange one or more plots side-by-side.

    If the plots have different heights, the shorter plots will be padded with
    blank space at the bottom to match the height of the tallest plot.

    Inputs:

    * *plots : plot.
        A sequence of plot objects to be horizontally stacked.
    """
    def __init__(
        self,
        *plots: plot,
    ):
        height = max(p.height for p in plots)
        padded_chars = [p.chars.pad(below=height-p.height) for p in plots]
        catted_chars = CharArray.map(
            lambda xs: np.concatenate(xs, axis=1), 
            padded_chars,
        )
        super().__init__(catted_chars)
        self.plots = plots

    def __repr__(self):
        return (
            f"hstack(height={self.height}, width={self.width}, "
            f"plots={self.plots!r})"
        )


class vstack(plot):
    """
    Vertically arrange one or more plots, one above the other.

    If the plots have different widths, the narrower plots will be padded with
    blank space on the right to match the width of the widest plot.

    Inputs:

    * *plots : plot.
        A sequence of plot objects to be vertically stacked.
    """
    def __init__(
        self,
        *plots: plot,
    ):
        width = max(p.width for p in plots)
        padded_chars = [p.chars.pad(right=width-p.width) for p in plots]
        catted_chars = CharArray.map(
            lambda xs: np.concatenate(xs, axis=0),
            padded_chars,
        )
        super().__init__(catted_chars)
        self.plots = plots

    def __repr__(self):
        return (
            f"vstack(height={self.height}, width={self.width}, "
            f"plots={self.plots!r})"
        )


class dstack(plot):
    """
    Overlay one or more plots on top of each other.

    The plots are layered in the order they are given, with later plots in the
    sequence drawn on top of earlier ones. The final size of the plot is
    determined by the maximum width and height among all input plots. Non-blank
    characters from upper layers will obscure characters from lower layers.

    Inputs:

    * *plots : plot.
        A sequence of plot objects to be overlaid.
    """
    def __init__(
        self,
        *plots: plot,
    ):
        height = max(p.height for p in plots)
        width = max(p.width for p in plots)
        # build the array front to back one plot at a time
        stacked_chars = CharArray.from_size(height=height, width=width)
        for p in plots:
            h = p.height
            w = p.width
            # keep new nonblank characters and foreground
            mask = p.chars.isnonblank()
            stacked_chars.codes[:h, :w][mask] = p.chars.codes[mask]
            stacked_chars.fg[:h, :w][mask] = p.chars.fg[mask]
            stacked_chars.fg_rgb[:h, :w][mask] = p.chars.fg_rgb[mask]
            # keep new background, or old background if no new background
            bgmask = mask & p.chars.bg
            stacked_chars.bg[:h, :w][bgmask] = True
            stacked_chars.bg_rgb[:h, :w][bgmask] = p.chars.bg_rgb[bgmask]

        super().__init__(stacked_chars)
        self.plots = plots
        
    def __repr__(self):
        return (
            f"dstack(height={self.height}, width={self.width}, "
            f"plots={self.plots!r})"
        )
        

class dstack2(dstack):
    """
    Overlay one or more plots on top of each other.

    The plots are layered in the order they are given, with later plots in the
    sequence drawn on top of earlier ones. The final size of the plot is
    determined by the maximum width and height among all input plots. Non-blank
    characters from upper layers will obscure characters from lower layers.

    Unlike dstack, every plot must carry a coordinate on both axes, and they
    must all share one window: the same intervals covered in the same number
    of character cells. Two plots covering the same intervals in different
    numbers of cells put the same coordinate in different places, and a
    rendered plot cannot be resampled to fix that, so it is refused.

    Inputs:

    * *plots : plot.
        A sequence of plot objects to be overlaid. At least one, all sharing
        one window.
    """
    def __init__(
        self,
        *plots: plot,
    ):
        if not plots:
            raise ValueError("no plots to overlay")
        shared = plots[0].window
        if shared is None or shared.xrange is None or shared.yrange is None:
            raise ValueError(
                f"{type(plots[0]).__name__} has no coordinates to be "
                "overlaid in"
            )
        for p in plots[1:]:
            if p.window != shared:
                raise ValueError(
                    f"cannot overlay {p.window!r} on {shared!r}"
                )

        super().__init__(*plots)
        self.window = shared

    def __repr__(self):
        return f"dstack2({self.window!r}, plots={self.plots!r})"


class wrap(plot):
    """
    Arrange a sequence of plots into a grid.

    The plots are arranged from left to right, wrapping to a new line when
    the specified number of columns is reached. All cells in the grid are
    padded to the size of the largest plot in the sequence.

    Inputs:

    * *plots : plot.
        A sequence of plot objects to be arranged in a grid.
    * cols : optional int.
        The number of columns in the grid. If not provided, it is automatically
        determined based on the terminal width and the width of the largest
        plot.
    * transpose: optional bool (default False).
        If False (default), the plots are arranged in reading order, from left
        to right and then from top to bottom. If True, the plots are arranged
        in column order, from top to bottom and then from left to right.
    """
    def __init__(
        self,
        *plots: plot,
        cols: int | None = None,
        transpose: bool = False,
    ):
        # determine and standardise cell size
        cell_height = max(p.height for p in plots)
        cell_width = max(p.width for p in plots)
        padded_chars = [
            p.chars.pad(
                below=cell_height - p.height,
                right=cell_width - p.width,
            ) for p in plots
        ]
        blank_cell = CharArray.from_size(
            height=cell_height,
            width=cell_width,
        )

        # determine grid size and initialise grid
        if cols is None:
            terminal_width = shutil.get_terminal_size(fallback=(80, 24)).columns
            cols = max(1, terminal_width // cell_width)
        n = len(padded_chars)
        full_rows, spare = divmod(n, cols)
        rows = full_rows + bool(spare)
        grid = [[blank_cell for _ in range(cols)] for _ in range(rows)]

        # populate grid
        for i, p in enumerate(padded_chars):
            if transpose:
                c, r = divmod(i, rows)
            else:
                r, c = divmod(i, cols)
            grid[r][c] = p

        # combine into new char array
        blocked_chars = CharArray(
            codes=np.block([[c.codes for c in row] for row in grid]),
            fg=np.block([[c.fg for c in row] for row in grid]),
            fg_rgb=np.block([[[c.fg_rgb] for c in row] for row in grid]),
            bg=np.block([[c.bg for c in row] for row in grid]),
            bg_rgb=np.block([[[c.bg_rgb] for c in row] for row in grid]),
        )
        super().__init__(blocked_chars)
        self.plots = plots

    def __repr__(self):
        return (
            f"wrap(height={self.height}, width={self.width}, "
            f"plots={self.plots!r})"
        )


class center(plot):
    """
    Pad a plot with blank space to center it within a larger area.

    If the specified `height` or `width` is smaller than the plot's dimensions,
    the larger dimension is used, effectively preventing the plot from being
    cropped.

    Inputs:

    * plot : plot.
        The plot object to be centered.
    * height : optional int.
        The target height of the new padded plot. If not provided, it defaults
        to the original plot's height (no vertical padding).
    * width : optional int.
        The target width of the new padded plot. If not provided, it defaults
        to the original plot's width (no horizontal padding).
    """
    def __init__(
        self,
        plot: plot,
        height: int | None = None,
        width: int | None = None,
    ):
        # decide padding amounts
        # vertical
        if height is None or height <= plot.height:
            above = 0
            below = 0
        else:
            hdiff = height - plot.height
            above = hdiff // 2
            below = above + (hdiff % 2)
        # horizontal
        if width is None or width <= plot.width:
            left = 0
            right = 0
        else:
            wdiff = width - plot.width
            left = wdiff // 2
            right = left + (wdiff % 2)
        # pad the character array
        padded_chars = plot.chars.pad(
            above=above,
            below=below,
            left=left,
            right=right,
        )
        super().__init__(padded_chars)
        self.plot = plot
    
    def __repr__(self):
        return (
            f"center(height={self.height}, width={self.width}, "
            f"plot={self.plot!r})"
        )
