"""
Plots that colour a grid of half-character pixels: images, heatmaps, the
plots that sample a function over a rectangle, and two-dimensional
histograms.

* `image`
* `heatmap`
* `function2`
* `vfunction2`
* `cfunction2`
* `histogram2`
"""
from __future__ import annotations

import einops
import numpy as np

from typing import Callable
from numpy.typing import ArrayLike
from matthewplotlib.colormaps import ColorMap, chroma, domain
from matthewplotlib.colors import parse_colors
from matthewplotlib.data import number
from matthewplotlib.scales import _value_range, _normalise
from matthewplotlib.window import window
from matthewplotlib.core import unicode_image
from matthewplotlib.plots.base import plot


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
    given a colorbar over the same numbers.
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

    def __repr__(self):
        return f"image({self.window!r})"


class heatmap(image):
    """
    Render a grid of values, colouring each by where it falls in an interval.

    The values are normalised onto the range 0.0 to 1.0 and handed to a
    colormap, so that the caller does not scale them by hand and the colours
    mean the same thing from one plot to the next. The interval is kept as
    `vrange`, so that a `colorbar` can be drawn over the same numbers.

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
    * xrange : (number, number).
        Lower and upper bounds on the x values to pass into the function.
    * yrange : (number, number).
        Lower and upper bounds on the y values to pass into the function.
    * width : int.
        The number of character columns in the plot. This will also become the
        number of grid squares along the x axis.
    * height : int.
        The number of character rows in the plot. This will also be half of the
        number of grid squares, since the result is an image plot with two
        half-character-pixels per row.
    * vrange : optional (number, number).
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
        xrange: tuple[number, number],
        yrange: tuple[number, number],
        width: int,
        height: int,
        vrange: tuple[number, number] | None = None,
        colormap: ColorMap | None = None,
        endpoints: bool = False,
    ):
        # the coordinates each grid square stands for, top row first
        w = window(xrange=xrange, yrange=yrange, width=width, height=height)
        XY = w.sample_points(endpoints=endpoints)

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


class vfunction2(image):
    """
    Colour field representing a 2d vector field over a rectangle.

    Every pixel is coloured by the vector the function returns there: the
    direction becomes the hue and the magnitude becomes the brightness. Unlike
    a field of arrows this shows a vector at every pixel, so the structure of
    the field---its sources, sinks, saddles and the channels between them---is
    visible at whatever resolution the terminal allows.

    Inputs:

    * F : float[batch, 2] -> float[batch, 2].
        The (vectorised) field to plot. The input is a batch of (x, y)
        positions. The output should be the batch of (u, v) vectors at those
        positions.
    * xrange : (number, number).
        Lower and upper bounds on the x values to pass into the function.
    * yrange : (number, number).
        Lower and upper bounds on the y values to pass into the function.
    * width : int.
        The number of character columns in the plot. This will also become the
        number of grid squares along the x axis.
    * height : int.
        The number of character rows in the plot. This will also be half of the
        number of grid squares, since the result is an image plot with two
        half-character-pixels per row.
    * vrange : optional (number, number).
        Expected lower and upper bounds on the *magnitude* of the vectors, used
        to scale them into the unit disc for the colormap. By default the lower
        bound is zero and the upper bound is the largest magnitude over the
        grid, so that the fastest part of the field is at full brightness.
        Magnitudes outside these bounds saturate at the nearest end.

        The lower bound is zero by default rather than the smallest magnitude,
        because a vector field's zeros are where its structure is, and they
        should come out black.
    * colormap : optional vector colormap (e.g. mp.chroma).
        Applied to the scaled field. Defaults to `mp.chroma`. A custom colormap
        receives the scaled `float[h, w, 2]` field and must return an RGB image
        of shape `[h, w, 3]`.
    * endpoints : bool (default: False).
        By default, the grid squares tile the ranges exactly and each one shows
        the value of the field at its own centre.

        If true, the field is instead sampled at points spread from one end of
        each range to the other, so that the four corner squares show the four
        corner combinations of xrange and yrange. The squares then reach half a
        square beyond the ranges, which the axes still report as the limits.
    """
    def __init__(
        self,
        F: Callable[[np.ndarray], np.ndarray],
        xrange: tuple[number, number],
        yrange: tuple[number, number],
        width: int,
        height: int,
        vrange: tuple[number, number] | None = None,
        colormap: ColorMap | None = None,
        endpoints: bool = False,
    ):
        # the coordinates each grid square stands for, top row first
        w = window(xrange=xrange, yrange=yrange, width=width, height=height)
        XY = w.sample_points(endpoints=endpoints)

        # sample the field
        UV = np.asarray(F(XY), dtype=float)
        if UV.shape != XY.shape:
            raise ValueError(
                f"expected the field to return one (u, v) vector per point, "
                f"an array of shape {XY.shape}, not {UV.shape}"
            )
        vgrid = einops.rearrange(
            UV,
            '(h w) uv -> h w uv',
            h=2*height,
            w=width,
        )

        # scale the magnitudes into [0, 1], leaving the directions alone
        magnitude = np.hypot(vgrid[..., 0], vgrid[..., 1])
        vrange = _value_range(
            vrange,
            magnitude,
            "vfunction2",
            from_zero=True,
        )
        scaled = _normalise(magnitude, vrange)
        with np.errstate(divide="ignore", invalid="ignore"):
            direction = vgrid / magnitude[..., np.newaxis]
        direction = np.where(np.isfinite(direction), direction, 0.)

        # create the image plot itself
        super().__init__(
            im=direction * scaled[..., np.newaxis],
            colormap=chroma if colormap is None else colormap,
            xrange=xrange,
            yrange=yrange,
        )
        self.name = getattr(F, '__name__', '?')
        self.vrange = vrange

    def __repr__(self):
        return f"vfunction2(f={self.name}, {self.window!r})"


class cfunction2(image):
    """
    Domain colouring of a complex function over a rectangle of the plane.

    Every pixel is coloured by the value the function takes there: the phase
    becomes the hue and the modulus becomes the lightness, so a zero of the
    function shows up as a black point, a pole as a white one, and the order of
    either can be counted off the number of times the colour wheel turns around
    it.

    Inputs:

    * F : complex[batch] -> complex[batch].
        The (vectorised) function to plot. The input is a batch of points of
        the complex plane. The output should be the batch of values there.
    * xrange : (number, number).
        Lower and upper bounds on the real part of the input.
    * yrange : (number, number).
        Lower and upper bounds on the imaginary part of the input.
    * width : int.
        The number of character columns in the plot. This will also become the
        number of grid squares along the real axis.
    * height : int.
        The number of character rows in the plot. This will also be half of the
        number of grid squares, since the result is an image plot with two
        half-character-pixels per row.
    * colormap : optional vector colormap (e.g. mp.domain).
        Applied to the values. Defaults to `mp.domain`. There is no range to
        configure, because a domain colouring puts the modulus on an absolute
        scale---the colormap owns it. A custom colormap receives the
        `complex[h, w]` values and must return an RGB image of shape
        `[h, w, 3]`.
    * endpoints : bool (default: False).
        By default, the grid squares tile the ranges exactly and each one shows
        the value of the function at its own centre.

        If true, the function is instead sampled at points spread from one end
        of each range to the other, so that the four corner squares show the
        four corner combinations of xrange and yrange. The squares then reach
        half a square beyond the ranges, which the axes still report as the
        limits.

    Sampling the function at the centre of each square is the default here for
    a further reason: a function with a pole at a round number, such as `1/z`
    at the origin, is then never evaluated exactly on it.
    """
    def __init__(
        self,
        F: Callable[[np.ndarray], np.ndarray],
        xrange: tuple[number, number],
        yrange: tuple[number, number],
        width: int,
        height: int,
        colormap: ColorMap | None = None,
        endpoints: bool = False,
    ):
        # the coordinates each grid square stands for, top row first
        w = window(xrange=xrange, yrange=yrange, width=width, height=height)
        XY = w.sample_points(endpoints=endpoints)

        # sample the function over the plane
        Z = np.asarray(F(XY[:, 0] + 1j * XY[:, 1]))
        if Z.shape != XY.shape[:1]:
            raise ValueError(
                f"expected the function to return one value per point, an "
                f"array of shape {XY.shape[:1]}, not {Z.shape}"
            )

        # create the image plot itself
        super().__init__(
            im=einops.rearrange(Z, '(h w) -> h w', h=2*height, w=width),
            colormap=domain if colormap is None else colormap,
            xrange=xrange,
            yrange=yrange,
        )
        self.name = getattr(F, '__name__', '?')

    def __repr__(self):
        return f"cfunction2(f={self.name}, {self.window!r})"


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
