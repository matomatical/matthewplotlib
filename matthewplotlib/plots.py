"""
A collection of building blots for plotting. There are lots of options---take a
look through this module. They are roughly grouped as follows.

Base class:

* `plot`: Every plot object inherits from this one. See this class for methods,
  properties, and shortcut operators available with every plot object.

Data plots:

* `scatter`
* `function`
* `scatter3`
* `image`
* `function2`
* `histogram2`
* `progress`
* `bars`
* `histogram`
* `columns`
* `vistogram`
* `hilbert`

Furnishing plots:

* `text`
* `border`

Arrangement plots:

* `blank`
* `hstack`
* `vstack`
* `dstack`
* `wrap`
* `center`
"""
import enum
import os
import numpy as np
import einops
import hilbert as _hilbert

from PIL import Image

from typing import Callable, Self, Sequence
from numpy.typing import ArrayLike, NDArray
from matthewplotlib.colormaps import ColorMap
from numbers import Number

from matthewplotlib.colors import ColorLike
from matthewplotlib.data import (
    number,
    Series,
    Series3,
    parse_range,
    parse_multiple_series,
    parse_multiple_series3,
    project3,
)
from matthewplotlib.core import (
    ColorLike,
    CharArray,
    BoxStyle,
    unicode_box,
    unicode_braille_array,
    unicode_bar,
    unicode_col,
    unicode_image,
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
        """
        return f"\x1b[{self.height}A\x1b[0J"


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
        image = Image.fromarray(image_data, mode='RGBA')
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
    
    
    def __add__(self: Self, other: Self) -> "hstack":
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


    def __truediv__(self: Self, other: Self) -> "vstack":
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


    def __or__(self: Self, other: Self) -> "vstack":
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


    def __matmul__(self: Self, other: Self) -> "dstack":
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
        xrange = parse_range(xs, xrange)
        yrange = parse_range(ys, yrange)
        
        # quantise 2d float coordinates to data grid
        counts, xedges, yedges = np.histogram2d(
            x=xs,
            y=ys,
            bins=(2*width, 4*height),
            range=(xrange, yrange),
        )

        # dots where counts > 0
        dots = counts > 0
        
        # determine colours for each position
        # 1: figure out which bins each point fell into
        ci = np.searchsorted(xedges, xs, side='right') - 1
        cj = np.searchsorted(yedges, ys, side='right') - 1
        ci[xs == xedges[-1]] = 2 * width - 1
        cj[ys == yedges[-1]] = 4 * height - 1
        valid = (ci >= 0) & (ci < 2*width) & (cj >= 0) & (cj < 4*height)
        # 2: average over colors in each cell
        total_colors = np.zeros((2*width, 4*height, 3))
        np.add.at(total_colors, (ci[valid], cj[valid]), cs[valid])
        total_colors[dots] /= counts[dots,None]
        # round to uint8
        dotc = total_colors.astype(np.uint8)
        dotw = counts
        
        # convert to Cartesian coordinates (+x right, -y down)
        dots = dots.T[::-1]
        if dotc is not None:
            dotc = dotc.transpose(1,0,2)[::-1]
            dotw = dotw.T[::-1]

        # render data grid as a grid of braille characters
        chars = unicode_braille_array(
            dots=dots,
            dotc=dotc,
            dotw=dotw,
        )
        super().__init__(chars)
        self.xrange = xrange
        self.yrange = yrange
        self.num_points = n

    def __repr__(self):
        return (
            f"scatter(height={self.height}, width={self.width}, "
            f"data=<{self.num_points} points on "
            f"[{self.xrange[0]:.2f},{self.xrange[1]:.2f}]x"
            f"[{self.yrange[0]:.2f},{self.yrange[1]:.2f}]>)"
        )


class scatter3(scatter):
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

    TODO:

    * Maybe allow configurable xyz ranges with clipping prior to projection?
    * Make sure this is not a subclass of scatter for the purposes of labelling
      axes as that would use projected coordinates.
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

        # create the scatter plot
        super().__init__(
            (xy[valid], cs[valid]),
            width=width,
            height=height,
            xrange=(-aspect_ratio, aspect_ratio),
            yrange=(-1.,1.),
        )
        
    def __repr__(self):
        return ("scatter3(TODO)")


class image(plot):
    """
    Render a small image or 2d array using a grid of unicode half-block
    characters.

    Represents an image by mapping pairs of vertically adjacent pixels to the
    foreground and background colors of a single character cell (this
    effectively doubles the vertical resolution in the terminal).

    Inputs:

    * im : float[h,w,3] | int[h,w,3] | float[h,w] | int[h,w].
        The image data. An array-like matching any of the following formats:
        * `float[h,w,3]`: A 2D array of RGB triples of floats in range [0,1].
        * `int[h,w,3]`: A 2D array of RGB triples of ints in range [0,255].
        * `float[h,w]`: A 2D array of scalars in the range [0,1]. If no
          colormap is provided, values are treated as greyscale (uniform
          colorisation). If a continuous colormap is provided, values are
          mapped to RGB values.
        * `int[h,w]`: A 2D array of ints. If no colormap is provided, values
          should be in the range [0,255], they are treated as greyscale
          (uniform colorisation). If a discrete colormap is provided, values
          should be in range as indices for the colormap, they will be mapped
          to RGB triples as such.
          
    * colormap : optional ColorMap.
        Function mapping (batches of) scalars to (batches of) RGB triples.
        Examples are provided by this library, such as:
        * continuous colormaps like `viridis : float[...] -> uint8[...,3]`, and
        * discrete colormaps like `pico8 : int[...] -> uint8[...,3]`.
        If `im` has no RGB dimension, it is transformed to a grid of RGB
        triples using one of these colormaps.

    TODO:

    * Offer normalisation?
    """
    def __init__(
        self,
        im: ArrayLike, # float[h,w] | float[h,w,rgb] | int[h,w] | int[h,w,rgb]
        colormap: ColorMap | None = None,
    ):
        # preprocessing: all inputs become float[h, w, rgb]
        im = np.asarray(im)
        if colormap is not None:
            # colormap provided: map image to u8 rgb
            im = colormap(im)
        if im.ndim == 2:
            # greyscale -> uniform colourisation
            im = einops.repeat(im, 'h w -> h w 3')
        if np.issubdtype(im.dtype, np.integer):
            # clip uint8
            im = np.clip(im, 0, 255).astype(np.uint8)
        if np.issubdtype(im.dtype, np.floating):
            # floats -> clipped uint8
            im = (255 * np.clip(im, 0., 1.)).astype(np.uint8)

        # construct the plot
        chars = unicode_image(im)

        # form a plot object
        super().__init__(chars)

    def __repr__(self):
        return f"image(height={self.height}, width={self.width})"


class function2(image):
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
    * zrange : optional (float, float).
        Expected lower and upper bounds on the f(x, y) values. Used for
        determining the bounds of the colour scale. By default, the minimum and
        maximum output over the grid are used.
    * colormap : optional colormap (e.g. mp.viridis).
        By default, the output will be in greyscale, with black corresponding
        to zrange[0] and white corresponding to zrange[1]. You can choose a
        different colormap (e.g. mp.reds, mp.viridis, etc.) here.
    * endpoints : bool (default: False).
        If true, endpoints are included from the linspaced inputs, and so the
        grid elements in each corner will represent the different combinations
        of xrange/yrange.
        
        If false (default), the endpoints are excluded, so the lower bounds are
        met but the upper bounds are not, meaning each grid square color shows
        the value of the function precisely at its lower left corner.
    """
    def __init__(
        self,
        F: Callable[[np.ndarray], np.ndarray],
        xrange: tuple[float, float],
        yrange: tuple[float, float],
        width: int,
        height: int,
        zrange: tuple[float, float] | None = None,
        colormap: ColorMap | None = None,
        endpoints: bool = False,
    ):
        # create a meshgrid with the required format and shape
        X, Y = np.meshgrid(
            np.linspace(*xrange, num=width, endpoint=endpoints),
            np.linspace(*yrange, num=2*height, endpoint=endpoints),
        ) # float[h, w] (x2)
        Y = Y[::-1] # correct Y direction for image plotting
        XY = einops.rearrange(np.dstack((X, Y)), 'h w xy -> (h w) xy')

        # sample the function
        Z = F(XY)

        # create the image array
        zgrid = einops.rearrange(Z, '(h w) -> h w', h=2*height, w=width)
        if zrange is None:
            zrange = (zgrid.min(), zgrid.max())
        if zrange[0] == zrange[1]:
            zgrid_norm = np.zeros_like(zgrid)
        else:
            zgrid_norm = (zgrid - zrange[0]) / (zrange[1] - zrange[0])

        # create the image plot itself
        super().__init__(
            im=zgrid_norm,
            colormap=colormap,
        )
        self.name = F.__name__
        self.xrange = xrange
        self.yrange = yrange
        self.zrange = zrange
        
    def __repr__(self):
        return ("function2("
                f"f={self.name}, "
                f"input=[{self.xrange[0]:.2f},{self.xrange[1]:.2f}]"
                f"x[{self.yrange[0]:.2f},{self.yrange[1]:.2f}]"
        ")")


class histogram2(image):
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
        
        # bin data
        hist, xbins, ybins = np.histogram2d(
            x=x,
            y=y,
            bins=(width, 2*height),
            range=(xrange, yrange),
            weights=weights,
            density=density,
        )

        # transform counts: scale and reorient
        if max_count is None:
            max_count = hist.max()
        hist /= max_count
        hist = hist.T[::-1]

        # construct the image
        super().__init__(
            im=hist,
            colormap=colormap,
        )
        self.xrange = xrange
        self.yrange = yrange
        self.xbins = xbins
        self.ybins = ybins
        self.num_points = len(x)
        
    def __repr__(self):
        return ("histogram2("
            f"height={self.height}, width={self.width}, "
            f"data=<{self.num_points} points on "
            f"[{self.xrange[0]:.2f},{self.xrange[1]:.2f}]x"
            f"[{self.yrange[0]:.2f},{self.yrange[1]:.2f}]>)"
        ")")


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
        all_chars.codes[0, :len(label)] = [ord(c) for c in label]
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
    * vrange : None | float | (float, float).
        Determine the scaling of the bars.
        * If omitted, the bars are scaled such that the bar(s) with the largest
          value occupy the whole width.
        * If a single number, then the bars are scaled so that bars with that
          value (or greater) would occupy the whole width.
        * If a pair of numbers, the bars are scaled so that bars with the first
          value (or less) would have zero width and bars with the second value
          (or greater) would occupy the whole width.
    * color : optional ColorLike.
        The color of the filled portion of the bars. Defaults to the terminal's
        default foreground color.

    TODO:

    * Make it possible to draw bars to the left for values below 0.
    * Make it possible to align all bars to the right rather than left.
    * Allow each bar to have its own colour.
    """
    def __init__(
        self,
        values: ArrayLike, # numeric[n]
        width: int = 30,
        bar_height: int = 1,
        bar_spacing: int = 0,
        vrange: None | number | tuple[number, number] = None,
        color: ColorLike | None = None,
    ):
        # standardise inputs
        values = np.asarray(values)
        vmin: number
        vmax: number
        if vrange is None:
            vmin = 0.0
            vmax = values.max()
        elif isinstance(vrange, Number):
            vmin = 0.0
            vmax = vrange
        elif isinstance(vrange, tuple):
            vmin, vmax = vrange
        num_bars = len(values)

        # compute the bar widths
        norm_values = (values - vmin) / (vmax - vmin + 1e-15)
        
        # construct the bars
        bars_chars = [
            unicode_bar(
                proportion=v,
                width=width,
                height=bar_height,
                fgcolor=color,
                bgcolor=None,
            ).pad(
                below=bar_spacing * (i==num_bars-1),
            )
            for i, v in enumerate(norm_values)
        ]
        all_chars = CharArray.map(
            lambda xs: np.concatenate(xs, axis=1), 
            bars_chars,
        )
        super().__init__(chars=all_chars)
        self.vmin = vmin
        self.vmax = vmax
        self.num_bars = num_bars

    def __repr__(self):
        return (
            f"bars(height={self.height}, width={self.width}, "
            f"values=<{self.num_bars} bars on "
            f"[{self.vmin:.2f},{self.vmax:.2f}]>)"
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
        
        # bin data
        hist, bins_ = np.histogram(
            a=data,
            bins=bins,
            range=xrange,
            weights=weights,
            density=density,
        )

        # build bar chart
        if max_count is None:
            max_count = hist.max()
        super().__init__(
            values=hist,
            width=width,
            bar_height=1,
            bar_spacing=0,
            vrange=max_count,
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
    * vrange : None | number | (number, number).
        Determine the scaling of the columns.
        * If omitted, the columns are scaled such that the columns(s) with the
          largest value occupy the whole width.
        * If a single number, then the columns are scaled so that columns with
          that value (or greater) would occupy the whole width.
        * If a pair of numbers, the columns are scaled so that columns with the
          first value (or less) would have zero width and columns with the
          second value (or greater) would occupy the whole width.
    * color : optional ColorLike.
        The color of the filled portion of the columns. Defaults to the
        terminal's default foreground color.

    TODO:

    * Make it possible to draw columns downward for values below 0.
    * Make it possible to align all columns to the top rather than bottom.
    * Allow each column to have its own color.
    """
    def __init__(
        self,
        values: ArrayLike, # number[n], actually int[n] will also work
        height: int = 10,
        column_width: int = 1,
        column_spacing: int = 0,
        vrange: None | number | tuple[number, number] = None,
        color: ColorLike | None = None,
    ):
        # standardise inputs
        values = np.asarray(values)
        vmin: number
        vmax: number
        if vrange is None:
            vmin = 0.0
            vmax = values.max()
        elif isinstance(vrange, Number):
            vmin = 0.0
            vmax = vrange
        elif isinstance(vrange, tuple):
            vmin, vmax = vrange
        num_cols = len(values)

        # compute the column heights
        norm_values = (values - vmin) / (vmax - vmin + 1e-15)
        
        # construct the columns
        cols_chars = [
            unicode_col(
                proportion=v,
                height=height,
                width=column_width,
                fgcolor=color,
                bgcolor=None,
            ).pad(
                right=column_spacing * (i==num_cols-1),
            )
            for i, v in enumerate(norm_values)
        ]
        all_chars = CharArray.map(
            lambda xs: np.concatenate(xs, axis=0), 
            cols_chars,
        )
        super().__init__(chars=all_chars)
        self.vmin = vmin
        self.vmax = vmax
        self.num_cols = num_cols

    def __repr__(self):
        return (
            f"columns(height={self.height}, width={self.width}, "
            f"values=<{self.num_cols} columns on "
            f"[{self.vmin:.2f},{self.vmax:.2f}]>)"
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
        
        # bin data
        hist, bins_ = np.histogram(
            a=data,
            bins=bins,
            range=xrange,
            weights=weights,
            density=density,
        )

        # build column chart
        if max_count is None:
            max_count = hist.max()
        super().__init__(
            values=hist,
            height=height,
            column_width=1,
            column_spacing=0,
            vrange=max_count,
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
    * color : optional ColorLike.
        The foreground color of the text. Defaults to the terminal's default
        foreground color.
    * bgcolor : optional ColorLike.
        The background color for the text. Defaults to a transparent
        background.
    
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
            chars.codes[i, :len(line)] = [ord(c) for c in line]
        
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
    * style : BoxStyle (default: BoxStyle.ROUND).
        The style of the border. Predefined styles are available in `BoxStyle`.
    * color : optional ColorLike.
        The color of the border characters. Defaults to the terminal's
        default foreground color.
    """
    def __init__(
        self,
        plot: plot,
        style: BoxStyle = BoxStyle.ROUND,
        color: ColorLike | None = None,
    ):
        chars = unicode_box(
            chars=plot.chars,
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
    """
    def __init__(
        self,
        *plots: plot,
        cols: int | None = None,
    ):
        # match size
        cell_height = max(p.height for p in plots)
        cell_width = max(p.width for p in plots)
        padded_chars = [
            p.chars.pad(
                below=cell_height - p.height,
                right=cell_width - p.width,
            ) for p in plots
        ]

        # wrap list
        if cols is None:
            cols = max(1, os.get_terminal_size()[0] // cell_width)
        n = len(padded_chars)
        wrapped_chars = [padded_chars[i:i+cols] for i in range(0, n, cols)]

        # correct final row
        if len(wrapped_chars) > 1 and len(wrapped_chars[-1]) < cols:
            buffer = CharArray.from_size(
                height=cell_height,
                width=cell_width * (cols - len(wrapped_chars[-1])),
            )
            wrapped_chars[-1].append(buffer)

        # combine into new char array
        blocked_chars = CharArray(
            codes=np.block([[c.codes for c in row] for row in wrapped_chars]),
            fg=np.block([[c.fg for c in row] for row in wrapped_chars]),
            fg_rgb=np.block([[[c.fg_rgb] for c in row] for row in wrapped_chars]),
            bg=np.block([[c.bg for c in row] for row in wrapped_chars]),
            bg_rgb=np.block([[[c.bg_rgb] for c in row] for row in wrapped_chars]),
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
        # horizontal
        if height is None or height <= plot.height:
            above = 0
            below = 0
        else:
            hdiff = height - plot.height
            above = hdiff // 2
            below = above + (hdiff % 2)
        # vertical
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


# # # 
# ANIMATIONS


def save_animation(
    plots: Sequence[plot], # non-empty
    filename: str,
    upscale: int = 1,
    downscale: int = 1,
    bgcolor: ColorLike | None = None,
    fps: int = 12,
    repeat: bool = True,
):
    """
    Supply a list of plots and a filename and this method will create an
    animated gif.
    
    Inputs:

    * plots : list[plot].
        The list of plots forming the frames of the animation.
    * filename : str.
        Where to save the gif. Should usually include a '.gif' extension.
    * upscale : int (>=1, default is 1).
        Represent each pixel with a square of side-length `upscale` pixels.
    * downscale : int (>=1, default is 1).
        Keep every `downscale`th pixel. Does not need to evenly divide the
        image height or width (think slice(0, height or width, downscale)).
        Applied after upscaling.
    * bgcolor : ColorLike | None.
        Default background colour. If none, a transparent background is used.
    * fps : int.
        Approximate frames per second encoded into the gif.
    * repeat : bool (default True).
        If true (default), the gif loops indefinitely. If false, the gif only
        plays once.

    Notes:

    * All plots should be the same size. If they are not, they will be aligned
      at the top left corner (padded with transparent pixels on the bottom and
      right). If you want different padding, add blank blocks before passing to
      this function.

    TODO:

    * Consider making this a plot aggregator and overriding .saveimg(). The
      only problem is that it's unclear what to use for renderimg and
      renderstr.
    """
    # render plots as image arrays
    frames = [
        plot.renderimg(
            upscale=upscale,
            downscale=downscale,
            bgcolor=bgcolor,
        ) for plot in plots
    ]
    
    # pad them to u8[height, width, RGBA]
    h = max(frame.shape[0] for frame in frames)
    w = max(frame.shape[1] for frame in frames)
    frames_uniform = [
        np.pad(
            frame,
            pad_width=((0,h-frame.shape[0]),(0,w-frame.shape[1]),(0,0)),
            mode='constant',
            constant_values=0,
        ) for frame in frames
    ]
    
    # convert to PIL images
    images = [
        Image.fromarray(frame, mode='RGBA') 
        for frame in frames_uniform
    ]

    # save
    images[0].save(
        filename,
        save_all=True,
        append_images=images[1:],
        duration=1000 // fps,
        loop=1-bool(repeat), # 1 = loop once, 0 = loop forever
    )
