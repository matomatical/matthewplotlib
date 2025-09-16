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

from typing import Callable, Self
from numpy.typing import ArrayLike
from matthewplotlib.colors import Color, ColorLike
from matthewplotlib.colormaps import ColorMap
from numbers import Number

from matthewplotlib.core import Char, BLANK, BoxStyle
from matthewplotlib.core import braille_encode, unicode_bar, unicode_col
from matthewplotlib.core import project3

type number = int | float | np.integer | np.floating


# # # 
# BASE PLOT CLASS WITH SHORTCUTS


class plot:
    """
    Abstract base class for all plot objects.

    A plot is essentially a 2D grid of `Char` objects. This class provides the
    core functionality for rendering and composing plots. Not typically
    instantiated directly, but it's useful to know its properties and methods.

    Properties:

    * height : int.
        The height of the plot in character lines.
    * width : int.
        The width of the plot in character columns.

    Methods:

    * renderstr() -> str.
        Returns a string representation of the plot with ANSI color codes,
        ready to be printed to a compatible terminal.
    * clearstr() -> str.
        Returns control characters that will clear the plot from the
        terminal after it has been printed.
    * saveimg(filename: str).
        Renders the plot to an image file (e.g., "plot.png") using a
        pixel font.

    Operators:
    
    * `str(plot)`: Shortcut for `plot.renderstr()`. This means you can render
       the plot just by calling `print(plot)`.
    * `-plot`: Shortcut for `plot.clearstr()`. Useful for animations.
    * `plot1 + plot2`: Horizontally stacks plots (see `hstack`).
    * `plot1 / plot2`: Vertically stacks plots (see `vstack`).
    * `plot1 | plot2`: Vertically stacks plots (see `vstack`).
    * `plot1 @ plot2`: Overlays plots (see `dstack`).
    """
    def __init__(self, array: list[list[Char]]):
        self.array = array


    @property
    def height(self) -> int:
        """
        Number of character rows in the plot.
        """
        return len(self.array)


    @property
    def width(self) -> int:
        """
        Number of character columns in the plot.
        """
        return len(self.array[0])


    def renderstr(self) -> str:
        """
        Convert the plot into a string for printing to the terminal.

        Note: plot.renderstr() is equivalent to str(plot).
        """
        return "\n".join(["".join([c.to_ansi_str() for c in l]) for l in self.array])


    def clearstr(self: Self) -> str:
        """
        Convert the plot into a string that, if printed immediately after
        plot.renderstr(), will clear that plot from the terminal.
        """
        return f"\x1b[{self.height}A\x1b[0J"


    def renderimg(
        self,
        scale_factor: int = 1,
    ) -> np.ndarray: # uint8[scale_factor * 16H, scale_factor * 8W, 4]
        """
        Convert the plot into an RGBA array for rendering with Pillow.
        """
        tiles = np.asarray(
            [[c.to_rgba_array() for c in l] for l in self.array],
        ) # uint8[H, W, 16, 8, 4]
        stacked = einops.rearrange(
            tiles,
            'H W h w rgba -> (H h) (W w) rgba',
        ) # uint8[16H, 8W, 4]
        if scale_factor == 1:
            scaled = stacked
        else:
            scaled = einops.repeat(
                stacked,
                'Hh Ww rgba -> (Hh scale1) (Ww scale2) rgba',
                scale1=scale_factor,
                scale2=scale_factor,
            )
        return scaled


    def saveimg(
        self,
        filename: str,
        scale_factor: int = 1,
    ):
        """
        Render the plot as an RGBA image and save it as a PNG file at the path
        `filename`.
        """
        image_data = self.renderimg(scale_factor=scale_factor)
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

    * data : number[n, 2].
        An array of n 2D points to plot. Each row is an (x, y) coordinate.
    * width : int (default: 30).
        The width of the plot in characters. The effective pixel width will be
        2 * width.
    * height : int (default: 10).
        The height of the plot in rows. The effective pixel height will be 4 *
        height.
    * xrange : optional (number, number).
        The x-axis limits `(xmin, xmax)`. If not provided, the limits are
        inferred from the min and max x-values in the data.
    * yrange : optional (number, number).
        The y-axis limits `(ymin, ymax)`. If not provided, the limits are
        inferred from the min and max y-values in the data.
    * color : optional ColorLike.
        The color of the plotted points (see `Color.parse`). Defaults to the
        terminal's default foreground color.
    * check_bounds : bool (default: False).
        If True, raises a `ValueError` if any data points fall outside the
        specified `xrange` or `yrange`.
    """
    def __init__(
        self,
        data: ArrayLike, # number[n, 2]
        width: int = 30,
        height: int = 10,
        xrange: tuple[number, number] | None = None,
        yrange: tuple[number, number] | None = None,
        color: ColorLike | None = None,
        check_bounds: bool = False,
    ):
        # preprocess and check shape
        data = np.asarray(data)
        n, _2 = data.shape
        assert _2 == 2
        color_ = Color.parse(color)

        # shortcut if no data
        if n == 0:
            array = [[BLANK] * width] * height
            super().__init__(array)
            self.xrange = xrange
            self.yrange = yrange
            self.num_points = len(data)
            return
        
        # determine data bounds
        xmin, ymin = data.min(axis=0)
        xmax, ymax = data.max(axis=0)
        if xrange is None:
            xrange = (xmin, xmax)
        else:
            xmin, xmax = xrange
        if yrange is None:
            yrange = (ymin, ymax)
        else:
            ymin, ymax = yrange
        # optional check
        if check_bounds:
            out_x = xmin < xrange[0] or xmax > xrange[1]
            out_y = ymin < yrange[0] or ymax > yrange[1]
            if out_x or out_y:
                raise ValueError("Scatter points out of range")
        
        # quantise 2d float coordinates to data grid
        dots, *_bins = np.histogram2d(
            x=data[:,0],
            y=data[:,1],
            bins=(2*width, 4*height),
            range=(xrange, yrange),
        )
        dots = dots.T     # we want y first
        dots = dots[::-1] # correct y for top-down drawing
        
        # render data grid as a grid of braille characters
        array = [[BLANK for _ in range(width)] for _ in range(height)]
        bgrid = braille_encode(dots > 0)
        for i in range(height):
            for j in range(width):
                if bgrid[i, j] > 0x2800:
                    braille_char = chr(bgrid[i, j])
                    array[i][j] = Char(braille_char, fg=color_)
        super().__init__(array)
        self.xrange = xrange
        self.yrange = yrange
        self.num_points = len(data)

    def __repr__(self):
        return (
            f"scatter(height={self.height}, width={self.width}, "
            f"data=<{self.num_points} points on "
            f"[{self.xrange[0]:.2f},{self.xrange[1]:.2f}]x"
            f"[{self.yrange[0]:.2f},{self.yrange[1]:.2f}]>)"
        )


class function(scatter):
    """
    Scatter plot representing a particular function.

    * F : float[batch] -> number[batch]
        
        The (vectorised) function to plot. The input should be a batch of
        floats x. The output should be a batch of scalars f(x).
    
    * xrange : (float, float)
        
        Lower and upper bounds on the x values to pass into the function.
    
    * width : int
        
        The number of character columns in the plot.
    
    * height : int
        
        The number of character rows in the plot.
    
    * yrange : optional (float, float)
        
        If provided, specifies the expected lower and upper bounds on the f(x)
        values. If not provided, they are automatically determined by using the
        minimum and maximum output over the inputs sampled.

    * color : optional ColorLike

        If provided, sets the colors for the scattered points. By default,
        foreground color is used.

    TODO:
    
    * More intelligent interpolation, like a proper line plot with a given
      thickness.
    """
    def __init__(
        self,
        F: Callable[[np.ndarray], np.ndarray],
        xrange: tuple[float, float],
        width: int,
        height: int,
        yrange: tuple[float, float] | None = None,
        color: None | ColorLike = None,
    ):
        # create a batch of inputs with the required format and shape
        X = np.linspace(*xrange, num=8*width, endpoint=False)
        
        # sample the function
        Y = F(X)

        # create the scatter plot
        super().__init__(
            data=np.c_[X, Y],
            width=width,
            height=height,
            xrange=xrange,
            yrange=yrange,
            color=color,
        )
        self.name = F.__name__
        self.xrange = xrange
        self.yrange = yrange
        
    def __repr__(self):
        return ("function("
                f"f={self.name}, "
                f"input=[{self.xrange[0]:.2f},{self.xrange[1]:.2f}]"
        ")")


class scatter3(scatter):
    """
    Scatter plot representing a 3d point cloud.

    * xyz: float[n, 3].
        The points to project, with columns corresponding to X, Y, and Z.
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
    * color : optional ColorLike.
        If provided, sets the colors for the scattered points. By default,
        foreground color is used.

    TODO:

    * Maybe allow configurable xyz ranges with clipping prior to projection?
    * Make sure this is not a subclass of scatter for the purposes of labelling
      axes as that would use projected coordinates.
    """
    def __init__(
        self,
        data: ArrayLike,                                        # float[n, 3]
        camera_position: np.ndarray = np.array([0., 0., 2.]),   # float[3]
        camera_target: np.ndarray = np.zeros(3),                # float[3]
        scene_up: np.ndarray = np.array([0.,1.,0.]),            # float[3]
        vertical_fov_degrees: float = 90.0,
        aspect_ratio: float | None = None,
        width: int = 30,
        height: int = 15,
        color: None | ColorLike = None,
    ):
        xyz = np.asarray(data)
        xy, valid = project3(
            xyz=xyz,
            camera_position=camera_position,
            camera_target=camera_target,
            scene_up=scene_up,
            fov_degrees=vertical_fov_degrees,
        )
        if aspect_ratio is None:
            aspect_ratio = width / (2*height)

        # create the scatter plot
        super().__init__(
            data=xy[valid],
            width=width,
            height=height,
            xrange=(-aspect_ratio, aspect_ratio),
            yrange=(-1.,1.),
            color=color,
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

    * im : float[h,w,3] | int[h,w,3] | float[h,w] | int[h,w] | ArrayLike.
        The image data. It can be in any of the following formats:
        * `float[h,w,3]`: A 2D array of RGB triples of floats in range [0,1].
        * `int[h,w,3]`: A 2D array of RGB triples of ints in range [0,255].
        * `float[h,w]`: A 2D array of scalars in the range [0,1]. If no
          colormap is provided, values are treated as greyscale (uniform
          colorisation). If a continuous colormap is provided, values are
          mapped to RGB values.
        * `int[h,w]`: A 2D array of scalars. If no colormap is provided,
          values should be in the range [0,255], they are treated as greyscale
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
    """
    def __init__(
        self,
        im: ArrayLike, # float[h,w] | float[h,w,rgb] | int[h,w] | int[h,w,rgb]
        colormap: ColorMap | None = None,
    ):
        # preprocessing: all inputs become float[h, w, rgb] with even h
        im = np.asarray(im)
        if len(im.shape) == 2 and colormap is None:
            # greyscale or indexed and no colormap -> uniform colourisation
            im = einops.repeat(im, 'h w -> h w 3')
        elif colormap is not None:
            # indexed, greyscale, or rgb and compatible colormap -> mapped rgb
            im = colormap(im)
        # pad to even height
        im = np.pad(
            array=im,
            pad_width=(
                (0, im.shape[0] % 2),
                (0, 0),
                (0, 0),
            ),
            mode='constant',
            constant_values=0.,
        )

        # processing: stack into fg/bg format
        stacked = einops.rearrange(im, '(h fgbg) w c -> h w fgbg c', fgbg=2)

        # render the image lines as unicode strings with ansi color codes
        array = [
            [
                Char("▀", Color.parse(fg), Color.parse(bg))
                for fg, bg in row
            ]
            for row in stacked
        ]

        # form a plot object
        super().__init__(array)

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
    * color : optional ColorLike.
        The color of the filled portion of the progress bar. Defaults to the
        terminal's default foreground color.
    """
    def __init__(
        self,
        progress: float,
        width: int = 40,
        color: ColorLike | None = None,
    ):
        color_ = Color.parse(color)
        progress = np.clip(progress, 0., 1.)

        # construct label
        label = f"{progress:4.0%}"
        label_chars = [Char(c) for c in label]
        
        # construct bar
        bar_chars = [
            Char(block, fg=color_)
            for block in unicode_bar(
                proportion=progress,
                total_width=width - 2 - len(label),
            )
        ]

        # put it together
        array = [
            [*label_chars, Char("["), *bar_chars, Char("]")]
        ]
        super().__init__(
            array=array,
        )
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
    * Allow each bar to have a height other than 1, and allow spacing.
    """
    def __init__(
        self,
        values: ArrayLike, # numeric[n]
        width: int = 30,
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
        color_ = Color.parse(color)

        # compute the bar widths
        norm_values = (values - vmin) / (vmax - vmin + 1e-15)
        
        # construct the bar chart!
        array = [
            [ Char(block, fg=color_) for block in unicode_bar(v, width) ]
            for v in norm_values
        ]
        super().__init__(
            array=array,
        )
        self.vmin = vmin
        self.vmax = vmax
        self.num_bars = len(values)

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
    * xrange : optional (number, number).
        If provided, bins range over this interval, and values outside the
        range are discarded. Same as np.histogram's range argument.
    * bins : optional int, sequence, or str.
        If provided, used to determine number of bins, bin boundaries, or bin
        boundary determination method. See np.histogram's bins argument for
        details.
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
        data: ArrayLike,    # number[n]
        bins = 10,          # as in np.histogram
        xrange = None,      # as 'range' parameter in np.histogram
        weights = None,     # as in np.histogram
        density = False,    # as in np.histogram
        max_count: None | number = None,
        width: int = 22,
        color: ColorLike | None = None,
    ):
        # prepare data
        data = np.asarray(data)
        
        # bin data
        hist, bins = np.histogram(
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
            vrange=max_count,
            color=color,
        )
        self.bins = bins

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
    * Allow each column to have a height other than 1, and allow spacing.
    """
    def __init__(
        self,
        values: ArrayLike, # number[n], actually int[n] will also work
        height: int = 10,
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
        color_ = Color.parse(color)

        # compute the column heights
        norm_values = (values - vmin) / (vmax - vmin + 1e-15)
        
        # construct the column chart!
        columns = [
            [ Char(block, fg=color_) for block in unicode_col(v, height) ]
            for v in norm_values
        ]
        array = [
            [ columns[j][i] for j in range(len(columns)) ]
            for i in range(height)
        ]
        super().__init__(array=array)
        self.vmin = vmin
        self.vmax = vmax
        self.num_cols = len(values)

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
    * xrange : optional (number, number).
        If provided, bins range over this interval, and values outside the
        range are discarded. Same as np.histogram's range argument.
    * bins : optional int, sequence, or str.
        If provided, used to determine number of bins, bin boundaries, or bin
        boundary determination method. See np.histogram's bins argument for
        details.
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
        data: ArrayLike,    # number[n]
        bins = 10,          # as in np.histogram
        xrange = None,      # as 'range' parameter in np.histogram
        weights = None,     # as in np.histogram
        density = False,    # as in np.histogram
        max_count: None | number = None,
        height: int = 10,
        color: ColorLike | None = None,
    ):
        # prepare data
        data = np.asarray(data)
        
        # bin data
        hist, bins = np.histogram(
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
            vrange=max_count,
            color=color,
        )
        self.bins = bins

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
    * dotcolor : optional ColorLike.
        The foreground color used for dots (points along the curve where `data`
        is `True`). Defaults to the terminal's default foreground color.
    * bgcolor : optional ColorLike.
        The background color for the entire path of the Hilbert curve (points
        along the curve where `data` is `False`, plus possibly some extra
        points if the curve does not exactly fit the last character cell).
        Defaults to a transparent background.
    * nullcolor : optional ColorLike.
        The background color for the grid area not occupied by the curve. This
        is relevant for non-square-power-of-2 data lengths. Defaults to a
        transparent background.
    """
    def __init__(
        self,
        data: ArrayLike, # bool[N]
        dotcolor: ColorLike | None = None,
        bgcolor: ColorLike | None = None,
        nullcolor: ColorLike | None = None,
    ):
        # preprocess and compute grid shape
        data = np.asarray(data)
        N, = data.shape
        n = max(2, ((N-1).bit_length() + 1) // 2)
        dotcolor_ = Color.parse(dotcolor)
        bgcolor_ = Color.parse(bgcolor)
        nullcolor_ = Color.parse(nullcolor)

        # compute grid positions for each data element
        all_coords: np.ndarray = _hilbert.decode(
            hilberts=np.arange(N),
            num_dims=2,
            num_bits=n,
        )
        lit_coords = all_coords[data]

        # make empty dot matrix
        all_grid = np.zeros((2**n,2**n), dtype=bool)
        all_grid[all_coords[:,1], all_coords[:,0]] = True
        lit_grid = np.zeros((2**n,2**n), dtype=bool)
        lit_grid[lit_coords[:,1], lit_coords[:,0]] = True
        
        # render data grid as a grid of braille characters
        width = int(2 ** (n-1))
        height = int(2 ** (n-2))
        null = Char(" ", bg=nullcolor_)
        array = [[null for _ in range(width)] for _ in range(height)]
        bg_grid = braille_encode(all_grid)
        fg_grid = braille_encode(lit_grid)
        for i in range(height):
            for j in range(width):
                if bg_grid[i, j] > 0x2800:
                    braille_char = chr(fg_grid[i, j])
                    array[i][j] = Char(
                        c=braille_char,
                        fg=dotcolor_,
                        bg=bgcolor_,
                    )
        super().__init__(array)
        self.num_points = len(lit_coords)
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
        color: ColorLike | None = None,
        bgcolor: ColorLike | None = None,
    ):
        color_ = Color.parse(color)
        bgcolor_ = Color.parse(bgcolor)

        lines = text.splitlines()
        height = len(lines)
        width = max(len(line) for line in lines)
        array = [
            [Char(c, fg=color_, bg=bgcolor_) for c in line]
            + [BLANK] * (width - len(line))
            for line in lines
        ]
        super().__init__(array=array)
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
        color_ = Color.parse(color)
        array = [
            # top row
            [
                Char(style.nw, fg=color_),
                *[Char(style.n, fg=color_)] * plot.width,
                Char(style.ne, fg=color_),
            ],
            # middle rows
            *[
                [
                    Char(style.w, fg=color_),
                    *row,
                    Char(style.w, fg=color_),
                ]
                for row in plot.array
            ],
            # bottom row
            [
                Char(style.sw, fg=color_),
                *[Char(style.s, fg=color_)] * plot.width,
                Char(style.se, fg=color_),
            ],
        ]
        super().__init__(array)
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
        array = [[BLANK] * width] * height
        super().__init__(array)

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
        width = sum(p.width for p in plots)
        # build array left to right one plot at a time
        array : list[list[Char]] = [[] for _ in range(height)]
        for p in plots:
            for i in range(p.height):
                array[i].extend(p.array[i])
            for i in range(p.height, height):
                array[i].extend([BLANK] * p.width)
        super().__init__(array)
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
        height = sum(p.height for p in plots)
        width = max(p.width for p in plots)
        # build the array top to bottom one plot at a time
        array = []
        for p in plots:
            for row in p.array:
                array.append(row + [BLANK] * (width - p.width))
        super().__init__(array)
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
        array = [ [BLANK for _ in range(width) ] for _ in range(height) ]
        for p in plots:
            for i, line in enumerate(p.array):
                for j, c in enumerate(line):
                    if c.isblank():
                        # keep underlying character
                        pass
                    elif c.bg is None:
                        # override, but use the (effective) background of the
                        # underlying char as the new background
                        array[i][j] = Char(c=c.c, fg=c.fg, bg=array[i][j].bg_)
                    else:
                        # simply override
                        array[i][j] = c
        super().__init__(array)
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
        cell_height = max(p.height for p in plots)
        cell_width = max(p.width for p in plots)
        if cols is None:
            cols = max(1, os.get_terminal_size()[0] // cell_width)
        # wrap list of plots into groups, of length `cols` (except last)
        wrapped_plots: list[list[plot]] = []
        for i, plot in enumerate(plots):
            if i % cols == 0:
                wrapped_plots.append([])
            wrapped_plots[-1].append(plot)
        # build the array left/right, top/down, one plot at a time
        array = []
        for group in wrapped_plots:
            row: list[list[Char]] = [[] for _ in range(cell_height)]
            for p in group:
                for i in range(p.height):
                    row[i].extend(p.array[i])
                    row[i].extend([BLANK] * (cell_width - p.width))
                for i in range(p.height, cell_height):
                    row[i].extend([BLANK] * cell_width)
            array.extend(row)
        # correction for the final row
        if len(group) < cols:
            buffer = [BLANK] * cell_width * (cols - len(group))
            for i in range(cell_height):
                array[-cell_height+i].extend(buffer)
        # done!
        super().__init__(array)
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
        height = plot.height if height is None else max(height, plot.height)
        width = plot.width if width is None else max(width, plot.width)
        def _center(inner_size, outer_size):
            diff = outer_size - inner_size
            left = diff // 2
            right = left + (diff % 2)
            return left, right
        left, right = _center(plot.width, width)
        above, below = _center(plot.height, height)
        array = (
            [[BLANK] * width] * above
            + [
                [BLANK] * left + row + [BLANK] * right for row in plot.array
            ]
            + [[BLANK] * width] * below
        )
        super().__init__(array)
        self.plot = plot
    
    def __repr__(self):
        return (
            f"center(height={self.height}, width={self.width}, "
            f"plot={self.plot!r})"
        )


