"""
Matthew's plotting library (matthewplotlib).

A python plotting library that isn't painful.

See https://github.com/matomatical/matthewplotlib
"""

import dataclasses
import math
import os
from typing import Callable, Self

import numpy as np
from numpy.typing import ArrayLike
import einops
import hilbert as _hilbert

from PIL import Image
import unscii


# # # 
# TYPES


ColorLike = (
    str
    | np.ndarray # float[3] (0 to 1) or uint8[3] (0 to 255)
    | tuple[int, int, int]
    | tuple[float, float, float]
    | None
)


ColorMap = (
    Callable[[ArrayLike], np.ndarray]   # float[...] -> uint8[...,3]
    | Callable[[ArrayLike], np.ndarray] # int[...] -> uint8[...,3]
)


# # # 
# UTILITY CLASSES


@dataclasses.dataclass(frozen=True)
class Color:
    """
    An RGB color triple.
    """
    r: int
    g: int
    b: int

    def __iter__(self) -> iter:
        return iter((self.r, self.g, self.b))

    @staticmethod
    def parse(color: ColorLike) -> Self | None:
        """
        Accept and standardise RGB triples in various formats.
        """
        if color is None:
            return None

        if isinstance(color, str):
            if color.startswith("#") and len(color) == 4:
                return Color(
                    r=17*int(color[1], base=16),
                    g=17*int(color[2], base=16),
                    b=17*int(color[3], base=16),
                )
            if color.startswith("#") and len(color) == 7:
                return Color(
                    r=int(color[1:3], base=16),
                    g=int(color[3:5], base=16),
                    b=int(color[5:7], base=16),
                )
            if color.lower() in KNOWN_COLORS:
                return KNOWN_COLORS[color.lower()]

        elif isinstance(color, (np.ndarray, tuple)):
            color_ = np.asarray(color)
            if color_.shape == (3,):
                if np.issubdtype(color_.dtype, np.floating):
                    rgb = (255*np.clip(color_, 0., 1.)).astype(int)
                    return Color(*rgb)
                if np.issubdtype(color_.dtype, np.integer):
                    rgb = np.clip(color_, 0, 255).astype(int)
                    return Color(*rgb)
        
        raise ValueError(f"invalid color {color!r}")


KNOWN_COLORS = {
    "black":    Color(  0,   0,   0),
    "red":      Color(255,   0,   0),
    "green":    Color(  0, 255,   0),
    "blue":     Color(  0,   0, 255),
    "cyan":     Color(  0, 255, 255),
    "magenta":  Color(255,   0, 255),
    "yellow":   Color(255, 255,   0),
    "white":    Color(255, 255, 255),
}
    

@dataclasses.dataclass(frozen=True)
class Char:
    """
    A single possibly-coloured character.
    """
    c: str = " "
    fg: Color | None = None
    bg: Color | None = None
    
    def __bool__(self):
        """
        False if the character is blank.
        """
        return bool(self.c != " " or self.bg is not None)


BLANK = Char(c=" ", fg=None, bg=None)


# # # 
# PLOT BASE CLASS


class plot:
    """
    Abstract base class for all plot objects.

    A plot is essentially a 2D grid of `Char` objects. This class provides the
    core functionality for rendering and composing plots. Not typically
    instantiated directly, but it's useful to know its properties and methods.

    Properties:

    * height : int
        The height of the plot in character lines.
    * width : int
        The width of the plot in character columns.

    Methods:

    * renderstr() -> str
        Returns a string representation of the plot with ANSI color codes,
        ready to be printed to a compatible terminal.
    * clearstr() -> str
        Returns control characters that will clear the plot from the
        terminal after it has been printed.
    * saveimg(filename: str)
        Renders the plot to an image file (e.g., "plot.png") using a
        pixel font.

    Operators:
    
    * `str(plot)`: Shortcut for `plot.renderstr()`. This means you can render
       the plot just by calling `print(plot)`.
    * `~plot`: Shortcut for `plot.clearstr()`. Useful for animations.
    * `plot1 | plot2`: Horizontally stacks plots (see `hstack`).
    * `plot1 ^ plot2`: Vertically stacks plots (see `vstack`).
    * `plot1 & plot2`: Overlays plots (see `dstack`).
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
        return "\n".join(["".join([to_ansi_str(c) for c in l]) for l in self.array])

    def clearstr(self: Self) -> str:
        """
        Convert the plot into a string that, if printed immediately after
        plot.renderstr(), will clear that plot from the terminal.
        """
        return f"\x1b[{self.height}A\x1b[0J"
    
    def saveimg(self, filename: str, scale_factor: int = 1):
        tiles = np.asarray([[to_rgba_array(c) for c in l] for l in self.array])
        # -> uint8[H, W, 16, 8, 4]
        stacked = einops.rearrange(tiles, 'H W h w rgba -> (H h) (W w) rgba')
        image = Image.fromarray(stacked, mode='RGBA')
        image.save(filename)
    
    def __str__(self) -> str:
        """
        Shortcut for the string for printing the plot.
        """
        return self.renderstr()
    
    def __invert__(self: Self) -> str:
        """
        Shortcut for the string for clearing the plot.
        """
        return self.clearstr()

    def __or__(self: Self, other: Self) -> Self:
        """
        Shortcut for horizontally stacking plots:

        plot1 | plot2 = hstack(plot1, plot2).
        """
        return hstack(self, other)

    def __xor__(self: Self, other: Self) -> Self:
        """
        Shortcut for vertically stacking plots:

        plot1 ^ plot2 = vstack(plot1, plot2).
        """
        return vstack(self, other)
    
    def __and__(self: Self, other: Self) -> Self:
        """
        Shortcut for depth-stacking plots:

        plot1 & plot2 = dstack(plot1, plot2).
        """
        return dstack(self, other)
    

# # # 
# DATA PLOTTING CLASSES


class image(plot):
    """
    Render a small image using a grid of unicode half-block characters.

    Represents an image by mapping pairs of vertically adjacent pixels to the
    foreground and background colors of a single character cell (this
    effectively doubles the vertical resolution in the terminal).

    Inputs:

    * im : float[h,w,3] | int[h,w,3] | float[h,w] | int[h,w] | ArrayLike
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
          
    * colormap : optional ColorMap
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
        colormap: ColorMap = None,
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


class fimage(image):
    """
    Heatmap representing the image of a 2d function over a square. Inputs:

    * F : float[batch, 2] -> float[batch]
        The (vectorised) function to plot. The input should be a batch of
        (x, y) vectors. The output should be a batch of scalars f(x, y).
    * xrange : (float, float)
        Lower and upper bounds on the x values to pass into the function.
    * yrange : (float, float)
        Lower and upper bounds on the y values to pass into the function.
    * width : int
        The number of grid squares along the x axis. This will also become the
        width of the plot.
    * height : int
        The number of grid squares along the y axis. This will become double
        the height of the plot in lines (since the result is an image plot with
        two pixels per line).
    * zrange : optional (float, float)
        Expected lower and upper bounds on the f(x, y) values. Used for
        determining the bounds of the colour scale. By default, the minimum and
        maximum output over the grid are used.
    * colormap : optional colormap (e.g. mp.viridis)
        By default, the output will be in greyscale, with black corresponding
        to zrange[0] and white corresponding to zrange[1]. You can choose a
        different colormap (e.g. mp.reds, mp.viridis, etc.) here.
    * endpoints : bool (default: False)
        If true, endpoints are included from the linspaced inputs, and so the
        grid elements in each corner will represent the different combinations
        of xrange/yrange.
        If false (default), the endpoints are excluded, so the lower bounds are
        met but the upper bounds are not, meaning each grid square color shows
        the value of the function precisely at its lower left corner.
    """
    def __init__(
        self,
        F: Callable[[ArrayLike], ArrayLike], # TODO: auto vectorise
        xrange: tuple[float, float],
        yrange: tuple[float, float],
        width: int,
        height: int,
        zrange: tuple[float, float] | None = None,
        colormap: Callable | None = None,
        endpoints: bool = False,
    ):
        # create a meshgrid with the required format and shape
        X, Y = np.meshgrid(
            np.linspace(*xrange, num=width, endpoint=endpoints),
            np.linspace(*yrange, num=height, endpoint=endpoints),
        ) # float[h, w] (x2)
        Y = Y[::-1] # correct Y direction for image plotting
        XY = einops.rearrange(np.dstack((X, Y)), 'h w xy -> (h w) xy')

        # sample the function
        Z = F(XY)

        # create the image array
        zgrid = einops.rearrange(Z, '(h w) -> h w', h=height, w=width)
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
        return ("fimage("
                f"f={self.name}, "
                f"input=[{self.xrange[0]:.2f},{self.xrange[1]:.2f}]"
                f"x[{self.yrange[0]:.2f},{self.yrange[1]:.2f}]"
        ")")


class scatter(plot):
"""
    Render a scatterplot using a grid of braille unicode characters.

    Each character cell in the plot corresponds to a 2x4 grid of sub-pixels,
    represented by braille dots.

    Inputs:

    * data : float[n, 2]
        An array of n 2D points to plot. Each row is an (x, y) coordinate.
    * height : int (default: 10)
        The height of the plot in rows. The effective pixel height will be 4 *
        height.
    * width : int (default: 30)
        The width of the plot in characters. The effective pixel width will be
        2 * width.
    * yrange : optional (float, float)
        The y-axis limits `(ymin, ymax)`. If not provided, the limits are
        inferred from the min and max y-values in the data.
    * xrange : optional (float, float)
        The x-axis limits `(xmin, xmax)`. If not provided, the limits are
        inferred from the min and max x-values in the data.
    * color : optional ColorLike
        The color of the plotted points (see `Color.parse`). Defaults to the
        terminal's default foreground color.
    * check_bounds : bool (default: False)
        If True, raises a `ValueError` if any data points fall outside the
        specified `xrange` or `yrange`.
    """
    def __init__(
        self,
        data: ArrayLike, # float[n, 2]
        height: int = 10,
        width: int = 30,
        yrange: tuple[float, float] | None = None,
        xrange: tuple[float, float] | None = None,
        color: ColorLike = None,
        check_bounds: bool = False,
    ):
        # preprocess and check shape
        data = np.asarray(data)
        n, _2 = data.shape
        assert _2 == 2
        color = Color.parse(color)

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
                    array[i][j] = Char(braille_char, fg=color)
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


class hilbert(plot):
    """
    Visualize a 1D boolean array along a 2D Hilbert curve.

    Maps a 1D sequence of data points to a 2D grid using a space-filling
    Hilbert curve, which helps preserve locality. The curve is rendered using
    braille unicode characters for increased resolution.

    Inputs:

    * data : bool[N]
        A 1D array of booleans. The length `N` determines the order of the
        Hilbert curve required to fit all points. True values are rendered as
        dots, and False values are rendered as blank spaces.
    * dotcolor : optional ColorLike
        The foreground color used for dots (points along the curve where `data`
        is `True`). Defaults to the terminal's default foreground color.
    * bgcolor : optional ColorLike
        The background color for the entire path of the Hilbert curve (points
        along the curve where `data` is `False`, plus possibly some extra
        points if the curve does not exactly fit the last character cell).
        Defaults to a transparent background.
    * nullcolor : optional ColorLike
        The background color for the grid area not occupied by the curve. This
        is relevant for non-square-power-of-2 data lengths. Defaults to a
        transparent background.
    """
    def __init__(
        self,
        data: ArrayLike, # bool[N]
        dotcolor: ColorLike = None,
        bgcolor: ColorLike = None,
        nullcolor: ColorLike = None,
    ):
        # preprocess and compute grid shape
        data = np.asarray(data)
        N, = data.shape
        n = max(2, ((N-1).bit_length() + 1) // 2)
        dotcolor = Color.parse(dotcolor)
        bgcolor = Color.parse(bgcolor)
        nullcolor = Color.parse(nullcolor)

        # compute grid positions for each data element
        all_coords = _hilbert.decode(
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
        null = Char(" ", bg=nullcolor)
        array = [[null for _ in range(width)] for _ in range(height)]
        bg_grid = braille_encode(all_grid)
        fg_grid = braille_encode(lit_grid)
        for i in range(height):
            for j in range(width):
                if bg_grid[i, j]:
                    braille_char = chr(fg_grid[i, j])
                    array[i][j] = Char(
                        c=braille_char,
                        fg=dotcolor,
                        bg=bgcolor,
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


class text(plot):
    """
    A plot object containing one or more lines of text.

    This class wraps a string in the plot interface, allowing it to be
    composed with other plot objects. It handles multi-line strings by
    splitting them at newline characters.

    Inputs:

    * text : str
        The text to be displayed. Newline characters (`\n`) will create
        separate lines in the plot.
    * color : optional ColorLike
        The foreground color of the text. Defaults to the terminal's default
        foreground color.
    * bgcolor : optional ColorLike
        The background color for the text. Defaults to a transparent
        background.
    
    TODO:

    * Allow alignment and resizing.
    * Account for non-printable and wide characters.
    """
    def __init__(
        self,
        text: str,
        color: ColorLike = None,
        bgcolor: ColorLike = None,
    ):
        color = Color.parse(color)
        bgcolor = Color.parse(bgcolor)
        lines = text.splitlines()
        height = len(lines)
        width = max(len(line) for line in lines)
        array = [
            [Char(c, fg=color, bg=bgcolor) for c in line]
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


class progress(plot):
    """
    A single-line progress bar.

    Displays a progress bar with a percentage label. The bar is rendered using
    block element characters to show fractional progress with finer granularity
    than a single character.

    Inputs:

    * progress : float
        The progress to display, as a float between 0.0 and 1.0. Values outside
        this range will be clipped.
    * width : int (default: 40)
        The total width of the progress bar plot in character columns,
        including the label and brackets.
    * color : optional ColorLike
        The color of the filled portion of the progress bar. Defaults to the
        terminal's default foreground color.
    """
    def __init__(
        self,
        progress: float,
        width: int = 40,
        color: ColorLike = None,
    ):
        progress = np.clip(progress, 0., 1.)
        # construct label
        label = f"{progress:4.0%}"
        label_chars = [Char(c) for c in label]
        # construct bar
        bar_width = width - 2 - len(label)
        fill_width = bar_width * progress
        bar_chars = [Char("█", fg=color)] * int(fill_width)
        marginal_width = int(8 * (fill_width % 1))
        if marginal_width > 0:
            bar_chars.append(Char(
                    [None, "▏", "▎", "▍", "▌", "▋", "▊", "▉"][marginal_width],
                    color,
            ))
        bar_chars.extend(
            [BLANK] * (bar_width - len(bar_chars))
        )
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


# # # 
# ARRANGEMENT CLASSES


class blank(plot):
     """
    Creates a rectangular plot composed entirely of blank space.

    Useful for adding padding or aligning items in a complex layout.

    Inputs:

    * height : optional int
        The height of the blank area in character rows. Default 1.
    * width : optional int
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

    * *plots : plot
        A sequence of plot objects to be horizontally stacked.
    """
    def __init__(
        self,
        *plots: plot,
    ):
        height = max(p.height for p in plots)
        width = sum(p.width for p in plots)
        # build array left to right one plot at a time
        array = [[] for _ in range(height)]
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

    * *plots : plot
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

    * *plots : plot
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
                    if c:
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

    * *plots : plot
        A sequence of plot objects to be arranged in a grid.
    * cols : optional int
        The number of columns in the grid. If not provided, it is automatically
        determined based on the terminal width and the width of the largest
        plot.
    """
    def __init__(
        self,
        *plots: plot,
        cols: int = None,
    ):
        cell_height = max(p.height for p in plots)
        cell_width = max(p.width for p in plots)
        if cols is None:
            cols = max(1, os.get_terminal_size()[0] // cell_width)
        # wrap list of plots into groups, of length `cols` (except last)
        wrapped_plots = []
        for i, plot in enumerate(plots):
            if i % cols == 0:
                wrapped_plots.append([])
            wrapped_plots[-1].append(plot)
        # build the array left/right, top/down, one plot at a time
        array = []
        for group in wrapped_plots:
            row = [[] for _ in range(cell_height)]
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


class border(plot):
    """
    Add a border around a plot using box-drawing characters.

    Inputs:

    * plot : plot
        The plot object to be enclosed by the border.
    * style : optional Style (default: Style.ROUND)
        The style of the border. Predefined styles are available in
        `border.Style`.
    * color : optional ColorLike
        The color of the border characters. Defaults to the terminal's
        default foreground color.
    """
    class Style:
        """
        An enum-like class defining preset styles for the `border` plot.

        Each style is a string of six characters representing the border
        elements in the following order: horizontal, vertical, top-left,
        top-right, bottom-left, and bottom-right.

        Available Styles:

        * `LIGHT`: A standard, single-line border.
        * `HEAVY`: A thicker, bold border.
        * `DOUBLE`: A double-line border.
        * `BLANK`: An invisible border, useful for adding padding around a
          plot while maintaining layout alignment.
        * `ROUND`: A single-line border with rounded corners.
        * `BUMPER`: A single-line border with corners made of blocks.

        Demo:

        ```
        ┌──────┐ ┏━━━━━━┓ ╔══════╗         ╭──────╮ ▛──────▜
        │LIGHT │ ┃HEAVY ┃ ║DOUBLE║  BLANK  │ROUND │ │BUMPER│
        └──────┘ ┗━━━━━━┛ ╚══════╝         ╰──────╯ ▙──────▟
        ```
        """
        LIGHT  = "─│┌┐└┘"
        HEAVY  = "━┃┏┓┗┛"
        DOUBLE = "═║╔╗╚╝"
        BLANK  = "      "
        ROUND  = "─│╭╮╰╯"
        BUMPER = "─│▛▜▙▟"


    def __init__(
        self,
        plot: plot,
        style: Style | None = None,
        color: ColorLike = None,
    ):
        if color is not None:
            color = Color.parse(color)
        if style is None:
            style = self.Style.ROUND
        array = [
            # top row
            [
                Char(style[2], fg=color),
                *[Char(style[0], fg=color)] * plot.width,
                Char(style[3], fg=color),
            ],
            # middle rows
            *[
                [
                    Char(style[1], fg=color),
                    *row,
                    Char(style[1], fg=color),
                ]
                for row in plot.array
            ],
            # bottom row
            [
                Char(style[4], fg=color),
                *[Char(style[0], fg=color)] * plot.width,
                Char(style[5], fg=color),
            ],
        ]
        super().__init__(array)
        self.style = style[2]
        self.plot = plot
    
    def __repr__(self):
        return f"border(style={self.style!r}, plot={self.plot!r})"


class center(plot):
    """
    Pad a plot with blank space to center it within a larger area.

    If the specified `height` or `width` is smaller than the plot's dimensions,
    the larger dimension is used, effectively preventing the plot from being
    cropped.

    Inputs:

    * plot : plot
        The plot object to be centered.
    * height : optional int
        The target height of the new padded plot. If not provided, it defaults
        to the original plot's height (no vertical padding).
    * width : optional int
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


# # # 
# CONTINUOUS COLORMAPS


def reds(
    x: ArrayLike    # float[...]
) -> np.ndarray:    # -> uint8[..., 3]
    """
    Red colormap. Simply embeds greyscale value into red channel.
    """
    x = np.asarray(x)
    rgb = np.zeros((*x.shape, 3), dtype=np.uint8)
    rgb[..., 0] = 255 * x
    return rgb


def greens(
    x: ArrayLike    # float[...]
) -> np.ndarray:    # -> uint8[..., 3]
    """
    Green colormap. Simply embeds greyscale value into green channel.
    """
    x = np.asarray(x)
    rgb = np.zeros((*x.shape, 3), dtype=np.uint8)
    rgb[..., 1] = 255 * x
    return rgb


def blues(
    x: ArrayLike    # float[...]
) -> np.ndarray:    # -> uint8[..., 3]
    """
    Blue colormap. Simply embeds greyscale value into blue channel.
    """
    x = np.asarray(x)
    rgb = np.zeros((*x.shape, 3), dtype=np.uint8)
    rgb[..., 2] = 255 * x
    return rgb


def yellows(
    x: ArrayLike    # float[...]
) -> np.ndarray:    # -> uint8[..., 3]
    """
    Yellow colormap. Simply embeds greyscale value into red and green channels.
    """
    x = np.asarray(x)
    rgb = np.zeros((*x.shape, 3), dtype=np.uint8)
    rgb[..., 0] = 255 * x
    rgb[..., 1] = 255 * x
    return rgb


def magentas(
    x: ArrayLike    # float[...]
) -> np.ndarray:    # -> uint8[..., 3]
    """
    Magenta colormap. Simply embeds greyscale value into red and blue
    channels.
    """
    x = np.asarray(x)
    rgb = np.zeros((*x.shape, 3), dtype=np.uint8)
    rgb[..., 0] = 255 * x
    rgb[..., 2] = 255 * x
    return rgb


def cyans(
    x: ArrayLike    # float[...]
) -> np.ndarray:    # -> uint8[..., 3]
    """
    Cyan colormap. Simply embeds greyscale value into green and blue
    channels.
    """
    x = np.asarray(x)
    rgb = np.zeros((*x.shape, 3), dtype=np.uint8)
    rgb[..., 1] = 255 * x
    rgb[..., 2] = 255 * x
    return rgb


def cyber(
    x: ArrayLike    # float[...]
) -> np.ndarray:    # -> uint8[..., 3]
    """
    Cyberpunk colormap. Uses greyscale value to interpolate between cyan and
    magenta.
    """
    x = np.asarray(x)
    rgb = np.zeros((*x.shape, 3), dtype=np.uint8)
    rgb[..., 0] = 255 * x
    rgb[..., 1] = 255 * (1-x)
    rgb[..., 2] = 255
    return rgb


def rainbow(
    x: ArrayLike    # float[...]
) -> np.ndarray:    # -> uint8[..., 3]
    """
    Rainbow colormap. Effectively embeds greyscale values as hue in HSV color
    space.
    """
    x = np.asarray(x)
    rgb = np.zeros((*x.shape, 3), dtype=np.uint8)
    # map [0, 1] into (i, r) coordinates where
    # * i is in {0, 1, 2, 3, 4, 5}, and
    # * r is in [0, 1].
    t = 6 * x
    i = t.astype(int) % 6
    r = t % 1
    # for each i, project t into the appropriate colour channels
    # i = 0 (red to red+green)
    i0 = (i == 0)
    rgb[i0, 0] = 255
    rgb[i0, 1] = 255 * r[i0]
    # i = 1 (red+green to green)
    i1 = (i == 1)
    rgb[i1, 0] = 255 - 255 * r[i1]
    rgb[i1, 1] = 255
    # i = 2 (green to green+blue)
    i2 = (i == 2)
    rgb[i2, 1] = 255
    rgb[i2, 2] = 255 * r[i2]
    # i = 3 (green+blue to blue)
    i3 = (i == 3)
    rgb[i3, 1] = 255 - 255 * r[i3]
    rgb[i3, 2] = 255
    # i = 4 (blue to blue+red)
    i4 = (i == 4)
    rgb[i4, 2] = 255
    rgb[i4, 0] = 255 * r[i4]
    # i = 5 (blue+red to red)
    i5 = (i == 5)
    rgb[i5, 2] = 255 - 255 * r[i5]
    rgb[i5, 0] = 255
    # done
    return rgb


def magma(
    x: ArrayLike    # float[...]
) -> np.ndarray:    # -> uint8[..., 3]
    """
    Magma colormap by Nathaniel J. Smith and Stefan van der Walt (see
    https://bids.github.io/colormap/).

    Discretised to 256 8-bit colours.
    """
    MAGMA = np.array([
        [  0,   0,   3], [  0,   0,   4], [  0,   0,   6], [  1,   0,   7],
        [  1,   1,   9], [  1,   1,  11], [  2,   2,  13], [  2,   2,  15],
        [  3,   3,  17], [  4,   3,  19], [  4,   4,  21], [  5,   4,  23],
        [  6,   5,  25], [  7,   5,  27], [  8,   6,  29], [  9,   7,  31],
        [ 10,   7,  34], [ 11,   8,  36], [ 12,   9,  38], [ 13,  10,  40],
        [ 14,  10,  42], [ 15,  11,  44], [ 16,  12,  47], [ 17,  12,  49],
        [ 18,  13,  51], [ 20,  13,  53], [ 21,  14,  56], [ 22,  14,  58],
        [ 23,  15,  60], [ 24,  15,  63], [ 26,  16,  65], [ 27,  16,  68],
        [ 28,  16,  70], [ 30,  16,  73], [ 31,  17,  75], [ 32,  17,  77],
        [ 34,  17,  80], [ 35,  17,  82], [ 37,  17,  85], [ 38,  17,  87],
        [ 40,  17,  89], [ 42,  17,  92], [ 43,  17,  94], [ 45,  16,  96],
        [ 47,  16,  98], [ 48,  16, 101], [ 50,  16, 103], [ 52,  16, 104],
        [ 53,  15, 106], [ 55,  15, 108], [ 57,  15, 110], [ 59,  15, 111],
        [ 60,  15, 113], [ 62,  15, 114], [ 64,  15, 115], [ 66,  15, 116],
        [ 67,  15, 117], [ 69,  15, 118], [ 71,  15, 119], [ 72,  16, 120],
        [ 74,  16, 121], [ 75,  16, 121], [ 77,  17, 122], [ 79,  17, 123],
        [ 80,  18, 123], [ 82,  18, 124], [ 83,  19, 124], [ 85,  19, 125],
        [ 87,  20, 125], [ 88,  21, 126], [ 90,  21, 126], [ 91,  22, 126],
        [ 93,  23, 126], [ 94,  23, 127], [ 96,  24, 127], [ 97,  24, 127],
        [ 99,  25, 127], [101,  26, 128], [102,  26, 128], [104,  27, 128],
        [105,  28, 128], [107,  28, 128], [108,  29, 128], [110,  30, 129],
        [111,  30, 129], [113,  31, 129], [115,  31, 129], [116,  32, 129],
        [118,  33, 129], [119,  33, 129], [121,  34, 129], [122,  34, 129],
        [124,  35, 129], [126,  36, 129], [127,  36, 129], [129,  37, 129],
        [130,  37, 129], [132,  38, 129], [133,  38, 129], [135,  39, 129],
        [137,  40, 129], [138,  40, 129], [140,  41, 128], [141,  41, 128],
        [143,  42, 128], [145,  42, 128], [146,  43, 128], [148,  43, 128],
        [149,  44, 128], [151,  44, 127], [153,  45, 127], [154,  45, 127],
        [156,  46, 127], [158,  46, 126], [159,  47, 126], [161,  47, 126],
        [163,  48, 126], [164,  48, 125], [166,  49, 125], [167,  49, 125],
        [169,  50, 124], [171,  51, 124], [172,  51, 123], [174,  52, 123],
        [176,  52, 123], [177,  53, 122], [179,  53, 122], [181,  54, 121],
        [182,  54, 121], [184,  55, 120], [185,  55, 120], [187,  56, 119],
        [189,  57, 119], [190,  57, 118], [192,  58, 117], [194,  58, 117],
        [195,  59, 116], [197,  60, 116], [198,  60, 115], [200,  61, 114],
        [202,  62, 114], [203,  62, 113], [205,  63, 112], [206,  64, 112],
        [208,  65, 111], [209,  66, 110], [211,  66, 109], [212,  67, 109],
        [214,  68, 108], [215,  69, 107], [217,  70, 106], [218,  71, 105],
        [220,  72, 105], [221,  73, 104], [222,  74, 103], [224,  75, 102],
        [225,  76, 102], [226,  77, 101], [228,  78, 100], [229,  80,  99],
        [230,  81,  98], [231,  82,  98], [232,  84,  97], [234,  85,  96],
        [235,  86,  96], [236,  88,  95], [237,  89,  95], [238,  91,  94],
        [238,  93,  93], [239,  94,  93], [240,  96,  93], [241,  97,  92],
        [242,  99,  92], [243, 101,  92], [243, 103,  91], [244, 104,  91],
        [245, 106,  91], [245, 108,  91], [246, 110,  91], [246, 112,  91],
        [247, 113,  91], [247, 115,  92], [248, 117,  92], [248, 119,  92],
        [249, 121,  92], [249, 123,  93], [249, 125,  93], [250, 127,  94],
        [250, 128,  94], [250, 130,  95], [251, 132,  96], [251, 134,  96],
        [251, 136,  97], [251, 138,  98], [252, 140,  99], [252, 142,  99],
        [252, 144, 100], [252, 146, 101], [252, 147, 102], [253, 149, 103],
        [253, 151, 104], [253, 153, 105], [253, 155, 106], [253, 157, 107],
        [253, 159, 108], [253, 161, 110], [253, 162, 111], [253, 164, 112],
        [254, 166, 113], [254, 168, 115], [254, 170, 116], [254, 172, 117],
        [254, 174, 118], [254, 175, 120], [254, 177, 121], [254, 179, 123],
        [254, 181, 124], [254, 183, 125], [254, 185, 127], [254, 187, 128],
        [254, 188, 130], [254, 190, 131], [254, 192, 133], [254, 194, 134],
        [254, 196, 136], [254, 198, 137], [254, 199, 139], [254, 201, 141],
        [254, 203, 142], [253, 205, 144], [253, 207, 146], [253, 209, 147],
        [253, 210, 149], [253, 212, 151], [253, 214, 152], [253, 216, 154],
        [253, 218, 156], [253, 220, 157], [253, 221, 159], [253, 223, 161],
        [253, 225, 163], [252, 227, 165], [252, 229, 166], [252, 230, 168],
        [252, 232, 170], [252, 234, 172], [252, 236, 174], [252, 238, 176],
        [252, 240, 177], [252, 241, 179], [252, 243, 181], [252, 245, 183],
        [251, 247, 185], [251, 249, 187], [251, 250, 189], [251, 252, 191],
    ])
    return MAGMA[(np.clip(x, 0., 1.) * 255).astype(np.uint8)]


def inferno(
    x: ArrayLike    # float[...]
) -> np.ndarray:    # -> uint8[..., 3]
    """
    Inferno colormap by Nathaniel J. Smith and Stefan van der Walt (see
    https://bids.github.io/colormap/).

    Discretised to 256 8-bit colours.
    """
    INFERNO = np.array([
        [  0,   0,   3], [  0,   0,   4], [  0,   0,   6], [  1,   0,   7],
        [  1,   1,   9], [  1,   1,  11], [  2,   1,  14], [  2,   2,  16],
        [  3,   2,  18], [  4,   3,  20], [  4,   3,  22], [  5,   4,  24],
        [  6,   4,  27], [  7,   5,  29], [  8,   6,  31], [  9,   6,  33],
        [ 10,   7,  35], [ 11,   7,  38], [ 13,   8,  40], [ 14,   8,  42],
        [ 15,   9,  45], [ 16,   9,  47], [ 18,  10,  50], [ 19,  10,  52],
        [ 20,  11,  54], [ 22,  11,  57], [ 23,  11,  59], [ 25,  11,  62],
        [ 26,  11,  64], [ 28,  12,  67], [ 29,  12,  69], [ 31,  12,  71],
        [ 32,  12,  74], [ 34,  11,  76], [ 36,  11,  78], [ 38,  11,  80],
        [ 39,  11,  82], [ 41,  11,  84], [ 43,  10,  86], [ 45,  10,  88],
        [ 46,  10,  90], [ 48,  10,  92], [ 50,   9,  93], [ 52,   9,  95],
        [ 53,   9,  96], [ 55,   9,  97], [ 57,   9,  98], [ 59,   9, 100],
        [ 60,   9, 101], [ 62,   9, 102], [ 64,   9, 102], [ 65,   9, 103],
        [ 67,  10, 104], [ 69,  10, 105], [ 70,  10, 105], [ 72,  11, 106],
        [ 74,  11, 106], [ 75,  12, 107], [ 77,  12, 107], [ 79,  13, 108],
        [ 80,  13, 108], [ 82,  14, 108], [ 83,  14, 109], [ 85,  15, 109],
        [ 87,  15, 109], [ 88,  16, 109], [ 90,  17, 109], [ 91,  17, 110],
        [ 93,  18, 110], [ 95,  18, 110], [ 96,  19, 110], [ 98,  20, 110],
        [ 99,  20, 110], [101,  21, 110], [102,  21, 110], [104,  22, 110],
        [106,  23, 110], [107,  23, 110], [109,  24, 110], [110,  24, 110],
        [112,  25, 110], [114,  25, 109], [115,  26, 109], [117,  27, 109],
        [118,  27, 109], [120,  28, 109], [122,  28, 109], [123,  29, 108],
        [125,  29, 108], [126,  30, 108], [128,  31, 107], [129,  31, 107],
        [131,  32, 107], [133,  32, 106], [134,  33, 106], [136,  33, 106],
        [137,  34, 105], [139,  34, 105], [141,  35, 105], [142,  36, 104],
        [144,  36, 104], [145,  37, 103], [147,  37, 103], [149,  38, 102],
        [150,  38, 102], [152,  39, 101], [153,  40, 100], [155,  40, 100],
        [156,  41,  99], [158,  41,  99], [160,  42,  98], [161,  43,  97],
        [163,  43,  97], [164,  44,  96], [166,  44,  95], [167,  45,  95],
        [169,  46,  94], [171,  46,  93], [172,  47,  92], [174,  48,  91],
        [175,  49,  91], [177,  49,  90], [178,  50,  89], [180,  51,  88],
        [181,  51,  87], [183,  52,  86], [184,  53,  86], [186,  54,  85],
        [187,  55,  84], [189,  55,  83], [190,  56,  82], [191,  57,  81],
        [193,  58,  80], [194,  59,  79], [196,  60,  78], [197,  61,  77],
        [199,  62,  76], [200,  62,  75], [201,  63,  74], [203,  64,  73],
        [204,  65,  72], [205,  66,  71], [207,  68,  70], [208,  69,  68],
        [209,  70,  67], [210,  71,  66], [212,  72,  65], [213,  73,  64],
        [214,  74,  63], [215,  75,  62], [217,  77,  61], [218,  78,  59],
        [219,  79,  58], [220,  80,  57], [221,  82,  56], [222,  83,  55],
        [223,  84,  54], [224,  86,  52], [226,  87,  51], [227,  88,  50],
        [228,  90,  49], [229,  91,  48], [230,  92,  46], [230,  94,  45],
        [231,  95,  44], [232,  97,  43], [233,  98,  42], [234, 100,  40],
        [235, 101,  39], [236, 103,  38], [237, 104,  37], [237, 106,  35],
        [238, 108,  34], [239, 109,  33], [240, 111,  31], [240, 112,  30],
        [241, 114,  29], [242, 116,  28], [242, 117,  26], [243, 119,  25],
        [243, 121,  24], [244, 122,  22], [245, 124,  21], [245, 126,  20],
        [246, 128,  18], [246, 129,  17], [247, 131,  16], [247, 133,  14],
        [248, 135,  13], [248, 136,  12], [248, 138,  11], [249, 140,   9],
        [249, 142,   8], [249, 144,   8], [250, 145,   7], [250, 147,   6],
        [250, 149,   6], [250, 151,   6], [251, 153,   6], [251, 155,   6],
        [251, 157,   6], [251, 158,   7], [251, 160,   7], [251, 162,   8],
        [251, 164,  10], [251, 166,  11], [251, 168,  13], [251, 170,  14],
        [251, 172,  16], [251, 174,  18], [251, 176,  20], [251, 177,  22],
        [251, 179,  24], [251, 181,  26], [251, 183,  28], [251, 185,  30],
        [250, 187,  33], [250, 189,  35], [250, 191,  37], [250, 193,  40],
        [249, 195,  42], [249, 197,  44], [249, 199,  47], [248, 201,  49],
        [248, 203,  52], [248, 205,  55], [247, 207,  58], [247, 209,  60],
        [246, 211,  63], [246, 213,  66], [245, 215,  69], [245, 217,  72],
        [244, 219,  75], [244, 220,  79], [243, 222,  82], [243, 224,  86],
        [243, 226,  89], [242, 228,  93], [242, 230,  96], [241, 232, 100],
        [241, 233, 104], [241, 235, 108], [241, 237, 112], [241, 238, 116],
        [241, 240, 121], [241, 242, 125], [242, 243, 129], [242, 244, 133],
        [243, 246, 137], [244, 247, 141], [245, 248, 145], [246, 250, 149],
        [247, 251, 153], [249, 252, 157], [250, 253, 160], [252, 254, 164],
    ])
    return INFERNO[(np.clip(x, 0., 1.) * 255).astype(np.uint8)]


def plasma(
    x: ArrayLike    # float[...]
) -> np.ndarray:    # -> uint8[..., 3]
    """
    Plasma colormap by Nathaniel J. Smith and Stefan van der Walt (see
    https://bids.github.io/colormap/).

    Discretised to 256 8-bit colours.
    """
    PLASMA = np.array([
        [ 12,   7, 134], [ 16,   7, 135], [ 19,   6, 137], [ 21,   6, 138],
        [ 24,   6, 139], [ 27,   6, 140], [ 29,   6, 141], [ 31,   5, 142],
        [ 33,   5, 143], [ 35,   5, 144], [ 37,   5, 145], [ 39,   5, 146],
        [ 41,   5, 147], [ 43,   5, 148], [ 45,   4, 148], [ 47,   4, 149],
        [ 49,   4, 150], [ 51,   4, 151], [ 52,   4, 152], [ 54,   4, 152],
        [ 56,   4, 153], [ 58,   4, 154], [ 59,   3, 154], [ 61,   3, 155],
        [ 63,   3, 156], [ 64,   3, 156], [ 66,   3, 157], [ 68,   3, 158],
        [ 69,   3, 158], [ 71,   2, 159], [ 73,   2, 159], [ 74,   2, 160],
        [ 76,   2, 161], [ 78,   2, 161], [ 79,   2, 162], [ 81,   1, 162],
        [ 82,   1, 163], [ 84,   1, 163], [ 86,   1, 163], [ 87,   1, 164],
        [ 89,   1, 164], [ 90,   0, 165], [ 92,   0, 165], [ 94,   0, 165],
        [ 95,   0, 166], [ 97,   0, 166], [ 98,   0, 166], [100,   0, 167],
        [101,   0, 167], [103,   0, 167], [104,   0, 167], [106,   0, 167],
        [108,   0, 168], [109,   0, 168], [111,   0, 168], [112,   0, 168],
        [114,   0, 168], [115,   0, 168], [117,   0, 168], [118,   1, 168],
        [120,   1, 168], [121,   1, 168], [123,   2, 168], [124,   2, 167],
        [126,   3, 167], [127,   3, 167], [129,   4, 167], [130,   4, 167],
        [132,   5, 166], [133,   6, 166], [134,   7, 166], [136,   7, 165],
        [137,   8, 165], [139,   9, 164], [140,  10, 164], [142,  12, 164],
        [143,  13, 163], [144,  14, 163], [146,  15, 162], [147,  16, 161],
        [149,  17, 161], [150,  18, 160], [151,  19, 160], [153,  20, 159],
        [154,  21, 158], [155,  23, 158], [157,  24, 157], [158,  25, 156],
        [159,  26, 155], [160,  27, 155], [162,  28, 154], [163,  29, 153],
        [164,  30, 152], [165,  31, 151], [167,  33, 151], [168,  34, 150],
        [169,  35, 149], [170,  36, 148], [172,  37, 147], [173,  38, 146],
        [174,  39, 145], [175,  40, 144], [176,  42, 143], [177,  43, 143],
        [178,  44, 142], [180,  45, 141], [181,  46, 140], [182,  47, 139],
        [183,  48, 138], [184,  50, 137], [185,  51, 136], [186,  52, 135],
        [187,  53, 134], [188,  54, 133], [189,  55, 132], [190,  56, 131],
        [191,  57, 130], [192,  59, 129], [193,  60, 128], [194,  61, 128],
        [195,  62, 127], [196,  63, 126], [197,  64, 125], [198,  65, 124],
        [199,  66, 123], [200,  68, 122], [201,  69, 121], [202,  70, 120],
        [203,  71, 119], [204,  72, 118], [205,  73, 117], [206,  74, 117],
        [207,  75, 116], [208,  77, 115], [209,  78, 114], [209,  79, 113],
        [210,  80, 112], [211,  81, 111], [212,  82, 110], [213,  83, 109],
        [214,  85, 109], [215,  86, 108], [215,  87, 107], [216,  88, 106],
        [217,  89, 105], [218,  90, 104], [219,  91, 103], [220,  93, 102],
        [220,  94, 102], [221,  95, 101], [222,  96, 100], [223,  97,  99],
        [223,  98,  98], [224, 100,  97], [225, 101,  96], [226, 102,  96],
        [227, 103,  95], [227, 104,  94], [228, 106,  93], [229, 107,  92],
        [229, 108,  91], [230, 109,  90], [231, 110,  90], [232, 112,  89],
        [232, 113,  88], [233, 114,  87], [234, 115,  86], [234, 116,  85],
        [235, 118,  84], [236, 119,  84], [236, 120,  83], [237, 121,  82],
        [237, 123,  81], [238, 124,  80], [239, 125,  79], [239, 126,  78],
        [240, 128,  77], [240, 129,  77], [241, 130,  76], [242, 132,  75],
        [242, 133,  74], [243, 134,  73], [243, 135,  72], [244, 137,  71],
        [244, 138,  71], [245, 139,  70], [245, 141,  69], [246, 142,  68],
        [246, 143,  67], [246, 145,  66], [247, 146,  65], [247, 147,  65],
        [248, 149,  64], [248, 150,  63], [248, 152,  62], [249, 153,  61],
        [249, 154,  60], [250, 156,  59], [250, 157,  58], [250, 159,  58],
        [250, 160,  57], [251, 162,  56], [251, 163,  55], [251, 164,  54],
        [252, 166,  53], [252, 167,  53], [252, 169,  52], [252, 170,  51],
        [252, 172,  50], [252, 173,  49], [253, 175,  49], [253, 176,  48],
        [253, 178,  47], [253, 179,  46], [253, 181,  45], [253, 182,  45],
        [253, 184,  44], [253, 185,  43], [253, 187,  43], [253, 188,  42],
        [253, 190,  41], [253, 192,  41], [253, 193,  40], [253, 195,  40],
        [253, 196,  39], [253, 198,  38], [252, 199,  38], [252, 201,  38],
        [252, 203,  37], [252, 204,  37], [252, 206,  37], [251, 208,  36],
        [251, 209,  36], [251, 211,  36], [250, 213,  36], [250, 214,  36],
        [250, 216,  36], [249, 217,  36], [249, 219,  36], [248, 221,  36],
        [248, 223,  36], [247, 224,  36], [247, 226,  37], [246, 228,  37],
        [246, 229,  37], [245, 231,  38], [245, 233,  38], [244, 234,  38],
        [243, 236,  38], [243, 238,  38], [242, 240,  38], [242, 241,  38],
        [241, 243,  38], [240, 245,  37], [240, 246,  35], [239, 248,  33],
    ])
    return PLASMA[(np.clip(x, 0., 1.) * 255).astype(np.uint8)]


def viridis(
    x: ArrayLike    # float[...]
) -> np.ndarray:    # -> uint8[..., 3]
    """
    Viridis colormap by Nathaniel J. Smith, Stefan van der Walt, and Eric
    Firing (see https://bids.github.io/colormap/).

    Discretised to 256 8-bit colours.
    """
    VIRIDIS = np.array([
        [ 68,   1,  84], [ 68,   2,  85], [ 68,   3,  87], [ 69,   5,  88],
        [ 69,   6,  90], [ 69,   8,  91], [ 70,   9,  92], [ 70,  11,  94],
        [ 70,  12,  95], [ 70,  14,  97], [ 71,  15,  98], [ 71,  17,  99],
        [ 71,  18, 101], [ 71,  20, 102], [ 71,  21, 103], [ 71,  22, 105],
        [ 71,  24, 106], [ 72,  25, 107], [ 72,  26, 108], [ 72,  28, 110],
        [ 72,  29, 111], [ 72,  30, 112], [ 72,  32, 113], [ 72,  33, 114],
        [ 72,  34, 115], [ 72,  35, 116], [ 71,  37, 117], [ 71,  38, 118],
        [ 71,  39, 119], [ 71,  40, 120], [ 71,  42, 121], [ 71,  43, 122],
        [ 71,  44, 123], [ 70,  45, 124], [ 70,  47, 124], [ 70,  48, 125],
        [ 70,  49, 126], [ 69,  50, 127], [ 69,  52, 127], [ 69,  53, 128],
        [ 69,  54, 129], [ 68,  55, 129], [ 68,  57, 130], [ 67,  58, 131],
        [ 67,  59, 131], [ 67,  60, 132], [ 66,  61, 132], [ 66,  62, 133],
        [ 66,  64, 133], [ 65,  65, 134], [ 65,  66, 134], [ 64,  67, 135],
        [ 64,  68, 135], [ 63,  69, 135], [ 63,  71, 136], [ 62,  72, 136],
        [ 62,  73, 137], [ 61,  74, 137], [ 61,  75, 137], [ 61,  76, 137],
        [ 60,  77, 138], [ 60,  78, 138], [ 59,  80, 138], [ 59,  81, 138],
        [ 58,  82, 139], [ 58,  83, 139], [ 57,  84, 139], [ 57,  85, 139],
        [ 56,  86, 139], [ 56,  87, 140], [ 55,  88, 140], [ 55,  89, 140],
        [ 54,  90, 140], [ 54,  91, 140], [ 53,  92, 140], [ 53,  93, 140],
        [ 52,  94, 141], [ 52,  95, 141], [ 51,  96, 141], [ 51,  97, 141],
        [ 50,  98, 141], [ 50,  99, 141], [ 49, 100, 141], [ 49, 101, 141],
        [ 49, 102, 141], [ 48, 103, 141], [ 48, 104, 141], [ 47, 105, 141],
        [ 47, 106, 141], [ 46, 107, 142], [ 46, 108, 142], [ 46, 109, 142],
        [ 45, 110, 142], [ 45, 111, 142], [ 44, 112, 142], [ 44, 113, 142],
        [ 44, 114, 142], [ 43, 115, 142], [ 43, 116, 142], [ 42, 117, 142],
        [ 42, 118, 142], [ 42, 119, 142], [ 41, 120, 142], [ 41, 121, 142],
        [ 40, 122, 142], [ 40, 122, 142], [ 40, 123, 142], [ 39, 124, 142],
        [ 39, 125, 142], [ 39, 126, 142], [ 38, 127, 142], [ 38, 128, 142],
        [ 38, 129, 142], [ 37, 130, 142], [ 37, 131, 141], [ 36, 132, 141],
        [ 36, 133, 141], [ 36, 134, 141], [ 35, 135, 141], [ 35, 136, 141],
        [ 35, 137, 141], [ 34, 137, 141], [ 34, 138, 141], [ 34, 139, 141],
        [ 33, 140, 141], [ 33, 141, 140], [ 33, 142, 140], [ 32, 143, 140],
        [ 32, 144, 140], [ 32, 145, 140], [ 31, 146, 140], [ 31, 147, 139],
        [ 31, 148, 139], [ 31, 149, 139], [ 31, 150, 139], [ 30, 151, 138],
        [ 30, 152, 138], [ 30, 153, 138], [ 30, 153, 138], [ 30, 154, 137],
        [ 30, 155, 137], [ 30, 156, 137], [ 30, 157, 136], [ 30, 158, 136],
        [ 30, 159, 136], [ 30, 160, 135], [ 31, 161, 135], [ 31, 162, 134],
        [ 31, 163, 134], [ 32, 164, 133], [ 32, 165, 133], [ 33, 166, 133],
        [ 33, 167, 132], [ 34, 167, 132], [ 35, 168, 131], [ 35, 169, 130],
        [ 36, 170, 130], [ 37, 171, 129], [ 38, 172, 129], [ 39, 173, 128],
        [ 40, 174, 127], [ 41, 175, 127], [ 42, 176, 126], [ 43, 177, 125],
        [ 44, 177, 125], [ 46, 178, 124], [ 47, 179, 123], [ 48, 180, 122],
        [ 50, 181, 122], [ 51, 182, 121], [ 53, 183, 120], [ 54, 184, 119],
        [ 56, 185, 118], [ 57, 185, 118], [ 59, 186, 117], [ 61, 187, 116],
        [ 62, 188, 115], [ 64, 189, 114], [ 66, 190, 113], [ 68, 190, 112],
        [ 69, 191, 111], [ 71, 192, 110], [ 73, 193, 109], [ 75, 194, 108],
        [ 77, 194, 107], [ 79, 195, 105], [ 81, 196, 104], [ 83, 197, 103],
        [ 85, 198, 102], [ 87, 198, 101], [ 89, 199, 100], [ 91, 200,  98],
        [ 94, 201,  97], [ 96, 201,  96], [ 98, 202,  95], [100, 203,  93],
        [103, 204,  92], [105, 204,  91], [107, 205,  89], [109, 206,  88],
        [112, 206,  86], [114, 207,  85], [116, 208,  84], [119, 208,  82],
        [121, 209,  81], [124, 210,  79], [126, 210,  78], [129, 211,  76],
        [131, 211,  75], [134, 212,  73], [136, 213,  71], [139, 213,  70],
        [141, 214,  68], [144, 214,  67], [146, 215,  65], [149, 215,  63],
        [151, 216,  62], [154, 216,  60], [157, 217,  58], [159, 217,  56],
        [162, 218,  55], [165, 218,  53], [167, 219,  51], [170, 219,  50],
        [173, 220,  48], [175, 220,  46], [178, 221,  44], [181, 221,  43],
        [183, 221,  41], [186, 222,  39], [189, 222,  38], [191, 223,  36],
        [194, 223,  34], [197, 223,  33], [199, 224,  31], [202, 224,  30],
        [205, 224,  29], [207, 225,  28], [210, 225,  27], [212, 225,  26],
        [215, 226,  25], [218, 226,  24], [220, 226,  24], [223, 227,  24],
        [225, 227,  24], [228, 227,  24], [231, 228,  25], [233, 228,  25],
        [236, 228,  26], [238, 229,  27], [241, 229,  28], [243, 229,  30],
        [246, 230,  31], [248, 230,  33], [250, 230,  34], [253, 231,  36],
    ])
    return VIRIDIS[(np.clip(x, 0., 1.) * 255).astype(np.uint8)]


# # # 
# DISCRETE COLORMAPS


def sweetie16(
    x: ArrayLike    # int[...]
) -> np.ndarray:    # -> uint8[..., 3]
    """
    Sweetie-16 colour palette by GrafxKid (see
    https://lospec.com/palette-list/sweetie-16).

    Input should be an array of indices in the range [0,15].
    """
    return np.array([
        [ 26,  28,  44], [ 93,  39,  93], [177,  62,  83], [239, 125,  87],
        [255, 205, 117], [167, 240, 112], [ 56, 183, 100], [ 37, 113, 121],
        [ 41,  54, 111], [ 59,  93, 201], [ 65, 166, 246], [115, 239, 247],
        [244, 244, 244], [148, 176, 194], [ 86, 108, 134], [ 51,  60,  87],
    ])[x]


def pico8(
    x: ArrayLike    # int[...]
) -> np.ndarray:    # -> uint8[..., 3]
    """
    PICO-8 colour palette (see https://pico-8.fandom.com/wiki/Palette).
    
    Input should be an array of indices in the range [0,15].
    """
    return np.array([
        [  0,   0,   0], [ 29,  43,  83], [126,  37,  83], [  0, 135,  81],
        [171,  82,  54], [ 95,  87,  79], [194, 195, 199], [255, 241, 232],
        [255,   0,  77], [255, 163,   0], [255, 236,  39], [  0, 228,  54],
        [ 41, 173, 255], [131, 118, 156], [255, 119, 168], [255, 204, 170],
    ])[x]


# # # 
# BRAILLE HELPER FUNCTIONS


BRAILLE_MAP = np.array([
    [0, 3],
    [1, 4],
    [2, 5],
    [6, 7],
], dtype=np.uint8)


def braille_encode(a: ArrayLike) -> np.ndarray:
    """
    Turns a HxW array of booleans into a (H//4)x(W//2) array of braille
    binary codes.

    Inputs:

    * 

    Here is a visual explanation of this function:

    ```
    Start with an array with height divisible by 4, width divisible by 2:
        ____
       [1  0] 0  1  0  1  1  1  1  0  1  0  0  0  0  1  0  0  0  0  0  1  1  0
       [1  0] 0  1  0  1  0  0  0  0  1  0  0  0  0  1  0  0  0  0  1  0  0  1
     .-[1  0] 0  1  0  1  0  0  0  0  1  0  0  0  0  1  0  0  0  0  1  0  0  1
     | [1__0] 0  1  0  1  0  0  0  0  1  0  0  0  0  1  0  0  0  0  1  0  0  1
     |  1  1  1  1  0  1  1  1  1  0  1  0  0  0  0  1  0  0  0  0  1  0  0  1
     |  1  0  0  1  0  1  0  0  0  0  1  0  0  0  0  1  0  0  0  0  1  0  0  1
     |  1  0  0  1  0  1  0  0  0  0  1  0  0  0  0  1  0  0  0  0  1  0  0  1
     |  1  0  0  1  0  1  1  1  1  0  1  1  1  1  0  1  1  1  1  0  0  1  1  0
     |
     | take each 4x2 subarray and ...
     |                                                               braille
     | identify the 4x2 bits with the                                unicode
     | eight numbered braille dots:                                  start pt
     |                                                               |
     |  (dot 1) 1 0 (dot 4)     convert to                           v
     `> (dot 2) 1 0 (dot 5) -----------------> 0 b 0 1 0 0 0 1 1 1 + 0x2800 -.
        (dot 3) 1 0 (dot 6)    braille code        | | | | | | | |           |
        (dot 7) 1 0 (dot 8)                    dot 8 7 6 5 4 3 2 1           |
                                                                             |
      convert the braille code to a unicode character and collate into array |
     .-----------------------------------------------------------------------'
     |  '''
     `->⡇⢸⢸⠉⠁⡇⠀⢸⠀⠀⡎⢱  (Note: this function returns codepoints, use `chr()`
        ⡏⢹⢸⣉⡁⣇⣀⢸⣀⡀⢇⡸  to convert these into braille characters for printing.)
        '''
    ```
    """
    # process input
    array = np.asarray(a, dtype=bool)
    H, W = array.shape
    h, w = H // 4, W // 2
    # create a view that chunks it into 4x2 cells
    cells = array.reshape(h, 4, w, 2)
    # convert each bit in each cell into a mask and combine into code array
    masks = np.left_shift(cells, BRAILLE_MAP.reshape(1,4,1,2), dtype=np.uint16)
    codes = np.bitwise_or.reduce(masks, axis=(1,3))
    # unicode braille block starts at 0x2800
    unicodes = 0x2800 + codes
    return unicodes


# # # 
# ANSI RENDERING HELP


def to_ansi_str(char: Char) -> str:
    """
    If necessary, wrap a Char in ANSI control codes that switch the color into
    the given fg and bg colors; plus a control code to switch back to default
    mode.
    """
    ansi_controls = []
    if char.fg is not None:
        ansi_controls.extend([38, 2, *char.fg])
    if char.bg is not None:
        ansi_controls.extend([48, 2, *char.bg])
    if ansi_controls:
        return f"\x1b[{";".join(map(str, ansi_controls))}m{char.c}\x1b[0m"
    else:
        return char.c


# # # 
# PNG RENDERING HELP


def to_rgba_array(char: Char) -> np.ndarray: # uint8[16,8,4]
    """
    Convert a Char to a small RGBA image patch, with the specified foreground
    color (or white) and background color (or a transparent background).
    """
    # bitmap : b[16, 8]
    rows = np.array(FONT_UNSCII_16.get_char(char.c), dtype=np.uint8)
    bits = np.unpackbits(rows[:,None], axis=1, bitorder='big').astype(bool)
    
    # colors
    if char.fg is not None:
        fg = np.array([*char.fg, 255], dtype=np.uint8)
    else:
        fg = np.array([255, 255, 255, 255], dtype=np.uint8)
    if char.bg is not None:
        bg = np.array([*char.bg, 255], dtype=np.uint8)
    else:
        bg = np.array([0, 0, 0, 0], dtype=np.uint8)

    # rgb array : uint8[16,8,4]
    img = np.where(bits[..., np.newaxis], fg, bg)

    return img


FONT_UNSCII_16 = unscii.UnsciiFont("unscii_16")


