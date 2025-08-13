"""
Dead-simple terminal plotting library by matthew.
"""

import math
import os
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike
import einops
import hilbert as _hilbert

from PIL import Image
import unscii


# # # 
# COLOURED CHARACTER UTILITY CLASS


class colorchar:
    """
    Class representing a possibly-coloured character. Provides methods for
    converting to a string, for use in rendering.
    """
    def __init__(
        self,
        character: str = " ",
        fgcolor: ArrayLike | None = None, # float[3] (rgb 0 to 1)
        bgcolor: ArrayLike | None = None, # float[3] (rgb 0 to 1)
    ):
        self.c = character
        if fgcolor is not None:
            self.fg = np.asarray(fgcolor)
        else:
            self.fg = None
        if bgcolor is not None:
            self.bg = np.asarray(bgcolor)
        else:
            self.bg = None


    def __str__(self) -> str:
        """
        If necessary, issue ANSI control codes that switch the color into the
        given fg and bg colors. Then print the character. Then, if necessary,
        switch back to default mode.
        """
        # needs fg?
        if self.fg is not None:
            fgr, fgg, fgb = (255 * self.fg).astype(np.uint8)
            fgcode = f"\033[38;2;{fgr};{fgg};{fgb}m"
        else:
            fgcode = ""
        # needs bg?
        if self.bg is not None:
            bgr, bgg, bgb = (255 * self.bg).astype(np.uint8)
            bgcode = f"\033[48;2;{bgr};{bgg};{bgb}m"
        else:
            bgcode = ""
        # needs reset?
        if self.fg is not None or self.bg is not None:
            rcode = "\033[0m"
        else:
            rcode = ""
        return f"{fgcode}{bgcode}{self.c}{rcode}"


    def to_rgba_array(self) -> ArrayLike: # u8[16,8,4]
        # bitmap
        char = FONT_UNSCII_16.get_char(self.c)
        rows = np.array(char, dtype=np.uint8).reshape(16, 1)
        bits = np.unpackbits(rows, axis=1, bitorder='big').astype(bool)
        # -> b[16, 8]
        
        # rgb array
        if self.fg is not None:
            fg = np.array([*(255*self.fg), 255], dtype=np.uint8)
        else:
            fg = np.array([255, 255, 255, 255], dtype=np.uint8)
        if self.bg is not None:
            bg = np.array([*(255*self.bg), 255], dtype=np.uint8)
        else:
            bg = np.array([  0,   0,   0,   0], dtype=np.uint8)
        img = np.where(bits[..., np.newaxis], fg, bg)
        # -> u8[16,8,4]

        return img

    
    def __bool__(self):
        return bool(
            self.c.strip()
            or self.fg is not None
            or self.bg is not None
        )


BLANK = colorchar(character=" ")


# # # 
# PLOT BASE CLASS


class plot:
    """
    Base class representing a 2d character array as a list of lines.
    Provides methods for converting to a string, along with operations
    for horizontal (|), vertical (^), and distal (&) stacking.
    """
    def __init__(self, array: list[colorchar]):
        self.array = array

    @property
    def height(self):
        return len(self.array)

    @property
    def width(self):
        return len(self.array[0])

    def __str__(self) -> str:
        return "\n".join(["".join([str(c) for c in l]) for l in self.array])

    def saveimg(self, filename: str, scale_factor: int = 1):
        tiles = np.asarray([[c.to_rgba_array() for c in l] for l in self.array])
        # -> u8[H, W, 16, 8, 4]
        stacked = einops.rearrange(tiles, 'H W h w rgba -> (H h) (W w) rgba')
        image = Image.fromarray(stacked, mode='RGBA')
        image.save(filename)

    def __or__(self, other):
        return hstack(self, other)

    def __xor__(self, other):
        return vstack(self, other)
    
    def __and__(self, other):
        return dstack(self, other)


# # # 
# DATA PLOTTING CLASSES


class image(plot):
    """
    Render a small image using a grid of unicode half-characters with
    different foreground and background colours to represent pairs of
    pixels.

    TODO: document input and colormap formats.
    """
    def __init__(
        self,
        im,
        colormap=None,
    ):
        # preprocessing: all inputs become float[h, w, rgb] with even h, w
        im = np.asarray(im)
        if len(im.shape) == 2 and colormap is None:
            # greyscale or indexed and no colormap -> uniform colourisation
            im = einops.repeat(im, 'h w -> h w 3')
        elif colormap is not None:
            # indexed, greyscale, or rgb and compatible colormap -> mapped rgb
            im = colormap(im)
        # pad to even height and width (width is not strictly necessary)
        im = np.pad(
            array=im,
            pad_width=(
                (0, im.shape[0] % 2),
                (0, im.shape[1] % 2),
                (0, 0),
            ),
            mode='constant',
            constant_values=0.,
        )

        # processing: stack into fg/bg format
        stacked = einops.rearrange(im, '(h fgbg) w c -> h w fgbg c', fgbg=2)

        # render the image lines as unicode strings with ansi color codes
        array = [
            [colorchar("▀", fg, bg) for fg, bg in row]
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
        F: Callable[[ArrayLike], ArrayLike], # TODO: vectorise herein?
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
            zrange = (zgrid.min(), zgrid.max() + 1e-6)
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
    """
    def __init__(
        self,
        data: ArrayLike, # float[n, 2]
        height: int = 10,
        width: int = 30,
        yrange: tuple[float, float] | None = None,
        xrange: tuple[float, float] | None = None,
        color: ArrayLike | None = None,            # float[3] (rgb 0 to 1)
        check_bounds: bool = False,
    ):
        # preprocess and check shape
        data = np.asarray(data)
        n, _2 = data.shape
        assert _2 == 2

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
                if bgrid[i, j]:
                    braille_char = chr(0x2800+bgrid[i, j])
                    array[i][j] = colorchar(braille_char, color)
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
    Render a list of bools long a hilbert curve using a grid of braille unicode
    characters.
    """
    def __init__(
        self,
        data: ArrayLike,                    # bool[N]
        dotcolor: ArrayLike | None = None,  # float[3] (rgb 0 to 1)
        bgcolor: ArrayLike | None = None,   # float[3] (rgb 0 to 1)
        nullcolor: ArrayLike | None = None, # float[3] (rgb 0 to 1)
    ):
        # preprocess and compute grid shape
        data = np.asarray(data)
        N, = data.shape
        n = max(2, ((N-1).bit_length() + 1) // 2)

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
        null = colorchar(" ", bgcolor=nullcolor)
        array = [[null for _ in range(width)] for _ in range(height)]
        bg_grid = braille_encode(all_grid)
        fg_grid = braille_encode(lit_grid)
        for i in range(height):
            for j in range(width):
                if bg_grid[i, j]:
                    braille_char = chr(0x2800+fg_grid[i, j])
                    array[i][j] = colorchar(
                        character=braille_char,
                        fgcolor=dotcolor,
                        bgcolor=bgcolor,
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
    One or more lines of ASCII text.
    TODO: allow alignment and resizing.
    TODO: account for non-printable and wide characters.
    """
    def __init__(
        self,
        text: str,
        color: ArrayLike | None = None,            # float[3] (rgb 0 to 1)
        bgcolor: ArrayLike | None = None,          # float[3] (rgb 0 to 1)
    ):
        lines = text.splitlines()
        height = len(lines)
        width = max(len(line) for line in lines)
        array = [
            [colorchar(c, color, bgcolor) for c in line]
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
    A progress bar.
    """
    def __init__(
        self,
        progress: float,
        width: int = 40,
        color: ArrayLike | None = None,            # float[3] (rgb 0 to 1)
    ):
        progress = np.clip(progress, 0., 1.)
        # construct label
        label = f"{progress:4.0%}"
        label_chars = [colorchar(c) for c in label]
        # construct bar
        bar_width = width - 2 - len(label)
        fill_width = bar_width * progress
        bar_chars = [colorchar("█", color)] * int(fill_width)
        marginal_width = int(8 * (fill_width % 1))
        if marginal_width > 0:
            bar_chars.append(colorchar(
                    [None, "▏", "▎", "▍", "▌", "▋", "▊", "▉"][marginal_width],
                    color,
            ))
        bar_chars.extend(
            [BLANK] * (bar_width - len(bar_chars))
        )
        # put it together
        array = [
            [*label_chars, colorchar("["), *bar_chars, colorchar("]")]
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
    A rectangle of blank space.
    """
    def __init__(self, height: int, width: int):
        array = [[BLANK] * width] * height
        super().__init__(array)

    def __repr__(self):
        return f"blank(height={self.height}, width={self.width})"


class hstack(plot):
    """
    Horizontally arrange a group of plots.
    """
    def __init__(self, *plots):
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
    Vertically arrange a group of plots.
    """
    def __init__(self, *plots):
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
    Distally arrange a group of plots.
    """
    def __init__(self, *plots):
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
    Horizontally and vertically arrange a group of plots.
    """
    def __init__(self, *plots, cols=None):
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
    Put a unicode border around a plot.
    """
    class Style:
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
        color: ArrayLike | None = None,            # float[3] (rgb 0 to 1)
    ):
        if style is None:
            style = self.Style.ROUND
        array = [
            # top row
            [
                colorchar(style[2], color),
                *[colorchar(style[0], color)] * plot.width,
                colorchar(style[3], color),
            ],
            # middle rows
            *[
                [
                    colorchar(style[1], color),
                    *row,
                    colorchar(style[1], color),
                ]
                for row in plot.array
            ],
            # bottom row
            [
                colorchar(style[4], color),
                *[colorchar(style[0], color)] * plot.width,
                colorchar(style[5], color),
            ],
        ]
        super().__init__(array)
        self.style = style[2]
        self.plot = plot
    
    def __repr__(self):
        return f"border(style={self.style!r}, plot={self.plot!r})"


class center(plot):
    """
    Put blank space around a plot.
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
# COLORMAPS


def reds(x):
    """
    Red colormap. Simply embeds greyscale value into red channel.
    """
    x = np.asarray(x)
    rgb = np.zeros((*x.shape, 3))
    rgb[..., 0] = x
    return rgb


def greens(x):
    """
    Green colormap. Simply embeds greyscale value into green channel.
    """
    x = np.asarray(x)
    rgb = np.zeros((*x.shape, 3))
    rgb[..., 1] = x
    return rgb


def blues(x):
    """
    Blue colormap. Simply embeds greyscale value into blue channel.
    """
    x = np.asarray(x)
    rgb = np.zeros((*x.shape, 3))
    rgb[..., 2] = x
    return rgb


def yellows(x):
    """
    Yellow colormap. Simply embeds greyscale value into red and green
    channels.
    """
    x = np.asarray(x)
    rgb = np.zeros((*x.shape, 3))
    rgb[..., 0] = x
    rgb[..., 1] = x
    return rgb


def magentas(x):
    """
    Magenta colormap. Simply embeds greyscale value into red and blue
    channels.
    """
    x = np.asarray(x)
    rgb = np.zeros((*x.shape, 3))
    rgb[..., 0] = x
    rgb[..., 2] = x
    return rgb


def cyans(x):
    """
    Cyan colormap. Simply embeds greyscale value into green and blue
    channels.
    """
    x = np.asarray(x)
    rgb = np.zeros((*x.shape, 3))
    rgb[..., 1] = x
    rgb[..., 2] = x
    return rgb


def cool(x):
    """
    Cool colormap. Embeds greyscale value into interpolating between cyan and
    magenta.
    """
    x = np.asarray(x)
    rgb = np.zeros((*x.shape, 3))
    rgb[..., 0] = x
    rgb[..., 1] = 1-x
    rgb[..., 2] = 1
    return rgb


def rainbow(x):
    """
    Rainbow colormap. Effectively embeds greyscale values as hue in HSV color
    space.
    """
    x = np.asarray(x)
    rgb = np.zeros((*x.shape, 3))
    # map [0, 1] into (i, r) coordinates where
    # * i is in {0, 1, 2, 3, 4, 5}, and
    # * r is in [0, 1].
    t = 6 * x
    i = t.astype(int) % 6
    r = t % 1
    # for each i, project t into the appropriate colour channels
    # i = 0 (red to red+green)
    i0 = (i == 0)
    rgb[i0, 0] = 1.
    rgb[i0, 1] = r[i0]
    # i = 1 (red+green to green)
    i1 = (i == 1)
    rgb[i1, 0] = 1. - r[i1]
    rgb[i1, 1] = 1.
    # i = 2 (green to green+blue)
    i2 = (i == 2)
    rgb[i2, 1] = 1.
    rgb[i2, 2] = r[i2]
    # i = 3 (green+blue to blue)
    i3 = (i == 3)
    rgb[i3, 1] = 1. - r[i3]
    rgb[i3, 2] = 1.
    # i = 4 (blue to blue+red)
    i4 = (i == 4)
    rgb[i4, 2] = 1.
    rgb[i4, 0] = r[i4]
    # i = 5 (blue+red to red)
    i5 = (i == 5)
    rgb[i5, 2] = 1. - r[i5]
    rgb[i5, 0] = 1.
    # done
    return rgb


def viridis(x):
    """
    Viridis colormap.

    Details: https://youtu.be/xAoljeRJ3lU
    """
    return np.array([
        [.267,.004,.329],[.268,.009,.335],[.269,.014,.341],[.271,.019,.347],
        [.272,.025,.353],[.273,.031,.358],[.274,.037,.364],[.276,.044,.370],
        [.277,.050,.375],[.277,.056,.381],[.278,.062,.386],[.279,.067,.391],
        [.280,.073,.397],[.280,.078,.402],[.281,.084,.407],[.281,.089,.412],
        [.282,.094,.417],[.282,.100,.422],[.282,.105,.426],[.283,.110,.431],
        [.283,.115,.436],[.283,.120,.440],[.283,.125,.444],[.283,.130,.449],
        [.282,.135,.453],[.282,.140,.457],[.282,.145,.461],[.281,.150,.465],
        [.281,.155,.469],[.280,.160,.472],[.280,.165,.476],[.279,.170,.479],
        [.278,.175,.483],[.278,.180,.486],[.277,.185,.489],[.276,.190,.493],
        [.275,.194,.496],[.274,.199,.498],[.273,.204,.501],[.271,.209,.504],
        [.270,.214,.507],[.269,.218,.509],[.267,.223,.512],[.266,.228,.514],
        [.265,.232,.516],[.263,.237,.518],[.262,.242,.520],[.260,.246,.522],
        [.258,.251,.524],[.257,.256,.526],[.255,.260,.528],[.253,.265,.529],
        [.252,.269,.531],[.250,.274,.533],[.248,.278,.534],[.246,.283,.535],
        [.244,.287,.537],[.243,.292,.538],[.241,.296,.539],[.239,.300,.540],
        [.237,.305,.541],[.235,.309,.542],[.233,.313,.543],[.231,.318,.544],
        [.229,.322,.545],[.227,.326,.546],[.225,.330,.547],[.223,.334,.548],
        [.221,.339,.548],[.220,.343,.549],[.218,.347,.550],[.216,.351,.550],
        [.214,.355,.551],[.212,.359,.551],[.210,.363,.552],[.208,.367,.552],
        [.206,.371,.553],[.204,.375,.553],[.203,.379,.553],[.201,.383,.554],
        [.199,.387,.554],[.197,.391,.554],[.195,.395,.555],[.194,.399,.555],
        [.192,.403,.555],[.190,.407,.556],[.188,.410,.556],[.187,.414,.556],
        [.185,.418,.556],[.183,.422,.556],[.182,.426,.557],[.180,.429,.557],
        [.179,.433,.557],[.177,.437,.557],[.175,.441,.557],[.174,.445,.557],
        [.172,.448,.557],[.171,.452,.557],[.169,.456,.558],[.168,.459,.558],
        [.166,.463,.558],[.165,.467,.558],[.163,.471,.558],[.162,.474,.558],
        [.160,.478,.558],[.159,.482,.558],[.157,.485,.558],[.156,.489,.557],
        [.154,.493,.557],[.153,.497,.557],[.151,.500,.557],[.150,.504,.557],
        [.149,.508,.557],[.147,.511,.557],[.146,.515,.556],[.144,.519,.556],
        [.143,.522,.556],[.141,.526,.555],[.140,.530,.555],[.139,.533,.555],
        [.137,.537,.554],[.136,.541,.554],[.135,.544,.554],[.133,.548,.553],
        [.132,.552,.553],[.131,.555,.552],[.129,.559,.551],[.128,.563,.551],
        [.127,.566,.550],[.126,.570,.549],[.125,.574,.549],[.124,.578,.548],
        [.123,.581,.547],[.122,.585,.546],[.121,.589,.545],[.121,.592,.544],
        [.120,.596,.543],[.120,.600,.542],[.119,.603,.541],[.119,.607,.540],
        [.119,.611,.538],[.119,.614,.537],[.119,.618,.536],[.120,.622,.534],
        [.120,.625,.533],[.121,.629,.531],[.122,.633,.530],[.123,.636,.528],
        [.124,.640,.527],[.126,.644,.525],[.128,.647,.523],[.130,.651,.521],
        [.132,.655,.519],[.134,.658,.517],[.137,.662,.515],[.140,.665,.513],
        [.143,.669,.511],[.146,.673,.508],[.150,.676,.506],[.153,.680,.504],
        [.157,.683,.501],[.162,.687,.499],[.166,.690,.496],[.170,.694,.493],
        [.175,.697,.491],[.180,.701,.488],[.185,.704,.485],[.191,.708,.482],
        [.196,.711,.479],[.202,.715,.476],[.208,.718,.472],[.214,.722,.469],
        [.220,.725,.466],[.226,.728,.462],[.232,.732,.459],[.239,.735,.455],
        [.246,.738,.452],[.252,.742,.448],[.259,.745,.444],[.266,.748,.440],
        [.274,.751,.436],[.281,.755,.432],[.288,.758,.428],[.296,.761,.424],
        [.304,.764,.419],[.311,.767,.415],[.319,.770,.411],[.327,.773,.406],
        [.335,.777,.402],[.344,.780,.397],[.352,.783,.392],[.360,.785,.387],
        [.369,.788,.382],[.377,.791,.377],[.386,.794,.372],[.395,.797,.367],
        [.404,.800,.362],[.412,.803,.357],[.421,.805,.351],[.430,.808,.346],
        [.440,.811,.340],[.449,.813,.335],[.458,.816,.329],[.468,.818,.323],
        [.477,.821,.318],[.487,.823,.312],[.496,.826,.306],[.506,.828,.300],
        [.515,.831,.294],[.525,.833,.288],[.535,.835,.281],[.545,.838,.275],
        [.555,.840,.269],[.565,.842,.262],[.575,.844,.256],[.585,.846,.249],
        [.595,.848,.243],[.606,.850,.236],[.616,.852,.230],[.626,.854,.223],
        [.636,.856,.216],[.647,.858,.209],[.657,.860,.203],[.668,.861,.196],
        [.678,.863,.189],[.688,.865,.182],[.699,.867,.175],[.709,.868,.169],
        [.720,.870,.162],[.730,.871,.156],[.741,.873,.149],[.751,.874,.143],
        [.762,.876,.137],[.772,.877,.131],[.783,.879,.125],[.793,.880,.120],
        [.804,.882,.114],[.814,.883,.110],[.824,.884,.106],[.835,.886,.102],
        [.845,.887,.099],[.855,.888,.097],[.866,.889,.095],[.876,.891,.095],
        [.886,.892,.095],[.896,.893,.096],[.906,.894,.098],[.916,.896,.100],
        [.926,.897,.104],[.935,.898,.108],[.945,.899,.112],[.955,.901,.118],
        [.964,.902,.123],[.974,.903,.130],[.983,.904,.136],[.993,.906,.143],
    ])[(np.clip(x, 0., 1.) * (255)).astype(int)]


def sweetie16(x):
    """
    Sweetie-16 colour palette.

    Details: https://lospec.com/palette-list/sweetie-16
    """
    return np.array([
        [.101,.109,.172],[.364,.152,.364],[.694,.243,.325],[.937,.490,.341],
        [.999,.803,.458],[.654,.941,.439],[.219,.717,.392],[.145,.443,.474],
        [.160,.211,.435],[.231,.364,.788],[.254,.650,.964],[.450,.937,.968],
        [.956,.956,.956],[.580,.690,.760],[.337,.423,.525],[.2  ,.235,.341],
    ])[x]


def pico8(x):
    """
    PICO-8 colour palette.

    Details: https://pico-8.fandom.com/wiki/Palette
    """
    return (np.array([
        [  0,   0,   0], [ 29,  43,  83], [126,  37,  83], [  0, 135,  81],
        [171,  82,  54], [ 95,  87,  79], [194, 195, 199], [255, 241, 232],
        [255,   0,  77], [255, 163,   0], [255, 236,  39], [  0, 228,  54],
        [ 41, 173, 255], [131, 118, 156], [255, 119, 168], [255, 204, 170],
    ]) / 255)[x]


# # # 
# UNICODE HELPER FUNCTIONS


def braille_encode(a):
    """
    Turns a HxW array of booleans into a (H//4)x(W//2) array of braille
    binary codes (suitable for specifying unicode codepoints, just add
    0x2800).
    
    braille symbol:                 binary digit representation:
                    0-o o-1
                    2-o o-3   ---->     0 b  0 0  0 0 0  0 0 0
                    4-o o-5                  | |  | | |  | | |
                    6-o o-7                  7 6  5 3 1  4 2 0
    """
    r = einops.rearrange(a, '(h h4) (w w2) -> (h4 w2) h w', h4=4, w2=2)
    b = (
          r[0]      | r[1] << 3 
        | r[2] << 1 | r[3] << 4 
        | r[4] << 2 | r[5] << 5 
        | r[6] << 6 | r[7] << 7
    )
    return b


# # # 
# FONT STUFF

FONT_UNSCII_16 = unscii.UnsciiFont("unscii_16")
