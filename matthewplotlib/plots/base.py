"""
The base class every plot type inherits, and the arrangement plots that
compose plots into larger ones: stacked across the screen or layered in
depth, wrapped into a grid, centred, or cropped.

The two live together because they meet in the base class's operators:
`+`, `/`, `|`, and `@` are shortcuts for the stacking classes.

* `plot`: the base class. See it for the methods, properties, and shortcut
  operators available with every plot object.
* `blank`
* `hstack`
* `vstack`
* `dstack`
* `dstack2`
* `wrap`
* `center`
* `crop`
"""
from __future__ import annotations

import shutil

import einops
import numpy as np

from PIL import Image

from typing import Self
from matthewplotlib.colors import ColorLike
from matthewplotlib.terminal import terminal_size
from matthewplotlib.window import window
from matthewplotlib.core import (
    CharArray,
    _validate_text,
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
        See the `erase-granularity` note.
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


class crop(plot):
    """
    Limit a plot's extent to a given size, marking the edges it cut off.

    The top left rectangle of the given size is kept. Wherever content was cut
    off, the last row or column of the result is given over to a marker
    character, so that a truncated plot cannot be mistaken for a whole one.
    That row or column costs content: cropping a 10 row plot to 8 rows shows 7
    of them and a row of markers.

    A `height` or `width` that the plot already fits within leaves that
    direction alone. These are maximum sizes, not target sizes; `center` is the
    tool for the other direction.

    Inputs:

    * plot : plot.
        The plot object to be cropped.
    * height : optional int (>0).
        The maximum height of the result. Defaults to one row less than the
        attached terminal's, the tallest plot it can animate: printing H rows
        plus the newline `print` appends takes H+1 rows.
    * width : optional int (>0).
        The maximum width of the result. Defaults to the attached terminal's
        width, the widest plot that does not wrap.
    * marker : optional str (default: `"#"`).
        The single character marking each cut edge. `None` marks nothing and
        keeps the full rectangle of content instead, a plain slice that leaves
        no sign anything was cut.
    * fgcolor : optional ColorLike.
        The color of the markers. Defaults to the terminal's default foreground
        color.
    * bgcolor : optional ColorLike.
        The background color behind the markers. Defaults to a transparent
        background.

    A defaulted `height` or `width` has to measure the terminal, and is an
    error without one attached. A fallback size would quietly truncate a plot
    on its way into a file or a pipe, which is worse than being told to say
    what size was meant.

    Where there is no room for content at all -- a `height` or `width` of one
    in a direction that is cropped -- the result is the marker alone.

    TODO:

    * Configurable crop direction, one of nine. For now default to keep
      top-left rectangle.
    """
    def __init__(
        self,
        plot: plot,
        height: int | None = None,
        width: int | None = None,
        marker: str | None = "#",
        fgcolor: ColorLike | None = None,
        bgcolor: ColorLike | None = None,
    ):
        # decide the maximum size
        if height is None or width is None:
            size = terminal_size()
            if size is None:
                raise ValueError(
                    "crop has no terminal to measure, so its height and width "
                    "cannot be defaulted: pass them explicitly"
                )
            if height is None:
                # a plot of the terminal's full height cannot animate in it
                height = max(1, size.lines - 1)
            if width is None:
                width = size.columns
        if height <= 0:
            raise ValueError(f"crop height must be positive, not {height}")
        if width <= 0:
            raise ValueError(f"crop width must be positive, not {width}")

        # validate the marker, and reduce it to the codepoint to be written
        if marker is None:
            marker_code = None
        else:
            _validate_text(marker)
            if len(marker) != 1:
                raise ValueError(
                    f"crop marker must be a single character, not {marker!r}"
                )
            marker_code = ord(marker)

        # decide the size of the result, and of the content inside it: each
        # direction that is cut gives its last row or column to the marker
        out_height = min(height, plot.height)
        out_width = min(width, plot.width)
        mark_below = marker_code is not None and plot.height > height
        mark_right = marker_code is not None and plot.width > width
        content_height = out_height - mark_below
        content_width = out_width - mark_right

        if content_height <= 0 or content_width <= 0:
            # no room for content, so the result is marker all the way across
            chars = CharArray.from_size(
                height=out_height,
                width=out_width,
                fgcolor=fgcolor,
                bgcolor=bgcolor,
            )
            if marker_code is not None:
                chars.codes[:,:] = marker_code
        else:
            below = plot.height - content_height
            right = plot.width - content_width
            content = (
                plot.chars.crop(below=below, right=right)
                if below or right
                else plot.chars
            )
            if mark_below or mark_right:
                # padding the content back out to size is what copies it, and
                # a copy is what the markers need: a crop is a view of the
                # array it came from, and writing into that would reach back
                # into the plot being cropped
                chars = content.pad(
                    below=mark_below,
                    right=mark_right,
                    fgcolor=fgcolor,
                    bgcolor=bgcolor,
                )
                if mark_right:
                    chars.codes[:,-1] = marker_code
                if mark_below:
                    chars.codes[-1,:] = marker_code
            else:
                chars = content

        super().__init__(chars)
        self.plot = plot
        self.marker = marker

    def __repr__(self):
        return (
            f"crop(height={self.height}, width={self.width}, "
            f"marker={self.marker!r}, plot={self.plot!r})"
        )
