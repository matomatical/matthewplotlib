"""
Plots that dress other plots: text labels, borders, axes, and colorbars.

* `text`
* `border`
* `axes`
* `Side`: what an `axes` draws along one of its four sides.
* `colorbar`
* `Direction`: which way along the screen a `colorbar` runs.
"""
from __future__ import annotations

import numpy as np

from typing import Literal
from matthewplotlib.colormaps import ColorMap
from matthewplotlib.colors import ColorLike
from matthewplotlib.data import number
from matthewplotlib.core import (
    ords,
    _validate_text,
    Align,
    BoxStyle,
    LineStyle,
    unicode_box,
    unicode_text,
    unicode_frame,
)
from matthewplotlib.plots.base import plot
from matthewplotlib.plots.grids import heatmap
from matthewplotlib.scales import scale


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
    * height : int (default: 0).
        The least number of rows the plot takes. More are taken if the text has
        more lines than this.
    * width : int (default: 0).
        The least number of columns the plot takes. More are taken if a line is
        longer than this.
    * align : Align (default: "left").
        Where each line sits in the width. Only has room to act where a width
        is given that is wider than the longest line.
    * fgcolor : optional ColorLike.
        The foreground color of the text. Defaults to the terminal's default
        foreground color.
    * bgcolor : optional ColorLike.
        The background color for the text, the rows and columns no line reaches
        included. Defaults to a transparent background.

    Carriage returns and newlines separate lines. Other C0 and C1 control
    characters are rejected, including the escapes used for raw ANSI
    formatting: styling has to be part of the plot so that composition and
    rendering know its size.

    The empty string has no lines in it, and so is a plot of no rows, which
    stacks and composes as nothing. A single empty line is `"\\n"`.

    TODO:

    * Account for non-printable and wide characters.
    """
    def __init__(
        self,
        text: str,
        height: int = 0,
        width: int = 0,
        align: Align = "left",
        fgcolor: ColorLike | None = None,
        bgcolor: ColorLike | None = None,
    ):
        _validate_text(text, allow_line_breaks=True)
        lines = text.splitlines()
        chars = unicode_text(
            lines=lines,
            height=height,
            width=width,
            align=align,
            fgcolor=fgcolor,
            bgcolor=bgcolor,
        )

        # initialise
        super().__init__(chars=chars)
        first = lines[0] if lines else ""
        if chars.height > 1 or chars.width > 8:
            self.preview = first[:5] + "..."
        else:
            self.preview = first[:8]

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
        south, north = _infer_side_pair(
            present=w.xrange is not None,
            primary=south,
            secondary=north,
            frame=frame,
        )
        west, east = _infer_side_pair(
            present=w.yrange is not None,
            primary=west,
            secondary=east,
            frame=frame,
        )
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
            west_edge = left_gutter
            chars.codes[first_row, west_edge-len(yhi_):west_edge] = ords(yhi_)
            chars.codes[last_row, west_edge-len(ylo_):west_edge] = ords(ylo_)
        if east == "label":
            east_edge = chars.width - right_gutter
            chars.codes[first_row, east_edge:east_edge+len(yhi_)] = ords(yhi_)
            chars.codes[last_row, east_edge:east_edge+len(ylo_)] = ords(ylo_)

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
            (lo, lo_col), (hi, hi_col) = _end_labels(
                lo=xlo,
                hi=xhi,
                span=span,
                room=chars.width,
            )
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

    * source : plot | (number, number) | scale.
        The interval the bar covers. Any plot that kept one lends it---a
        `heatmap` and everything built on one, a `calendar`, a `weeks`, a
        `bars`---so that the numbers on the bar cannot drift from the numbers
        in the picture. An interval or a `scale` on its own works too, for a
        bar assembled by hand.

        A plot whose values are all the same settled on an interval covering
        nothing, which is one colour and no axis to label it along, so there is
        no bar to draw for it and it is refused.

        Whatever the spacing of the source's scale, the bar itself draws the
        colormap swept evenly from one end to the other, with the interval's
        limits at the ends: a bar for a `logscale` and a bar for a plain
        interval draw identically. Where the values in between fall is the
        scale's business, not the bar's, until there are ticks to mark them.
    * colormap : optional ColorMap.
        Maps each position along the bar onto its colour. By default the bar
        runs black to white.

        Name the same one the picture was drawn with: the bar is not told what
        the picture used, in the same way that nothing else in this library
        infers a colormap. A bar is a gradient, so a continuous colormap is
        what it can stand for; a palette wants a swatch beside each label,
        which is a different plot and is not built.
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
    mp.axes(heat, title="field") \
        + mp.axes(mp.colorbar(heat, colormap=mp.viridis), east="label")
    ```
    """
    def __init__(
        self,
        source: plot | tuple[number, number] | scale,
        colormap: ColorMap | None = None,
        direction: Direction = "up",
        length: int = 12,
        thickness: int = 1,
    ):
        if isinstance(source, plot):
            vscale = getattr(source, "vrange", None)
            if vscale is None:
                raise ValueError(
                    f"{type(source).__name__} carries no interval for a "
                    "colorbar to draw; pass one instead"
                )
            # the plots that lay values along a coordinate axis lend a plain
            # pair, which stands for a linear scale
            if not isinstance(vscale, scale):
                vscale = scale(vscale[0], vscale[1])
            # a plot whose values are all the same settles on an interval
            # covering nothing, which a bar cannot be a scale for: it has one
            # colour and no axis to label it along
            if vscale.lo == vscale.hi:
                raise ValueError(
                    f"every value in the {type(source).__name__} sits at "
                    f"{vscale.lo}, so there is no scale to draw a colorbar "
                    "for; pass an interval instead"
                )
        elif isinstance(source, scale):
            vscale = source
            if vscale.lo is None or vscale.hi is None:
                raise ValueError(
                    f"{vscale!r} has a missing endpoint, and a colorbar has "
                    "no data to complete it from; give both ends"
                )
            if vscale.lo == vscale.hi:
                raise ValueError(
                    f"{vscale!r} covers no interval, so there is no scale to "
                    "draw a colorbar for"
                )
        else:
            vscale = scale(source[0], source[1])
            if vscale.lo == vscale.hi:
                raise ValueError(
                    f"{source!r} covers no interval, so there is no scale to "
                    "draw a colorbar for"
                )
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
        first, second = vscale.interval
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
        # the ramp is normalised over the plain interval rather than through
        # the source's scale, so that the bar sweeps the colormap evenly
        # whatever the spacing: where the values in between fall is the
        # scale's business, not the bar's
        super().__init__(
            values=values,
            colormap=colormap,
            vrange=(first, second),
            xrange=None if vertical else span,
            yrange=span if vertical else None,
        )
        # the bar stands for the source's scale, not the linear one its ramp
        # was drawn through
        self.vrange = vscale
        self.direction = direction

    def __repr__(self):
        return f"colorbar({self.direction}, {self.window!r})"
