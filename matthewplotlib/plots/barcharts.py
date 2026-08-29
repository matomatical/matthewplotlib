"""
Plots that draw values as marks with length: progress bars, bar and column
charts, the histograms binning values into them, candlesticks, and box
plots.

* `progress`
* `bars`
* `histogram`
* `columns`
* `vistogram`
* `candles`
* `boxes`
"""
from __future__ import annotations

import math
import dataclasses

import numpy as np

from typing import Literal, cast
from collections.abc import Sequence
from numpy.typing import ArrayLike, NDArray
from matthewplotlib.colors import ColorLike, parse_color, parse_colors
from matthewplotlib.data import number
from matthewplotlib.scales import (
    scale,
    _resolve_vrange,
    _resolve_coordinate_range,
)
from matthewplotlib.window import window
from matthewplotlib.core import (
    CharArray,
    ords,
    LineStyle,
    Orientation,
    unicode_bar,
    unicode_col,
    unicode_boxes,
)
from matthewplotlib.plots.base import plot


# # #
# MARKS DRAWN AS FILLS

# A fill whose end lands part way into a cell it does not fill from the low
# edge is drawn as a negative, and a negative needs a color at each end of it:
# the mark's own, which goes to the cell's background, and the ground it is
# drawn against. These stand in where the caller named neither.
_FILL_BACKGROUND = (0.08, 0.09, 0.11)
_FILL_COLOR = (1.0, 1.0, 1.0)

# How far past an end of its interval an axis has to reach before it counts as
# having widened at all, as a fraction of the interval. An interval a baseline
# already divides into whole cells wants to be left exactly as it is, and the
# arithmetic that finds the cells lands a rounding error away from saying so.
# Well below an eighth of the narrowest cell any chart has.
_WIDENING_FLOOR = 1e-9


def _split_cells(
    below: float,
    above: float,
    length: int,
) -> tuple[int, int]:
    """
    Share `length` cells between the two sides of a baseline, so that the
    widest cell either side needs is as narrow as it can be.

    `below` and `above` are how far the axis reaches each way from the
    baseline. A side needing `d` of them across `n` cells needs cells of width
    `d/n`, so the cells are `max(below/n_below, above/n_above)` wide and the
    best split is the one where the two sides come closest to needing the same
    width. That is where the falling `below/n_below` crosses the rising
    `above/n_above`, so it is one of the two whole numbers either side of the
    crossing.

    A side the axis does not reach takes no cells at all. Where there is only
    one cell to share and the axis reaches both ways, it goes to the side that
    reaches further, and the other side is left with the baseline at the edge
    of the chart.
    """
    if below <= 0:
        return 0, length
    if above <= 0:
        return length, 0
    if length < 2:
        return (length, 0) if below >= above else (0, length)
    crossing = length * below / (below + above)
    candidates = {
        max(1, math.floor(crossing)),
        min(length - 1, math.ceil(crossing)),
    }
    best = min(candidates, key=lambda n: max(below / n, above / (length - n)))
    return best, length - best


def _align_to_baseline(
    vscale: scale,
    baseline: number,
    length: int,
) -> tuple[scale, int]:
    """
    Widen a value axis until its baseline sits on the edge between two cells.

    A bar grows out of the baseline, so the baseline has to lie where a glyph
    can start: on the edge between two character cells, never part way into
    one. That fixes a whole number of cells on each side of it. The cells are
    all one width, so an interval whose baseline does not already divide it
    into whole cells is covered by reaching a little past one of its ends.

    The cells are made as narrow as they can be while still covering the
    interval. That leaves the side needing the wider cells ending exactly
    where it was asked to and the other reaching past its end by less than one
    cell. Where the baseline is already at an end of the interval---which is
    every chart measured from zero over values on one side of it---one side
    takes no cells and the axis is exactly the interval it was given.

    Returns the axis it settled on, and the number of cells below the
    baseline.
    """
    # measure from the baseline in the axis's own units, where the interval
    # runs from 0.0 to 1.0 whichever way round its ends are
    base = float(np.clip(vscale.position(baseline), 0.0, 1.0))
    cells_below, cells_above = _split_cells(
        below=base,
        above=1.0 - base,
        length=length,
    )
    if cells_below == 0 or cells_above == 0:
        return vscale, cells_below

    # widen to the whole cells either side, in the space the scale spaces its
    # values in, so that a nonlinear axis grows by a cell of its own kind. The
    # side needing the wider cells is covered exactly and stays where it is;
    # only the other one reaches past its end, and neither does where the
    # baseline already divides the interval into whole cells.
    cell = max(base / cells_below, (1.0 - base) / cells_above)
    tlo, thi = vscale._transformed_ends()
    span = thi - tlo
    ends = []
    for end, reach, beyond in (
        (vscale.lo, base - cells_below * cell, 0.0),
        (vscale.hi, base + cells_above * cell, 1.0),
    ):
        if abs(reach - beyond) < _WIDENING_FLOOR:
            ends.append(float(cast(number, end)))
        else:
            ends.append(float(np.asarray(
                vscale._checked_inverse(tlo + reach * span), dtype=float,
            )))
    try:
        return dataclasses.replace(vscale, lo=ends[0], hi=ends[1]), cells_below
    except ValueError as e:
        raise ValueError(
            f"a baseline of {float(baseline):g} falls inside {vscale!r}, "
            f"which has to widen to {ends[0]:g} to {ends[1]:g} to put the "
            f"baseline on the edge between two of its {length} cells, and "
            f"{e}"
        ) from None


def _baseline_reaches(
    vscale: scale,
    values: NDArray,            # float[n]
    cells_below: int,
    length: int,
) -> tuple[NDArray, NDArray]:   # float[n], float[n]
    """
    How far each value reaches from the baseline, towards the low end of the
    axis and towards the high end, each as a proportion of that side.

    Only one of the two is ever more than zero. A value that is not a number
    reaches neither way, having no length to draw, and neither does any value
    at all along an axis with no cells to draw in.
    """
    if length < 1:
        return np.zeros(values.shape), np.zeros(values.shape)
    base = cells_below / length
    proportions = np.asarray(vscale(values), dtype=float)
    if base > 0:
        towards_low = np.clip((base - proportions) / base, 0.0, 1.0)
    else:
        towards_low = np.zeros(proportions.shape)
    if base < 1:
        towards_high = np.clip((proportions - base) / (1.0 - base), 0.0, 1.0)
    else:
        towards_high = np.zeros(proportions.shape)

    # a value that is not a number comes off the scale at its bottom, which is
    # a mark of full length where the baseline is not there
    drawn = np.isfinite(values)
    return (
        np.where(drawn, towards_low, 0.0),
        np.where(drawn, towards_high, 0.0),
    )


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
        An array of values to display.
    * width : int (default: 30).
        The total width of full bars.
    * bar_height: int (default: 1).
        The number of rows comprising each bar.
    * bar_spacing: int (default: 0).
        The number of rows between each bar.
    * vrange : optional (number, number) | scale.
        The interval of values the bars measure: a bar at the first value or
        below has zero width and one at the second value or above occupies the
        whole width. By default the interval runs from the baseline to the
        largest value, reaching down to the smallest where that falls below
        the baseline, so that the longest bar or bars fill the width.
        Measuring from a baseline rather than from the smallest value is what
        makes a bar's width readable on its own, and a chart of equal values a
        row of full bars.

        A `scale` says how the widths are spaced within the interval, where a
        plain pair spaces them linearly. An inferred interval still takes in
        the baseline, so a `logscale` here needs its own lower end:
        `vrange=mp.logscale(1, 1000)`.
    * baseline : number (default: 0).
        The value the bars are measured from. Where it falls inside the
        interval the chart diverges: the bar for a value below it reaches to
        the left and the bar for a value above it to the right.
    * mirror : bool (default: False).
        Turn the value axis around, so that the bars grow leftwards from the
        right edge. This reverses whatever interval the chart settled on, so
        mirroring a descending `vrange` gives an ascending one.
    * color : optional ColorLike.
        The color of the filled portion of the bars. Defaults to the terminal's
        default foreground color, or to white where the chart has bars growing
        leftwards, whose color has to be named for their negatives to be drawn
        against the background.
    * colors : optional ColorLike[n].
        The colours of the filled portion of each bar. Should be an array or
        list of the same length as `values`.
    * background : optional ColorLike.
        The color behind the bars, painting the whole rectangle, the rows
        between the bars included. A chart with bars growing leftwards
        defaults to a near-black, because such a bar reaches every eighth of a
        cell only by drawing one cell as a negative, which needs the
        background named. A chart whose bars all grow rightwards leaves the
        terminal's own background showing unless one is given.

    A value that is not a number is left out of an inferred interval, and its
    bar has zero width.

    The baseline sits on the edge between two character cells, so that every
    bar starts where a glyph can start and a short bar keeps its true length
    rather than shifting to a cell it does not reach. Where the interval does
    not already divide into whole cells that way, the chart widens it to the
    nearest interval that does: the cells are made as narrow as they can be
    while still covering the interval, leaving one end exactly where it was
    asked to be and the other reaching past its end by less than one cell. The
    interval the chart settled on is its `vrange`. A baseline at an end of the
    interval divides it into whole cells to begin with, so a chart measured
    from zero over values on one side of zero covers exactly what it was
    given.
    """
    def __init__(
        self,
        values: ArrayLike, # numeric[n]
        width: int = 30,
        bar_height: int = 1,
        bar_spacing: int = 0,
        vrange: tuple[number, number] | scale | None = None,
        baseline: number = 0,
        mirror: bool = False,
        color: ColorLike | None = None,
        colors: list[ColorLike | None] | None = None,
        background: ColorLike | None = None,
    ):
        # standardise inputs
        values = np.asarray(values, dtype=float)
        vscale = _resolve_vrange(vrange, values, "bars", from_zero=True)
        if mirror:
            vscale = dataclasses.replace(vscale, lo=vscale.hi, hi=vscale.lo)
        num_bars = len(values)

        # put the baseline on the edge between two cells, and measure every
        # bar from it
        vscale, left_width = _align_to_baseline(vscale, baseline, width)
        right_width = width - left_width
        leftwards, rightwards = _baseline_reaches(
            vscale=vscale,
            values=values,
            cells_below=left_width,
            length=width,
        )

        # determine the colors
        if colors is None:
            colors = [color for _ in range(num_bars)]
        if left_width:
            # a bar growing leftwards draws one cell as a negative, which
            # needs a color of its own and a background to draw it against
            background = _FILL_BACKGROUND if background is None else background
            colors = [_FILL_COLOR if c is None else c for c in colors]

        # construct the bars
        bars_chars = []
        for i in range(num_bars):
            pieces = []
            if left_width:
                pieces.append(unicode_bar(
                    proportion=leftwards[i],
                    width=left_width,
                    height=bar_height,
                    fgcolor=colors[i],
                    bgcolor=background,
                    anchor="high",
                ))
            if right_width or not pieces:
                pieces.append(unicode_bar(
                    proportion=rightwards[i],
                    width=right_width,
                    height=bar_height,
                    fgcolor=colors[i],
                    bgcolor=background,
                    anchor="low",
                ))
            bar_chars = CharArray.map(
                lambda xs: np.concatenate(xs, axis=1),
                pieces,
            )
            bars_chars.append(bar_chars.pad(
                below=bar_spacing * (i!=num_bars-1),
                bgcolor=background,
            ))
        all_chars = CharArray.map(
            lambda xs: np.concatenate(xs, axis=0),
            bars_chars,
        )
        super().__init__(chars=all_chars)
        self.vrange = vscale
        self.baseline = baseline
        self.num_bars = num_bars

    def __repr__(self):
        return (
            f"bars(height={self.height}, width={self.width}, "
            f"values=<{self.num_bars} bars on {self.vrange!r}>)"
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
    * mirror : bool (default: False).
        Turn the value axis around, so that the bars grow leftwards from the
        right edge.
    * color : optional ColorLike.
        The color of the filled portion of the bars. Defaults to the terminal's
        default foreground color, or to white where the bars grow leftwards.
    * background : optional ColorLike.
        The color behind the bars. Bars growing leftwards default to a
        near-black; bars growing rightwards leave the terminal's own
        background showing unless one is given.
    """
    def __init__(
        self,
        data: ArrayLike, # number[n]
        bins: int = 10,
        xrange: tuple[number, number] | None = None,
        weights: ArrayLike | None = None, # optional number[n]
        density: bool = False,
        max_count: number | None = None,
        width: int = 22,
        mirror: bool = False,
        color: ColorLike | None = None,
        background: ColorLike | None = None,
    ):
        # prepare data
        data = np.asarray(data)
        weights_ = None if weights is None else np.asarray(weights)
        
        # bin data
        hist, bins_ = np.histogram(
            a=data,
            bins=bins,
            # numpy's stubs ask for concrete floats, where the rest of the
            # library spells a range as a pair of any numbers
            range=cast("tuple[float, float] | None", xrange),
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
            mirror=mirror,
            color=color,
            background=background,
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
        An array of values to display.
    * height : int (default: 10).
        The total height of full columns.
    * column_width: int (default 1).
    * column_spacing: int (default 0).
    * vrange : optional (number, number) | scale.
        The interval of values the columns measure: a column at the first
        value or below has zero height and one at the second value or above
        occupies the whole height. By default the interval runs from the
        baseline to the largest value, reaching down to the smallest where
        that falls below the baseline, so that the tallest column or columns
        fill the height. Measuring from a baseline rather than from the
        smallest value is what makes a column's height readable on its own,
        and a chart of equal values a row of full columns.

        A `scale` says how the heights are spaced within the interval, where a
        plain pair spaces them linearly. An inferred interval still takes in
        the baseline, so a `logscale` here needs its own lower end:
        `vrange=mp.logscale(1, 1000)`.
    * baseline : number (default: 0).
        The value the columns are measured from. Where it falls inside the
        interval the chart diverges: the column for a value below it hangs
        downwards and the column for a value above it stands upwards.
    * mirror : bool (default: False).
        Turn the value axis around, so that the columns hang downwards from
        the top edge. This reverses whatever interval the chart settled on, so
        mirroring a descending `vrange` gives an ascending one.
    * color : optional ColorLike.
        The color of the filled portion of the columns. Defaults to the
        terminal's default foreground color, or to white where the chart has
        columns hanging downwards, whose color has to be named for their
        negatives to be drawn against the background.
    * colors : optional ColorLike[n].
        The colours of the filled portion of each column. Should be an array or
        list of the same length as `values`.
    * background : optional ColorLike.
        The color behind the columns, painting the whole rectangle, the
        columns between them included. A chart with columns hanging
        downwards defaults to a near-black, because such a column reaches
        every eighth of a cell only by drawing one cell as a negative, which
        needs the background named. A chart whose columns all stand upwards
        leaves the terminal's own background showing unless one is given.

    A value that is not a number is left out of an inferred interval, and its
    column has zero height.

    The baseline sits on the edge between two character cells, so that every
    column starts where a glyph can start and a short column keeps its true
    length rather than shifting to a cell it does not reach. Where the
    interval does not already divide into whole cells that way, the chart
    widens it to the nearest interval that does: the cells are made as narrow
    as they can be while still covering the interval, leaving one end exactly
    where it was asked to be and the other reaching past its end by less than
    one cell. The interval the chart settled on is its `vrange`. A baseline at
    an end of the interval divides it into whole cells to begin with, so a
    chart measured from zero over values on one side of zero covers exactly
    what it was given.
    """
    def __init__(
        self,
        values: ArrayLike, # number[n], actually int[n] will also work
        height: int = 10,
        column_width: int = 1,
        column_spacing: int = 0,
        vrange: tuple[number, number] | scale | None = None,
        baseline: number = 0,
        mirror: bool = False,
        color: ColorLike | None = None,
        colors: list[ColorLike | None] | None = None,
        background: ColorLike | None = None,
    ):
        # standardise inputs
        values = np.asarray(values, dtype=float)
        vscale = _resolve_vrange(vrange, values, "columns", from_zero=True)
        if mirror:
            vscale = dataclasses.replace(vscale, lo=vscale.hi, hi=vscale.lo)
        num_cols = len(values)

        # put the baseline on the edge between two cells, and measure every
        # column from it
        vscale, lower_height = _align_to_baseline(vscale, baseline, height)
        upper_height = height - lower_height
        downwards, upwards = _baseline_reaches(
            vscale=vscale,
            values=values,
            cells_below=lower_height,
            length=height,
        )

        # determine the colours
        if colors is None:
            colors = [color for _ in range(num_cols)]
        if lower_height:
            # a column hanging downwards draws one cell as a negative, which
            # needs a color of its own and a background to draw it against
            background = _FILL_BACKGROUND if background is None else background
            colors = [_FILL_COLOR if c is None else c for c in colors]

        # construct the columns, the upper region above the lower one
        cols_chars = []
        for i in range(num_cols):
            pieces = []
            if upper_height or not lower_height:
                pieces.append(unicode_col(
                    proportion=upwards[i],
                    height=upper_height,
                    width=column_width,
                    fgcolor=colors[i],
                    bgcolor=background,
                    anchor="low",
                ))
            if lower_height:
                pieces.append(unicode_col(
                    proportion=downwards[i],
                    height=lower_height,
                    width=column_width,
                    fgcolor=colors[i],
                    bgcolor=background,
                    anchor="high",
                ))
            col_chars = CharArray.map(
                lambda xs: np.concatenate(xs, axis=0),
                pieces,
            )
            cols_chars.append(col_chars.pad(
                right=column_spacing * (i!=num_cols-1),
                bgcolor=background,
            ))
        all_chars = CharArray.map(
            lambda xs: np.concatenate(xs, axis=1),
            cols_chars,
        )
        super().__init__(chars=all_chars)
        self.vrange = vscale
        self.baseline = baseline
        self.num_cols = num_cols

    def __repr__(self):
        return (
            f"columns(height={self.height}, width={self.width}, "
            f"values=<{self.num_cols} columns on {self.vrange!r}>)"
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
    * mirror : bool (default: False).
        Turn the value axis around, so that the bars hang downwards from the
        top edge.
    * color : optional ColorLike.
        The color of the filled portion of the bars. Defaults to the terminal's
        default foreground color, or to white where the bars hang downwards.
    * background : optional ColorLike.
        The color behind the bars. Bars hanging downwards default to a
        near-black; bars standing upwards leave the terminal's own background
        showing unless one is given.
    """
    def __init__(
        self,
        data: ArrayLike, # number[n]
        bins: int = 10,
        xrange: tuple[number, number] | None = None,
        weights: ArrayLike | None = None, # optional number[n]
        density: bool = False,
        max_count: None | number = None,
        height: int = 10,
        mirror: bool = False,
        color: ColorLike | None = None,
        background: ColorLike | None = None,
    ):
        # prepare data
        data = np.asarray(data)
        weights_ = None if weights is None else np.asarray(weights)
        
        # bin data
        hist, bins_ = np.histogram(
            a=data,
            bins=bins,
            # numpy's stubs ask for concrete floats, where the rest of the
            # library spells a range as a pair of any numbers
            range=cast("tuple[float, float] | None", xrange),
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
            mirror=mirror,
            color=color,
            background=background,
        )
        self.bins = bins_

    def __repr__(self):
        return (
            f"vistogram(height={self.height}, width={self.width}, "
            f"bins=<{len(self.bins)-1} on "
            f"[{self.bins[0]:.2f},{self.bins[-1]:.2f}]>)"
        )


class candles(plot):
    """
    A candlestick chart.

    Draw one candle per period, each a filled body spanning the opening and
    closing values with a thin wick reaching out of it to the high and the low.
    The body is colored by whether the period closed above or below where it
    opened.

    Inputs:

    * opens, highs, lows, closes : number[n].
        The four values of each period. Each high must be at least as large as
        the opening and closing values of its period, and each low at most as
        small, as the wick reaches out of the body rather than into it. Every
        value must be a number: a period one of whose four is unknown has no
        candle to draw.
    * length : int (default: 12).
        The number of character cells along the value axis.
    * body_thickness : int (default 1).
        The number of cells across each body. The wick runs along the middle
        one, so an even thickness leaves it off centre.
    * spacing : int (default 0).
        The number of blank cells between one candle and the next.
    * candle_direction : Orientation (default: "vertical").
        Which way one candle lies. Vertical candles stand up and march across
        the screen, which is the way a price series is usually read and so the
        default; horizontal candles lie flat and stack up it.
    * vrange : optional (number, number) | scale.
        The values at the ends of the value axis. By default, the lowest low
        and the highest high, so that every candle fits. Given a narrower
        interval, the candles outside it are clipped to it. A `scale` spaces
        values nonlinearly along the axis: `vrange=mp.logscale()` is a
        logarithmic value axis over an inferred interval.
    * rising : ColorLike (default: a green).
        The color of a candle that closed at or above its opening value.
    * falling : ColorLike (default: a red).
        The color of a candle that closed below its opening value.
    * wick : optional ColorLike.
        The color of the wicks. By default each wick takes the color of the
        body it belongs to.
    * background : ColorLike (default: a near-black).
        The color behind the candles. Unlike most plots, a candlestick chart
        paints its whole rectangle rather than leaving the terminal's
        background showing: a body is positioned to an eighth of a character
        cell, and reaching every eighth means drawing some bodies as a
        background-colored block over a body-colored cell, which needs the
        background named.
    * style : LineStyle (default: LineStyle.LIGHT).
        The weight of the wicks.

    The plot carries its value range on the axis its candles stand along and no
    coordinate on the other, since the candles are a sequence of periods rather
    than a measured axis. So `axes` labels its value axis and leaves the other
    three sides alone.

    A body is positioned to the nearest eighth of a character cell and a wick
    to the nearest half. A body always keeps its true length, and a candle that
    opened and closed at the same value still shows a hairline.

    A candle and a box are one mark with different switches thrown, so this is
    `boxes` with its caps, its median and its outlying points switched off, and
    the two share their drawing. See the `box-plots` note.
    """
    def __init__(
        self,
        opens: ArrayLike,   # number[n]
        highs: ArrayLike,   # number[n]
        lows: ArrayLike,    # number[n]
        closes: ArrayLike,  # number[n]
        length: int = 12,
        body_thickness: int = 1,
        spacing: int = 0,
        candle_direction: Orientation = "vertical",
        vrange: tuple[number, number] | scale | None = None,
        rising: ColorLike = (0.30, 0.78, 0.45),
        falling: ColorLike = (0.90, 0.32, 0.36),
        wick: ColorLike | None = None,
        background: ColorLike = (0.08, 0.09, 0.11),
        style: LineStyle = LineStyle.LIGHT,
    ):
        # standardise inputs
        values = [
            np.asarray(v, dtype=float)
            for v in (opens, highs, lows, closes)
        ]
        for name, value in zip(("opens", "highs", "lows", "closes"), values):
            if value.ndim != 1:
                raise ValueError(
                    f"{name} should be a sequence of numbers, but it has "
                    f"shape {value.shape}"
                )
        opens_, highs_, lows_, closes_ = values
        lengths = {v.shape[0] for v in values}
        if len(lengths) > 1:
            raise ValueError(
                "opens, highs, lows and closes should all be the same length, "
                f"but they are {', '.join(str(v.shape[0]) for v in values)}"
            )
        num_candles = opens_.shape[0]

        # a period with a value that is not a number has no candle to draw: it
        # passes the ordering check below silently, every comparison against it
        # being false, and then lands at the bottom of the scale claiming a
        # value it does not have
        for name, value in zip(("opens", "highs", "lows", "closes"), values):
            unknown = np.flatnonzero(~np.isfinite(value))
            if len(unknown):
                i = unknown[0]
                raise ValueError(
                    f"candle {i} has a {name[:-1]} of {value[i]}, which is "
                    "not a number"
                )

        # a high below the body, or a low above it, would leave the wick inside
        # the body; usually it means the four series arrived out of order
        too_low = np.flatnonzero(highs_ < np.maximum(opens_, closes_))
        too_high = np.flatnonzero(lows_ > np.minimum(opens_, closes_))
        for name, wrong in (("high", too_low), ("low", too_high)):
            if len(wrong):
                i = wrong[0]
                value = highs_[i] if name == "high" else lows_[i]
                raise ValueError(
                    f"candle {i} has a {name} of {value}, inside its opening "
                    f"value {opens_[i]} and closing value {closes_[i]}; the "
                    "arguments are opens, highs, lows, closes"
                )

        # determine the value range, and where each value sits within it
        if vrange is None and num_candles == 0:
            raise ValueError("cannot infer a value range with no candles")
        vscale = _resolve_coordinate_range(
            vrange,
            np.concatenate(values),
            "candles",
            "vrange",
        )
        proportions = [vscale(v) for v in values]

        # determine the colours
        rose = closes_ >= opens_
        body_colors = np.where(
            rose[:, None],
            parse_colors(rising, n=1),
            parse_colors(falling, n=1),
        ).astype(np.uint8)
        if wick is None:
            wick_colors = body_colors
        else:
            wick_colors = np.broadcast_to(
                parse_colors(wick, n=1), (num_candles, 3)
            )

        # construct the candles: an outer interval drawn thin and an inner one
        # drawn thick, which is a box plot with its caps, median and outlying
        # points all switched off
        opens_p, highs_p, lows_p, closes_p = proportions
        chars = unicode_boxes(
            outer_los=lows_p,
            outer_his=highs_p,
            inner_los=np.minimum(opens_p, closes_p),
            inner_his=np.maximum(opens_p, closes_p),
            length=length,
            box_colors=body_colors,
            outer_colors=wick_colors,
            direction=candle_direction,
            filled=True,
            thickness=body_thickness,
            spacing=spacing,
            caps=False,
            background=background,
            style=style,
        )

        # form a plot object
        super().__init__(chars)
        standing = candle_direction == "vertical"
        self.window = window(
            xrange=None if standing else vscale,
            yrange=vscale if standing else None,
            width=chars.width,
            height=chars.height,
        )
        self.num_candles = num_candles
        self.vrange = vscale

    def __repr__(self):
        return f"candles(<{self.num_candles} candles>, {self.window!r})"


class boxes(plot):
    """
    A box plot, one box per group of samples.

    Draw one box per group, spanning the first and third quartiles, divided at
    the median, with whiskers reaching out to the extremes of the group and any
    sample beyond them drawn as an individual point.

    Inputs:

    * data : sequence of number[k].
        The samples in each group. The groups need not be the same length. A 2d
        array works, one group per row. A sample that is not finite is a
        measurement that was not made: it is left out of the summary rather
        than shifting the quartiles or counting as a point beyond the whiskers,
        and a group with no finite samples at all is an error.
    * length : int (default: 30).
        The number of character cells along the value axis.
    * box_thickness : int (default: 3).
        The number of character cells across one box. At least 3 for an
        outlined box, which needs two edges and an interior between them, and
        at least 1 for a filled one.
    * box_spacing : int (default: 1).
        The number of blank cells between one box and the next.
    * box_direction : Orientation (default: "horizontal").
        Which way one box lies. Horizontal boxes lie flat and stack up the
        screen; vertical boxes stand up and march across it. Horizontal is the
        default because it gives the value axis both more cells and finer ones:
        terminals are wider than they are tall, and character cells are taller
        than they are wide.
    * filled : bool (default: False).
        Whether to draw each box as a solid fill rather than an outline. A fill
        reaches the nearest eighth of a cell where an outline is confined to
        whole cells, at the cost of needing a background color.
    * caps : bool (default: True).
        Whether to draw a cap across the end of each whisker.
    * median : bool (default: True).
        Whether to divide each box at its median. The mark is dropped from a
        box with no room for it regardless.
    * whisker_iqrs : optional number (default: 1.5).
        How far the whiskers reach, as a multiple of the interquartile range
        beyond the quartiles. Each whisker stops at the furthest sample within
        that reach, and every sample beyond it is drawn as a point: the default
        is Tukey's rule. Given None, the whiskers reach the smallest and
        largest samples instead and no points are drawn.
    * vrange : optional (number, number) | scale.
        The values at the ends of the value axis. By default, the smallest and
        largest samples, so that every group fits. Given a narrower interval,
        the boxes outside it are clipped to it and the points outside it are
        dropped. A `scale` spaces values nonlinearly along the axis:
        `vrange=mp.logscale()` is a logarithmic value axis over an inferred
        interval, and a point with a value the scale is not defined over is
        dropped the way a point outside the interval is.
    * color : optional ColorLike.
        The color of every box. Defaults to the terminal's foreground color,
        or to white for a filled box, whose color has to be named for the
        negatives to be drawn against it.
    * colors : optional ColorLike[n].
        The color of each box. Should be a list or array as long as `data`.
    * background : optional ColorLike.
        The color behind the boxes. A filled plot paints its whole rectangle,
        defaulting to a near-black, because a fill reaches every eighth of a
        cell only by drawing some eighths as negatives, which needs the
        background named. An outlined plot leaves the terminal's own background
        showing unless one is given.
    * style : LineStyle (default: LineStyle.LIGHT).
        The weight of the whiskers, the caps and an outlined box's outline.
    * median_style : optional LineStyle.
        The weight of a filled box's median. Defaults to light lying flat and
        heavy standing up, each being the one that matches the eighth blocks
        the median lands on at the edges of a cell. An outlined box's median
        joins its outline and so takes `style` instead.

    The plot carries its value range on one axis and no coordinate on the
    other, since the groups are a list of categories rather than a measured
    axis. So `axes` labels the value axis and leaves the other three sides
    alone.
    """
    def __init__(
        self,
        data: Sequence[ArrayLike], # sequence of number[k]
        length: int = 30,
        box_thickness: int = 3,
        box_spacing: int = 1,
        box_direction: Orientation = "horizontal",
        filled: bool = False,
        caps: bool = True,
        median: bool = True,
        whisker_iqrs: number | None = 1.5,
        vrange: tuple[number, number] | scale | None = None,
        color: ColorLike | None = None,
        colors: Sequence[ColorLike] | None = None,
        background: ColorLike | None = None,
        style: LineStyle = LineStyle.LIGHT,
        median_style: LineStyle | None = None,
    ):
        # standardise inputs
        groups = []
        for i, group in enumerate(data):
            samples = np.asarray(group, dtype=float)
            if samples.ndim == 0:
                raise ValueError(
                    f"group {i} is the single number {samples}; boxes takes a "
                    "sequence of samples for each group, so one group of "
                    "samples is [samples] rather than samples"
                )
            samples = samples.reshape(-1)
            if samples.size == 0:
                raise ValueError(f"group {i} has no samples to summarise")
            # a sample that is not finite is a measurement that was not made,
            # as it is for the plots that colour dated values, so it is left
            # out of the summary rather than poisoning it
            finite = samples[np.isfinite(samples)]
            if finite.size == 0:
                raise ValueError(
                    f"group {i} has no finite samples to summarise"
                )
            groups.append(finite)
        if not groups:
            raise ValueError("boxes needs at least one group of samples")
        num_boxes = len(groups)
        if whisker_iqrs is not None and whisker_iqrs < 0:
            raise ValueError(
                f"whisker_iqrs must be non-negative, not {whisker_iqrs}"
            )

        # summarise each group, and find the samples beyond its whiskers
        quartiles = np.array(
            [np.percentile(group, [25, 50, 75]) for group in groups]
        )
        first, medians, third = quartiles.T
        if whisker_iqrs is None:
            whisker_los = np.array([group.min() for group in groups])
            whisker_his = np.array([group.max() for group in groups])
            beyond = np.empty(0)
            beyond_boxes = np.empty(0, dtype=int)
        else:
            reach = whisker_iqrs * (third - first)
            whisker_los = np.empty(num_boxes)
            whisker_his = np.empty(num_boxes)
            outlying = []
            for i, group in enumerate(groups):
                low, high = first[i] - reach[i], third[i] + reach[i]
                within = (group >= low) & (group <= high)
                # the fence always admits a sample, since it contains the
                # quartiles and so cannot fall between two samples
                whisker_los[i] = group[within].min()
                whisker_his[i] = group[within].max()
                outlying.append(group[~within])
            beyond = np.concatenate(outlying)
            beyond_boxes = np.repeat(
                np.arange(num_boxes),
                [len(samples) for samples in outlying],
            )

        # determine the value range, and where each value sits within it
        vscale = _resolve_coordinate_range(
            vrange,
            np.concatenate(groups),
            "boxes",
            "vrange",
        )
        # a point outside the range is dropped rather than clipped, since a
        # point drawn at the end of the axis claims a value it does not have,
        # so these are placed without the saturation the extents get; a point
        # the scale is not defined over comes out unplaceable and is dropped
        # the same way
        outlying_proportions = vscale.position(beyond)
        inside = (outlying_proportions >= 0) & (outlying_proportions <= 1)

        # determine the colours
        box_colors: NDArray | None
        if colors is not None:
            if len(colors) != num_boxes:
                raise ValueError(
                    f"there are {num_boxes} groups but {len(colors)} colors"
                )
            box_colors = np.array(
                [parse_color(c) for c in colors], dtype=np.uint8
            )
        elif color is not None:
            box_colors = np.broadcast_to(
                parse_colors(color, n=1), (num_boxes, 3)
            )
        elif filled:
            box_colors = np.full((num_boxes, 3), 255, dtype=np.uint8)
        else:
            box_colors = None
        if filled and background is None:
            background = (0.08, 0.09, 0.11)
        if median_style is None:
            median_style = (
                LineStyle.LIGHT
                if box_direction == "horizontal"
                else LineStyle.HEAVY
            )

        # construct the boxes
        chars = unicode_boxes(
            outer_los=vscale(whisker_los),
            outer_his=vscale(whisker_his),
            inner_los=vscale(first),
            inner_his=vscale(third),
            interiors=vscale(medians) if median else None,
            outliers=outlying_proportions[inside],
            outlier_boxes=beyond_boxes[inside],
            length=length,
            box_colors=box_colors,
            direction=box_direction,
            filled=filled,
            thickness=box_thickness,
            spacing=box_spacing,
            caps=caps,
            background=background,
            style=style,
            interior_style=median_style,
        )

        # form a plot object
        super().__init__(chars)
        horizontal = box_direction == "horizontal"
        self.window = window(
            xrange=vscale if horizontal else None,
            yrange=None if horizontal else vscale,
            width=chars.width,
            height=chars.height,
        )
        self.num_boxes = num_boxes
        self.vrange = vscale

    def __repr__(self):
        return f"boxes(<{self.num_boxes} groups>, {self.window!r})"
