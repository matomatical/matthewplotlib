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

import numpy as np

from typing import Literal, cast
from collections.abc import Sequence
from numpy.typing import ArrayLike, NDArray
from matthewplotlib.colors import ColorLike, parse_color, parse_colors
from matthewplotlib.data import number
from matthewplotlib.scales import (
    scale,
    _resolve_vrange,
    _resolve_linear_vrange,
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
    * vrange : optional (number, number) | scale.
        The interval of values the bars measure: a bar at the first value or
        below has zero width and one at the second value or above occupies the
        whole width. By default the interval runs from zero to the largest
        value, so that the largest bar or bars fill the width. Measuring from
        zero rather than from the smallest value is what makes a bar's width
        readable on its own, and a chart of equal values a row of full bars.

        A `scale` says how the widths are spaced within the interval, where a
        plain pair spaces them linearly. An inferred interval still starts at
        zero, so a `logscale` here needs its own lower end:
        `vrange=mp.logscale(1, 1000)`.
    * color : optional ColorLike.
        The color of the filled portion of the bars. Defaults to the terminal's
        default foreground color.
    * colors : optional ColorLike[n].
        The colours of the filled portion of each bar. Should be an array or
        list of the same length as `values`.

    A value that is not a number is left out of an inferred interval, and its
    bar has zero width.

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
        vrange: tuple[number, number] | scale | None = None,
        color: ColorLike | None = None,
        colors: list[ColorLike | None] | None = None,
    ):
        # standardise inputs
        values = np.asarray(values, dtype=float)
        vscale = _resolve_vrange(vrange, values, "bars", from_zero=True)
        num_bars = len(values)

        # compute the bar widths
        norm_values = vscale(values)

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
        self.vrange = vscale
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
    * color : optional ColorLike.
        The color of the filled portion of the bars. Defaults to the terminal's
        default foreground color.
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
        color: ColorLike | None = None,
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
    * vrange : optional (number, number) | scale.
        The interval of values the columns measure: a column at the first value
        or below has zero height and one at the second value or above occupies
        the whole height. By default the interval runs from zero to the largest
        value, so that the tallest column or columns fill the height. Measuring
        from zero rather than from the smallest value is what makes a column's
        height readable on its own, and a chart of equal values a row of full
        columns.

        A `scale` says how the heights are spaced within the interval, where a
        plain pair spaces them linearly. An inferred interval still starts at
        zero, so a `logscale` here needs its own lower end:
        `vrange=mp.logscale(1, 1000)`.
    * color : optional ColorLike.
        The color of the filled portion of the columns. Defaults to the
        terminal's default foreground color.
    * colors : optional ColorLike[n].
        The colours of the filled portion of each column. Should be an array or
        list of the same length as `values`.

    A value that is not a number is left out of an inferred interval, and its
    column has zero height.

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
        vrange: tuple[number, number] | scale | None = None,
        color: ColorLike | None = None,
        colors: list[ColorLike | None] | None = None,
    ):
        # standardise inputs
        values = np.asarray(values, dtype=float)
        vscale = _resolve_vrange(vrange, values, "columns", from_zero=True)
        num_cols = len(values)

        # compute the column heights
        norm_values = vscale(values)

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
        self.vrange = vscale
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
    * color : optional ColorLike.
        The color of the filled portion of the bars. Defaults to the terminal's
        default foreground color.
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
        color: ColorLike | None = None,
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
            color=color,
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
    * vrange : optional (number, number).
        The values at the ends of the value axis. By default, the lowest low
        and the highest high, so that every candle fits. Given a narrower
        interval, the candles outside it are clipped to it.
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
        vrange: tuple[number, number] | None = None,
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

        # determine the value range, and where each value sits within it. The
        # range is a coordinate interval, handed to the window below, so it is
        # a plain pair: a scale that moved the candles would part them from
        # the labels on their axis
        if vrange is None and num_candles == 0:
            raise ValueError("cannot infer a value range with no candles")
        vscale = _resolve_linear_vrange(
            vrange,
            np.concatenate(values),
            "candles",
            allow_flat=False,
        )
        vrange = vscale.interval
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
            xrange=None if standing else vrange,
            yrange=vrange if standing else None,
            width=chars.width,
            height=chars.height,
        )
        self.num_candles = num_candles
        self.vrange = vrange

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
    * vrange : optional (number, number).
        The values at the ends of the value axis. By default, the smallest and
        largest samples, so that every group fits. Given a narrower interval,
        the boxes outside it are clipped to it and the points outside it are
        dropped.
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
        vrange: tuple[number, number] | None = None,
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

        # determine the value range, and where each value sits within it. The
        # range is a coordinate interval, handed to the window below, so it is
        # a plain pair: a scale that moved the boxes would part them from the
        # labels on their axis
        vscale = _resolve_linear_vrange(
            vrange,
            np.concatenate(groups),
            "boxes",
            allow_flat=False,
        )
        vrange = vscale.interval
        # a point outside the range is dropped rather than clipped, since a
        # point drawn at the end of the axis claims a value it does not have,
        # so these are placed without the saturation the extents get
        vmin, vmax = vrange
        outlying_proportions = (beyond - vmin) / (vmax - vmin)
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
            xrange=vrange if horizontal else None,
            yrange=None if horizontal else vrange,
            width=chars.width,
            height=chars.height,
        )
        self.num_boxes = num_boxes
        self.vrange = vrange

    def __repr__(self):
        return f"boxes(<{self.num_boxes} groups>, {self.window!r})"
