"""
Plots that lay values out over the days of a calendar: a block per month,
or one unbroken strip of weeks.

* `calendar`
* `weeks`
"""
from __future__ import annotations

import calendar as _calendar
import datetime

import numpy as np

from typing import NamedTuple
from numpy.typing import NDArray
from matthewplotlib.colormaps import ColorMap
from matthewplotlib.colors import ColorLike, parse_color, parse_colors
from matthewplotlib.data import (
    number,
    DateLike,
    DateSeries,
    parse_date,
    parse_date_series,
)
from matthewplotlib.scales import scale, _resolve_vrange
from matthewplotlib.core import CharArray, ords
from matthewplotlib.plots.base import plot, wrap


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
    vrange: scale


def _color_days(
    data: DateSeries,
    vrange: tuple[number, number] | scale | None,
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
    vscale = _resolve_vrange(vrange, levels, what)
    rgb = parse_colors(
        vscale(levels),
        n=len(levels),
        colormap=colormap,
    )
    return _ColoredDays(
        colors={date: color for (date, _), color in zip(drawn, rgb)},
        first=first,
        last=last,
        vrange=vscale,
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
    * vrange : optional (number, number) | scale.
        The interval of values the colormap covers: values at the first limit
        or below come out at the bottom of the colormap and values at the
        second or above come out at the top. By default the interval runs from
        the lowest to the highest value among the days drawn, so that the
        colours span the data. A `scale` says how the colours are spaced
        within the interval, where a plain pair spaces them linearly.
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
        vrange: tuple[number, number] | scale | None = None,
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
        columns = filled * cell_width - month_spacing * day_width
        super().__init__(grid.crop(
            below=month_spacing,
            right=grid.width - columns,
        ))
        self.vrange = dated.vrange
        self.daterange = (first, last)
        self.num_days = len(colors)

    def __repr__(self):
        first, last = self.daterange
        return (
            f"calendar(height={self.height}, width={self.width}, "
            f"data=<{self.num_days} days from {first} to {last} on "
            f"{self.vrange!r}>)"
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
    * vrange : optional (number, number) | scale.
        The interval of values the colormap covers: values at the first limit
        or below come out at the bottom of the colormap and values at the
        second or above come out at the top. By default the interval runs from
        the lowest to the highest value among the days drawn, so that the
        colours span the data. A `scale` says how the colours are spaced
        within the interval, where a plain pair spaces them linearly.
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
        vrange: tuple[number, number] | scale | None = None,
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
                        chars=chars,
                        row=0,
                        column=column,
                        caption=f"{year:4d}",
                        width=band_width,
                    )
                if month_labels:
                    _write_caption(
                        chars=chars,
                        row=int(year_labels),
                        column=column,
                        caption=month_caption,
                        width=band_width,
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
        self.daterange = (first, last)
        self.num_days = len(colors)
        self.num_weeks = num_weeks

    def __repr__(self):
        first, last = self.daterange
        return (
            f"weeks(height={self.height}, width={self.width}, "
            f"data=<{self.num_days} days over {self.num_weeks} weeks from "
            f"{first} to {last} on {self.vrange!r}>)"
        )
