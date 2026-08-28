"""
Specifying the data that goes into a plot.

Plot constructors are deliberately permissive about how data arrives: a single
array of points, a pair of coordinate sequences, or an axis object standing in
for one of the coordinates, each optionally paired with colors; a mapping from
dates to values, or a grid of them keyed by column. This module defines what is
accepted and normalises it before plotting.

Types:

* `number`: A scalar, Python or NumPy.
* `Series` and `Series3`: The accepted shapes for 2d and 3d point data. See
  these aliases for the full list of forms.
* `DateSeries`: The accepted shapes for values observed on dates.
* `TableData`: The accepted shapes for a grid of values to tabulate.

Special series:

* `axis`, and its subclasses `xaxis`, `yaxis` and `zaxis`: Stand-ins for a
  coordinate that runs over a range, so that a series can be given as one
  sequence of values against an axis rather than as two sequences.

Parsers:

* `parse_series`, `parse_series3`, and their `parse_multiple_*` variants: Turn
  any accepted form into arrays of points and colors.
* `parse_date` and `parse_date_series`: Turn any accepted form of dated data
  into a list of dates and an array of values.
* `parse_table_data`: Turn any accepted form of tabular data into the names of
  its columns and a list of rows, and `parse_per_column` spread a setting given
  for the table, per column, or by column name over one entry per column.

For turning 3d data into positions on a camera's film, see
`matthewplotlib.camera`.
"""

from __future__ import annotations

import dataclasses
import datetime
from collections.abc import Mapping
from typing import Any, Sequence, cast

import numpy as np
from numpy.typing import NDArray, ArrayLike

from matthewplotlib.colors import ColorSpec, parse_colors


# # # 
# Types


type number = int | float | np.integer | np.floating


type Series = (
    NDArray                                     # number[n,2]
    | tuple[NDArray, ColorSpec]                 # number[n,2], colors
    | tuple[ArrayLike, ArrayLike]               # number[n]^2
    | tuple[ArrayLike, ArrayLike, ColorSpec]    # number[n]^2, colors
    | axis                                      # axis
    | tuple[axis, ColorSpec]                    # axis, colors
)
"""
The accepted shapes for 2d point data.

Any of the following, for a series of n points, where the colors are one
`ColorSpec` for the whole series:

* `number[n,2]`: An array of coordinate pairs.
* `(number[n,2], colors)`: The same, coloured.
* `(number[n], number[n])`: The x and y coordinates as separate sequences.
* `(number[n], number[n], colors)`: The same, coloured.
* `axis`: An axis, standing in for points spaced along it, with the other
  coordinate held at zero.
* `(axis, colors)`: The same, coloured.
"""


type Series3 = (
    NDArray                                             # number[n,3]
    | tuple[NDArray, ColorSpec]                         # number[n,3], colors
    | tuple[ArrayLike, ArrayLike, ArrayLike]            # number[n]^3
    | tuple[ArrayLike, ArrayLike, ArrayLike, ColorSpec] # number[n]^3, colors
    | axis                                              # axis
    | tuple[axis, ColorSpec]                            # axis, uint8[n,rgb]
)
"""
The accepted shapes for 3d point data.

As `Series`, with a third coordinate:

* `number[n,3]`: An array of coordinate triples.
* `(number[n,3], colors)`: The same, coloured.
* `(number[n], number[n], number[n])`: The three coordinates as separate
  sequences.
* `(number[n], number[n], number[n], colors)`: The same, coloured.
* `axis`: An axis, standing in for points spaced along it, with the other two
  coordinates held at zero.
* `(axis, colors)`: The same, coloured.
"""


type DateLike = datetime.date | datetime.datetime | np.datetime64 | str
"""
The accepted spellings of a single date.

A `datetime.date`, a `datetime.datetime` or a NumPy `datetime64` (in each of
the latter two cases the time of day is discarded), or a string in ISO 8601
format such as `"2025-01-01"`.
"""


type DateSeries = (
    Mapping[DateLike, number]                   # {date: value}
    | tuple[Sequence[DateLike], ArrayLike]      # date[n], number[n]
    | tuple[DateLike, ArrayLike]                # first date, number[n]
)
"""
The accepted shapes for values observed on dates.

Any of the following, for n dated values:

* `{date: value}`: A mapping from dates to values.
* `(date[n], number[n])`: The dates and the values as separate sequences.
* `(date, number[n])`: One date, standing in for the n consecutive days
  starting there, and the values on those days.

The dates need not be sorted or contiguous, but no date may appear twice. Each
of them is any `DateLike`. A value that is not finite marks a date whose value
is unknown, as distinct from one whose value is zero.
"""


type TableData = (
    Sequence[Mapping[Any, Any]]         # a row a mapping, keyed by column
    | Mapping[Any, Sequence[Any]]       # a column a sequence, keyed by column
    | Sequence[Sequence[Any]]           # a row a sequence of values
    | NDArray                           # or a 2d array of them
)
"""
The accepted shapes for a grid of values to tabulate.

Any of the following:

* A sequence of mappings, one per row. The columns are the keys, in the order
  they are first seen, and a row missing one of them leaves that cell blank.
* A mapping from column to the values down it. A column shorter than the
  longest is blank where it runs out.
* A sequence of sequences, or a 2d array, one row of values each. These name
  no columns of their own.

The first two name their columns and the last does not, which is what decides
whether a `headers` argument picks columns out or names them.
"""


# # #
# Parsers


def parse_date(
    date: DateLike,
) -> datetime.date:
    """
    Reduce any accepted spelling of a date to a `datetime.date`.
    """
    match date:
        # datetime is a subclass of date, so it has to be matched first.
        case datetime.datetime():
            return date.date()
        case datetime.date():
            return date
        case np.datetime64():
            return date.astype("datetime64[D]").astype(datetime.date)
        case str():
            return datetime.date.fromisoformat(date)
        case _:
            raise TypeError(f"Invalid date {date!r}")


def parse_date_series(
    series: DateSeries, # DateSeries<n>
) -> tuple[
    list[datetime.date],    # date[n]
    NDArray,                # float[n]
]:
    """
    Turn any accepted form of dated data into a list of dates and an array of
    values, ordered by date.
    """
    match series:
        case Mapping():
            dates = [parse_date(date) for date in series]
            values = np.asarray(list(series.values()), dtype=float)
        case (first, values_):
            values = np.asarray(values_, dtype=float)
            if values.ndim != 1:
                raise ValueError(
                    f"expected one axis of values, not {values.ndim}"
                )
            # One date stands in for the consecutive days from there, whereas a
            # sequence of them names each day itself.
            if _is_date(first):
                start = parse_date(cast(DateLike, first))
                dates = [
                    start + datetime.timedelta(days=i)
                    for i in range(len(values))
                ]
            else:
                dates = [
                    parse_date(date)
                    for date in cast(Sequence[DateLike], first)
                ]
        case _:
            raise TypeError(f"Invalid DateSeries {series!r}")

    if len(dates) != len(values):
        raise ValueError(
            f"expected as many values as dates, not {len(values)} and "
            f"{len(dates)}"
        )
    order = sorted(range(len(dates)), key=dates.__getitem__)
    dates = [dates[i] for i in order]
    values = values[order]
    for earlier, later in zip(dates, dates[1:]):
        if earlier == later:
            raise ValueError(f"date {earlier} appears more than once")
    return dates, values


def _is_date(spec: object) -> bool:
    """Whether a single date is spelled here, rather than a sequence of them."""
    return isinstance(spec, (datetime.date, np.datetime64, str))


def _select_columns(
    available: list[Any],
    headers: Sequence[Any] | Mapping[Any, str] | None,
) -> tuple[list[Any], list[str]]:
    """
    Choose which of the keys carried by the data become columns, and what each
    one is called. A mapping renames as it selects; a sequence takes the keys
    as they are.
    """
    if headers is None:
        return list(available), [str(key) for key in available]
    if isinstance(headers, Mapping):
        keys = list(headers.keys())
        names = [str(name) for name in headers.values()]
    else:
        keys = list(headers)
        names = [str(key) for key in keys]
    for key in keys:
        if key not in available:
            raise ValueError(f"no column {key!r} in the data")
    return keys, names


def parse_table_data(
    data: TableData,
    headers: Sequence[Any] | Mapping[Any, str] | None,
) -> tuple[list[str] | None, list[list[Any]]]:
    """
    Standardise the accepted spellings of tabular data into the names of the
    columns, or None where the data carries none, and a list of rows.
    """
    # a mapping of columns to their values
    if isinstance(data, Mapping):
        keys, names = _select_columns(list(data.keys()), headers)
        columns = [list(data[key]) for key in keys]
        num_rows = max((len(column) for column in columns), default=0)
        rows = [
            [column[i] if i < len(column) else None for column in columns]
            for i in range(num_rows)
        ]
        return names, rows

    try:
        records = list(data)
    except TypeError:
        raise ValueError(
            "a table takes a list of dicts, a dict of lists, or a 2d array"
        )

    # a sequence of mappings, one per row, not necessarily sharing their keys
    if records and isinstance(records[0], Mapping):
        available: list[Any] = []
        for record in records:
            for key in record:
                if key not in available:
                    available.append(key)
        keys, names = _select_columns(available, headers)
        return names, [[record.get(key) for key in keys] for record in records]

    # a sequence of rows of values, which name no columns of their own
    if isinstance(headers, Mapping):
        raise ValueError(
            "headers renames columns the data names, but a 2d table names "
            "none: pass a list of names instead"
        )
    rows = []
    for i, record in enumerate(records):
        if isinstance(record, str) or not hasattr(record, "__iter__"):
            raise ValueError(f"row {i} of the table is not a row of values")
        rows.append(list(record))
    num_columns = len(rows[0]) if rows else 0
    for i, row in enumerate(rows):
        if len(row) != num_columns:
            raise ValueError(
                f"row {i} has {len(row)} values, but row 0 has {num_columns}"
            )
    if headers is None:
        return None, rows
    names = [str(name) for name in headers]
    if rows and len(names) != num_columns:
        raise ValueError(
            f"got {len(names)} headers for {num_columns} columns"
        )
    return names, rows


def parse_per_column(
    spec: Any,
    names: list[str] | None,
    num_columns: int,
    what: str,
) -> list[Any]:
    """
    Spread a specification given for the whole table, for each column in turn,
    or for columns picked out by name, into one entry per column.
    """
    if spec is None:
        return [None] * num_columns
    if isinstance(spec, Mapping):
        if names is None:
            raise ValueError(
                f"{what} was given per column name, but the table has no "
                "headers"
            )
        for name in spec:
            if name not in names:
                raise ValueError(
                    f"{what} was given for {name!r}, which is not a column"
                )
        return [spec.get(name) for name in names]
    if isinstance(spec, str) or callable(spec):
        return [spec] * num_columns
    entries = list(spec)
    if len(entries) != num_columns:
        raise ValueError(
            f"{what} has {len(entries)} entries for {num_columns} columns"
        )
    return entries

def parse_series(
    series: Series, # Series<n>
) -> tuple[
    NDArray,        # number[n]
    NDArray,        # number[n]
    NDArray,        # uint8[n,3]
]:
    match series:
        case axis() as a:
            xs = a.xs
            ys = a.ys
            cs = parse_colors(None, a.n)
        case (axis() as a, cs_):
            xs = a.xs
            ys = a.ys
            cs = parse_colors(cast(ColorSpec, cs_), a.n)
        case np.ndarray(shape=(n, 2)) as a:
            xs = a[:, 0]
            ys = a[:, 1]
            cs = parse_colors(None, n)
        case (np.ndarray(shape=(n, 2)) as a, cs_):
            xs = a[:, 0]
            ys = a[:, 1]
            cs = parse_colors(cast(ColorSpec, cs_), n)
        case (xs_, ys_):
            xs = np.asarray(xs_)
            ys = np.asarray(ys_)
            n, = xs.shape
            cs = parse_colors(None, n)
        case (xs_, ys_, cs_):
            xs = np.asarray(xs_)
            ys = np.asarray(ys_)
            n, = xs.shape
            cs = parse_colors(cast(ColorSpec, cs_), n)
        case _:
            raise TypeError(f"Invalid Series {series!r}")
    return xs, ys, cs

            
def parse_multiple_series(
    *seriess: Series,
) -> tuple[
    NDArray,    # number[N]
    NDArray,    # number[N]
    NDArray,    # uint8[N,3]
]:
    xss, yss, css = zip(*map(parse_series, seriess))
    return (
        np.concatenate(xss),
        np.concatenate(yss),
        np.concatenate(css),
    )

    
def parse_segments(
    *seriess: Series,
) -> tuple[
    NDArray,    # number[m, 2]
    NDArray,    # number[m, 2]
    NDArray,    # uint8[m, 3]
    NDArray,    # uint8[m, 3]
]:
    """
    Turn series into the segments joining their consecutive points, with the
    colors at each end of each.

    Where `parse_multiple_series` pools every series into one cloud of points,
    this pairs the points up within each series first: a series is one stroke
    of the pen, and the last point of one is never joined to the first point of
    the next.
    """
    return _pair_up([parse_series(s) for s in seriess], dimensions=2)


def parse_series3(
    series: Series3, # Series3<n>
) -> tuple[
    NDArray,        # number[n]
    NDArray,        # number[n]
    NDArray,        # number[n]
    NDArray,        # uint8[n,3]
]:
    match series:
        case axis() as a:
            xs = a.xs
            ys = a.ys
            zs = a.zs
            cs = parse_colors(None, a.n)
        case (axis() as a, cs_):
            xs = a.xs
            ys = a.ys
            zs = a.zs
            cs = parse_colors(cast(ColorSpec, cs_), a.n)
        case np.ndarray(shape=(n, 3)) as a:
            xs = a[:, 0]
            ys = a[:, 1]
            zs = a[:, 2]
            cs = parse_colors(None, n)
        case (np.ndarray(shape=(n, 3)) as a, cs_):
            xs = a[:, 0]
            ys = a[:, 1]
            zs = a[:, 2]
            cs = parse_colors(cast(ColorSpec, cs_), n)
        case (xs_, ys_, zs_):
            xs = np.asarray(xs_)
            ys = np.asarray(ys_)
            zs = np.asarray(zs_)
            n, = xs.shape
            cs = parse_colors(None, n)
        case (xs_, ys_, zs_, cs_):
            xs = np.asarray(xs_)
            ys = np.asarray(ys_)
            zs = np.asarray(zs_)
            n, = xs.shape
            cs = parse_colors(cast(ColorSpec, cs_), n)
        case _:
            raise TypeError(f"Invalid Series3 {series!r}")
    return xs, ys, zs, cs


def parse_multiple_series3(
    *seriess: Series3,
) -> tuple[
    NDArray,    # number[N]
    NDArray,    # number[N]
    NDArray,    # number[N]
    NDArray,    # uint8[N,3]
]:
    xss, yss, zss, css = zip(*map(parse_series3, seriess))
    return (
        np.concatenate(xss),
        np.concatenate(yss),
        np.concatenate(zss),
        np.concatenate(css),
    )

    
def parse_segments3(
    *seriess: Series3,
) -> tuple[
    NDArray,    # number[m, 3]
    NDArray,    # number[m, 3]
    NDArray,    # uint8[m, 3]
    NDArray,    # uint8[m, 3]
]:
    """
    Turn 3d series into the segments joining their consecutive points, with the
    colors at each end of each. See `parse_segments`.
    """
    return _pair_up([parse_series3(s) for s in seriess], dimensions=3)


def _pair_up(
    polylines: Sequence[tuple[NDArray, ...]],
    dimensions: int,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """The consecutive pairs of each parsed series, pooled after pairing."""
    starts = []
    ends = []
    start_colors = []
    end_colors = []
    for polyline in polylines:
        *coordinates, colors = polyline
        points = np.stack(coordinates, axis=1)
        starts.append(points[:-1])
        ends.append(points[1:])
        start_colors.append(colors[:-1])
        end_colors.append(colors[1:])
    empty_points = np.zeros((0, dimensions))
    empty_colors = np.zeros((0, 3), dtype=np.uint8)
    return (
        np.concatenate([*starts, empty_points]),
        np.concatenate([*ends, empty_points]),
        np.concatenate([*start_colors, empty_colors]),
        np.concatenate([*end_colors, empty_colors]),
    )


# # # 
# Special series


@dataclasses.dataclass(frozen=True)
class axis:
    a: number = 0.
    b: number = 1.
    n: int = 10
    
    @property
    def xs(self) -> NDArray:
        return np.zeros(self.n)

    @property
    def ys(self) -> NDArray:
        return np.zeros(self.n)
    
    @property
    def zs(self) -> NDArray:
        return np.zeros(self.n)


class xaxis(axis):
    """
    A series of `n` points evenly spaced from `a` to `b` along the x
    axis, with the y and z coordinates held at zero.

    Accepted wherever a series is, so that something can be plotted against an
    axis without building coordinates for it by hand.
    """

    @property
    def xs(self) -> NDArray:
        return np.linspace(self.a, self.b, self.n)


class yaxis(axis):
    """
    A series of `n` points evenly spaced from `a` to `b` along the y
    axis, with the x and z coordinates held at zero.

    Accepted wherever a series is, so that something can be plotted against an
    axis without building coordinates for it by hand.
    """

    @property
    def ys(self) -> NDArray:
        return np.linspace(self.a, self.b, self.n)


class zaxis(axis):
    """
    A series of `n` points evenly spaced from `a` to `b` along the z
    axis, with the x and y coordinates held at zero.

    Accepted wherever a series is, so that something can be plotted against an
    axis without building coordinates for it by hand.
    """

    @property
    def zs(self) -> NDArray:
        return np.linspace(self.a, self.b, self.n)
