"""
Values arranged in labelled rows and columns.

* `table`
* `Rule`: what a `table` draws between its cells.
"""
from __future__ import annotations

import numpy as np

from typing import Any, Callable, Literal
from collections.abc import Mapping, Sequence
from numbers import Number
from numpy.typing import ArrayLike
from matthewplotlib.colormaps import ColorMap
from matthewplotlib.colors import ColorLike, parse_colors
from matthewplotlib.data import (
    TableData,
    parse_table_data,
    parse_per_column,
)
from matthewplotlib.core import (
    _validate_text,
    Align,
    LineStyle,
    unicode_text,
    unicode_grid,
)
from matthewplotlib.plots.base import plot


type Rule = Literal["skip", "blank", "single", "double"]
"""
What is drawn along one of a `table`'s rules, in increasing order of what it
costs and what it shows.

* `"skip"`: nothing at all, taking no row or column.
* `"blank"`: a row or column of space, which any rule crossing it still runs
  through.
* `"single"`: a light line.
* `"double"`: a double line.
"""


_RULE_WEIGHTS: dict[Rule, tuple[bool, LineStyle | None]] = {
    "skip":   (False, None),
    "blank":  (True,  None),
    "single": (True,  LineStyle.LIGHT),
    "double": (True,  LineStyle.DOUBLE),
}


def _format_cell(value: Any, spec: str | Callable[[Any], str] | None) -> str:
    """
    Turn one value into the text that fills its cell. A missing value is
    blank, whatever the format asked for.
    """
    if value is None:
        return ""
    if spec is None:
        if isinstance(value, (float, np.floating)):
            return f"{value:.4g}"
        return str(value)
    if callable(spec):
        return str(spec(value))
    if "{" in spec:
        return spec.format(value)
    return format(value, spec)


def _auto_align(values: list[Any]) -> Align:
    """
    Right-align a column of numbers, so that its digits line up, and anything
    else to the left. A column of nothing but blanks aligns left.
    """
    numeric = False
    for value in values:
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, Number):
            return "left"
        numeric = True
    return "right" if numeric else "left"


def _cell_lines(text: str, max_width: int | None) -> list[str]:
    """
    Split the text of one cell into its lines, each cut to the widest a column
    may be, with an ellipsis marking what was taken off.
    """
    _validate_text(text, allow_line_breaks=True)
    lines = text.splitlines() or [""]
    if max_width is None:
        return lines
    if max_width < 1:
        raise ValueError(f"a column cannot be {max_width} characters wide")
    return [
        line if len(line) <= max_width else line[:max_width - 1] + "…"
        for line in lines
    ]


class table(plot):
    """
    A grid of values, formatted into aligned columns and ruled.

    Inputs:

    * data : list of dicts | dict of lists | 2d array.
        The values to tabulate, in any of three spellings:
        * A sequence of mappings, one per row. The columns are the keys, in
          the order they are first seen, and a row missing one of them leaves
          that cell blank.
        * A mapping from column to the values down it. A column shorter than
          the longest is blank where it runs out.
        * A sequence of sequences, or a 2d array, one row of values each.
          These name no columns of their own.
    
    * headers : optional list of str, or dict.
        The columns to show. Where the data names its columns, a list picks
        them out and orders them, and a mapping from key to name does that and
        renames them as well. Where the data names none, a list names them,
        and without one the table has no header row.
    
    * index : optional list of str.
        Labels for the rows, drawn in a column of their own before the first.
        The columns everything else is specified per-column for do not count
        this one.
    
    * index_name : str (default: "").
        The header over the index column.
    
    * formats : optional str | callable | list | dict.
        How to turn a value into the text of its cell. One of:
        * A format spec, as `format` takes, like `".3f"`.
        * A template with a field in it, as `str.format` takes, like
          `"{:.1%}"`.
        * A function from a value to a string.
        * A list of any of those, one per column, or None to leave a column
          formatted as it would be by default.
        * A mapping from a column's header to any of those.

        By default a float is shown to four significant figures, and anything
        else as `str` shows it. A value of None is blank whatever the format.

    * aligns : optional Align | list | dict.
        Where to put a value in its cell. One for the whole table, a list with
        one per column, or a mapping from a column's header. By default a
        column holding nothing but numbers is aligned right, so that its digits
        line up, and every other column left. A header follows its column.
    
    * toprule : optional Rule.
        The rule above the header, `"single"` by default.
    
    * midrule : optional Rule.
        The rule between the header and the body, `"double"` by default. Only
        a table with a header row has one.
    
    * rowrule : optional Rule.
        The rule between one body row and the next, `"skip"` by default.
    
    * bottomrule : optional Rule.
        The rule below the body, `"single"` by default.
    
    * leftrule : optional Rule.
        The rule down the left of the table, `"skip"` by default.
    
    * indexrule : optional Rule.
        The rule between the index column and the body, `"skip"` by default.
        Only a table with an index has one.
    
    * colrule : optional Rule.
        The rule between one column and the next, `"skip"` by default.
    
    * rightrule : optional Rule.
        The rule down the right of the table, `"skip"` by default.
    
    * max_col_width : optional int.
        The widest a column of text may be. Anything longer is cut, with an
        ellipsis marking what was taken off. By default a column is as wide as
        the longest thing in it.
    
    * cell_padding : int (default: 1).
        Columns of space between a value and the rule on each side of it. Two
        neighbouring cells give twice this much space between columns that
        have no rule between them. An outer edge with no rule on it is not
        padded, so that the table starts flush with its first column.
    
    * color : optional ColorLike.
        The color of everything in the table that is not given a color of its
        own. Defaults to the terminal's default foreground color.
    
    * bgcolor : optional ColorLike.
        The color behind the whole table. Defaults to a transparent background.
    
    * header_color : optional ColorLike.
        The color of the header row and the index column. Defaults to `color`.
    
    * rule_color : optional ColorLike.
        The color of the rules. Defaults to `color`.
    
    * colors : optional ColorLike[nrows, ncols].
        The color of the text in each body cell, not counting the header row
        or the index column.
    
    * bgcolors : optional ColorLike[nrows, ncols].
        The color behind each body cell. Together with a colormap this shades
        a table by its values, so that it reads as a heatmap that can still be
        read off exactly.
    
    * colormap : optional ColorMap.
        Applied to `colors` and `bgcolors` before either is read as colors, so
        that they can be given as the data the table is showing.


    Notes:

    * A cell whose text has newlines in it takes as many lines as it needs.
      Every other cell in its row grows to match.
    """
    def __init__(
        self,
        data: TableData,
        headers: Sequence[Any] | Mapping[Any, str] | None = None,
        index: Sequence[str] | None = None,
        index_name: str = "",
        formats: (
            str
            | Callable[[Any], str]
            | Sequence[str | Callable[[Any], str] | None]
            | Mapping[str, str | Callable[[Any], str]]
            | None
        ) = None,
        aligns: (
            Align
            | Sequence[Align | None]
            | Mapping[str, Align]
            | None
        ) = None,
        toprule:    Rule | None = None,
        midrule:    Rule | None = None,
        rowrule:    Rule | None = None,
        bottomrule: Rule | None = None,
        leftrule:   Rule | None = None,
        indexrule:  Rule | None = None,
        colrule:    Rule | None = None,
        rightrule:  Rule | None = None,
        max_col_width: int | None = None,
        cell_padding: int = 1,
        color: ColorLike | None = None,
        bgcolor: ColorLike | None = None,
        header_color: ColorLike | None = None,
        rule_color: ColorLike | None = None,
        colors: ArrayLike | None = None,
        bgcolors: ArrayLike | None = None,
        colormap: ColorMap | None = None,
    ):
        names, body = parse_table_data(data, headers)
        num_rows = len(body)
        num_columns = len(body[0]) if body else len(names or ())
        if num_columns == 0:
            raise ValueError("a table needs at least one column")
        if index is not None and len(index) != num_rows:
            raise ValueError(
                f"got {len(index)} index labels for {num_rows} rows"
            )
        if index_name and index is None:
            raise ValueError("index_name was given but the table has no index")
        if cell_padding < 0:
            raise ValueError(f"cannot pad a cell by {cell_padding} columns")

        # what fills each cell, and where in it
        column_formats = parse_per_column(
            formats, names, num_columns, "formats",
        )
        column_aligns = parse_per_column(
            aligns, names, num_columns, "aligns",
        )
        for j, align in enumerate(column_aligns):
            if align is None:
                column_aligns[j] = _auto_align([row[j] for row in body])
            elif align not in ("left", "center", "right"):
                raise ValueError(f"cannot align a column {align!r}")

        # the text of every cell, the header row and index column included
        text: list[list[list[str]]] = []
        if names is not None:
            text.append([_cell_lines(name, max_col_width) for name in names])
        for row in body:
            text.append([
                _cell_lines(
                    _format_cell(value, column_formats[j]),
                    max_col_width,
                )
                for j, value in enumerate(row)
            ])
        if index is not None:
            labels = [str(label) for label in index]
            if names is not None:
                labels.insert(0, index_name)
            for cells, label in zip(text, labels):
                cells.insert(0, _cell_lines(label, max_col_width))
            column_aligns.insert(0, "left")

        # which rules the table has, and which line each one draws
        has_header = names is not None
        has_midrule = has_header and num_rows > 0
        grid_rows = has_header + num_rows
        grid_columns = (index is not None) + num_columns
        if midrule is not None and not has_header:
            raise ValueError(
                "midrule was given but the table has no header row"
            )
        if indexrule is not None and index is None:
            raise ValueError("indexrule was given but the table has no index")
        horizontal = []
        for i in range(grid_rows + 1):
            if i == 0:
                rule = toprule if toprule is not None else "single"
            elif i == grid_rows:
                rule = bottomrule if bottomrule is not None else "single"
            elif i == 1 and has_midrule:
                rule = midrule if midrule is not None else "double"
            else:
                rule = rowrule if rowrule is not None else "skip"
            horizontal.append(rule)
        vertical = []
        for j in range(grid_columns + 1):
            if j == 0:
                rule = leftrule if leftrule is not None else "skip"
            elif j == grid_columns:
                rule = rightrule if rightrule is not None else "skip"
            elif j == 1 and index is not None:
                rule = indexrule if indexrule is not None else "skip"
            else:
                rule = colrule if colrule is not None else "skip"
            vertical.append(rule)
        for rule in (*horizontal, *vertical):
            if rule not in _RULE_WEIGHTS:
                raise ValueError(
                    f"a rule is skip, blank, single or double, not {rule!r}"
                )
        hcells, hrules = zip(*(_RULE_WEIGHTS[rule] for rule in horizontal))
        vcells, vrules = zip(*(_RULE_WEIGHTS[rule] for rule in vertical))

        # a cell is padded away from the rule on each side of it, and an outer
        # edge with no rule on it has nothing to be held away from
        pads_left = [cell_padding] * grid_columns
        pads_right = [cell_padding] * grid_columns
        if vertical[0] == "skip":
            pads_left[0] = 0
        if vertical[-1] == "skip":
            pads_right[-1] = 0

        # every cell in a row is as tall as the tallest, and every cell in a
        # column as wide as the widest
        heights = [max(len(cell) for cell in cells) for cells in text]
        widths = [
            max(max((len(line) for line in cells[j]), default=0)
                for cells in text)
            for j in range(grid_columns)
        ]

        # the colors of the body cells, which the header and index do not take
        body_colors = None if colors is None else parse_colors(
            colors,
            shape=(num_rows, num_columns),
            colormap=colormap,
        )
        body_bgcolors = None if bgcolors is None else parse_colors(
            bgcolors,
            shape=(num_rows, num_columns),
            colormap=colormap,
        )
        if header_color is None:
            header_color = color
        if rule_color is None:
            rule_color = color

        # draw each cell into the size its row and column settled on
        cells_chars = []
        for i, cells in enumerate(text):
            row_chars = []
            for j, lines in enumerate(cells):
                is_label = (has_header and i == 0) or (
                    index is not None and j == 0
                )
                # where the body starts, once the header row and the index
                # column have been counted out of the grid
                body_row = i - has_header
                body_column = j - (index is not None)
                if is_label:
                    fgcolor = header_color
                    cell_bgcolor = bgcolor
                else:
                    fgcolor = (
                        color if body_colors is None
                        else body_colors[body_row, body_column]
                    )
                    cell_bgcolor = (
                        bgcolor if body_bgcolors is None
                        else body_bgcolors[body_row, body_column]
                    )
                # the cell at the size its row and column settled on, then
                # held away from the rule on each side of it
                row_chars.append(unicode_text(
                    lines=lines,
                    height=heights[i],
                    width=widths[j],
                    align=column_aligns[j],
                    fgcolor=fgcolor,
                    bgcolor=cell_bgcolor,
                ).pad(
                    left=pads_left[j],
                    right=pads_right[j],
                    fgcolor=fgcolor,
                    bgcolor=cell_bgcolor,
                ))
            cells_chars.append(row_chars)

        chars = unicode_grid(
            cells=cells_chars,
            hcells=hcells,
            hrules=hrules,
            vcells=vcells,
            vrules=vrules,
            fgcolor=rule_color,
            bgcolor=bgcolor,
        )
        super().__init__(chars=chars)
        self.headers = names
        self.num_rows = num_rows
        self.num_columns = num_columns

    def __repr__(self):
        return (
            f"table(height={self.height}, width={self.width}, "
            f"data=<{self.num_rows} rows by {self.num_columns} columns>)"
        )
