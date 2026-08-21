# Tables, and the characters where their rules cross

Designed 2026-08-20 by MFR and Claude (Opus 5), written up by Claude, while
adding `table`. The decisions are MFR's; the Unicode coverage below is
Claude's, and is reproducible from the snippets.

There was no way to put a table of numbers next to a plot. The roadmap asked
for one taking a list of dicts, a dict of lists, or a 2d array, with
configurable format strings. What made it more than a formatting exercise is
that a table has rules inside it, and nothing in the library could draw those.

## Why `BoxStyle` is the wrong shape for it

A `BoxStyle` is eight characters: four edges and four corners. A table needs
eleven more --- `┬ ┴ ├ ┤ ┼` and the rest --- because its rules meet in the
middle, not only at the corners.

The obvious fix, a `TableStyle` of eleven characters, was drafted and thrown
away. `axes` had just stopped reading its characters out of a style string for
the same reason: a corner's glyph depends on which of its neighbours are
drawn, and a string cannot say that. `unicode_frame` derives each character
from the set of arms meeting in its cell, indexing a `LineStyle` by a four-bit
mask, and a table wants exactly that mechanism one dimension further in. See
the `axes-sides` note for where the arm model came from.

So `unicode_grid` generalises `unicode_frame` from four sides to a grid of
`nrows + 1` horizontal rules by `ncols + 1` vertical ones. Each rule is two
things: whether it takes a row or column of cells, and which weight of line,
if any, is drawn in it. A rule that takes no cells does not appear; one that
takes cells and draws nothing is blank space, which any rule crossing it still
runs through.

## Rules span, so the arms are simpler than a frame's

`unicode_frame` finds a cell's arms by looking at its neighbours, because its
ruled cells form a thin loop. In a grid a drawn rule occupies its whole row or
column, cells and all, so a cell's arms follow from the rules through it
without looking anywhere:

    ruled[y, x] = h_drawn[y] or v_drawn[x]
    left, right arms  where h_drawn[y]
    up, down arms     where v_drawn[x]

with one exception at the boundary. A rule that ends at the edge of the grid
should fill its last cell: `╶────╴` leaves a visible gap at both ends in most
fonts, and a booktabs rule wants to run edge to edge. A rule that *turns* at
the edge should not, or the corner cannot be a corner. The two are told apart
by whether the boundary cell belongs to a drawn rule of the other orientation:

    drop the left arm at x = 0 only if v_drawn[0]

which is three lines and covers every case. The same vocabulary `LineStyle`
uses for `DOUBLE` --- a line that ends "runs to the edge of its final cell
instead of stopping halfway" --- applies here for the same reason.

## Why single and double, and no other pair

A table crosses weights where a frame never has to: a light column rule
passing through a double midrule needs `╪`, and a double left rule meeting a
light row rule needs `╟`. Neither weight's own `LineStyle` has those.

Unicode has the complete mixed set for single and double, in both directions.
Every character the two sets need exists, and each crossing is named for the
two weights meeting in it:

    python3 - <<'EOF'
    import unicodedata
    for s in (" ╵╷│═╛╕╡═╘╒╞═╧╤╪", " ║║║╴╜╖╢╶╙╓╟─╨╥╫"):
        print([unicodedata.name(c) for c in s if c != " "])
    EOF

What Unicode does *not* have is a cell whose two arms along one axis differ in
weight --- up single and down double, say. That is not a gap a rectangular
table can fall into: each horizontal rule has one weight along its whole
length and each vertical rule likewise, so the two vertical arms of any cell
always agree, and so do the two horizontal ones. Corners, where only one arm
per axis exists, are covered outright (`╒ ╓ ╕ ╖ ╘ ╙ ╛ ╜`).

Light and heavy has the same complete coverage and would extend the same way.
Round does not: round differs from light only at its corners, and there are no
round-and-double corners, so a mixture would silently lose the rounding.
`unicode_grid` therefore refuses anything but `LIGHT` and `DOUBLE`, by name,
rather than drawing something subtly wrong.

## The rule vocabulary

Eight rules, each `None` to infer or one of four words:

| word | means |
| --- | --- |
| `"skip"` | no row or column at all |
| `"blank"` | a row or column of space, which crossing rules run through |
| `"single"` | a light line |
| `"double"` | a double line |

`toprule`, `midrule`, `rowrule`, `bottomrule` across; `leftrule`, `indexrule`,
`colrule`, `rightrule` down. Unspecified, they give booktabs: a line above, a
double line under the header, a line below, and nothing else.

`midrule` and `indexrule` are the two that name a particular boundary rather
than a class of them --- header from body, labels from values. `indexrule` was
not in the first draft, and without it `colrule` is the only way to rule after
the row labels, which also rules between every data column. It was added for
the symmetry: `midrule` and `indexrule` separate the labels, `rowrule` and
`colrule` are the interior, and the other four are the edges.

Asking for a rule the table has no boundary for raises rather than being
ignored: `midrule` with no header row, `indexrule` with no index. A rule that
simply has nowhere to go this time --- `rowrule` on a table with one row ---
does not, because how many rows the data has is not a mistake.

Rejected: a single `rules="none" | "header" | "rows" | "grid"` enum plus a
`frame` flag. It cannot express a double midrule inside a light grid, which is
the default this design settled on, and every table wanting something slightly
off the menu would have needed a new word on it.

## Padding

`cell_padding` holds a value away from the rule on each side of it, so two
neighbouring cells give a two-column gutter where no rule separates them. At
an outer edge with no rule there is nothing to be held away from, so the
padding is dropped and the table starts flush with its first column. Uniform
padding put a stray space down the left of every unruled table.

## Defaults that are opinions

* A float shows four significant figures. Unformatted floats in a table are
  unreadable, and `str` is not a defensible default just because it is the
  neutral one.
* A column holding nothing but numbers is aligned right, so its digits line
  up; everything else left. Blanks are ignored when deciding, so one missing
  value does not left-align a column of numbers. A header follows its column.
* `None` is a blank cell whatever the format says, rather than `"None"` or a
  raised `TypeError` from `format(None, ".3f")`.

## Not in this design

* **A plot in a cell**, so a row could carry a sparkline. The row heights and
  column widths are computed per cell and a cell is already a `CharArray`, so
  the machinery would take it; what is missing is what a plot's width means
  when the column is sized to its contents.
* **Heavy and round rules.** Heavy would be a third entry in the joint table;
  round cannot be mixed with double at all.
* **An array of colour names.** `colors` and `bgcolors` go through
  `parse_colors`, which takes one `ColorLike` broadcast over the table or an
  array of numbers, but not an array of names: `[["cyan", "cyan"]]` raises
  `invalid colors of type <U7`. That belongs to the colour-specification
  roadmap item, not here.
* **Wrapping a long cell.** `max_col_width` cuts and marks with an ellipsis.
  Wrapping would make a row's height depend on its columns' widths, which are
  themselves computed from the contents.
