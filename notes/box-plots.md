# Box plots, and candles as one of them

Designed 2026-08-20 by MFR and Claude (Opus 5), in conversation, written up by
Claude. Agreed between us; none of it is built. The measurements are Claude's
and are reproducible from the snippets at the end. `candles` exists already and
this design absorbs it.

## One mark, six switches

A candlestick and a box plot are the same picture with different switches
thrown. Per category, a mark is:

* an **outer interval**, drawn thin;
* an **inner interval**, drawn thick;
* optionally **caps** across the ends of the outer interval;
* optionally an **interior mark** at one value inside the inner interval;
* optionally **outlier points** beyond the outer interval;
* a **colour**.

|              | outer      | inner      | caps | interior | outliers | colour from |
|--------------|------------|------------|------|----------|----------|-------------|
| candle       | low–high   | open–close | no   | none     | no       | direction   |
| box plot     | whiskers   | Q1–Q3      | yes  | median   | yes      | the group   |

Intermediate settings mean things too: caps without an interior mark is a range
plot. So the switches are the design, and the two named plots are presets over
it.

## Which way a box lies

The parameter is spelled by the direction of **one box**, not of the row of
them:

    box_direction = "horizontal"    boxes lie flat, stacked up the screen
    box_direction = "vertical"      boxes stand up, marching across

Verbose on purpose. `"horizontal"` alone is ambiguous --- it does not say
whether it describes a box or the series of boxes --- and naming the axis
instead (`values="x"`) was rejected because MFR would rather the library not
name axes `x` and `y` where it can avoid it.

`candles` inherits this and so gains a horizontal form for nothing.

**Horizontal is the default for box plots**, for two independent reasons, both
about resolution along the value axis:

* terminals are wider than they are tall, so a value axis laid across 80
  columns has 80 mark positions where one laid down 24 rows has 24;
* character cells are taller than they are wide --- the bundled font is 8 by
  16, which `CharArray.to_rgba_array` shows by returning
  `uint8[height*16, width*8, 4]` --- so a horizontal value axis resolves twice
  as finely per unit of distance on screen, whatever shape the terminal is.

It is paid for in category capacity. A box needs three cells across it plus a
gap, so an 80 by 24 terminal holds about 6 horizontal boxes against about 20
vertical ones. Box plots usually have a handful of groups and want the quartiles
readable, so the trade goes this way; `candles` has many periods and short
bodies, so its default goes the other way.

## Filled or outlined

Both, chosen by a parameter, defaulting to **outlined**.

**Outlined** is box-drawing characters, the corners and junctions derived from
an arm mask exactly as `unicode_frame` does it for `axes`. One mark position per
cell. Needs no background, so it blends into the terminal like every other plot.
It is also the shape people recognise as a box plot, which is why it is the
default even though it is the coarser of the two. Minimum thickness 3, since a
box needs two edges and an interior.

**Filled** is the machinery `candles` already has: partial blocks, half of the
eighths drawn as negatives against a named background. Eight mark positions per
cell. Minimum thickness 1, so distributions can be stacked densely. Requires a
background colour, for the reason `notes/candlesticks.md` sets out.

A thickness below the minimum is an error rather than a silent promotion.

## The median, which is the interesting part

The obstacle looked fatal and is not. A body's edge cell can only show two
regions, because a partial block is a contiguous fill; but a **line** glyph is a
thin band with background on either side, so one glyph gives three regions. The
median is that line, drawn in its own colour over a cell whose background is the
box colour. In a filled box it needs no gap in the fill and no extra pass.

Measured in the bundled font, the thin bands available inside one cell:

    ▔ U+2594 upper one eighth   rows  0-1      ▏ U+258F left one eighth   col  0
    ─ U+2500 light horizontal   rows  7-8      │ U+2502 light vertical    cols 3-4
    ▁ U+2581 lower one eighth   rows 14-15     ▕ U+2595 right one eighth  col  7

So a median has **three positions per character cell**: against the cell's near
edge, in the middle, against its far edge. The two sets mirror each other, so
the mechanism is the same whichever way the box lies.

That leaves filled boxes at eighths for their edges and thirds for their median.
The mismatch is mild and in the tolerable direction: over a box four cells long
that is twelve median positions, and the median rounds by at most 0.19 of a cell
against the edges' 0.06. A median has one job, which is to show where the middle
of the distribution sits between the quartiles, and twelve positions do it.

**Where there is no room, the median is dropped.** A box shorter than the
positions available cannot place a line inside itself, and a line drawn anyway
would be a line about nothing. `candles` already does this with the wick that
disappears into the cell holding a body edge, so it is a precedent and not a new
exception.

**The median's weight should be a `LineStyle`, not a fixed choice.** The pairing
of the mid-cell line against the eighth slivers is not the same in every font.
In `unscii16` the horizontal set is exactly matched, all three bands two pixels
of sixteen, while the vertical mid-cell line is two pixels against the slivers'
one and there is nothing lighter to reach for. In an ordinary terminal font the
discrepancy runs the other way, `─` being a hairline where an eighth block is a
bar. Since `axes` and the candle wicks already take a `LineStyle`, so should
this, and the default should be chosen by looking at it in a real terminal
rather than derived from either font alone.

## Statistics

Raw samples only: a sequence of one-dimensional arrays, possibly ragged, with
the quantiles computed for you. This matches `histogram`, which takes values and
bins them. Accepting precomputed five-number summaries can come later if
something wants it.

Whiskers take a parameter, `whis`, defaulting to `1.5`: Tukey's rule, whiskers
reaching the furthest sample within 1.5 times the interquartile range of the
quartiles, with everything beyond drawn individually as outlier points.
`whis=None` gives min-to-max whiskers and no outliers. The default draws marks
some readers will not expect, which is accepted because it is the standard and
because the outliers are usually the informative part.

## Order of work, and why

Build the box plot first, then move `candles` onto whatever the two turn out to
share. Not the other way around: the shared base should be extracted from two
working plots rather than guessed from one, and the guess is especially unsafe
here, because how much is shared depends on the fill-or-outline choice that only
the box plot makes. `candles` is safe to refactor --- 44 unit tests and an
example snapshot fence its behaviour, and it is unreleased, so its signature is
still free to change.

## Still open

* **The size vocabulary.** `width` and `height` cannot survive a parameterised
  orientation, since each would mean a different thing per setting. Proposed:
  `length` along the value axis, `thickness` across one mark, `spacing`
  between marks. `length` is the unloved one; `extent` and `span` are the
  alternatives. A fixed-orientation preset like `candles` may be better off
  keeping `height` and `body_width` at its own call sites, on the grounds that
  once the orientation is known the screen words are the clearer ones.
* **What `candles` calls `box_direction`,** given a candle is not a box.
* **The class name:** `boxes`, `boxplot`, or `whiskers`.
* **The default median weight,** per the paragraph above.

## Not in this design

* **Labelling the category axis.** Box plots want their group names beside them
  and cannot have them yet, for the reason in `notes/categorical-axes.md`. The
  value axis is labelled by `axes` already, since a box plot carries a range on
  that axis and none on the other.
* **A two-tone median**, splitting a filled box into a lighter and a darker
  half at the median rather than drawing a line. Reaches all eight positions,
  because it is a boundary between two fills rather than a band inside one, but
  changes what a box plot looks like enough to want its own discussion.

## Reproducing the measurements

Glyph geometry, for any of the characters above:

```python
from matthewplotlib.unscii16 import bitmaps
import numpy as np
bitmap = bitmaps(np.array([[ord("▔")]], dtype=np.uint32))[0, 0]
np.flatnonzero(bitmap.any(axis=1)), np.flatnonzero(bitmap.any(axis=0))
```

The cell aspect ratio is in the shape `CharArray.to_rgba_array` returns, sixteen
pixel rows and eight columns per cell. The resolution comparison for filled
marks is the one in `notes/candlesticks.md`.
