# Box plots, and candles as one of them

Designed 2026-08-20 by MFR and Claude (Opus 5), in conversation, written up by
Claude. The questions it left open were settled and the names fixed in a second
conversation on 2026-08-21, and building started from there; the sections marked
below carry that second round. Agreed between us. The measurements are Claude's
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

## How the two orientations are drawn

Added 2026-08-21.

The two orientations share their arithmetic and differ only in which glyphs they
spend it on. Everything is computed in orientation-neutral integers --- sub-cell
positions along the value axis, whole cells across the thickness, arm masks for
rules, counts of eighths for fills --- and the glyphs are chosen last, from a
table per orientation:

|                      | boxes lying flat        | boxes standing up     |
|----------------------|-------------------------|-----------------------|
| filled interval      | `PARTIAL_BLOCKS_ROW`    | `PARTIAL_BLOCKS_COL`  |
| thin interval, caps  | arm bits left and right | arm bits up and down  |
| interior mark        | `▏│▕`                   | `▔─▁`                 |

An outlined box is the one piece that is genuinely two-dimensional, since a
corner reaches two ways at once. It gets its arm mask built as a small 2d array
and transposed by permuting the arm bits before a `LineStyle` is indexed by it,
which is exact and needs no table, because every style carries all sixteen
combinations.

**Rejected: drawing in one orientation and rotating the character array.** A
rotation by a quarter turn clockwise sends each cell's bottom edge to its left
edge, and Unicode has bottom-anchored eighths and left-anchored eighths, so
`▁▂…█` maps onto `▏▎…█` and increasing-upward maps onto increasing-rightward,
which is what the other orientation wants. It would have let one drawing routine
serve both. It fails because character cells are not square, so a quarter turn
is not an isometry of the marks:

* the interior mark's three positions are three uniform eighths of a cell
  standing up (`▔─▁`, two pixels of sixteen each) and three uneven ones lying
  flat (`▏│▕`, one, two and one of eight), so a rotated median arrives with its
  middle position at twice the weight of its edges;
* the thin interval is a hairline an eighth of a cell thick as `─` and a stroke
  a quarter of a cell thick as `│`, so the thin-against-thick contrast that is
  the whole grammar of the mark comes out at a different ratio each way;
* the eighths themselves quantise at two pixels standing up and one lying flat.

It was also a smaller saving than it looked. The glyph-choosing helpers differ
by a constant apiece --- which block table, which pair of arm bits --- against a
rotation needing a codepoint translation table. And the table above is the thing
a rotation would have forbidden: with the orientations parameterised rather than
derived, details are free to differ where the cell shape says they must, which
is what lets the interior mark default to a different weight each way.

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
edge, in the middle, against its far edge. The two sets mirror each other in
where the three positions fall, so the mechanism is the same whichever way the
box lies; they do not mirror each other in weight, which is what the choice of
`LineStyle` below is about.

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

**The median's weight is a `LineStyle`, light lying flat and heavy standing
up.** Only the middle of the three positions is a line glyph; the two at the
cell's edges are eighth blocks and cannot be styled. So the weight is not a
matter of taste: the median has to read as the same mark at all three of its
positions as it slides between them, which means the line has to match the
slivers it alternates with.

Measured in `unscii16`, against the eighth blocks the median lands on at a
cell's edges:

    standing box, band across a 16px cell    flat box, band across an 8px cell
      ▔  2px            slivers                ▏  1px            slivers
      ─  2px    LIGHT                          │  2px    LIGHT
      ━  4px    HEAVY                          ┃  4px    HEAVY
      ▁  2px            slivers                ▕  1px            slivers

A flat box is forced: light already overshoots the slivers two to one and there
is nothing lighter to reach for. A standing box is the only orientation where
the choice is live, and there the two fonts disagree. `unscii16` matches light
exactly and doubles it at heavy, so a heavy median visibly fattens whenever it
lands mid-cell rather than on a cell edge. An ordinary terminal font inverts the
ratio, `─` being a hairline where an eighth of a cell is a bar, so there light
is the one that flickers and heavy is the match.

The default serves the terminal, because that is where plots are read;
`unscii16` only surfaces in image export, so the exported images and the
documentation carry the mismatch instead. There is no middle setting to escape
into: `ROUND` is light's weight on a straight run, and `DOUBLE` draws two bands,
which is a different mark.

## Statistics

Raw samples only: a sequence of one-dimensional arrays, possibly ragged, with
the quantiles computed for you. This matches `histogram`, which takes values and
bins them. Accepting precomputed five-number summaries can come later if
something wants it.

Whiskers take a parameter, `whisker_iqrs`, defaulting to `1.5`: Tukey's rule,
whiskers reaching the furthest sample within 1.5 times the interquartile range
of the quartiles, with everything beyond drawn individually as outlier points.
`whisker_iqrs=None` gives min-to-max whiskers and no outliers. The default draws
marks some readers will not expect, which is accepted because it is the standard
and because the outliers are usually the informative part.

Matplotlib spells this parameter `whis`. That name is not carried over: it
abbreviates heavily, and it abbreviates the wrong noun, since the number is a
multiple of the interquartile range rather than a property of the whiskers.

## Order of work, and why

Build the box plot first, then move `candles` onto whatever the two turn out to
share. Not the other way around: the shared base should be extracted from two
working plots rather than guessed from one, and the guess is especially unsafe
here, because how much is shared depends on the fill-or-outline choice that only
the box plot makes. `candles` is safe to refactor --- 44 unit tests and an
example snapshot fence its behaviour, and it is unreleased, so its signature is
still free to change.

## What was shared, in the end

Built on 2026-08-21, in that order. `unicode_boxes` draws the general mark and
both plots are settings of it: `boxes` turns every switch on, `candles` passes
its low and high as the outer interval, its open and close as the inner one, and
switches the caps, the interior mark and the points off. `unicode_candles` and
its four helpers are gone, their behaviour absorbed and their tests retargeted.

The prediction held. Of the candlestick machinery, the fill was reused almost
verbatim --- eighths of a cell, half of them drawn as negatives --- and so was
the half-cell outer interval, while everything the outlined mode needed was new,
which is the dependency that made building the box plot first the right order.

One thing the candlestick wanted that the box plot had not asked for: a colour of
its own for the outer interval, since `candles` lets every wick be drawn in one
neutral colour whatever its body. That is `outer_colors`. In a filled mark it is
a pass of its own; in an outlined mark it colours only the cells the outer
interval has to itself, the ones it shares with the outline going to the mark's
own colour, because a cell shows one colour and the structural part is the one
worth keeping.

The 19 tests over the `candles` plot passed unmodified apart from the two
renamed parameters, which is what they were there for. One behavioural change
survived them, and it is worth recording because it is not a bug: sub-cells are
now counted from the low end of the value axis where `unicode_candles` counted
from the high end. Over 200000 random intervals that shifts about 2% of them,
always by exactly one eighth and never changing a length; a zero-length body
shifts almost always, again by one eighth, because the eighth it rounds into is
now the one above the value rather than the one below. Both sit inside the
"nearest eighth" the docstring promises. One candle in the example snapshot moved
by a cell as a result.

`candles` also picked up the horizontal form the design said it would get for
nothing, which cost it `height` and `body_width`: with the orientation a
parameter rather than a fact, the screen words no longer name anything fixed, so
it spells its sizes `length`, `body_thickness` and `spacing` like `boxes` does.

## The names, settled

Fixed on 2026-08-21.

The class is **`boxes`**, matching the plural nouns the other repeated-mark
plots go by, `bars` and `columns` and `candles`, and leaving `boxplot` and
`whiskers` unclaimed.

The sizes are **`length`** along the value axis, **`box_thickness`** across one
box and **`box_spacing`** between them. That maps exactly onto what `bars`
already does --- one plot-wide dimension, then the mark's own dimension and gap
prefixed by the mark --- so `width`, `bar_height`, `bar_spacing` becomes
`length`, `box_thickness`, `box_spacing`. `extent` and `span` were the
alternatives to `length` and both read worse, `span` especially, in a plot made
of spans.

`candles` spells its orientation **`candle_direction`**, so the concept is
spelled twice. The alternative was a bare `direction` shared by both, and the
argument against it is the one that named `box_direction` in the first place:
`direction="horizontal"` alone does not say whether it describes the mark or the
row of them.

Outliers are drawn as `·`, and whether they appear follows from `whisker_iqrs`
rather than a switch of their own.

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
