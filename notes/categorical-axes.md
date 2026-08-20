# The axis that is a list, not an interval

Raised 2026-08-20 by MFR, on adding `candles` and noticing that its horizontal
axis cannot be described at all. Nothing here is built, and nothing here is
designed: the observation is MFR's, the survey of what stands in the way is
Claude's (Opus 5), and the shape of the answer is still open.

## The observation

A candlestick chart's horizontal axis is a sequence of periods. A bar chart's
is a list of names. Neither is an interval of numbers, and the library has no
way to say so. Every plot that lays marks out side by side is in the same
position:

    bars, columns       one bar per value
    histogram, vistogram    one bar per bin
    candles             one candle per period
    box plots           one box per group

`window` offers `xrange: (number, number) | None`, so each of these leaves it
`None`, and `axes` then has nothing to label. `candles` gets away with it
because the axis worth labelling is its *value* axis, which is a genuine
interval, and the four-sided `axes` will label that one side and leave the rest
alone. The horizontal axis is simply absent. `bars` and `columns` have it worse:
they carry no window at all, so they cannot be given an axis on either side.

This is the unchecked roadmap item "Labels and ticks for bar/column charts,
histograms, box plots", seen from the data end rather than the drawing end.

## Why the existing machinery does not stretch to it

Two obstacles, both real, neither obviously fatal.

**A window range is an interval, and these plots do not have one.** The
histograms are the interesting case: a `vistogram` genuinely does cover a
numeric interval, the range its bins partition, and yet what a reader wants
labelled is often the bins rather than the interval. So "categorical" and
"numeric" are not a clean partition of these plots; one plot can want both
readings of the same axis.

**`axes` labels the two ends of a side, not n positions along it.** By design:
`notes/axes-sides.md` puts interior ticks out of scope, on the grounds that a
tick anywhere else needs a position along the axis and a rule for choosing tick
values. A categorical axis supplies exactly that missing thing --- the
positions are the marks, and the values are their names --- so it is the case
that would motivate interior ticks rather than being blocked by their absence.

Then the arithmetic that follows is the awkward part, and it is the same
arithmetic in every one of these plots: a mark is `width` cells wide with
`spacing` between, so its centre is at a known column, and a name longer than
its mark has to be truncated, rotated, staggered, or thinned out. `bars` and
`hist` in `notes/reference/myplot.py` did a version of this before the rewrite
and are worth reading first.

## What would want deciding

Not decided here, only listed, so that the next session does not have to
rediscover the questions:

* whether a categorical axis is a third kind of range inside `window`, a
  separate field beside the two ranges, or something the plots hold themselves;
* whether an axis carrying names also carries the geometry --- mark width and
  spacing --- since `axes` needs the column of each mark's centre and only the
  plot knows it;
* what happens to a name too long for its mark, which is the same family of
  problem as the limits that will not fit in `notes/axes-sides.md`, and probably
  wants the same kind of answer: never wrong, never resized, never raised;
* whether the histograms label bins, their interval, or either on request.
