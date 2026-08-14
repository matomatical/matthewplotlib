# The axis series, after line plots

Raised 2026-08-14 by MFR, on the observation that an axis drawn in a scatter
plot is a hack from before line plots existed. Nothing here is built yet.

`data.axis`, with its `xaxis`/`yaxis`/`zaxis` subclasses, stands in for a
coordinate running over a range, so that `mp.scatter3((mp.xaxis(), "red"), ...)`
draws the X axis. It samples the interval at `n` points, ten by default, because
ten dots was the closest thing to a line that a scatter could draw.

## What already works

A line plot interprets it as a segment with no change at all, since
`parse_segments` pairs the consecutive points of a series:

    line(mp.xaxis())      ⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤
    scatter(mp.xaxis())   ⠄ ⠠  ⠄  ⠄ ⠠  ⠄ ⠠  ⠠  ⠄ ⠠

So the hack is already retired wherever a line draws it. `teapot.py`'s three
coloured axes would come out solid if it drew them with `line3`.

## What the hack leaves behind

`n` is a rendering parameter inside a data description, and its only purpose was
ever to fake a line out of dots. For a line it is inert: ten collinear points
draw exactly what two would. Three ways to retire it:

1. Document that `n` only matters when the axis is scattered. Free.
2. Let the parser sample: an `axis` describes an interval, `parse_segments`
   takes two points from it and `parse_multiple_series` takes `n`. The sampling
   decision moves to the primitive being built. Small and local.
3. Let the plot sample at its own dot resolution, so that a scattered axis comes
   out solid too, one sample per dot. This retires the concept rather than the
   default: an axis is an interval, and each plot draws it as well as its medium
   allows. It needs the plot's resolution where the parsing happens, which is
   what the window in `notes/plot-windows.md` would carry, so it belongs after
   that and not before it.

## The form that is documented but missing

`data`'s module docstring says a series can be given as "one sequence of values
against an axis rather than as two sequences". That form does not exist:

    line((mp.xaxis(0, 10), ys))
    ValueError: invalid color array([0., 0.0123, ...])

`case (axis() as a, cs_)` matches any two-tuple beginning with an axis and reads
the second element as colors. So the most ordinary line chart there is, y
against an index or a time range, is documented, unimplemented, and fails as a
color error. The docstring is knowingly left over-promising until this is
settled; that is the reason for this note.

It cannot simply be added, because the grammar is ambiguous: `(mp.xaxis(0, 1,
3), [255, 0, 0])` is both a valid three-point series of values and a valid
single color. The candidate spellings:

* a three-tuple, `(axis, ys, colors)`, leaving the two-tuple as it is.
  Unambiguous, but asymmetric with `(xs, ys)`;
* resolving by shape, where a one-dimensional numeric array of length other
  than three means values. Clever, and therefore fragile;
* making the two-tuple mean values and requiring the three-tuple for colors.
  The cleanest grammar, and it breaks `teapot.py`'s `(mp.xaxis(), "red")`;
* a distinct spelling, such as `mp.xaxis(0, 10).against(ys)`. Explicit, nothing
  breaks, and nothing about it is ambiguous.

Worth noting the capability is already there: `line((np.arange(n), ys))` draws
it with one extra array. This is ergonomics.

## And the furnishing

Separately: now that segments can be drawn, the `axes` furnishing could draw
real axis lines inside the plot area -- through the origin, with braille ticks
-- instead of box-drawing characters around it. That is a redesign of something
the README already calls not final, and it wants the window too.
