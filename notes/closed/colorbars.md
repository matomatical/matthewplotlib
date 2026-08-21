# Colorbars, and the mapping they are a picture of

Written 2026-08-20 by Claude (Opus 5) in conversation with MFR, over the
session that built `heatmap` and `colorbar`. The open version of this note
proposed a `norm` value type; what got built is smaller, and the reasoning for
the difference is the point of keeping this. The scale work that was deferred
has its own note, `scales`.

## What the job turned out to be

A colorbar needs one thing from the plot it describes: the interval of values
its colours cover. It needs a colormap too, but that comes from the caller ---
see the last decision below. Everything else followed from finding out where
the interval lived.

They were in four places, spelled three ways:

| plot | input | kept |
|------|-------|------|
| `image` | --- (caller pre-scaled to [0,1]) | nothing |
| `function2` | `zrange=(lo, hi)` | `self.zrange` |
| `histogram2` | `max_count=hi` | nothing |
| `calendar`, `weeks` | `vrange: None \| hi \| (lo, hi)` | `vmin`, `vmax` |
| `bars`, `columns` | the same `vrange` | `vmin`, `vmax` |

`bars` and `columns` are the reason `vrange` won. They spell the interval the
same way, and document it in the same words, for a visual channel that is not
colour at all: the length of a bar. So `vrange` already meant "the interval of
values this plot's visual channel covers", and colour is one such channel. The
library had the convention; it had just not finished applying it.

`histogram2` was the case that could not be drawn at all before this: it
divided its counts by `max_count` and then kept `xbins`, `ybins` and
`num_points`, so the plot no longer knew what its own colours meant.

## Log scales cost the colorbar nothing

This is the fact that made the job separable, and it was not obvious at the
start.

The ramp is `linspace(0, 1)` in *normalised* space, which is linear in screen
position whatever transform sits behind the normalisation. And `axes` labels
only the two ends of a coordinate, which are always the two limits of the
interval. So a log colorbar and a linear colorbar over the same interval draw
identically, character for character.

Nonlinear colour scales are therefore not a prerequisite for colorbars. They
are their own piece of work, and they went to the `scales` note.

## Why `image` did not grow a `vrange`

`image`'s array argument means five different things, and the examples use all
five:

    mp.image(rgb)                                 # doomfire.py    colours
    mp.image(image / 15)                          # image.py:35    greys
    mp.image(image / 15, colormap=mp.viridis)     # image.py:36    continuous
    mp.image(image // 2, colormap=mp.sweetie16)   # image.py:39    indices
    mp.image(state, colormap=life_palette)        # life.py:153    a palette

Normalising suits rows two and three. In row four it is destructive: `pico8`
does `np.asarray(x, np.uint) % 16`, so [0,1] floats collapse every index to 0
or 1. And the ambiguity is deliberate --- `examples/colormaps.py` puts the same
array through both readings, `mp.image(im_discrete16 / 16)` for greys at line
35 and `mp.image(im_discrete16, colormap=c)` for indices at line 59.

Nothing distinguishes the two kinds of colormap at runtime, either:
`ContinuousColorMap` and `DiscreteColorMap` are both
`Callable[[ArrayLike], ndarray]`, one type with two docstrings. So `image`
could not have checked that a `vrange` suited the colormap it was handed.

Four designs were weighed:

* **A `vrange` on `image`, applied only when given.** Smallest diff, but
  `mp.image(indices, colormap=mp.pico8, vrange=(0, 15))` would have been
  silently wrong with no way to catch it, and `vrange=None` would have meant
  "identity" here and "infer" everywhere else.
* **Normalise always, defaulting to the unit interval.** Breaks the
  int-greyscale form (ints in 0 to 255 clip to 0 or 1) and every discrete
  colormap. Non-starter.
* **Mark the colormaps** so continuous and discrete are distinguishable, and
  check. Touches all twenty public names in `colormaps.py`.
* **Split the two jobs into two plots.** What was built.

`heatmap` is values on a colour scale; `image` is pixels. `function2` and
`histogram2` moved onto `heatmap` and stopped carrying their own copies of the
normalisation. Marking the colormaps stayed unnecessary, but for a different
reason than first thought: not because the plot types sort themselves out, but
because a colorbar never inspects a colormap at all. See below.

The name was already in the code. `function2`'s docstring opens "Heatmap
representing the image of a 2d function over a square" and `histogram2`'s
"Heatmap representing the density of a collection of 2d points". `heatmap` is
the word for what they have in common, and is now their base class.

The cost is one redundancy: `mp.image(a01, colormap=mp.viridis)` and
`mp.heatmap(a01, vrange=(0, 1), colormap=mp.viridis)` draw the same picture,
and only the second carries a scale. "Image draws pixels, heatmap draws
values" is the line between them.

## The rejected spelling for composition

MFR proposed letting a colormap be a tuple, meaning composition:
`colormap=(np.log, mp.scale(0, 255), mp.magma)`.

It does not hold, for one decisive reason: with the transform applied first,
the interval is no longer in data space, so `mp.scale(0, 255)` would mean
"log-space values from 0 to 255" and the colorbar could not print the numbers
the caller wrote. **The transform has to live inside the thing that carries the
interval**, which makes the composition `scale(lo, hi, transform)` followed by
a colormap --- a pair, not an open-ended tuple.

Three lesser objections, kept because they apply to any future attempt to put
a scale in the `colormap` slot:

* Nothing enforces the order. `(mp.magma, mp.scale(0, 255))` type-checks
  identically and produces garbage, and `parse_colors` validates only that the
  last element returns `[h,w,3]`.
* It collides with `vrange`, and needs a precedence rule for
  `heatmap(v, vrange=(0, 10), colormap=(mp.scale(0, 255), mp.magma))`.
* It does not serve the plots that infer. `function2` and `histogram2` derive
  their interval from data they compute, so a colormap tuple without a scale
  in it would mean "insert an inferred one", and the plot would be rewriting
  the caller's colormap.

## Decisions worth naming

* **`direction` names both the axis and the sense**, so there is no separate
  orientation to contradict it. The first limit of the interval sits at the end
  the direction points away from. The window's coordinate always runs from the
  low end of the screen axis to the high end, which for `"down"` and `"left"`
  means a descending range, and inverting an axis is a thing `window` already
  did.
* **The resolution is asymmetric and stays that way.** A cell holds two
  half-block pixels vertically and one horizontally, so a vertical bar of a
  given length has twice the gradient steps of a horizontal one. Nothing short
  of dithering changes it. Documented rather than fixed.
* **The scalar `vrange` went.** MFR did not like `vrange=50`. Its only internal
  users were `histogram` and `vistogram` passing `max_count` down to `bars`
  and `columns`, which now pass `(0, max_count)`. `max_count` survives on all
  three histograms, where the lower bound is always zero and the name says
  what it counts.
* **A degenerate `vrange` is an error when the caller wrote it**, matching
  `window`, which raises on a range covering no interval. Inferred from
  constant data it puts everything at the bottom of the colormap, there being
  nothing else it could mean.
* **A non-finite value comes out at the bottom.** `heatmap` leaves it out of an
  inferred interval and maps it to zero, which is what happened before by
  accident, minus the NumPy cast warning. `image` still warns on the same
  input; that is the "robust input validation" roadmap item, not this one.
* **A colorbar never derives its colormap.** The interval comes from the plot;
  the colours are named at the bar. Two attempts to be cleverer than that both
  failed, and the reason they failed is the useful part.

  Deriving the colormap from the plot means deciding whether that colormap is
  one a gradient can stand for, and **that is a property of the colormap, not
  of the plot.** `viridis` takes a scalar in [0,1] and a gradient is exactly
  right; `pico8` takes indices and wants a swatch beside each label; `chroma`
  and `domain` take plane vectors and no one-dimensional bar represents them.
  A marker on the plot classes---mixed into `heatmap`, `calendar` and
  `weeks`---looked like it settled this, and did not: `heatmap(values,
  colormap=mp.pico8)` carried the marker and drew a "gradient" of three
  identical black cells. It blocked a colour field for being the wrong plot
  class while admitting the same mistake through the class it was built for.

  Marking the colormaps instead would be on the right axis, and
  `colormaps.py` now names three flavours---continuous, discrete and
  vector---in its type aliases, so the taxonomy exists on paper. Making it real
  at runtime is the `scales` note business, along with the swatch plot the
  discrete flavour wants.

  Deleting the derivation dissolves all of it. Nothing else in the library
  infers a colormap: hand `viridis` an index array and you get whatever you
  get. A bar is no different, the caller names what the picture was drawn with,
  and a plot's `vrange` can then be borrowed by anything without the bar having
  to judge it. That last part is a gain rather than a loss: a colour field's
  interval really is its brightness scale, and a bar of it with a colormap the
  caller chooses is a legitimate legend, where the marker had refused one.

## Left undone

* Clipping is silent. `function2` and `heatmap` saturate values outside the
  interval at the ends of the scale, and the bar says nothing about it --- no
  arrow, no different end cap.
* Discrete colour scales. A scale for `pico8` or `sweetie16` is a column of
  swatches with a label beside each, not a gradient with labelled ends. That is
  a different plot type, and probably the same one legends want.
* `animation` (`animations.py`) takes a colormap and an array with a time axis,
  and has the same shape as `image`: it asks the caller to pre-scale. The
  animated counterpart of `heatmap` has not been built.
* `examples/time_series_histogram.py:71` carries MFR's `# TODO: colorbar`
  against a `histogram2`, which can now be given one.
