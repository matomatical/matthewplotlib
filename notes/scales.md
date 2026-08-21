# Scales: the interval as a value, and nonlinear colour axes

Written 2026-08-20 by Claude (Opus 5), at MFR's direction, out of the session
that built `heatmap` and `colorbar`. MFR asked for the scale to be developed as
a first-class object in a session of its own and threaded through the plots
that need it, rather than smuggled in behind the colorbars. The design below
was worked out with him in conversation and then deliberately not built; the
reasoning for the parts that were settled is in the `colorbars` note.

Nothing here is built. The roadmap entries are "Configurable colour scales and
normalisation" and, under advanced image options, "Nonlinear normalisation".

## Where the library stands

A colour scale has two halves: data onto [0,1], and [0,1] onto colours. The
second half is a first-class value --- `mp.viridis` is a function you pass
around. The first half is a pair of numbers and a division.

Every plot that maps values onto colours takes `vrange: (number, number) |
None` and keeps the pair it settled on as `plot.vrange`. `_value_range` and
`_normalise` in `plots.py` are the whole of it: clip to the interval, divide,
saturate at the ends. `colorbar` borrows that pair off any plot that kept one,
and is told its colormap outright.

So the interval travels with the plot and the numbers cannot drift. What is
missing is any way to say that the colours are not spaced linearly in the
data.

## The shape the value type wants

    scale(lo, hi, transform=None)

Clip to the interval, transform, then map affinely onto [0,1]:

    v -> (t(clip(v)) - t(lo)) / (t(hi) - t(lo))

Four things recommend this over a class per norm:

* **Clipping before transforming** keeps `np.log` from ever seeing a negative,
  and costs nothing since values outside the interval saturate anyway.
* **The interval stays in data space**, which is what makes the labels right.
  This is the whole reason a transform composed *upstream* of the interval does
  not work; see the `colorbars` note for the argument, which is worth
  reading before proposing anything tuple-shaped again.
* **`transform=None` is the linear case**, so a bare pair promotes to a linear
  `scale` and there is one stored type rather than two.
* **Log, symlog, sqrt, cbrt and anything else monotonic** are arguments rather
  than subclasses. Named shorthands (`mp.logscale`) can follow if the spelling
  gets tiresome.

The transform must be monotonic on the interval, and `t(lo) != t(hi)`. Worth
validating at construction, where the error can name the transform, rather than
leaving a caller to wonder why a picture came out flat.

## How it threads through

The keyword already exists, so widen it rather than adding a second one:

    vrange: tuple[number, number] | scale | None

    mp.heatmap(m, colormap=mp.magma)                      # infer, linear
    mp.heatmap(m, vrange=(0, 255), colormap=mp.magma)     # linear
    mp.heatmap(m, vrange=mp.scale(1, 255, transform=np.log), colormap=mp.magma)
    mp.colorbar(that, colormap=mp.magma)                  # labels 1 and 255

Plots to reach: `heatmap` and therefore `function2` and `histogram2`;
`calendar` and `weeks`; `bars` and `columns`, whose `vrange` scales a length
rather than a colour but is the same idea and would read the same way. MFR has
flagged `hilbert` as wanting a colormap and a scale of its own; it currently
takes `bool[N]` and one `color`.

`colorbar` should need no change at all. Its ramp is `linspace(0, 1)` in
normalised space, which is linear in screen position whatever the transform,
and `axes` labels only the two ends, which are always `lo` and `hi`. A log bar
and a linear bar over the same interval draw identically. This was checked
during the colorbar work and is the reason the two pieces separate cleanly.

## Open questions

**Where it lives.** `colors.py` is about the spellings a colour arrives in and
`colormaps.py` about mapping [0,1] onto colours; a scale is a third thing. A
new `matthewplotlib/scales.py` of eighty-odd lines fits the habit `window.py`
set at 159. Putting it in `colormaps.py` would muddy a file that is 640 lines
of palette data behind a docstring promising "a collection of pre-defined
colormaps".

**Whether the colormap belongs inside it.** A `scale` holding both halves would
let `mp.colorbar(plot)` colour itself correctly without being told anything,
which it deliberately does not do today: the colormap is named at the bar
because judging whether a given colormap is one a gradient can stand for is a
question nobody has answered yet (see below, and
the `colorbars` note). A `scale` carrying its own colormap would be the
place that judgement finally lands. Against: the two halves are chosen
independently, and `bars` wants the interval with no colormap at all.

**Intermediate ticks.** The claim that a log bar draws like a linear one holds
only because `axes` labels the two ends. The moment a colorbar carries ticks in
between, the transform decides where they fall, and the scale needs to say
where --- matplotlib puts log-spaced ticks on a log colorbar. This is also
the `axis-series` note's territory.

**Axis transformations.** The roadmap wants logarithmic *coordinate* axes
separately from colour scales. A `scale` is very nearly the same object as the
mapping a `window` performs from data onto cells, with a transform in the
middle. Worth looking at whether one type serves both before building two.

**Which colormaps a bar can stand for.** A gradient stands for a continuous
colormap. `pico8` and `sweetie16` map integers to colours by lookup, so
`mp.pico8(np.arange(4))` is four unrelated colours rather than a ramp, and a
scale for one of those is a column of swatches with a label beside each ---
a different plot type, and probably the same one legends want. `chroma` and
`domain` take plane vectors, and no one-dimensional bar represents them at all.

`colormaps.py` names these three flavours in its type aliases, but they are all
the same `Callable` at runtime, so nothing can check them. Whoever makes that
taxonomy real gets three things at once: a bar that can refuse a colormap it
cannot draw, the swatch plot the discrete flavour wants, and the option of
letting a `scale` carry its own colormap. Until then a colorbar is told its
colours and asks no questions, which is what the rest of the library does with
colormaps too.

## Not in this design

Legends. They share the "swatch beside a label" shape with discrete colour
scales, and the two are worth looking at together once either has been built.
