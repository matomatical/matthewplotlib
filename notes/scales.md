# Scales: the interval as a value, and nonlinear colour axes

Written 2026-08-20 by Claude (Opus 5), at MFR's direction, out of the session
that built `heatmap` and `colorbar`. Revised 2026-08-27 by Claude (Fable 5) in
discussion with MFR, replacing the earlier sketch with a settled design ---
named scale classes --- and a checked account of how the same type later
serves coordinate axes. The first phase, scales for the colour and length
channels, was built the same day, answering the roadmap entries "Configurable
colour scales and normalisation" and "Nonlinear normalisation". Revised again
2026-08-28 by Claude (Fable 5) after the second phase --- scales on
coordinate axes --- was built to the general-case design below; the section
"What the second phase settled" records the decisions the build added.

## Where the library stands

A colour scale has two halves: data onto [0,1], and [0,1] onto colours. The
second half is a first-class value --- `mp.viridis` is a function you pass
around. The first half is a pair of numbers and a division: `_value_range` and
`_normalise` in the `scales` module clip to the interval, divide, and saturate
at the ends. Every plot that maps values onto colours or lengths takes
`vrange: (number, number) | None` and keeps the pair it settled on as
`plot.vrange`, and `colorbar` borrows that pair off any plot that kept one, so
the interval travels with the plot and the numbers cannot drift. What is
missing is any way to say that the colours are not spaced linearly in the
data.

## The design: a family of named scales

A scale is the first half as a value, mirroring the second: a colormap is a
function from [0,1] to colours that you pass around, and a scale is a callable
from data to [0,1] that you pass around. It is a *named* value: a family of
frozen dataclasses, one per spacing, each knowing its own transform, its
inverse, and its domain.

    scale(lo=None, hi=None)          linear, and the base class
    logscale(lo=None, hi=None)       logarithmic; wants an interval with
                                     0 < lo, hi
    symlogscale(lo=None, hi=None,    arcsinh(v / linear_width): logarithmic far
        linear_width=1.0)            from zero, linear near it, spans zero
    powscale(lo=None, hi=None,       v ** exponent; wants lo, hi >= 0; sqrt is
        exponent=...)                powscale(exponent=0.5)

Normalisation is clip, transform, then affine onto [0,1]:

    v -> (t(clip(v)) - t(lo)) / (t(hi) - t(lo))

Clipping first keeps `np.log` from ever seeing a value outside the validated
interval, and costs nothing, since values outside it saturate at the ends
anyway. The interval stays in data space, which is what makes the labels
right: the transform has to live inside the thing that carries the interval
(see the `colorbars` note for the argument, which killed a tuple-shaped
alternative). A value that is not a number comes out at the bottom, and an
interval given descending turns the scale around --- the affine step flips
sign, so this needs no special case even with a transform.

The pieces of the type:

* `transform(v)` and `inverse(u)` are methods, identity on the base class,
  overridden by each subclass. They are the whole contract for a custom
  scale: subclass `scale`, override `transform` (vectorised, strictly
  monotone on any interval the scale will be given) and `inverse` (consulted
  only once scales reach coordinate axes; see below). This is the declared
  API, the way colormaps declare a function shape.
* `__call__(values)` is the saturating normalisation above --- what every
  colour and length channel wants.
* `__iter__` yields `lo, hi`, so a scale unpacks as its interval and
  `vmin, vmax = plot.vrange` keeps working wherever it is written.

**Partial scales.** `lo` and `hi` are each optional, and the plot *completes*
a partial scale against its data, so `mp.logscale()` means "log, interval
inferred" --- the common case for log colour. The inference policies stay
with the plots, as they are today: `from_zero` for `bars`, `columns` and
`vfunction2` magnitudes, `allow_flat` where positions need an interval to sit
in. Completion is `dataclasses.replace`, which preserves the subclass and its
extra fields, so a partial `symlogscale(linear_width=0.1)` completes with no
per-class code. Per-endpoint partiality matches the precedent `parse_range`
set for `scatter` and `line`, whose `xrange` already accepts `(None, 5)`.

**Validation.** At completion, each class checks its own domain exactly:
`logscale` refuses an interval touching zero or spanning it, `powscale`
refuses negatives, `symlogscale` accepts all reals and requires
`linear_width > 0`, and every scale requires `t(lo) != t(hi)`. The errors can
name the scale and the offending endpoint. There is no monotonicity probing
of arbitrary functions --- there are no arbitrary functions.

**How it threads through.** The keyword widens rather than multiplying:

    vrange: tuple[number, number] | scale | None

A bare pair promotes to a linear `scale`, `None` to an empty one, and the
plot stores the completed scale as `plot.vrange` --- one stored type, the
transform riding along as part of what the plot's colours mean. Internally
`_value_range` and `_normalise` collapse into "complete the scale, call it".

    mp.heatmap(m)                                # linear, inferred (unchanged)
    mp.heatmap(m, vrange=(1, 255))               # linear (unchanged)
    mp.heatmap(m, vrange=mp.logscale(1, 255))    # log over [1, 255]
    mp.heatmap(m, vrange=mp.logscale())          # log, interval inferred
    mp.bars(v, vrange=mp.powscale(exponent=0.5)) # lengths on a sqrt scale

**Equality and repr come free.** Dataclass equality on a subclass compares
class identity plus fields, so `logscale(1, 255) == logscale(1, 255)` by
construction and `logscale(0, 1) != scale(0, 1)`. The dataclass repr
(`logscale(lo=1, hi=255)`) lands in plot reprs and error messages unasked.

**Log needs no base parameter.** The normalisation is invariant to affine
maps of the transform, and log bases differ by a constant factor, so log2, ln
and log10 draw the identical picture; worth a docstring sentence, since it
preempts the first feature request. The same argument does not collapse
`linear_width` or `exponent` --- those change the picture.

## Why named classes, reversing the earlier sketch

The earlier version of this note proposed `scale(lo, hi, transform=np.log)`,
with named subclasses explicitly rejected: "log, symlog, sqrt and anything
else monotonic are arguments rather than subclasses", guarding against
matplotlib's `Normalize` zoo. MFR pushed back on the argument form, and the
guard turns out to have misfired: what makes that zoo painful is statefulness
and API sprawl, not subclassing itself. A frozen two-field base with three
small subclasses is not a zoo, the user had to name their scale either way,
and naming it by class brings along what naming it by function argument could
not:

* exact per-class domain validation, where a function argument forced a
  monotonicity probe over samples --- heuristic, and falsely reassuring;
* the inverse, bundled, so nobody supplies `inverse=np.exp` by hand;
* equality by value, where functions compare by identity and two lambdas
  spelling the same transform would have refused to overlay;
* one spelling from day one, where `transform=np.log` today and a `logscale`
  shorthand tomorrow would have shipped a churn.

The set is deliberately small and close to complete for the medium: log
covers most real use, symlog (in the smooth arcsinh formulation --- matplotlib
added an `asinh` scale after its piecewise symlog's kink proved a problem)
covers data spanning zero, and `powscale` covers the root-and-gamma family. A
separate `sqrtscale` was considered and dropped as redundant next to
`powscale(exponent=0.5)`; matplotlib's remaining scales (logit, function
scales) have no terminal-plotting constituency, and that tail is exactly who
the subclass contract serves.

## The colorbar: picture unchanged, code not quite

The earlier version claimed `colorbar` needs no change at all. The *picture*
does not: the bar should be linear in screen position whatever the transform,
with the colormap evenly swept and `lo` and `hi` at the ends, so a log bar
and a linear bar over the same interval draw identically, character for
character. But the code builds its ramp as `np.linspace` in *data* space and
lets `heatmap` normalise it, which is only equivalent while the scale is
linear; push a log scale through unchanged and the gradient crowds toward one
end. So `colorbar` must draw its ramp in normalised space explicitly (a
linear scale over the same interval), and its `source` widens to
`plot | tuple | scale`.

One honest caveat for its docstring: with ends-only labels, a reader
linearly interpolating a log bar reads wrong intermediate values. That is
inherent until intermediate ticks exist --- and ticks need only the *forward*
transform: the tick for value 10 goes at fraction `scale(10)` of the bar.
A scale never needs its inverse for anything a colour axis does.

## The general case: scales on coordinate axes

Worked through in discussion ahead of the first build so that nothing in it
foreclosed this, and built as the second phase on 2026-08-28. The observation
that makes it cohere:
`window` already *is* two linear scales plus a raster. `window.dots` inlines
two affine maps, and `pixel_edges` inlines their inverses as `linspace`. So
`scale` is the one-axis atom, `window` is two of them plus the placement
conventions and the cell counts, and a coordinate axis and a colour axis
become the same kind of value --- which is why the colorbar sits at their
intersection. Walking every consumer of a window axis:

* **Forward placement (`scatter`, `line`).** A colour channel *saturates*;
  a coordinate *culls* --- a point past the edge falls off the raster rather
  than smearing onto the border. So the core mapping is the unclipped
  affine-of-transform, kept factorable behind the saturating `__call__`
  (phase 2 adds an unclipped `position` method; nothing else changes).
  Pleasingly, culling handles log's danger zone by itself: log of a negative
  coordinate is NaN under `errstate`, and the point plots already cull
  non-finite points. A negative value on a log axis is correctly an
  undrawable point, not an error.
* **Inversion (`pixel_centres`, `sample_points`, `pixel_edges`).** Sampling
  plots ask what value a pixel stands for, so `function2` on a log axis
  samples log-spaced points and `histogram2` bins on log-spaced edges ---
  both genuinely wanted. Only these plots invert; the bundled scales all
  provide `inverse`, and a custom scale that omits it gets a clear error
  from the sampling plots alone.
* **Overlay compatibility (`dstack2`).** It refuses on `p.window != shared`,
  which is dataclass equality, so windows carrying scales compare class and
  interval for free: two log plots overlay, log-on-linear is refused, and
  the scale's repr shows why.
* **End labels (`axes`).** `lo` and `hi` are data space and sit at the ends
  whatever the transform --- the same argument as the colorbar. Marking
  log-ness on the axis line, and intermediate ticks, are future work (the
  `axis-series` note's territory), already served by the forward map.
* **The microcosm resolves.** The first phase had `colorbar` put a plain
  pair in its window; the second puts the scale there, so when `axes` grows
  intermediate ticks, a log colorbar gets log-spaced ticks with no
  colorbar-specific code. Likewise `candles` and `boxes`, whose `vrange`
  becomes a window coordinate and so is *this* problem rather than the
  colour-channel one: excluded from the first phase, dissolved by the
  second.

## What the second phase settled

Decisions the build added to the design above, each discussed with MFR at the
level of policy (the widening, the inference domain, the stored types) with
the mechanics left to the build.

* **The window stores promoted scales.** `window.xrange` and `yrange` accept
  a pair or a scale and keep a completed scale either way, so there is one
  stored type, both spellings build equal windows, and `dstack2` compares
  spacing for free. The visible change: `plot.window.xrange` still unpacks
  as a pair but compares equal to `mp.scale(lo, hi)` rather than `(lo, hi)`.
  Teaching `scale.__eq__` to match tuples was considered and rejected: it
  would break the equality/hash contract for a cosmetic convenience.
* **The window works in transformed space, in the old expressions.** `dots`,
  `pixel_edges`, `pixel_centres` and `sample_points` apply the transform,
  run the affine arithmetic they always ran, and invert where they used to
  read data coordinates directly. With the identity transform this is
  operation-for-operation the previous floating point, so every golden
  stood, the same argument that kept the colorbar's ramp in data space in
  the first phase.
* **Edges are pinned, not round-tripped.** The outermost pixel edges are
  assigned the interval's own ends rather than `inverse(transform(end))`,
  which floating point can land a whisker away --- enough to drop a sample
  sitting exactly at the limit of a log `histogram2` out of its own bin.
* **A flat inferred coordinate interval widens along the scale.** Half a
  unit each way in transformed space, which is the old `lo - 0.5, hi + 0.5`
  exactly for linear axes and a factor of sqrt(e) each way for log. Only
  endpoints that were left to inference move: one the caller wrote stays
  written. An *explicit* flat coordinate range is now an error, as it always
  was for `vrange` (previously `parse_range` silently widened it), and
  `parse_range` itself dissolved into the scales module's coordinate
  resolver.
* **Inference sees only the scale's domain.** Completing a partial scale
  infers over the values it could place, so a zero in the data no longer
  makes `logscale()` an error; on a coordinate axis the zeros were culled
  anyway, and on a colour axis they saturate at the bottom. This applies to
  the colour channel too, reversing the first phase's error --- flagged
  under "Left open" as policy to revisit before v1.0.
* **An unwritten `inverse` is refused, not inherited.** A subclass
  overriding `transform` alone would silently inherit the base identity
  `inverse`, so the sampling paths check for exactly that shape and raise,
  which is what makes the docstring's "may leave it unwritten" true.

## What the first phase reaches

Scales for the channels that are not coordinates: `heatmap` (and through it
`function2` and `histogram2`), `vfunction2` magnitudes, `calendar` and
`weeks` through `_color_days`, and `bars` and `columns` lengths (their
histogram and box relatives ride along). `candles` and `boxes` keep plain
pairs, per the above. `image` stays scale-free ("image draws pixels, heatmap
draws values" --- the `colorbars` note). `hilbert` growing a colormap and a
scale, and the animated counterpart of `heatmap`, remain future features
that will slot in.

One implementation note from the build: the colorbar's ramp stayed in data
space, normalised through a plain linear scale over the source's interval,
rather than moving to normalised space. The two are mathematically identical,
but the data-space arithmetic reproduces the previous floating-point results
bit for bit, which kept every golden snapshot standing.

## Decided along the way

* **No colormap inside the scale.** It would reopen "colorbar derives its
  colormap", which the `colorbars` note killed for good reasons, and `bars`
  wants the interval with no colormap at all. The two halves stay
  independently chosen. (A composed spelling, `colormap=viridis.over(...)`,
  was also weighed and dropped: the scale must exist alone for `bars`
  anyway, so composition just adds a second spelling and resurrects the
  precedence problem of the rejected tuple.)
* **No separate `norm` or per-plot `transform` keyword.** The interval and
  its spacing travel together or they drift; `vrange` widens instead.
* **`symlogscale`, not `asinhscale`.** The name states the need (symmetric,
  log-like, spans zero); the docstring states the formulation.

## Left open, unchanged by this design

* The global inference-domain policy, before v1.0 (MFR, 2026-08-28): the
  second phase settled on filtering inference to the scale's domain
  uniformly, for coordinates and colour alike, but whether out-of-domain
  data should infer silently, warn, or error deserves one deliberate pass
  across the library before the interface freezes.
* Intermediate ticks, and how a bar or axis marks its spacing visually.
* Discrete colour scales: a scale for `pico8` or `sweetie16` is a column of
  swatches with a label beside each --- a different plot type, probably the
  one legends want, and the place the colormap taxonomy (continuous,
  discrete, vector --- named in `colormaps` type aliases, unenforced at
  runtime) would finally become real.
