# The window a plot draws in

Raised 2026-08-14 by MFR, on seeing two asymmetries in the plot classes that
came out of the line-plots work: `scatter` mapped its data onto dots inline
while `line` used a helper, and `scatter3` subclassed `scatter` while `line3`
subclassed `plot`.

Both had one cause. The 2d constructors take the series grammar; the 3d ones,
after projecting, hold primitives in plot coordinates. A projected point cloud
happens to be expressible as a series, so `scatter3` could hand its result to
`scatter` and inherit the rest. Projected segments cannot be, so `line3` had to
map and draw them itself.

## What was done

Each drawing primitive got one rasteriser in `core`, in dot coordinates
(`rasterise_points` beside `rasterise_segments`, sharing `accumulate_dots`), and
all four classes became the same four steps: parse the series, decide the
ranges, map to dots with `plots._to_dots`, draw. `scatter3` is no longer a
`scatter`, which its own TODO had been asking for, and `axes` no longer accepts
it -- projected coordinates are not the data's own and would be labelled as if
they were.

The cost was paid once: a scatter's points move by at most one dot, since the
data's limits now land on the centres of the outermost dots rather than on the
outer edges of the outermost bins. Fourteen example goldens moved with it.

## What is deferred

The mapping from data coordinates onto the grid is still derived per plot type.
`scatter` and `line` now share `_to_dots`, but `function2` and `histogram2` each
do their own, and `axes` and `dstack2` reach for `.xrange` and `.yrange`
afterwards by duck-typing, with a union of plot types spelled out in their
signatures -- a union that has to be widened whenever a plot type is added, as
it was for `line`.

That is a concept without a name. Giving it one:

    @dataclasses.dataclass
    class window:
        xrange: tuple[number, number]
        yrange: tuple[number, number]
        width: int
        height: int
        def dots(self, points): ...

Then `_to_dots` is `window.dots`, every 2d plot holds a window, the 3d ones
build one from the camera's aspect ratio, and `axes` and `dstack2` take anything
that has one instead of a listed union. It is deferred because it changes how
every 2d plot is built, which deserves its own change rather than riding along
behind line plots.

The refactor should also make a window the unit of compatibility for
coordinate-aware overlays. `dstack2` currently compares only `xrange` and
`yrange`, and does so with `assert`, so the checks disappear under `python -O`.
It does not compare rendered dimensions at all. Two plots can therefore report
the same ranges while mapping a coordinate to different cells because their
widths or heights differ, after which `dstack2` simply overlays them from the
top left.

Once plots carry windows, `dstack2` should require every child to have the same
window as the first: both ranges and both rendered dimensions. It should reject
an empty input and mismatches with explicit `ValueError`s, not assertions. A
rendered plot no longer contains enough information to be faithfully resampled,
so differently sized windows should be rejected rather than padded or scaled.
Tests should cover range and dimension mismatches and run the validation under
`python -O`; if resampling is wanted later, it should be a separate operation
on plots that retain enough source data to do it deliberately.

## Also considered

* **Stroke bundles.** Add `float[strokes, points, dims]` to the series grammar,
  making segments expressible, so `line3` could delegate to `line` exactly as
  `scatter3` used to delegate to `scatter`, and a mesh could be passed as one
  array. Kept for whenever the mesh question comes up, since that is what it is
  really for; it fixes the symmetry only as a side effect.
* **Leaving it.** The asymmetry was explicable, but "the grammar happens to
  reach one of the two cases" is a thin reason for two shapes of code, and it
  left the `scatter3` TODO standing.

## What was built

All of it, on 2026-08-20. `matthewplotlib.window` holds the type, every 2d plot
carries one, and `axes` and `dstack2` take anything that has one. Two things
came out differently from the sketch above:

* Either range may be `None`, for an axis that carries no coordinate, so that a
  plain image has a window too and a gradient can be labelled along the one
  side that means anything. The sketch assumed both.
* The window owns the grid as well as the dot mapping. `pixel_edges` and
  `pixel_centres` define the tiling that `function2` samples and `histogram2`
  bins, which used to be the same arithmetic written out twice.

`dstack2` compares whole windows, ranges and rendered dimensions alike, and
raises `ValueError` rather than asserting. Nothing in the tests or the examples
had been relying on the looser check.
