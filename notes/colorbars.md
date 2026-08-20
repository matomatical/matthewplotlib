# Colorbars, and the mapping they are a picture of

Written 2026-08-20 by Claude (Opus 5) at MFR's direction, as the axes work
landed, to carry the rest of the colorbar job into a fresh session. The shape
of the `colorbar` constructor below was settled with MFR in conversation; the
normalisation half was not, and the questions at the end are genuinely open.

Nothing here is built.

## What is already in place

* A `window` may carry a coordinate on one axis and none on the other, so a
  gradient one cell wide is a plot with a scale.
* `axes` labels each side independently, and infers the minimal treatment for a
  window with a single coordinate: the one side that means anything, and
  nothing on the other three.
* A descending range inverts an axis, so a bar can run in any of four
  directions.
* `image` accepts `xrange` and `yrange`.

So a colorbar already draws, by hand. This is what `examples/axes_gallery.py`
does:

    ramp = np.linspace(1.0, 0.0, 2 * length)[:, None].repeat(thickness, axis=1)
    mp.axes(mp.image(ramp, colormap=mp.viridis, yrange=zrange))

Two things are missing: a constructor, so nobody writes that; and a reason for
`zrange` to be the range of the plot the bar describes, rather than a number
the caller repeats.

## Part one: the constructor

    colorbar(
        zrange: tuple[number, number],
        colormap: ColorMap | None = None,
        direction: Literal["up", "down", "left", "right"] = "up",
        length: int = 12,        # cells along the scale
        thickness: int = 1,      # cells across it
    )

`direction` names both the axis and the sense, so there is no separate
orientation to contradict it. The window's range always runs from the low end
of the screen axis to the high end, and the ramp always runs zmin to zmax in
the direction named:

| direction | ramp                   | window               |
|-----------|------------------------|----------------------|
| `"up"`    | top row is zmax        | `yrange=(zmin,zmax)` |
| `"down"`  | top row is zmin        | `yrange=(zmax,zmin)` |
| `"right"` | leftmost column is zmin| `xrange=(zmin,zmax)` |
| `"left"`  | leftmost column is zmax| `xrange=(zmax,zmin)` |

    class colorbar(image):
        def __init__(self, zrange, colormap=None, direction="up",
                     length=12, thickness=1):
            vertical = direction in ("up", "down")
            ramp = np.linspace(0., 1., 2 * length if vertical else length)
            if direction in ("up", "left"):
                ramp = ramp[::-1]
            ramp = ramp[:, None].repeat(thickness, axis=1) if vertical \
              else ramp[None, :].repeat(2 * thickness, axis=0)
            lo, hi = zrange
            span = (hi, lo) if direction in ("down", "left") else (lo, hi)
            super().__init__(im=ramp, colormap=colormap,
                             xrange=None if vertical else span,
                             yrange=span if vertical else None)

One asymmetry to document rather than fix: half-blocks give two pixels per cell
vertically and one horizontally, so a vertical bar of a given length has twice
the gradient resolution of a horizontal one. Nothing short of dithering changes
that.

## Part two: the mapping a colorbar is a picture of

`parse_colors` hands a colormap data that is already in [0,1]; the colormaps in
`matthewplotlib.colormaps` take nothing else. So the mapping from a value to a
position on the ramp lives in each caller, three times over, and one of them
throws it away:

* `image` does not normalise at all. The caller scales before calling.
* `function2` clips to `zrange`, and keeps it as `self.zrange`.
* `histogram2` divides by `max_count` and keeps `xbins`, `ybins` and
  `num_points` --- not `max_count`. Its attributes are exactly
  `chars, window, xbins, ybins, num_points`.

A colorbar for a `histogram2` therefore cannot be drawn at all: the plot no
longer knows what its colours mean.

What is wanted is a `norm`: a value type holding the interval and how values
outside it behave, which a plot keeps, which `parse_colors` composes with the
colormap, and which a colorbar renders. Then:

    mp.colorbar(plot)           # takes its range from the plot's own norm

and a log or symlog colour scale is a different norm rather than a different
code path, which is the roadmap's "configurable colour scales and
normalisation".

## Open questions

**Where the norm lives.** `colors.py` is about the spellings a colour arrives
in; `colormaps.py` about mapping [0,1] to colours. A norm is a third thing, and
may want its own module, the way `window.py` turned out to.

**What `colorbar` accepts.** A plot, so that it can read the norm off it; a
norm on its own; or a bare range and colormap, which is all the sketch above
takes. Probably all three, but the overload wants designing rather than
accreting.

**Whether `image` should normalise.** It is the one plot that asks the caller
to pre-scale. Giving it a norm would make it consistent with the two plots
built on it, and would change what every existing caller passes.

**Discrete colormaps.** `pico8` and `sweetie16` map integers to colours by
lookup --- `mp.pico8(np.arange(4))` returns four unrelated colours, not a ramp.
A scale for one of those is a column of swatches with a label beside each, not
a gradient with labelled ends. That is a different plot type, and possibly the
thing that legends want too.

**Clipping.** `function2` saturates values outside `zrange` at the ends of the
scale. Whether a colorbar should show that --- an arrow, a different end cap ---
or stay silent about it.

## Not in this design

Legends, which the roadmap wants and which share the "swatch beside a label"
shape with discrete colour scales. Worth looking at together, once one of them
has been built.
