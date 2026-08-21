# Hilbert curves as an encoding, not a plot type

Designed 2026-08-20 by MFR and Claude (Opus 5), written up by Claude, from a
session that started at the roadmap's non-square and 3d Hilbert curves and
turned into an overhaul of what `hilbert` is for. Nothing is built. MFR wants to
come back to it fresh before any code is written, so read this as a proposal to
be re-argued, not a decision.

The shape of the design is MFR's: the principle that a plot type commits to one
rendering, the split of `mask` from `data`, the restriction to even-sided
rectangles, `trail` as a plot type in its own right, and the scepticism about 3d.
The measurements, the gilbert verification and the `dots` gap are Claude's, and
reproduce from the snippets named at the end.

## Where it started

`plots.hilbert` takes `bool[N]`, rounds `N` up to the next `4**n` cells, asks
`numpy-hilbert-curve` for the `2**n` by `2**n` curve, and lights a braille dot at
every cell whose value is `True`, all in one colour.

* It cannot show a value, so the colormaps never reach the one plot type whose
  purpose is laying a long sequence out compactly.
* It cannot choose its shape. `N = 15565` claims 16384 cells; `N = 16385` claims
  65536. No rectangle is reachable, though a terminal is one.
* It cannot be asked for the curve. `examples/colormaps.py` wants the layout, not
  the plot, so it imports `hilbert` and calls `decode` itself. When an example
  reaches around a plot type to the dependency behind it, the useful thing is the
  layout, and the layout is not exported.
* It has no third dimension, and no route to one.

## The principle: a plot type commits to one rendering

The library already works this way. `scatter` is braille and `image` is half
blocks, and you choose the plot type in order to choose the primitive; there is
no `scatter(style=blocks)`. So the first draft of this design, a `hilbert` with
`style=DOTS|BLOCKS|PATH`, was against the grain, and so is spelling the same
switch as three plot types named `hilbert_dots`, `hilbert_blocks` and
`hilbert_lines`, which also gives one curve far more of the namespace than it has
earned.

Applying the principle instead: the renderings are plot types, the curve is an
encoding, and pictures are compositions.

| rendering | plot type | data cells per character |
| --- | --- | --- |
| braille dots | `dots` (new) | 8 |
| half blocks | `image` | 2 |
| box drawing | `trail` (new) | 1/2 |
| stroked line | `line`, `line3` | as spread out as the canvas allows |

```python
mp.dots(mp.fold(mask))                              # what hilbert draws today
mp.image(mp.fold(values), colormap=mp.viridis)      # values, and a colorbar
mp.trail(mp.hilbert_curve(16, 16), values)          # the box-drawing picture
mp.line3(mp.hilbert_curve3(4, 4, 4))                # the 3d curve
```

Hilbert then costs two names, `hilbert_curve` and `fold`, and neither is a plot
type. Two of the four rows are general-purpose beyond this note, which is the
main argument for the whole rearrangement: `trail` draws any route through grid
cells, and `dots` draws any dot matrix.

## The encoding

```python
type Curve = Callable[[int, int], NDArray]      # (width, height) -> int[w*h, 2]

def hilbert_curve(width, height) -> NDArray             # int[width*height, 2]
def hilbert_curve3(width, height, depth) -> NDArray     # int[w*h*d, 3]
def snake_curve(width, height) -> NDArray               # the baseline to compare
def curve_shape(n, width=None, height=None) -> tuple[int, int]
def fold(values, width=None, height=None, curve=hilbert_curve, fill=nan) -> NDArray
```

Home: a new `curves.py`, since none of it parses user input (`data.py`) or draws
characters (`core.py`).

Coordinates are `(x, y)` with `y` upward, because that is what `scatter`, `line`
and `line3` take. `fold` is the one that transposes and flips into array order,
so `mp.image(mp.fold(values))` puts the start of the sequence at the bottom left.
Cells past the end of the sequence get `fill`.

`curve_shape(n)` takes `width = ceil(sqrt(n))` and `height = ceil(n / width)`,
each rounded up to even (see below). `n = 15565` gives 126 by 126 with 311 spare
cells, against today's 16384 with 819 spare.

### The algorithm, and what was checked

The generalisation is Červený's *gilbert* construction (BSD-2-Clause,
`github.com/jakubcerveny/gilbert`); the implementation would be ours, from the
published algorithm, credited in the module docstring. It carries a major axis
vector and an orthogonal vector rather than a power-of-two order, splitting a
rectangle in two along the major axis when it is more than 1.5 times as long as
it is wide and into three otherwise, recursing until one side is 1.

Checked over every shape from 1 by 1 to 25 by 25:

* **It is a bijection onto the rectangle.** All 625 shapes.
* **It is the Hilbert curve on a power-of-two square,** elementwise, not merely
  up to symmetry: `hilbert_curve(2**n, 2**n)` equals `hilbert.decode(hilberts=
  arange(4**n), num_dims=2, num_bits=n)` for `n` up to 5. So
  `numpy-hilbert-curve` can go, and anything drawing a square today keeps
  drawing exactly what it draws.
* **On an even-sided rectangle it is continuous.** Elsewhere it can contain one
  diagonal step, never more, and only when one side is even and the other odd
  -- which is also exactly when no continuous curve of that shape exists, by
  the checkerboard argument: a path of `N` cells alternates colours, so its ends
  differ in colour when `N` is even, and the corners the curve runs between do
  not. Hence even sides only, and no caveat to document.

Cost: pure Python recursion, one call per cell, 15625 cells in 19 ms and a
million in 1.2 s. Fine for anything a terminal shows; a vectorised power-of-two
path is available later if it ever matters.

The 3d construction is the same idea with three vectors and many more cases, and
is the part of this most likely to be wrong first time. The tests that catch it
are cheap: bijection onto the box, adjacency, and agreement with
`numpy-hilbert-curve` on `2**n` cubes.

## `dots`

`core.unicode_braille_array` renders a dot matrix with optional per-dot colours,
and nothing public sits on it: every braille plot type in the library goes
through `unicode_braille_points` or `unicode_braille_segments` instead, which
take data coordinates and a window. So a caller holding an array of dots -- a
Life board, a bitmap, braille art, or a folded Hilbert layout -- has to reach
into `core`.

```python
class dots(plot):
    def __init__(
        self,
        mask,               # bool[H,W]: where the dots go
        data=None,          # number[H,W]: what colour they are
        vrange=None,
        colormap=None,
        color=None,
        bgcolor=None,
    )
```

This is the braille counterpart of `image`: an array in, glyphs out, no
coordinates and no window. Eight dots share a character's foreground, so with
`data` the colours mix per character, as they do in `scatter`.

## `trail`

```python
class trail(plot):
    def __init__(
        self,
        cells,              # int[N,2], in the order they are visited
        data=None,          # a colour per cell
        vrange=None,
        colormap=None,
        color=None,
        style=LineStyle.LIGHT,
        cell_width=2,
    )
```

One cell per two characters, which is what makes a cell square, since a character
is twice as tall as it is wide. Every glyph is derived the way `unicode_frame`
derives its own: a cell reaches towards the cell before it and the cell after it,
and the four direction bits index the `LineStyle`. Three things follow for free.
A cell visited twice accumulates arms and draws a junction, `├` or `┼`, so a
self-crossing walk is correct. The two ends of the route get the half-length
stubs, `╵` and `╶`. And a break in the route, wherever it comes from, draws as
two stubs, which reads as an honest seam.

One detail the prototype got wrong and the implementation should not: the
character filling the gap between two cells joined left to right is
`style[LEFT|RIGHT]`, not a hardcoded `─`. Hardcode it and `HEAVY` draws thin gaps
between thick corners.

Beyond curves, this is the plot type for a maze solution, a tour, or the path a
piece took through a board.

## Masks, values and colour

`mask` and `data` are separate parameters: `mask` says which cells are drawn,
`data` says what colour they get. That is better than the first draft's reading
of a `bool` array as a mask and a float array as values, it composes (a mask
*and* values), and it survives translation to each rendering. Values follow
`calendar`: a scale set by `vrange`, `parse_colors` for the colours, `vmin` and
`vmax` kept on the object so a colorbar can label it.

The renderings do not leave holes equally well. `dots` and `trail` can genuinely
leave a cell empty. `image` cannot: it paints two pixels into every character
cell and has no per-cell transparency, so a masked-out cell has to be given a
colour. Teaching `image` to treat a non-finite value as transparent would close
the gap, and would serve `function2` and heatmaps generally -- worth its own
roadmap line rather than being smuggled in here.

`fold` stays out of it: it writes `fill` for the cells past the end of the
sequence, and a caller who wants a mask writes `np.where(mask, values, np.nan)`.

## Three dimensions

A dense 3d curve is mush: `line3` on `hilbert_curve3(8, 8, 8)` is 512 nodes of
tangled braille in a 60 by 30 canvas and reads as a coloured blob. What reads is
the sparse version -- nodes far enough apart to see the edges between them --
and that is what `line3` already does, since the canvas resolution sets the
spacing. 64 nodes at 4 by 4 by 4 reads; 512 does not.

So 3d needs no plot type and no new rendering. It needs `hilbert_curve3` and a
small example. It is also the only proposed use of `hilbert_curve3`, which is an
argument for building the 3d construction last.

## The example

Two candidates were prototyped.

**A byte map of a file**, the binvis idea, does not survive contact with the
repository, and is recorded here so it is not proposed again. Byte maps are
striking on heterogeneous binaries, where headers, string tables, code and
padding all look different. Every file we could ship is homogeneous:
`matthewplotlib/unscii16.py` renders as a uniform green field and
`images/hilbert_curve.png` as uniform noise. Pointing the example at a system
binary would make the golden snapshot depend on the machine.

**What the curve is for**, which the pictures make in one glance. Colour a
sequence by which block of 64 it belongs to and fold it into 64 by 32: under the
Hilbert curve every block is a compact 8 by 8 square, under a snake curve every
block is a 64 by 1 stripe. Mean distance between the cells holding indices 64
apart is 9.95 against the snake's 33.00. So the example is one figure of three
panels -- the curve itself through `trail`, the same data folded both ways
through `image`, and a dense mask through `dots` -- which covers all three new
plot types, the non-square case, and the reason any of it exists. It needs no
data files, and it replaces `examples/hilbert_curve.py`.

## To settle when we come back

* **Does `mp.hilbert` retire?** Under this design it becomes `mp.dots(mp.fold(
  mask))` and the name goes. Claude leans towards retiring it: `dots` covers the
  picture, and `hilbert` is the one call in the library where the encoding and
  the rendering are welded together. The alternative is keeping it as the braille
  plot only, which respects the one-rendering principle but spends a third name
  on the curve and keeps the weld.
* **Is `fold` the right name**, and does it belong beside `hilbert_curve` in
  `curves.py` or with the parsers?
* **Does `dots` land in this change or its own?** It is the piece least connected
  to Hilbert curves and the piece most likely to be wanted elsewhere.
* **How much of `snake_curve` is wanted** beyond the one example that compares
  against it.

## Checked with

Prototypes in the session scratchpad, none of it library code, quoted here for
the numbers: `gilbert.py` (the construction, and the bijection and adjacency
sweep), `parity.py` (the checkerboard argument against the observed breaks),
`vscheck.py` (elementwise agreement with `numpy-hilbert-curve`, and the
timings), `preview.py` (the renderings), `locality.py` (the block pictures and
the distances), `demo.py` (all of the pictures in one run).
