# Quiver plots — framing notes

Written 2026-08-20 by Claude, when Matthew picked up the arrow half of the
roadmap's vector field entry, straight after the colour half (`vfunction2` and
the `chroma` colormap) landed. Not designed. This note records what the colour
half established, and what half an hour of prototyping measured, so that the
design does not start from scratch.

The prototype is kept at `notes/reference/quiver-prototype.py`; run it to
recheck any number below.

## What already exists

`examples/boids.py` draws arrows today. Each boid becomes a two-point `line`
series from `pos - vel * 2.5` to `pos`, coloured by its heading through
`mp.rainbow`. No arrowheads. It is the closest thing to a quiver in the repo
and it is worth looking at before designing one.

**The batched primitive is already there, and `line` already goes through it.**
`line.__init__` pools every series it is given through `parse_segments` and
makes a single `unicode_braille_segments(starts[n,2], ends[n,2], ...)` call, so
drawing n arrows costs one rasterisation, with `thickness`, round caps and
filled joins included. Drawing is not the design problem. Deciding what
segments to draw is.

**A quiver is a complement to `vfunction2`, not an alternative.** The colour
field gives a vector to every pixel and loses the sense of the arrow --- hue
tells you the direction, but nothing in it reads as pointing. A quiver shows
the sense and loses everything between the arrows. Both carry a `window`, so
layering one over the other with `@` should work today; that is worth trying
before designing anything that assumes the two must become one plot.

## What the prototype measured

Unit-length arrows on a lattice over a Taylor-Green vortex field, in a 44 by 12
cell plot, which is 88 by 48 braille dots. All lengths in dots.

| shaft | head | lattice | pitch | reads as |
| --- | --- | --- | --- | --- |
| 5 | none | 20 x 12 | 4.4 x 4 | texture; no direction at all |
| 8 | none | 14 x 8 | 6.3 x 6 | clean streamline texture; still no sense |
| 8 | 3 | 14 x 8 | 6.3 x 6 | sense readable, but arrows collide |
| 12 | 4 | 10 x 6 | 8.8 x 8 | best of the four; heads distinct |

Four findings.

1. **No head, no vector.** Without an arrowhead the picture is a *line* field:
   the axis of the flow is legible and which way it runs along that axis is
   not. That is a real loss, and it is the whole reason to draw arrows instead
   of colouring pixels. Colour could carry the sense instead, but then the
   arrows are doing no work that `vfunction2` does not already do better.
2. **A head costs three or four dots, and needs a shaft of eight to twelve.**
   Shorter and the head merges into the shaft.
3. **So the lattice pitch is around six cells by three rows.** Twelve dots is
   six cells across (two dots per column) and three rows down (four dots per
   row). An 80 by 24 terminal therefore holds roughly 13 by 8 arrows: about a
   hundred samples, against `vfunction2`'s roughly four thousand at the same
   size. That budget is the central constraint on the whole design.
4. **Arrows overran their lattice cells in every panel that had a head**, the
   best one included, because the prototype never clamped a length against the
   pitch. Whatever the API turns out to be, something has to.

## Questions to answer first

* **Does length carry magnitude?** The prototype drew unit arrows. If length
  carries magnitude then a slow arrow is a stub with no room for a head, which
  is finding 1 again for exactly the part of the field where the structure
  usually is. Options: a length with both a floor and a cap; a fixed length
  with magnitude in the colour, through `chroma`; or both at once, redundantly.
* **What is a length measured in?** The prototype worked in dots, where arrows
  come out isotropic because a braille dot is roughly square when a character
  cell is one by two. Lengths in data coordinates would shear every arrow
  whenever the window is not square. Screen space is probably right, which
  would make a quiver's geometry unlike that of every other plot here.
* **Who chooses the lattice?** A spacing in cells, a count of arrows, or
  positions from the caller? Sampling a function mirrors `vfunction2`; taking
  positions and vectors mirrors `scatter`. The library already splits this way
  once, between `function2` and `image`, so both may be wanted.
* **Arrowhead geometry is untuned.** The prototype used two barbs at 150
  degrees of three or four dots. A filled head is not available at this
  resolution.

## Not tried

* **Arrow glyphs** from the arrows block (← ↑ → ↓ ↖ ↗ ↘ ↙). One cell each, so
  the lattice pitch drops from six by three cells to one by one --- eighteen
  times the arrows --- at the cost of eight fixed directions and no positioning
  within a cell. Whether the fonts in the support matrix carry them is unknown;
  see `docs/src/compatibility.md`.
* **Streamlines**: arrows placed along traced trajectories rather than on a
  lattice. This is what makes matplotlib's `streamplot` legible at densities
  where its `quiver` is a thicket, and finding 3 says this library will be at
  those densities on an ordinary terminal.
