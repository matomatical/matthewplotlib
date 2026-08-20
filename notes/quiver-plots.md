# Quiver plots — framing notes

Written 2026-08-20 by Claude, when Matthew picked up the arrow half of the
roadmap's vector field entry, straight after the colour half (`vfunction2` and
the `chroma` colormap) landed. Parked the same day; see the status below.

The measurements are Claude's, the verdict is Matthew's, and both are from that
one afternoon. The prototype behind every number is at
`notes/reference/quiver-prototype.py`; run it to recheck any of them.

## Status: parked on legibility

Matthew looked at the renderings from both studies below and read all of them
as essentially unreadable, for a field whose underlying structure is simple.
Not worth pursuing now. `vfunction2` covers the vector field entry in the
meantime, and covers it well.

This is where it was left on the day, not a proof that arrows cannot work in a
terminal. Whether any of these variants can be made to read is exactly the open
question. If quiver plots are wanted badly enough to come back to, **the
challenge to beat is legibility, and it is not a machinery problem** --- the
drawing primitives are already in place and already fast, as below. Something
has to make a few dozen short strokes read as a field, and nothing tried here
did.

## What already exists

`examples/boids.py` draws arrows today. Each boid becomes a two-point `line`
series from `pos - vel * 2.5` to `pos`, coloured by its heading through
`mp.rainbow`. No arrowheads. It is the closest thing to a quiver in the repo.

**The batched primitive is already there, and `line` already goes through it.**
`line.__init__` pools every series it is given through `parse_segments` and
makes a single `unicode_braille_segments(starts[n,2], ends[n,2], ...)` call, so
drawing n arrows costs one rasterisation, with `thickness`, round caps and
filled joins included.

**A quiver would be a complement to `vfunction2`, not an alternative.** The
colour field gives a vector to every pixel and loses the sense of the arrow ---
hue tells you the direction, but nothing in it reads as *pointing*. A quiver
shows the sense and loses everything between the arrows. Both carry a `window`,
so layering them with `@` should work today; worth trying before designing
anything that assumes the two must become one plot.

## Study one: how small an arrowhead can get

Unit-length arrows on a lattice over a Taylor-Green vortex field, in a 44 by 12
cell plot, which is 88 by 48 braille dots. All lengths in dots.

| shaft | head | lattice | pitch | reads as |
| --- | --- | --- | --- | --- |
| 5 | none | 20 x 12 | 4.4 x 4 | texture; no direction at all |
| 8 | none | 14 x 8 | 6.3 x 6 | clean streamline texture; still no sense |
| 8 | 3 | 14 x 8 | 6.3 x 6 | sense readable, but arrows collide |
| 12 | 4 | 10 x 6 | 8.8 x 8 | best of the four; heads distinct |

* **A head costs three or four dots, and needs a shaft of eight to twelve.**
  Shorter and the head merges into the shaft.
* **So an arrowhead wants a lattice pitch around six cells by three rows.**
  Twelve dots is six cells across (two dots per column) and three rows down
  (four dots per row). An 80 by 24 terminal holds roughly 13 by 8 of them:
  about a hundred samples, against `vfunction2`'s roughly four thousand at the
  same size.
* **Arrows overran their lattice cells in every panel that had a head**, the
  best one included, because nothing clamped a length against the pitch.

## Study two: what can carry the sense instead

A bare stroke is symmetric, so it cannot say which of two opposite directions
it means. That is true by construction, and it is the whole argument for a
head --- but only if the geometry is the only thing available. It is not. Five
ways of carrying the sense, all on the 8-dot, 14 by 8 lattice from above, which
fits 112 arrows against the headed version's 60.

| variant | carries the sense with | outcome |
| --- | --- | --- |
| plain stroke | nothing | a *line* field: the axis reads, the sense is absent |
| coloured by direction | hue, via `chroma` | sense reads; spends the colour channel |
| dark tail, bright tip | brightness | sense reads; costs nothing, and leaves hue free |
| thick tail, thin tip | stroke weight | sense reads, but heavy ends blot together |
| arrowhead | geometry | collides at this pitch; needs the sparser one |

The brightness gradient is the cheapest of these by some way: `line` already
interpolates colour along a segment, so it is a two-colour series and no new
machinery at all. It also survives in monochrome, and it leaves hue free for
`chroma` to carry magnitude --- which is otherwise the thing a quiver has no
room to show, since putting magnitude in the length leaves a slow arrow as a
stub.

None of which was enough. Legible in isolation is not the same as legible as a
field, and the verdict above is about the field.

## If it is picked up again

* **The two studies above are about a lattice.** Streamlines --- arrows placed
  along traced trajectories rather than on a grid --- are what make matplotlib's
  `streamplot` legible at densities where its `quiver` is a thicket, and study
  one says this library lives at those densities on an ordinary terminal. This
  is the largest untried idea, and the one most likely to change the verdict.
* **Arrow glyphs** from the arrows block (← ↑ → ↓ ↖ ↗ ↘ ↙). One cell each, so
  the pitch drops from six by three cells to one by one --- eighteen times the
  arrows --- at the cost of eight fixed directions and no positioning within a
  cell. Whether the fonts in the support matrix carry them is unknown; see
  `docs/src/compatibility.md`.
* **Fewer, larger arrows.** Everything tried here filled the plot. A dozen big
  arrows over a colour field may say more than a hundred small ones.
* **Lengths would want to be in dots, not data.** The prototype worked in dots,
  where arrows come out isotropic because a braille dot is roughly square when
  a character cell is one by two. Lengths in data coordinates shear every arrow
  whenever the window is not square, which would make a quiver's geometry
  unlike that of every other plot here.
