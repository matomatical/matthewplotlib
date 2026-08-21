# Drawing lines

Written 2026-08-14 by Claude (Opus 5), working with MFR, who asked for the
roadmap's line plots and wanted the design to leave 3d wireframes cheap to add
afterwards. Not reviewed line by line at the time of writing.

The result is `core.rasterise_segments`, with `plots.line` and `plots.line3` on
top of it and `camera.project3_segments` between them for the 3d case.

## Bresenham is the wrong shape for this

Bresenham's algorithm exists to avoid division and floating point: it carries an
integer error term and decides each step by its sign. Neither constraint applies
here, and the shape it imposes -- a sequential loop along one segment, inside a
loop over segments -- is exactly what NumPy is bad at.

Sampling does the same job. For a segment whose endpoints are `steps =
ceil(max(|dr|, |dc|))` dots apart along its major axis, take `steps + 1` samples
at `t = k/steps` and floor each to a dot. Sample spacing is then at most one dot
in the Chebyshev metric, so consecutive samples land in the same dot or an
8-adjacent one, which is the same no-gaps guarantee Bresenham gives.

The two agree exactly, and the tests check it:
`tests/test_core.py::TestRasteriseSegmentsThin::test_matches_bresenham` runs a
textbook integer Bresenham as an oracle over random segments and asserts the
same set of dots. They can only disagree where a sample falls exactly on a
boundary between two dots, which is a matter of convention; `has_tie` in that
file detects those segments and the exact comparison skips them. The remaining
properties (one dot per step along the major axis, both ends covered, no gap
between consecutive dots) are asserted for every segment including the tied
ones.

So Bresenham stays in the repository, as the test oracle rather than the
implementation.

## Flattening the ragged result

Segments differ in length, so their sample counts differ, and padding them to a
common length wastes whatever the longest segment costs. The flattening avoids
it:

    segment = np.repeat(np.arange(len(samples)), samples)
    within = np.arange(samples.sum()) - np.repeat(
        np.cumsum(samples) - samples, samples)
    t = within / np.maximum(samples - 1, 1)[segment]

`within` counts each sample's position inside its own segment, by subtracting
the segment's start offset from a flat index. One allocation, exactly the size
the work needs, and `segment` doubles as the gather index for per-segment data
(the clipped endpoints, and the colours at each end).

What this buys, measured on 2000 random segments across a 200x200 dot grid, each
about 100 dots long, with a colour interpolated along it:

    vectorised       41 ms
    per-segment     969 ms      (23x slower)

The comparison is against a Python loop doing the *same* job -- accumulating
coverage counts and colour sums into the grid -- because a loop that only
computes a list of dots and throws it away comes out only 2-3x behind, which
would flatter the loop by omitting most of the work.

For the two examples that use this, `vaporwave.py` draws 853 segments in 4.6 ms
a frame, backdrop and all, and `starburst.py` 24 rays at thickness 4 in 2.3 ms.
Both sit inside a 20 fps budget with a factor of ten to spare.

## Thickness is a Minkowski sum

Bresenham does not widen well. The published extensions (Murphy's modified
Bresenham, or stepping parallel offset lines) are intricate, and they still
leave the joins between consecutive segments to be reasoned about separately.

A thick line is the segment's Minkowski sum with a disc, and the samples are
already about one dot apart, so stamping a disc of integer offsets at every
sample *is* that sum -- one broadcast, `dot[:, None, :] + offsets[None, :, :]`.
Round caps follow, and so do the joins: the union of two capsules that share an
endpoint is already the right shape, so there is no mitre or bevel case at all.
`disc_offsets` is the disc, and a thickness of 1 selects the single dot `(0, 0)`
without needing a special case for thin lines.

The cost is the number of dots in the disc, which grows quadratically with
thickness:

    thickness 1     15 ms      (2000 segments, no colour)
    thickness 2     50 ms
    thickness 4     97 ms
    thickness 8    338 ms

Rejected alternative: a signed distance field, testing every dot in a segment's
bounding box for distance to the segment. It is the more usual answer and it
yields antialiasing for free, but the bounding box of a long diagonal segment is
`L` by `L`, so it costs quadratically in length where stamping costs linearly.
Antialiasing would not pay for itself here in any case: braille dots are on or
off, and colour belongs to the character cell rather than the dot, so there is
nowhere to put partial coverage.

## Clipping, twice, for two different reasons

Segments are clipped to the dot grid (Liang and Barsky's algorithm, vectorised
over segments as four constraints on the parameter interval) *before* they are
sampled. This is not an optimisation. Perspective projection sends a point close
to the camera arbitrarily far from the view, so without clipping first, the
sample count of a single segment is unbounded and one wire pointed near the
camera exhausts memory. Clipping caps it at the diagonal of the grid.

Then `camera.project3_segments` clips in camera space, against a near plane just
in front of the camera, before projecting at all. A segment with one end behind
the camera cannot be projected endpoint-wise: dividing by a negative depth
reflects that end through the centre of the image, and the wire is drawn across
the wrong half of the view. Cutting it at the near plane keeps the part that is
really visible. `project3` gains nothing from this and is left alone. What the
two do share is factored out: `camera.view_matrix` into the camera's
coordinates, `camera.perspective` out of them onto the film, so the only
difference between the two functions is the cut in the middle.

## The data limits land on the outermost dot centres

`plots._to_dots` maps the data range onto `[0.5, dots - 0.5]`, so the
extremes of the data sit on the centres of the outermost dots rather than on the
outer edge of the grid.

The first attempt mapped onto `[0, dots]` and treated a coordinate of exactly
`dots` as belonging to the last dot, mirroring how `scatter` clamps a point
sitting on the top bin edge. That interacted badly with clipping: a clipped
endpoint lands exactly *on* the boundary by construction, so the rule fired for
segments merely passing out of view and pulled a spurious dot back inside the
grid. Diagonal segments leaving through a corner grew a stray dot in the last
row. Mapping to dot centres removes the case rather than special-casing it, and
it is what a line plot should do anyway: a line between the extremes of its data
runs corner to corner of the plot.

`scatter` was brought onto the same mapping rather than left on its own, which
it had by way of `np.histogram2d`'s bins. Its points move by at most one dot,
and in exchange the two agree exactly: a line now passes through every dot a
scatter of its points marks, which
`tests/test_plots.py::TestLine::test_covers_the_scatter_of_the_same_points`
asserts.

## Edge soup underneath, polylines on top

The rasteriser takes arrays of independent segments, not a polyline. A polyline
is the easy case of that, and a mesh is not expressible as one at all, so the
lower layer is the general one. `data.parse_segments` turns series into
segments, taking each series separately so that the last point of one is never
joined to the first point of the next -- where `parse_multiple_series` pools
them all into one cloud, which is what a scatter wants.

Separate lines are separate series, which is what `vaporwave.py` does: a series
per wire of the mesh, all projected in one call. Gaps are for the other job,
holes *inside* one line, as in the sparsely measured curve in `lines.py`. The
two are not alternatives: a series padded out with holes has no two consecutive
points left to join.

Open, and the reason the internals are shaped this way:

* An explicit edge list (`vertices`, `edges`) would project each vertex once
  instead of once per wire through it. `vaporwave.py` currently projects every
  mesh vertex twice, once for its row and once for its column. At this size it
  does not matter; for a real mesh it would.
* A bundle of equal-length strokes, `float[strokes, points, 3]`, would save a
  mesh from being unpacked into one series per wire. `parse_series3` takes two
  dimensions only.
* Hidden line removal needs a depth per dot, which the dot grid has no room
  for. Colouring by depth, as `vaporwave.py` does, is the cheap stand-in.

## `line3` is not a subclass of `line`

`scatter3` subclasses `scatter`, and carries a TODO saying it should not,
because it inherits `xrange` and `yrange` and so can be passed to `axes`, which
then labels the axes with projected coordinates that mean nothing. `line3`
therefore extends `plot` directly and shares the drawing tail with `line` as a
module-private function instead. The cost is that the two 3d plot types are now
inconsistent with each other until `scatter3` is changed the same way.
