# World maps, and which way the projection runs

Designed 2026-08-20 by MFR and Claude (Opus 5), written up by Claude, before
building any of it, prompted by the roadmap's world maps entry. The shape of the
public surface is MFR's decision; the measurements are Claude's, and each one
below names the formula or dataset it came from so it can be rechecked.

Two examples were built against this design first, both carrying their own
coarse outline data, so that the library could gain world maps without the
library gaining a dataset. `examples/globe.py` inverts a projection at every
pixel; `examples/world_map.py` runs five of them forwards over a list of
coordinates, and prototypes the `projection` value type proposed below. Nothing
in the library itself is built.

## The terminal's scale is world scale

Natural Earth publishes its vectors at three levels of generalisation, named
for the map scale each is drawn for. Counting the vertices of the land polygons
and the angular length of the segments between them:

    tier          vertices   median segment   90th percentile
    1:110,000,000    5,143            0.82°             1.66°
    1:50,000,000    60,669            0.10°             0.28°
    1:10,000,000   446,175            0.02°             0.06°

Against that, what one braille dot is worth. An equirectangular map `width`
cells across is `2 * width` dots across, and covers 360° of longitude:

    width 40 cells     4.50° per dot
    width 80 cells     2.25° per dot
    width 120 cells    1.50° per dot
    width 200 cells    0.90° per dot

So the coarsest tier published is already about the right density for a map that
fills a terminal, and is under-sampled by a map 200 cells wide. There is no
detail here to throw away, and nothing finer to gain by carrying more.

The same numbers say what world maps in a terminal are *not* for. Cropped to
Europe, 1:110m coastlines come out as long straight runs with Britain and
Ireland as stick figures; a regional map wants the 1:50m tier, which is 1.6 MB
of GeoJSON against 138 KB. A projection and a renderer do not care which tier
they are handed, so a caller who wants a regional map can pass their own rings.
Nothing in the design should assume the data is the one we ship, if we ship any.

## A projection runs both ways, and the inverse is the one that matters

A projection is a pair of maps between the sphere and the plane, and the two
plot styles want opposite directions.

* An **outline** map projects *vertices*: each ring's points go forward through
  the projection, and the segments between them are drawn into the dot grid.
  `window.dots` already turns data coordinates into that grid.
* A **filled** map inverts *pixels*: each pixel's centre goes backward through
  the projection to the point of the sphere seen there, and what is at that
  point decides the colour. `window.pixel_centres` already produces those
  coordinates.

The second is worth more than it looks. It makes a filled map the same shape of
problem as `function2` --- evaluate something at every pixel centre and colour
by the result --- so land and sea, a choropleth by country, and a field like
distance-from-a-point are all one mechanism with different lookups. It also
handles the edge of the map without a special case: a pixel with no preimage on
the sphere, which is every pixel outside a globe's disc or outside a sinusoidal
map's lens, comes back as `nan` and is left blank.

And it is what makes a globe cheap. Projecting sphere points forward onto the
screen sends *both* hemispheres into the same unit disc, so the far side lands
on top of the near side and has to be culled. Inverting per pixel never
computes the far side at all: `orthographic` inverted at a position in the disc
yields the one point of the *near* hemisphere seen there, and the far side has
no representation to bleed through. Checking a forward projection against its
inverse over a 1°-grid of the whole sphere shows this directly --- the
round-trip is exact to 1e-10 degrees on the near hemisphere and off by up to
180° on the far one, those being the antipodes that share a screen position.

A second dividend: the inverse hands back a latitude and longitude, hence the
unit surface normal at that pixel, so shading a globe by a sun direction is a
dot product on data the renderer already has.

## Which projections, and which are refused

Those with an inverse in closed form, or one cheap iteration away. Extents are
of the projected plane, in the units the formulae below give.

    projection       x extent    y extent   inverse
    equirectangular      2 pi          pi   identity
    sinusoidal           2 pi          pi   lon = x / cos y
    mercator             2 pi     6.2626*   lat = 2 atan(e^y) - pi/2
    mollweide         4 sqrt2     2 sqrt2   closed form, forward iterates
    hammer            4 sqrt2     2 sqrt2   closed form
    orthographic            2           2   closed form, near side only

    * cut at +-85° latitude; the true extent is infinite

Robinson and Winkel tripel are refused, for now. Both are defined by a table of
coefficients interpolated between parallels, so an inverse means interpolating
the table backwards, and neither the forward nor the backward direction is a
formula anyone can check by reading it. They are worth adding only if someone
wants that particular look enough to carry the table.

Mercator has no pole, so a cut is not optional: latitude 90° projects to
infinity. Cutting at ±85° puts the last parallel at y = 3.13, making the map
very nearly square, and loses nothing a terminal could have drawn.

## Mollweide needs more Newton steps than it looks

Mollweide's forward map needs an auxiliary angle `t` solving

    2t + sin 2t = pi sin(lat)

which has no closed form. Newton from `t = lat` is the textbook advice and it is
a trap: at the poles `t = ±pi/2` and the derivative `2 + 2 cos 2t` vanishes
there, so the iteration divides by zero at the pole itself and converges only
linearly near it. Measuring the error against a bisection reference, in units of
one braille dot on a width-80 map (1/160th of the map's x extent, or 0.0354 map
units):

    steps    from t = lat    from the cube-root seed
        1         30 dots                   0.39 dots
        2         20 dots                 0.0006 dots
        3         13 dots                 0.0005 dots
        5        5.8 dots                 0.0005 dots
       10       0.56 dots                 0.0005 dots

The seed that fixes it comes from the pole itself. Writing `t = pi/2 - u`, the
equation becomes `2u - sin 2u = pi (1 - sin lat)`, and `2u - sin 2u` is
`(4/3) u^3` to leading order, so

    u = cbrt(3 pi (1 - |sin lat|) / 4)

lands within a fraction of a dot before Newton starts, and is close enough near
the pole that the linear convergence there never gets a chance to matter. Away
from the pole the seed is mediocre --- off by 0.23 rad at the equator --- but
the convergence there is quadratic and cleans it up in one step. Three steps
from this seed is the whole solve, with no convergence test and no loop over
points.

The floor at 0.0005 dots is not the iteration's; it is the conditioning of the
equation at the pole. Since `f` is cubic in the distance from the pole, an error
of machine epsilon in `f` is an error of its cube root, about 5e-6, in `u`. The
pole's own coordinate is therefore only knowable to about 1/2000th of a dot,
which is 2000 times better than the picture needs.

## A path has to be broken where the map is

Every projection here cuts the sphere somewhere --- along the antimeridian, for
all five --- and any path crossing that cut arrives at one edge of the map and
leaves from the other. Joined up, it draws a line straight back across the
whole picture. A flight from Los Angeles to Sydney does this, and so does
Antarctica, whose ring runs through 180 degrees on its way round the pole.

The rule that fixes it, from `split_at_seam` in `examples/world_map.py`: insert
a gap wherever consecutive projected points jump more than half the map's
width, since `line` already treats a non-finite coordinate as a break in the
stroke. Half the width is chosen for margin, not for principle. Densely sampled
paths never approach it --- an arc of 96 samples steps under four degrees at a
time --- and the widest step any of the coarse coastlines takes is a sixth of
the map, in Antarctica.

It is not exact, and the exception is worth writing down before it is found the
hard way. Under an interrupted projection the map's width is not the same at
every latitude: sinusoidal is `x = lon cos(lat)`, so a wrap at sixty degrees
jumps `2 pi cos(60) = pi`, which is exactly the threshold. A library doing this
properly should ask the projection where its cut is and split against that,
rather than guessing from the size of the jump.

## A projection is a value

    mp.worldmap(projection=mp.mollweide, width=80)
    mp.worldmap(projection=mp.orthographic(lon0=0, lat0=30), width=40)

MFR's call, over accepting a string. The parameterised projections settle it:
`orthographic` is a family, one per point the globe is centred on, and a string
naming it would need somewhere else to put the two numbers. Values also let a
caller write their own and pass it, which is the same freedom `colormaps`
already gives --- a `ColorMap` is a value there, not a name, for the same
reason.

What a projection has to carry is therefore: the forward map, the inverse map,
and the extents of the projected plane, which is what lets a plot choose its own
height.

## Height follows from width

A character cell is about twice as tall as it is wide. Both dense grids the
library draws in divide that cell so that their own units come out square: the
dot grid is 2 across by 4 down, giving dots half a cell wide and half a cell
tall; the pixel grid is 1 across by 2 down, giving square pixels.

So a plot `width` cells across and `height` cells down is a rectangle `width`
by `2 * height` in units of cell width, and it shows the projection undistorted
when

    height = width * y_extent / (2 * x_extent)

which is the same expression for outline maps and filled maps both, since the
two grids agree on the shape of the cell they divide. A map and an outline of
the same projection at the same width therefore land on the same rectangle, and
can be overlaid. In particular an equirectangular, sinusoidal, mollweide or
hammer map is a quarter as tall as it is wide, an orthographic globe is half as
tall as it is wide, and a mercator map cut at ±85° is very nearly half.

## Where the data comes from is still open

Deferred, deliberately. Bundling has been costed but not decided: the 1:110m
land and coastline vectors pack into 20 KB as `int16` hundredths of a degree
(5,143 vertices at four bytes each), whose 0.005° quantisation is 1/450th of a
dot at width 80, and the package already carries a comparable weight of data in
`unscii16`. Countries with names and ISO codes would add another 25 KB and would
be what makes a choropleth possible, at the cost of the library shipping an
opinion about where borders are.

Until that is decided, the examples hard-code their own outlines at very low
resolution, which is enough to demonstrate every part of the design above and
leaves the question to be answered on its merits rather than by default.

## Open questions

* Whether to bundle vectors at all, and if so whether political boundaries.
* Whether a filled map takes its data as an array over pixels, a lookup per
  country, or a function on the sphere. The rasteriser does not care; the
  constructor has to pick a spelling.
* Whether the 3d globe is `worldmap` under `orthographic`, which is a globe
  seen from infinitely far away, or a separate plot that puts the sphere in
  front of a `camera` and gains perspective and a free viewpoint. The examples
  should say which is worth having.
* Whether a projection carries its own seam, so that paths can be split
  against it exactly rather than by the size of the jump they take.
* How data in longitude and latitude reaches a map that is already drawn.
  `dstack2` demands one shared window, which is available --- the projection
  and the width fix it --- but nothing yet makes it convenient.
