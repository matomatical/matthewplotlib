"""
A collection of building blocks for plotting. There are lots of options---take
a look around this package. It is organised as one module per plot family,
and everything is re-exported here, so no family module ever has to be named
by hand.

Base class and arrangement plots ([`base`][matthewplotlib.plots.base]):

* [`plot`][matthewplotlib.plots.base.plot]: Every plot object inherits from
  this one. See this class for methods, properties, and shortcut operators
  available with every plot object.
* [`blank`][matthewplotlib.plots.base.blank],
  [`hstack`][matthewplotlib.plots.base.hstack],
  [`vstack`][matthewplotlib.plots.base.vstack],
  [`dstack`][matthewplotlib.plots.base.dstack],
  [`dstack2`][matthewplotlib.plots.base.dstack2],
  [`wrap`][matthewplotlib.plots.base.wrap],
  [`center`][matthewplotlib.plots.base.center], and
  [`crop`][matthewplotlib.plots.base.crop].

Data plots:

* [`points`][matthewplotlib.plots.points]:
  [`scatter`][matthewplotlib.plots.points.scatter],
  [`scatter3`][matthewplotlib.plots.points.scatter3],
  [`line`][matthewplotlib.plots.points.line],
  [`line3`][matthewplotlib.plots.points.line3], and
  [`hilbert`][matthewplotlib.plots.points.hilbert].
* [`grids`][matthewplotlib.plots.grids]:
  [`image`][matthewplotlib.plots.grids.image],
  [`heatmap`][matthewplotlib.plots.grids.heatmap],
  [`function2`][matthewplotlib.plots.grids.function2],
  [`vfunction2`][matthewplotlib.plots.grids.vfunction2],
  [`cfunction2`][matthewplotlib.plots.grids.cfunction2], and
  [`histogram2`][matthewplotlib.plots.grids.histogram2].
* [`barcharts`][matthewplotlib.plots.barcharts]:
  [`progress`][matthewplotlib.plots.barcharts.progress],
  [`bars`][matthewplotlib.plots.barcharts.bars],
  [`histogram`][matthewplotlib.plots.barcharts.histogram],
  [`columns`][matthewplotlib.plots.barcharts.columns],
  [`vistogram`][matthewplotlib.plots.barcharts.vistogram],
  [`candles`][matthewplotlib.plots.barcharts.candles], and
  [`boxes`][matthewplotlib.plots.barcharts.boxes].
* [`calendars`][matthewplotlib.plots.calendars]:
  [`calendar`][matthewplotlib.plots.calendars.calendar] and
  [`weeks`][matthewplotlib.plots.calendars.weeks].
* [`tables`][matthewplotlib.plots.tables]:
  [`table`][matthewplotlib.plots.tables.table], and the
  [`Rule`][matthewplotlib.plots.tables.Rule] type naming what it draws
  between its cells.

Furnishing plots ([`furnishings`][matthewplotlib.plots.furnishings]):

* [`text`][matthewplotlib.plots.furnishings.text] and
  [`border`][matthewplotlib.plots.furnishings.border].
* [`axes`][matthewplotlib.plots.furnishings.axes], and the
  [`Side`][matthewplotlib.plots.furnishings.Side] type naming what it draws
  along each of its four sides.
* [`colorbar`][matthewplotlib.plots.furnishings.colorbar], and the
  [`Direction`][matthewplotlib.plots.furnishings.Direction] type naming
  which way along the screen it runs.

The third stacking operation, `tstack`, arranges plots in time rather than
across the screen, and lives with the rest of the animation machinery in
`matthewplotlib.animations`.

The forms the data itself may arrive in are named in `matthewplotlib.data`;
`Orientation`, which the plots drawn either way about take, in
`matthewplotlib.core` with the drawing routine that reads it.
"""
from matthewplotlib.plots.base import (
    plot,
    blank,
    hstack,
    vstack,
    dstack,
    dstack2,
    wrap,
    center,
    crop,
)
from matthewplotlib.plots.points import (
    scatter,
    scatter3,
    line,
    line3,
    hilbert,
)
from matthewplotlib.plots.grids import (
    image,
    heatmap,
    function2,
    vfunction2,
    cfunction2,
    histogram2,
)
from matthewplotlib.plots.barcharts import (
    progress,
    bars,
    histogram,
    columns,
    vistogram,
    candles,
    boxes,
)
from matthewplotlib.plots.calendars import (
    calendar,
    weeks,
)
from matthewplotlib.plots.tables import (
    table,
    Rule,
)
from matthewplotlib.plots.furnishings import (
    text,
    border,
    axes,
    Side,
    colorbar,
    Direction,
)
