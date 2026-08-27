"""
A collection of building blocks for plotting. There are lots of options---take
a look around this package. It is organised as one module per plot family,
and everything is re-exported here, so no family module ever has to be named
by hand.

Base class and arrangement plots ([`base`][matthewplotlib.plots.base]):

* `plot`: Every plot object inherits from this one. See this class for
  methods, properties, and shortcut operators available with every plot
  object.
* `blank`, `hstack`, `vstack`, `dstack`, `dstack2`, `wrap`, `center`, `crop`.

Data plots:

* [`points`][matthewplotlib.plots.points]: `scatter`, `scatter3`, `line`,
  `line3`, and `hilbert`.
* [`grids`][matthewplotlib.plots.grids]: `image`, `heatmap`, `function2`,
  `vfunction2`, `cfunction2`, and `histogram2`.
* [`barcharts`][matthewplotlib.plots.barcharts]: `progress`, `bars`,
  `histogram`, `columns`, `vistogram`, `candles`, and `boxes`.
* [`calendars`][matthewplotlib.plots.calendars]: `calendar` and `weeks`.
* [`tables`][matthewplotlib.plots.tables]: `table`, and the `Rule` type
  naming what it draws between its cells.

Furnishing plots ([`furnishings`][matthewplotlib.plots.furnishings]):

* `text` and `border`.
* `axes`, and the `Side` type naming what it draws along each of its four
  sides.
* `colorbar`, and the `Direction` type naming which way along the screen it
  runs.

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
