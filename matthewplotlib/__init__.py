"""
Matthew's plotting library.

Everything the library offers is re-exported here, so one import reaches all of
it. Write `mp.scatter`, `mp.animate`, `mp.viridis`, and so on; no submodule ever
has to be imported by hand.

```python
import matthewplotlib as mp
```

The code behind that namespace is organised into submodules, and the API
reference documents it one submodule at a time.

The plots themselves:

* [`plots`][matthewplotlib.plots]: All of the plot types, and the operators
  that compose them into larger plots. The place to start.
* [`animations`][matthewplotlib.animations]: A sequence of plots as a value,
  and the terminal session that shows one.

What a plot accepts:

* [`data`][matthewplotlib.data]: The forms in which point data can arrive, and
  how each is normalised before plotting.
* [`colors`][matthewplotlib.colors]: The spellings a color can arrive in, one
  for a whole series or one per point.
* [`colormaps`][matthewplotlib.colormaps]: Ready-made continuous and discrete
  colormaps to map data onto colors.

What draws them:

* [`core`][matthewplotlib.core]: The grid of coloured characters every plot is
  ultimately drawn on, and the glyphs dense enough to draw with.
* [`window`][matthewplotlib.window]: The interval of data a plot covers on
  each axis, and how that lands on the grid it is drawn in.
* [`camera`][matthewplotlib.camera]: Projecting points and lines in space onto
  a viewing plane, for the plot types that take three dimensions.
* [`unscii16`][matthewplotlib.unscii16]: The bitmap font that renders a plot
  when it is exported to an image or a gif.
"""


__version__ = "0.7.0"


from matthewplotlib.plots import (
    plot,
    scatter,
    scatter3,
    line,
    line3,
    image,
    heatmap,
    function2,
    vfunction2,
    cfunction2,
    histogram2,
    progress,
    bars,
    histogram,
    columns,
    vistogram,
    candles,
    boxes,
    hilbert,
    calendar,
    weeks,
    table,
    Rule,
    text,
    border,
    axes,
    Side,
    colorbar,
    Direction,
    blank,
    hstack,
    vstack,
    dstack,
    dstack2,
    wrap,
    center,
    crop,
)


from matthewplotlib.animations import (
    tstack,
    animation,
    animate,
)


from matthewplotlib.colors import (
    ColorLike,
)


from matthewplotlib.core import (
    Align,
    BoxStyle,
    LineStyle,
    Orientation,
)


from matthewplotlib.window import (
    window,
)


from matthewplotlib.data import (
    Series,
    Series3,
    DateLike,
    DateSeries,
    TableData,
    xaxis,
    yaxis,
    zaxis,
)


from matthewplotlib.colormaps import (
    ContinuousColorMap,
    DiscreteColorMap,
    VectorColorMap,
    ColorMap,
    reds,
    greens,
    blues,
    yellows,
    magentas,
    cyans,
    divreds,
    divgreens,
    divblues,
    cyber,
    rainbow,
    magma,
    inferno,
    plasma,
    viridis,
    sweetie16,
    pico8,
    tableau,
    nouveau,
    chroma,
    domain,
)
