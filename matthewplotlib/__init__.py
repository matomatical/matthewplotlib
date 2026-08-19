"""
Top-level module. Imports various documented items from other modules and makes
them available under the top-level namespace.
"""


__version__ = "0.6.2"


from matthewplotlib.plots import (
    plot,
    scatter,
    scatter3,
    line,
    line3,
    image,
    function2,
    histogram2,
    progress,
    bars,
    histogram,
    columns,
    vistogram,
    hilbert,
    text,
    border,
    axes,
    blank,
    hstack,
    vstack,
    dstack,
    dstack2,
    wrap,
    center,
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
    BoxStyle,
)


from matthewplotlib.data import (
    Series,
    Series3,
    xaxis,
    yaxis,
    zaxis,
)


from matthewplotlib.colormaps import (
    ContinuousColorMap,
    DiscreteColorMap,
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
)
