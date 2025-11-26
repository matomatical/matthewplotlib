"""
Top-level module. Imports various documents items from other modules and makes
them available under the top-level namespace.
"""


__version__ = "0.3.2"


from matthewplotlib.plots import (
    plot,
    scatter,
    scatter3,
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
    wrap,
    center,
    save_animation,
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
