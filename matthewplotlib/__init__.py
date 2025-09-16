"""
Top-level module. Imports various documents items from other modules and makes
them available under the top-level namespace.
"""


__version__ = "0.1.2"


from matthewplotlib.plots import (
    plot,
    scatter,
    function,
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
    blank,
    hstack,
    vstack,
    dstack,
    wrap,
    center,
)


from matthewplotlib.colors import (
    Color,
    ColorLike,
)


from matthewplotlib.core import (
    BoxStyle,
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
