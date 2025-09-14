"""
A Python plotting library that aspires to *not be painful.* See
[README.md](README.md) for overview.
"""


__version__ = "0.1.1"


from matthewplotlib.plots import (
    plot,
    scatter,
    function,
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
