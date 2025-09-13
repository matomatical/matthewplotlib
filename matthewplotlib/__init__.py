"""
Matthew's plotting library (matthewplotlib).

A python plotting library that isn't painful.

See https://github.com/matomatical/matthewplotlib
"""


__version__ = "0.1.1"


from matthewplotlib.plots import (
    plot,
    image,
    fimage,
    scatter,
    hilbert,
    text,
    progress,
    blank,
    hstack,
    vstack,
    dstack,
    wrap,
    border,
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
)
