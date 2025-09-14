import numpy as np
import matthewplotlib as mp


print(mp.border(mp.dstack(*[
    mp.function(
        lambda X: (0.5 * y + 0.5) * np.sin(X + (2*y-1)*2*np.pi),
        xrange=(-2*np.pi, 2*np.pi),
        width=72,
        height=8,
        yrange=(-1,1),
        color=mp.cyber(y),
    )
    for y in np.linspace(0., 1., num=10)
])))

print(mp.border(mp.function2(
    lambda XY: np.sin(XY.sum(axis=-1)),
    xrange=(-2*np.pi, 2*np.pi),
    yrange=(-np.pi, np.pi),
    width=72,
    height=18,
    colormap=mp.cyber,
)))
