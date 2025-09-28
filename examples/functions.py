import numpy as np
import matthewplotlib as mp

xs = np.linspace(-2*np.pi, 2*np.pi, num=154)

print(mp.border(
    mp.scatter(
        mp.data.xaxis(-2*np.pi, 2*np.pi, 154),
        mp.data.yaxis(-1, 1, 32),
        *[
            (xs, (0.5 * y + 0.5) * np.sin(xs + (2*y-1)*2*np.pi), mp.cyber(y))
            for y in np.linspace(0., 1., num=5, endpoint=False)
        ],
        xrange=(-2*np.pi, 2*np.pi),
        width=72,
        height=8,
        yrange=(-1,1),
    )
))

print(mp.border(mp.function2(
    lambda XY: np.sin(XY.sum(axis=-1)),
    xrange=(-2*np.pi, 2*np.pi),
    yrange=(-np.pi, np.pi),
    width=72,
    height=18,
    colormap=mp.cyber,
)))
