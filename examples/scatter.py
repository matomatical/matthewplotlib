import numpy as np

import matthewplotlib as mp

np.random.seed(42)

ts = np.linspace(0, 8*np.pi, 1000)
rs = np.linspace(0, 1, 1000)
es = 0.01 * 1/(rs+0.1) * np.random.normal(size=(2, 1000))
xs = rs * np.cos(ts) + es[0]
ys = rs * np.sin(ts) + es[1]
cs = mp.viridis(1-rs)

plot = mp.axes(
    mp.scatter(
        (xs, ys, cs),
        height=20,
        width=40,
        xrange=(-1.05, 1.05),
        yrange=(-1.05, 1.05),
    ),
    title=" scatter example ",
    ylabel="y",
    xlabel="x",
)
print(plot)

print("saving...")
plot.saveimg("images/scatter.png")
