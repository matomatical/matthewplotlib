"""
Interpretation of https://matplotlib.org/stable/gallery/statistics/time_series_histogram.html
"""

import numpy as np
import matthewplotlib as mp


# config

np.random.seed(19680801)
num_points = 158
num_series = 1000
SNR = 0.10

# generate data

# time
x = np.linspace(0, 4 * np.pi, num_points)
X = np.broadcast_to(x, (num_series, num_points))

# unbiased Gaussian random walks
num_noise = round((1-SNR) * num_series)
Y_random_walk = np.cumsum(
    np.random.randn(num_series, num_points),
    axis=-1,
)[:num_noise] # generate more to preserve random state cf. original
# slightly noisy sinusoidal signals
num_signal = round(SNR * num_series)
phi = (np.pi / 8) * np.random.randn(num_signal, 1)
scale = np.sqrt(np.arange(num_points))
noise = 0.05 * np.random.randn(num_signal, num_points)
Y_sinusoidal = scale * (np.sin(x - phi) + noise)
# combine
Y = np.concatenate([Y_random_walk, Y_sinusoidal])

# Stacked scatter plots series.
plot1 = mp.border(mp.dstack(*[
    mp.scatter(
        data=np.c_[x, Y[i]],
        color=mp.tableau(0),
        width=78,
        height=11,
        yrange=(Y.min(), Y.max())
    )
    for i in range(num_series)
]))
print(plot1)

# Pooled scatter plots series.
plot2 = mp.border(mp.scatter(
    data=np.c_[X.ravel(), Y.ravel()],
    color=mp.tableau(0),
    width=78,
    height=11,
))
print(plot2)

# TODO: log scale

# Histogram 2d, linear colour scale
plot3 = mp.border(mp.histogram2(
    x=X.ravel(),
    y=Y.ravel(),
    width=78,
    height=11,
    colormap=mp.plasma,
))
# TODO: colorbar
print(plot3)


