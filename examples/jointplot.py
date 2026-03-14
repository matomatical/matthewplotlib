"""
Joint distribution plot with marginal histograms.

Demonstrates: scatter, histogram, vistogram, columns, bars, hstack, vstack.

Author: Claude (claude.ai)
"""

import tyro
import numpy as np

import matthewplotlib as mp


def main(save: str | None = None):
    """Joint distribution plot with marginal histograms."""
    # --- generate bivariate data from a mixture of two Gaussians ---

    np.random.seed(42)

    n = 2000
    # cluster 1: top-right
    x1 = np.random.normal(1.5, 0.8, n // 2)
    y1 = np.random.normal(1.0, 0.6, n // 2)
    # cluster 2: bottom-left
    x2 = np.random.normal(-1.0, 0.5, n // 2)
    y2 = np.random.normal(-0.5, 1.0, n // 2)

    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])

    xrange = (-3.5, 4.0)
    yrange = (-4.0, 3.5)
    scatter_width = 50
    scatter_height = 20
    margin_size = 6

    # --- colors from density ---

    c = mp.viridis(np.concatenate([np.ones(n // 2) * 0.3, np.ones(n // 2) * 0.7]))


    # --- build the joint plot ---

    # central scatter plot
    main_plot = mp.scatter(
        (x, y, c),
        width=scatter_width,
        height=scatter_height,
        xrange=xrange,
        yrange=yrange,
    )

    # top margin: vistogram (vertical columns showing x distribution)
    # bins = scatter_width so each column aligns with one character of scatter
    top = mp.vistogram(
        x,
        bins=scatter_width,
        xrange=xrange,
        height=margin_size,
        color="white",
    )

    # right margin: histogram (horizontal bars showing y distribution)
    # bins = scatter_height so each bar aligns with one row of scatter
    right = mp.histogram(
        y,
        bins=scatter_height,
        xrange=yrange,
        width=margin_size,
        color="white",
    )

    # bottom: columns showing per-bin mean of y given x
    hist_x, bin_edges = np.histogram(x, bins=scatter_width, range=xrange)
    bin_indices = np.clip(np.digitize(x, bin_edges) - 1, 0, scatter_width - 1)
    mean_y = np.array([
        y[bin_indices == i].mean() if hist_x[i] > 0 else 0
        for i in range(scatter_width)
    ])
    bottom = mp.columns(
        np.clip(mean_y - yrange[0], 0, None),
        height=margin_size,
        column_width=1,
        vrange=yrange[1] - yrange[0],
        color="white",
    )

    # left: bars showing per-bin mean of x given y
    hist_y, bin_edges_y = np.histogram(y, bins=scatter_height, range=yrange)
    bin_indices_y = np.clip(np.digitize(y, bin_edges_y) - 1, 0, scatter_height - 1)
    mean_x = np.array([
        x[bin_indices_y == i].mean() if hist_y[i] > 0 else 0
        for i in range(scatter_height)
    ])
    left = mp.bars(
        np.clip(mean_x - xrange[0], 0, None)[::-1],
        width=margin_size,
        vrange=xrange[1] - xrange[0],
        color="white",
    )

    # --- assemble layout ---
    #
    #   [blank] [  top  ]  [blank]
    #   [left ] [ main  ]  [right]
    #   [blank] [bottom ]  [blank]
    #

    corner = mp.blank(margin_size, margin_size)

    plot = mp.border(
        (corner + top       + corner)
        /
        (left   + main_plot + right)
        /
        (corner + bottom    + corner),
        title=" joint distribution ",
    )

    print(plot)
    if save:
        plot.saveimg(save)

if __name__ == "__main__":
    tyro.cli(main)
