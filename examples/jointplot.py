"""
Joint distribution plot with marginal and conditional panels.

A bivariate mixture of two Gaussians in the middle, with four panels around
it. Above and to the left are the marginal distributions, each anchored
against the scatter and growing away from it---the left one mirrored, so that
its bars run leftwards. Below and to the right are the conditional means, one
per bin, drawn as diverging charts measured from zero: a bar reaches right of
the baseline where the mean is positive and left of it where the mean is
negative, and a bin holding no samples has no mean and draws nothing.

Demonstrates: scatter, histogram, vistogram, columns, bars, mirrored and
diverging bar charts, hstack, vstack.

By Claude Opus 4.6, reworked by Claude Opus 5.
"""

import tyro
import numpy as np

import matthewplotlib as mp


# the panels around the scatter paint their own background, since a bar
# growing towards the low end of its axis is drawn against one
PANEL = (0.08, 0.09, 0.11)
RISING = (0.30, 0.78, 0.45)
FALLING = (0.90, 0.32, 0.36)


def conditional_means(
    values: np.ndarray,
    given: np.ndarray,
    bins: int,
    range: tuple[float, float],
) -> np.ndarray:
    """
    The mean of `values` within each bin of `given`, and nan where a bin holds
    no samples at all, having no mean to report.
    """
    edges = np.linspace(*range, bins + 1)
    which = np.clip(np.digitize(given, edges) - 1, 0, bins - 1)
    return np.array([
        values[which == i].mean() if (which == i).any() else np.nan
        for i in np.arange(bins)
    ])


def main(save: str | None = None):
    """Joint distribution plot with marginal and conditional panels."""
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
    # a character cell is twice as tall as it is wide, so the panels beside
    # the scatter need twice as many cells across as the ones above and below
    # it need down, to take up the same depth on the screen
    margin_rows = 6
    margin_columns = 2 * margin_rows

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

    # top margin: the distribution of x, one column per column of the scatter,
    # standing up out of the scatter's top edge
    top = mp.vistogram(
        x,
        bins=scatter_width,
        xrange=xrange,
        height=margin_rows,
        color="white",
        background=PANEL,
    )

    # left margin: the distribution of y, one bar per row of the scatter,
    # mirrored so that the bars run left out of the scatter's left edge. The
    # counts are turned over because a bar chart draws its first value at the
    # top and the y axis climbs the other way.
    counts_y, _ = np.histogram(y, bins=scatter_height, range=yrange)
    left = mp.bars(
        counts_y[::-1],
        width=margin_columns,
        mirror=True,
        color="white",
        background=PANEL,
    )

    # right margin: the mean x within each row of the scatter, measured from
    # zero, so that a row whose samples sit left of the origin reaches left.
    # The interval is symmetric about zero, putting the baseline down the
    # middle of the panel and giving each side the whole of its half.
    mean_x = conditional_means(x, y, bins=scatter_height, range=yrange)[::-1]
    reach_x = np.nanmax(np.abs(mean_x))
    right = mp.bars(
        mean_x,
        width=margin_columns,
        vrange=(-reach_x, reach_x),
        colors=[FALLING if m < 0 else RISING for m in mean_x],
        background=PANEL,
    )

    # bottom margin: the mean y within each column of the scatter, measured
    # from zero the same way, hanging below the baseline where it is negative
    mean_y = conditional_means(y, x, bins=scatter_width, range=xrange)
    reach_y = np.nanmax(np.abs(mean_y))
    bottom = mp.columns(
        mean_y,
        height=margin_rows,
        column_width=1,
        vrange=(-reach_y, reach_y),
        colors=[FALLING if m < 0 else RISING for m in mean_y],
        background=PANEL,
    )

    # --- assemble layout ---
    #
    #   [blank] [  top  ]  [blank]
    #   [left ] [ main  ]  [right]
    #   [blank] [bottom ]  [blank]
    #

    corner = mp.blank(height=margin_rows, width=margin_columns)

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
