"""
Power laws, hidden and found.

A Zipfian sample looked at four ways. Counting how often each value appears
gives a histogram whose linear columns show a few bars and then nothing: the
tail is there, but a linear height has nothing to spend on it. Log-scaled
columns give every decade of count the same share of the height, and the
whole staircase appears. Ranking the same counts largest-first and plotting
count against rank gives the classic rank-frequency scatter: an L hugging
the axes on linear coordinates, and a straight line once both axes are
logarithmic --- the power law, visible only in the view built to see it.

By Claude Fable 5.
"""

import tyro
import numpy as np
import matthewplotlib as mp


def main(
    size: int = 100_000,
    exponent: float = 2.0,
    seed: int = 42,
    save: str | None = None,
):
    """A Zipfian sample's power law, hidden and found."""
    rng = np.random.default_rng(seed)
    sample = rng.zipf(exponent, size=size)

    # counting how often each value appears is the histogram, and ranking
    # the same counts largest-first is the rank-frequency law
    frequencies = np.sort(np.unique(sample, return_counts=True)[1])[::-1]
    ranks = np.arange(1, len(frequencies) + 1)
    head = np.bincount(sample[sample <= 30], minlength=31)[1:]

    green = (0.30, 0.78, 0.45)
    scatters = mp.hstack(
        mp.axes(
            mp.scatter((ranks, frequencies, "cyan"), width=26, height=8),
            title=" count against rank ",
            xfmt="{x:g}",
            yfmt="{y:g}",
        ),
        mp.blank(1, 2),
        mp.axes(
            mp.scatter(
                (ranks, frequencies, "cyan"),
                xrange=mp.logscale(),
                yrange=mp.logscale(),
                width=26,
                height=8,
            ),
            title=" the same, log-log ",
            xfmt="{x:g}",
            yfmt="{y:g}",
        ),
    )
    # the same 50% gray the axes below default to, so the frames also
    # survive export onto a white background
    gray = (0.5, 0.5, 0.5)
    histograms = mp.hstack(
        mp.border(
            mp.columns(head, height=8, color=green),
            title=" counts of the values 1 to 30 ",
            color=gray,
        ),
        mp.blank(1, 2),
        mp.border(
            mp.columns(head, height=8, vrange=mp.logscale(1), color=green),
            title=" the same, log heights ",
            color=gray,
        ),
    )
    plot = mp.vstack(
        mp.center(histograms, width=scatters.width),
        mp.blank(1, 1),
        scatters,
    )
    print(plot)
    if save:
        plot.saveimg(save)


if __name__ == "__main__":
    tyro.cli(main)
