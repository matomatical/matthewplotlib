"""
Candlesticks: a simulated price series, four numbers to a period.

A candle's body spans the value a period opened at and the value it closed at,
coloured by which of the two was higher, and its wick reaches out of the body
to the highest and lowest values the period touched. The series here is a
random walk rather than any real market, so the picture is a demonstration of
the plot type and not of anything that happened.

Unlike most plots in the library, a candlestick chart paints its own
background. That is what buys it the vertical resolution: a body is placed to
an eighth of a character cell, and half of those eighths are drawn as a
background-coloured block over a body-coloured cell, which only works with the
background named. The bottom panel shows the same thing on a pale background to
make the point that the colour is a choice.

By Claude Opus 5.
"""

import numpy as np
import tyro

import matthewplotlib as mp


def walk(
    periods: int,
    seed: int,
    start: float = 100.0,
    drift: float = 0.01,
    volatility: float = 0.25,
):
    """A random walk, sampled four ways per period.

    Each period is walked in small steps; where it began and ended are its
    opening and closing values, and the extremes it reached along the way are
    its high and its low. Sampling a path rather than drawing four independent
    numbers is what makes the wicks sit outside the bodies, as they must.
    """
    rng = np.random.default_rng(seed)
    steps = rng.normal(drift, volatility, size=(periods, 16))
    path = start + np.cumsum(steps.reshape(-1)).reshape(periods, 16)
    return dict(
        opens=path[:, 0],
        highs=path.max(axis=1),
        lows=path.min(axis=1),
        closes=path[:, -1],
    )


def main(
    periods: int = 56,
    seed: int = 11,
    height: int = 14,
    save: str | None = None,
):
    """A candlestick chart of a simulated price series.

    Draws the whole series one column to a period, the last stretch of it with
    room for wider bodies, and the whole series again on a pale background.
    """
    series = walk(periods=periods, seed=seed)
    recent = {name: values[-12:] for name, values in series.items()}

    whole = mp.axes(
        mp.candles(**series, height=height),
        title=f"{periods} periods, one column each",
        ylabel="value",
        yfmt="{y:.0f}",
    )
    zoomed = mp.axes(
        mp.candles(**recent, height=height, body_width=3, spacing=1),
        title="the last 12, three columns each",
        ylabel="value",
        yfmt="{y:.0f}",
    )
    pale = mp.axes(
        mp.candles(
            **series,
            height=height // 2,
            background=(0.96, 0.95, 0.93),
            rising=(0.11, 0.35, 0.55),
            falling=(0.85, 0.45, 0.10),
            wick=(0.45, 0.45, 0.45),
        ),
        title="a pale background, and colours to suit it",
        yfmt="{y:.0f}",
    )

    plot = whole / zoomed / pale
    print(plot)
    if save:
        plot.saveimg(save)


if __name__ == "__main__":
    tyro.cli(main)
