"""
What line plots are for, and what a thicker pen does to one.

Above: two curves of the kind that made connecting the dots worth
implementing. A scatter of the same points leaves the eye to guess the order
they came in; a line states it. The lower curve is measured less often than the
upper one and has a stretch missing entirely, which is drawn as a gap rather
than as a straight line across the missing part: a non-finite coordinate ends
one stroke, and the next point starts another.

Below: one spiral, drawn four times with the pen set wider each time. The
stroke is the curve widened by a disc, so it stays the same width around the
tightest part of the turn as it is along the outside, and the joins between the
segments approximating the curve fill in rather than showing as notches.
"""

import tyro
import numpy as np

import matthewplotlib as mp


STEPS = 240
GAP = (150, 190)

TRAIN = (120, 220, 255)
TEST = (255, 120, 190)


def main(
    # the default is the width at which the chart comes out exactly as wide as
    # the row of spirals underneath it: four panels of sixteen, plus a column
    # of border either side of each, less the axis gutter and border of the
    # chart itself
    width: int = 67,
    height: int = 12,
    seed: int = 0,
    save: str | None = None,
):
    """A line chart, and the same spiral drawn with four widths of pen."""
    steps, train, sparse_steps, sparse_test = curves(seed)

    chart = mp.axes(
        mp.line(
            (steps, train, TRAIN),
            (sparse_steps, sparse_test, TEST),
            xrange=(0, STEPS - 1),
            width=width,
            height=height,
        ),
        title="loss",
        xlabel="step",
        ylabel="loss",
    )
    legend = mp.text("train", fgcolor=TRAIN) + mp.text("  test", fgcolor=TEST)

    pens = mp.hstack(*[
        mp.border(
            mp.line(
                spiral(),
                xrange=(-1.05, 1.05),
                yrange=(-1.05, 1.05),
                width=16,
                height=8,
                thickness=thickness,
            ) / mp.center(mp.text(f"{thickness:.0f}"), width=16),
            style=mp.BoxStyle.ROUND,
            color=(0.35, 0.35, 0.35),
        )
        for thickness in (1, 2, 3, 4)
    ])

    plot = chart / mp.center(legend, width=chart.width) / mp.center(
        pens,
        width=chart.width,
    )
    print(plot)

    if save:
        plot.saveimg(save, bgcolor="black")


def curves(seed: int) -> tuple[np.ndarray, ...]:
    """Two loss curves: one measured every step, one every eighth of one.

    A line joins the consecutive points of a series, so a curve measured less
    often is its own shorter series rather than a long one padded out with
    holes -- padding it would leave no two consecutive points to join at all.
    The one real hole, where the sparse curve stopped being measured, is the
    one non-finite point.
    """
    rng = np.random.default_rng(seed)
    steps = np.arange(STEPS, dtype=float)

    settling = np.exp(-steps / 70)
    train = 0.15 + 2.4 * settling + 0.12 * settling * rng.standard_normal(STEPS)

    sparse_steps = steps[::8]
    sparse_settling = settling[::8]
    sparse_test = (
        0.42 + 2.3 * sparse_settling
        + 0.05 * rng.standard_normal(len(sparse_steps))
    )
    # a stretch in the middle that was never measured: one gap, not a straight
    # line drawn across the missing part
    sparse_test[(sparse_steps >= GAP[0]) & (sparse_steps < GAP[1])] = np.nan
    return steps, train, sparse_steps, sparse_test


def spiral(turns: float = 2.0, points: int = 200) -> tuple[np.ndarray, ...]:
    """An Archimedean spiral, as a series: tight turns at one end, wide at the
    other, so one pen has to cope with both."""
    angle = np.linspace(0, turns * 2 * np.pi, points)
    radius = angle / angle[-1]
    return radius * np.cos(angle), radius * np.sin(angle)


if __name__ == "__main__":
    tyro.cli(main)
