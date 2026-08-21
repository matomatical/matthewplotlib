"""
Box plots: four distributions that a five-number summary cannot tell apart.

Each of the four groups is built to share a median and an interquartile range
with the others, so the four boxes come out very nearly the same. The samples
they are drawn from are not the same at all: one is symmetric, one is skewed to
the right, one is split into two clusters with nothing in the middle, and one is
symmetric but heavy in the tails. The only part of the picture that gives the
difference away is the outlying points, which is a fair reason for Tukey's rule
to be the default.

A box plot lies flat by default, which puts the value axis across the terminal
where it has both more character cells and finer ones. Standing the boxes up
instead is `box_direction="vertical"`, and fits many more groups on a screen.

The group names are stacked up beside the boxes by hand, because the axis along
a row of boxes is a list of names rather than an interval of numbers, and the
library has no way to describe one yet.

By Claude Opus 5.
"""

import numpy as np
import tyro

import matthewplotlib as mp


def shapes(samples: int, seed: int):
    """Four differently shaped groups sharing a five-number summary.

    Each is drawn from its own distribution, then shifted and scaled so that
    its quartiles land on the same two values as everyone else's. That leaves
    the median free to differ a little, and leaves the tails entirely free,
    which is the point.
    """
    rng = np.random.default_rng(seed)
    half = samples // 2
    raw = {
        "symmetric": rng.normal(0.0, 1.0, samples),
        "skewed": rng.lognormal(0.0, 0.6, samples),
        "bimodal": np.concatenate([
            rng.normal(-2.0, 0.35, half),
            rng.normal(2.0, 0.35, samples - half),
        ]),
        "heavy tailed": rng.standard_t(4.0, samples),
    }
    groups = {}
    for name, values in raw.items():
        first, third = np.percentile(values, [25, 75])
        groups[name] = 100.0 + 20.0 * (values - first) / (third - first)
    return groups


def labels(names: list[str], colors: list[str], thickness: int, spacing: int):
    """The group names, each on the middle row of the box it belongs to.

    Coloured to match their boxes, which ties each name to its own box and, as
    a side effect, keeps them visible in an exported image: text left to the
    terminal's own foreground colour is exported as white.
    """
    width = max(len(name) for name in names)
    rows = []
    for i, name in enumerate(names):
        label = mp.text(name.rjust(width), fgcolor=colors[i])
        rows.append(mp.center(label, height=thickness))
        if i < len(names) - 1:
            rows.append(mp.blank(height=spacing, width=width))
    return mp.vstack(*rows)


def main(
    samples: int = 400,
    seed: int = 3,
    length: int = 56,
    box_thickness: int = 3,
    box_spacing: int = 1,
    filled: bool = False,
    save: str | None = None,
):
    """A box plot of four distributions with the same quartiles.

    `--filled` draws the boxes as solid fills instead of outlines, which places
    their edges to an eighth of a character cell rather than a whole one, at the
    cost of painting the plot's own background.
    """
    groups = shapes(samples=samples, seed=seed)
    palette = ["#7fb3d5", "#7dcea0", "#f5b041", "#e59866"]

    plot = mp.hstack(
        labels(
            names=list(groups),
            colors=palette,
            thickness=box_thickness,
            spacing=box_spacing,
        ),
        mp.blank(height=1, width=1),
        mp.axes(
            mp.boxes(
                list(groups.values()),
                length=length,
                box_thickness=box_thickness,
                box_spacing=box_spacing,
                filled=filled,
                colors=palette,
            ),
            xlabel="value",
        ),
    )

    print(plot)
    if save:
        plot.saveimg(save)


if __name__ == "__main__":
    tyro.cli(main)
