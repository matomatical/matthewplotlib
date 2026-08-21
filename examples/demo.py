"""
General demonstration combining images, borders, and scatter plots.

By Matthew Farrugia-Roberts.
"""

import tyro
import numpy as np
import matthewplotlib as mp


def main(save: str | None = None):
    """Demo plot with images, borders, and scatter."""
    size = 14
    np.random.seed(42)
    u = np.random.rand(size**2).reshape(size,size)
    i = np.eye(size)
    g = np.clip(np.random.normal(size=(size, size)) + 3, 0, 6)
    plot = (
        mp.border(
            mp.center(mp.text("G'day matthewplotlib"), height=3, width=46),
            style=mp.BoxStyle.DOUBLE,
        )
        |
        mp.border(
            mp.text("uniform:") | mp.heatmap(u, colormap=mp.reds, vrange=(0, 1)),
            style=mp.BoxStyle.LIGHT,
        ) + mp.border(
            mp.text("identity:") | mp.heatmap(i, colormap=mp.greens, vrange=(0, 1)),
            style=mp.BoxStyle.HEAVY,
        ) + mp.border(
            mp.text("gaussian:") | mp.heatmap(g, colormap=mp.blues, vrange=(0, 6)),
            style=mp.BoxStyle.DOUBLE,
        )
        |
        mp.border(
            mp.text("uniform:") | mp.heatmap(u, colormap=mp.yellows, vrange=(0, 1)),
            style=mp.BoxStyle.ROUND,
        ) + mp.border(
            mp.text("identity:") | mp.heatmap(i, colormap=mp.cyber, vrange=(0, 1)),
            style=mp.BoxStyle.BLANK,
        ) + mp.border(
            mp.text("gaussian:") | mp.heatmap(g, colormap=mp.cyans, vrange=(0, 6)),
            style=mp.BoxStyle.BUMPER,
        )
        |
        mp.border(
            mp.scatter(
                (np.random.normal(size=(300,2)), 'green'),
                height=18,
                width=46,
                xrange=(-5, +5),
                yrange=(-4, +4),
            ),
            style=mp.BoxStyle.ROUND,
        )
    )
    print(repr(plot))
    print(plot)
    if save:
        plot.saveimg(save)

if __name__ == "__main__":
    tyro.cli(main)
