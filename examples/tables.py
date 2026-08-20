"""
A report on a small hyperparameter sweep, laid out in tables.

Demonstrates: table, line, axes, text, hstack, vstack, viridis.

By Claude Opus 5.
"""

import tyro
import numpy as np

import matthewplotlib as mp


LEARNING_RATES = [1e-4, 3e-4, 1e-3, 3e-3]
WEIGHT_DECAYS = [0.0, 0.01, 0.1]

RUN_NAMES = ["baseline", "wider", "deeper", "longboi", "tiny"]


def sweep_losses():
    """A bowl in log learning rate, tilted by weight decay."""
    lr = np.log10(np.array(LEARNING_RATES))[None, :]
    wd = np.array(WEIGHT_DECAYS)[:, None]
    return 0.34 + 0.28 * (lr + 3.1) ** 2 + 1.4 * wd * (lr + 4.2)


def loss_curve(final, learning_rate):
    """A run's loss on its way down to where it finished, faster for a
    larger learning rate."""
    t = np.linspace(0.0, 1.0, 60)
    rate = 2.0 + 1.5 * (np.log10(learning_rate) + 4)
    return final + (1.6 - final) * np.exp(-rate * t) + 0.012 * np.sin(37 * t)


def readable_on(background):
    """Black or white per cell, whichever the background can be read against.

    Shading a table by its values only works if the values stay legible, and
    the dark end of a colormap and the light end do not want the same ink.
    """
    luminance = background @ np.array([0.299, 0.587, 0.114])
    ink = np.where(luminance > 140, 0, 255).astype(np.uint8)
    return np.repeat(ink[..., np.newaxis], 3, axis=-1)


def main(save: str | None = None):
    """A hyperparameter sweep report laid out in tables."""
    losses = sweep_losses()

    # --- the five best cells of the sweep, as the runs that were kept ---

    order = np.argsort(losses, axis=None)[:len(RUN_NAMES)]
    runs = []
    for name, cell in zip(RUN_NAMES, order):
        row, column = np.unravel_index(cell, losses.shape)
        loss = losses[row, column]
        runs.append({
            "run": name,
            "lr": LEARNING_RATES[column],
            "wd": WEIGHT_DECAYS[row],
            "steps": 5000 * (1 + column),
            "loss": loss,
            "acc": 0.99 - 0.28 * loss,
        })

    # --- the runs, ruled the way a paper rules a table ---

    runs_table = mp.table(
        runs,
        formats={
            "lr": "{:.0e}",
            "wd": ".3f",
            "steps": ",d",
            "loss": ".3f",
            "acc": "{:.1%}",
        },
    )

    # --- the sweep itself, shaded by the losses it is showing ---

    shades = mp.viridis(1.0 - (losses - losses.min()) / np.ptp(losses))
    sweep_table = mp.table(
        losses,
        headers=[f"{lr:.0e}" for lr in LEARNING_RATES],
        index=[f"{wd:.2f}" for wd in WEIGHT_DECAYS],
        index_name="wd",
        formats=".3f",
        colors=readable_on(shades),
        bgcolors=shades,
        leftrule="single",
        indexrule="single",
        rightrule="single",
    )

    # --- what the runs were trained on, in a box of its own ---

    data_table = mp.table(
        {
            "split": ["train", "valid", "test"],
            "examples": [45000, 5000, 10000],
            "batches": [352, 40, 79],
        },
        formats={"examples": ",d"},
        midrule="single",
        leftrule="single",
        colrule="single",
        rightrule="single",
    )

    # --- the undecayed row of the sweep, on its way down ---

    # the same shades the table gave those cells, so that the two read
    # together without a word passing between them
    curve_colors = shades[0]
    steps = np.linspace(0.0, 1.0, 60)
    curves = mp.dstack2(*[
        mp.line(
            (steps, loss_curve(losses[0, column], learning_rate), color),
            width=34,
            height=9,
            xrange=(0.0, 1.0),
            yrange=(0.3, 1.2),
        )
        for column, (learning_rate, color) in enumerate(
            zip(LEARNING_RATES, curve_colors)
        )
    ])
    curves_plot = mp.axes(curves, xlabel="training", ylabel="loss")

    # --- a legend is a table whose rows are colored one at a time ---

    legend = mp.table(
        {
            "lr": [f"{lr:.0e}" for lr in LEARNING_RATES],
            "final": losses[0],
        },
        formats={"final": ".3f"},
        colors=np.stack([curve_colors, curve_colors], axis=1),
        toprule="skip",
        bottomrule="skip",
    )

    # --- assemble the report ---

    plot = mp.vstack(
        mp.text("sweep report", fgcolor="white"),
        mp.blank(),
        mp.text("runs kept, best first"),
        runs_table,
        mp.blank(),
        mp.hstack(
            mp.vstack(
                mp.text("validation loss"),
                sweep_table,
                mp.blank(),
                mp.text("data"),
                data_table,
            ),
            mp.blank(width=3),
            mp.vstack(
                mp.text("loss curves at wd 0.00"),
                curves_plot,
                mp.blank(),
                legend,
            ),
        ),
    )

    print(plot)
    if save:
        plot.saveimg(save, bgcolor="black")


if __name__ == "__main__":
    tyro.cli(main)
