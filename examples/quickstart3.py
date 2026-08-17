"""
The same animation as quickstart2.py, with the library running the loop.

quickstart2.py is the mechanism: `print(plot - prev)`, with the previous frame,
the clock and the frame count all kept by hand. This is the same animation handed
to `mp.animate`, which keeps them instead.
"""

import tyro
import numpy as np
import matthewplotlib as mp


def main(
    num_frames: int = 0,
    fps: int = 20,
    period: float = 3.0,
    save: str | None = None,
):
    """Animated cosine wave with shifting phase and amplitude."""
    x = np.linspace(-2*np.pi, +2*np.pi, 150)

    animation = mp.animate(
        fps=fps,
        record=save is not None,
        stop_on_interrupt=True,
    )
    with animation as anim:
        frame = 0
        while num_frames == 0 or frame < num_frames:
            k = (frame / fps % period) / period * 6
            A = 0.85 + 0.15 * np.cos(2*np.pi*k/6)
            y = A * np.cos(x - 2*np.pi*k/6)
            c = mp.rainbow(1-k/6)
            anim.update(mp.axes(
                mp.scatter(
                    (x, y, c),
                    width=75,
                    height=10,
                    yrange=(-1,1),
                ),
                title=f" y = {A:.2f} cos(x + 2π*{k:.2f}/6) ",
                xlabel="x",
                ylabel="y",
            ))
            frame += 1

    if save:
        anim.frames.savegif(save, bgcolor="black")


if __name__ == "__main__":
    tyro.cli(main)
