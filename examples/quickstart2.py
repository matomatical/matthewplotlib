import time
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

    prev = None
    frames = [] if save else None

    frame = 0
    while num_frames == 0 or frame < num_frames:
        k = (frame / fps % period) / period * 6
        A = 0.85 + 0.15 * np.cos(k)
        y = A * np.cos(x - 2*np.pi*k/6)
        c = mp.rainbow(1-k/6)
        plot = mp.axes(
            mp.scatter(
                (x, y, c),
                width=75,
                height=10,
                yrange=(-1,1),
            ),
            title=f" y = {A:.2f} cos(x + 2π*{k:.2f}/6) ",
            xlabel="x",
            ylabel="y",
        )
        print(plot - prev)
        prev = plot
        if frames is not None: frames.append(plot)
        frame += 1
        time.sleep(1/fps)

    if save and frames:
        mp.tstack(*frames, fps=fps).savegif(save, bgcolor="black")


if __name__ == "__main__":
    try:
        tyro.cli(main)
    except KeyboardInterrupt:
        print()
