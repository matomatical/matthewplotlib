import tyro
import numpy as np
import matthewplotlib as mp


CX, CY = -0.743643887037151, 0.131825904205330
ZOOM_START = 3
ZOOM_END = 0.000000000001


def main(
    num_frames: int = 200,
    fps: float = 20.0,
    width: int = 80,
    height: int = 40,
    max_iter: int = 2000,
    save: str | None = None,
):
    """Mandelbrot set zoom animation."""
    # The gif is of the fractal alone, so the frames are collected here rather
    # than with mp.animate(record=True) -- what is shown includes a progress bar
    # that would be meaningless in a saved animation.
    frames = []
    zoom_factors = np.geomspace(ZOOM_START, ZOOM_END, num_frames)

    print(f"Generating {num_frames} frames for Mandelbrot zoom...")

    with mp.animate(fps=fps, stop_on_interrupt=True) as anim:
        for i, zoom in enumerate(zoom_factors):
            # zoom
            xrange = (CX - zoom, CX + zoom)
            aspect_ratio = width / (2 * height)
            yzoom = zoom / aspect_ratio
            yrange = (CY - yzoom, CY + yzoom)

            # compute
            frame = mp.function2(
                lambda xy: max_iter-escape_time(xy[:,0] + 1j * xy[:,1], max_iter),
                xrange=xrange,
                yrange=yrange,
                width=width,
                height=height,
                zrange=(0, max_iter),
                colormap=mp.magma,
            )
            frames.append(frame)

            # plot
            anim.update(
                mp.vstack(mp.progress((i+1)/num_frames, width=width), frame)
            )

    if save and frames:
        mp.tstack(*frames, fps=fps).savegif(save, downscale=8)


def escape_time(c: np.ndarray, max_iter: int) -> np.ndarray:
    z = np.zeros_like(c)
    n = np.zeros(c.shape, dtype=int)
    mask = np.ones(c.shape, dtype=bool)
    for i in range(max_iter):
        z[mask] = z[mask]**2 + c[mask]
        mask[np.abs(z) > 2] = False
        n[mask] = i
        if not mask.any():
            break
    return n


if __name__ == "__main__":
    tyro.cli(main)
