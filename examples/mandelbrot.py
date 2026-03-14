import time
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
    frames = []
    zoom_factors = np.geomspace(ZOOM_START, ZOOM_END, num_frames)

    print(f"Generating {num_frames} frames for Mandelbrot zoom...")
    
    plot = None
    for i, zoom in enumerate(zoom_factors):
        start = time.perf_counter()

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
        if plot is not None:
            print(-plot, end="")
        plot = mp.vstack(mp.progress((i+1)/num_frames, width=width), frame)
        print(plot, sep="")

        # wait
        time.sleep(max(0, start + 1/fps - time.perf_counter()))

    if save:
        mp.save_animation(frames, save, fps=fps, downscale=8)


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
