import time
import numpy as np
import matthewplotlib as mp


WIDTH = 48
HEIGHT = 16
MAX_ITER = 2000
FRAMES = 200
FPS = 20.
FILENAME = "images/mandelbrot.gif"

CX, CY = -0.743643887037151, 0.131825904205330
ZOOM_START = 3
ZOOM_END = 0.000000000001


def main():
    frames = []
    zoom_factors = np.geomspace(ZOOM_START, ZOOM_END, FRAMES)
    
    print(f"Generating {FRAMES} frames for Mandelbrot zoom...")
    
    plot = None
    for i, zoom in enumerate(zoom_factors):
        start = time.perf_counter()

        # zoom
        xrange = (CX - zoom, CX + zoom)
        aspect_ratio = WIDTH / (2 * HEIGHT)
        yzoom = zoom / aspect_ratio
        yrange = (CY - yzoom, CY + yzoom)
        
        # compute
        frame = mp.function2(
            lambda xy: MAX_ITER-escape_time(xy[:,0] + 1j * xy[:,1]),
            xrange=xrange,
            yrange=yrange,
            width=WIDTH,
            height=HEIGHT,
            zrange=(0, MAX_ITER),
            colormap=mp.magma,
        )
        frames.append(frame)

        # plot
        if plot is not None:
            print(-plot, end="")
        plot = mp.vstack(mp.progress((i+1)/FRAMES, width=WIDTH), frame)
        print(plot, sep="")

        # wait
        time.sleep(max(0, start + 1/FPS - time.perf_counter()))
        
    print(f"saving to '{FILENAME}'...")
    mp.save_animation(frames, FILENAME, fps=FPS)
    print("Done!")


def escape_time(c: np.ndarray) -> np.ndarray:
    z = np.zeros_like(c)
    n = np.zeros(c.shape, dtype=int)
    mask = np.ones(c.shape, dtype=bool)
    for i in range(MAX_ITER):
        z[mask] = z[mask]**2 + c[mask]
        mask[np.abs(z) > 2] = False
        n[mask] = i
        if not mask.any():
            break
    return n


if __name__ == "__main__":
    main()
