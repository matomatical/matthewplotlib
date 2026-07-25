import time
import tyro
import numpy as np
import matthewplotlib as mp


# States a cell can be in, and the colour each is drawn with. Conway's rules
# only distinguish alive from dead; the rest is memory, so that the eye can see
# *what just happened* as well as what is there.
DEAD, EMBER, ASH, STABLE, NEWBORN = range(5)

PALETTE = np.array([
    [ 12,  12,  20],    # DEAD    -- the background
    [ 26,  28,  44],    # EMBER   -- died two generations ago
    [ 54,  46,  84],    # ASH     -- died last generation
    [ 56, 183, 100],    # STABLE  -- alive, and was alive before
    [244, 244, 244],    # NEWBORN -- alive as of this generation
], dtype=np.uint8)


def life_palette(
    x,              # int[...]
) -> np.ndarray:    # -> uint8[..., 3]
    """
    Discrete colormap over the cell states above. Any `int[...] -> uint8[...,3]`
    function is a colormap as far as `mp.image` is concerned.
    """
    return PALETTE[np.asarray(x, np.uint) % len(PALETTE)]


def step(alive: np.ndarray) -> np.ndarray:
    """
    One generation of Conway's Game of Life, on a torus.

    Counting the eight neighbours by summing shifted copies of the board is the
    same trick as convolving with [[1,1,1],[1,0,1],[1,1,1]], without needing
    scipy. A cell with exactly three neighbours lives (whether it was alive or
    not), and an already-live cell with two neighbours survives.
    """
    n = sum(
        np.roll(alive, (dy, dx), (0, 1))
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dy, dx) != (0, 0)
    )
    return (alive & (n == 2)) | (n == 3)


def main(
    num_frames: int = 0,
    fps: int = 12,
    width: int = 76,
    height: int = 20,
    density: float = 0.28,
    seed: int = 0,
    save: str | None = None,
):
    """Conway's Game of Life, as a showcase for differential rendering.

    Half-block characters give one cell per half-row, so the board is twice as
    tall as the plot. Braille would pack in four times as many cells again, but
    a braille cell carries no colour of its own, and here the colour is the
    point.

    The footer keeps score against a full redraw. What it shows is that a frame
    costs in proportion to the number of cells that *changed*, not to the size
    of the board: about 1x while the soup is boiling and everything is moving,
    climbing as it settles into still lifes and a few blinkers -- around 5x for
    this board after a thousand generations, and far more on a smaller one,
    where less is left oscillating.
    """
    rng = np.random.default_rng(seed)
    alive = rng.random((2 * height, width)) < density
    was_alive = np.zeros_like(alive)
    died_recently = np.zeros_like(alive)

    prev = None
    frames = [] if save else None
    saved = 0.0

    frame = 0
    while num_frames == 0 or frame < num_frames:
        deadline = time.perf_counter() + 1 / fps

        # classify every cell by what just happened to it
        state = np.where(died_recently, EMBER, DEAD)
        state = np.where(was_alive & ~alive, ASH, state)
        state = np.where(alive & was_alive, STABLE, state)
        state = np.where(alive & ~was_alive, NEWBORN, state)

        score = f"{saved:.0f}x smaller than a redraw" if saved else "..."
        footer = f" gen {frame}  {alive.sum():>5d} alive  {score}"
        plot = (
            mp.border(
                mp.image(state, colormap=life_palette),
                title=" Conway's Game of Life ",
            )
            / mp.text(footer.ljust(width + 2)[:width + 2])
        )

        update = plot - prev
        print(update)
        # what the same frame would have cost as a clear and a full redraw
        if prev is not None:
            redraw = len(prev.clearstr()) + 1 + len(plot.renderstr())
            saved = redraw / max(len(update), 1)
        prev = plot
        if frames is not None: frames.append(plot)

        was_alive, died_recently = alive, was_alive & ~alive
        alive = step(alive)
        frame += 1
        time.sleep(max(0, deadline - time.perf_counter()))

    if save and frames:
        mp.save_animation(frames, save, bgcolor="black", fps=fps)


if __name__ == "__main__":
    try:
        tyro.cli(main)
    except KeyboardInterrupt:
        print()
