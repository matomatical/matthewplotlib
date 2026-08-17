"""
Conway's Game of Life with extra colours for newly alive/dead cells. The panels
underneath track the cell counts and, on the right, number of terminal bytes
written using different rendering methods.

By Claude Opus 5.
"""

import collections
import math
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

SENT = (255, 163, 0)        # bytes actually written, in the byte panel
REDRAW = (110, 110, 130)    # what a full redraw would have cost


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


# Two digits minimum, so the y tick gutter is the same width whether the counts
# are in single or double digits. Otherwise the panels would change width as
# the population crosses ten, and the whole dashboard would jiggle.
YFMT = "{y:2.0f}"


def axes_overhead(height: int, ylabel: str, ymax: float) -> int:
    """
    How many columns `mp.axes` adds around a scatter, given the widest tick
    label it will have to print. Measured once rather than hardcoded, since it
    depends on the label text.
    """
    probe = mp.axes(
        mp.scatter(([0], [ymax]), xrange=(0, 1), yrange=(0, ymax),
                   width=10, height=height),
        xlabel="x", ylabel=ylabel, xfmt="{x:.0f}", yfmt=YFMT,
    )
    return probe.width - 10


def main(
    num_frames: int = 0,
    fps: int = 12,
    width: int = 72,
    height: int = 16,
    panel_height: int = 4,
    density: float = 0.28,
    seed: int = 0,
    save: str | None = None,
):
    """Conway's Game of Life, as a showcase for differential rendering.

    Underneath, two time series say what the board is doing and what it costs
    to draw. The right-hand panel is the whole argument for differential
    rendering in one picture: what was actually written each frame, against
    what a clear-and-redraw would have cost. The two curves start together
    while the soup is boiling and every cell is changing, and separate as the
    board settles into still lifes and blinkers, because a frame costs in
    proportion to the cells that *changed*, not to the size of the board.
    """
    # np.random.seed rather than default_rng: NumPy guarantees the legacy
    # stream across releases and explicitly does not guarantee Generator's, and
    # this board is pinned by the example snapshot tests.
    np.random.seed(seed)
    alive = np.random.rand(2 * height, width) < density
    was_alive = np.zeros_like(alive)
    died_recently = np.zeros_like(alive)

    # Fixed axis ranges, so the tick labels never change width and the whole
    # composition keeps a constant size. (A resize would render correctly --
    # that is what `plot - prev` falls back to -- but it would jitter.)
    board_bytes = len(mp.image(alive.astype(int), colormap=life_palette).renderstr())
    # at generation zero every live cell counts as newborn, which is as large
    # as any single category ever gets
    cells_max = 50 * math.ceil(2 * height * width * density / 50)
    kb_max = max(10, math.ceil(board_bytes * 1.3 / 1000))

    # split the width under the board between the two panels
    left_total = (width + 2) // 2
    right_total = (width + 2) - left_total
    left_width = left_total - axes_overhead(panel_height, "cells", cells_max)
    right_width = right_total - axes_overhead(panel_height, "kB", kb_max)

    # one datum per braille column, so the window is twice the panel's width
    span = 2 * min(left_width, right_width)
    # counts are known for the frame being drawn; its cost is only known once it
    # has been written, so the cost series always trails the counts by one
    counts: collections.deque = collections.deque(maxlen=span)
    costs: collections.deque = collections.deque(maxlen=span)

    prev = None
    frames = [] if save else None

    frame = 0
    while num_frames == 0 or frame < num_frames:
        deadline = time.perf_counter() + 1 / fps

        # classify every cell by what just happened to it
        state = np.where(died_recently, EMBER, DEAD)
        state = np.where(was_alive & ~alive, ASH, state)
        state = np.where(alive & was_alive, STABLE, state)
        state = np.where(alive & ~was_alive, NEWBORN, state)

        counts.append([int((state == s).sum()) for s in (NEWBORN, STABLE, ASH)])
        tc = np.arange(len(counts))
        tb = np.arange(len(costs))
        series = [np.array([c[i] for c in counts]) for i in range(3)]
        sent = np.array([c[0] for c in costs])
        redraw = np.array([c[1] for c in costs])

        board = mp.border(
            mp.image(state, colormap=life_palette),
            title=f" Conway's Game of Life · gen {frame} ",
        )
        # no legends in the library yet, so the x label names the series in the
        # order they are drawn, and the colours match the board above
        cells_panel = mp.axes(
            mp.scatter(
                *((tc, s, tuple(PALETTE[c]))
                  for s, c in zip(series, (NEWBORN, STABLE, ASH))),
                xrange=(0, span), yrange=(0, cells_max),
                width=left_width, height=panel_height,
            ),
            xlabel="newborn · stable · ash", ylabel="cells",
            xfmt="{x:.0f}", yfmt=YFMT,
        )
        bytes_panel = mp.axes(
            mp.scatter(
                (tb, redraw, REDRAW),
                (tb, sent, SENT),
                xrange=(0, span), yrange=(0, kb_max),
                width=right_width, height=panel_height,
            ),
            xlabel="redraw (grey) · diff (orng)", ylabel="kB",
            xfmt="{x:.0f}", yfmt=YFMT,
        )
        plot = board / (cells_panel + bytes_panel)

        update = plot - prev
        print(update)

        # score this frame, now that it has been written, for the next to draw
        if prev is not None:
            costs.append((
                len(update) / 1000,
                (len(prev.clearstr()) + 1 + len(plot.renderstr())) / 1000,
            ))
        prev = plot
        if frames is not None: frames.append(plot)

        was_alive, died_recently = alive, was_alive & ~alive
        alive = step(alive)
        frame += 1
        time.sleep(max(0, deadline - time.perf_counter()))

    if save and frames:
        mp.tstack(*frames, fps=fps).savegif(save, bgcolor="black")


if __name__ == "__main__":
    try:
        tyro.cli(main)
    except KeyboardInterrupt:
        print()
