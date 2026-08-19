"""
Doom Fire Effect.

A vectorized implementation of the classic 1997 PSX Doom fire effect.
Heat propagates upwards and dissipates, pulling from a glowing hot
bottom row. Visualized using `mp.image` and a custom 37-color palette.

By Gemini 3.1 Pro.
"""

import tyro
import numpy as np
import matthewplotlib as mp


# Fabien Sanglard's classic 37-color Doom Fire palette
DOOM_PALETTE = np.array([
    [0, 0, 0], [31, 7, 7], [47, 15, 7], [71, 15, 7], [87, 23, 7],
    [103, 31, 7], [119, 31, 7], [143, 39, 7], [159, 47, 7], [175, 63, 7],
    [191, 71, 7], [199, 71, 7], [223, 79, 7], [223, 87, 7], [223, 87, 7],
    [215, 95, 7], [215, 95, 7], [215, 103, 15], [207, 111, 15], [207, 119, 15],
    [207, 127, 15], [207, 135, 23], [199, 135, 23], [199, 143, 23], [199, 151, 31],
    [191, 159, 31], [191, 159, 31], [191, 167, 39], [191, 167, 39], [191, 175, 47],
    [183, 175, 47], [183, 183, 47], [183, 183, 55], [207, 207, 111], [223, 223, 159],
    [239, 239, 199], [255, 255, 255]
], dtype=np.uint8)


def do_fire_step(grid):
    height, width = grid.shape
    # Iterate from top to bottom. y=0 is the top of the fire.
    # We pull heat from the row below it (y+1).
    for y in range(height - 1):
        # A random shift: mostly 0, occasionally -1 or 1 (wind/turbulence)
        shift = np.random.randint(-1, 2, size=width)
        # Ensure we don't read out of bounds
        src_x = np.clip(np.arange(width) + shift, 0, width - 1)
        
        # Heat dissipates randomly by 0 or 1 as it rises
        decay = np.random.randint(0, 2, size=width)
        
        # Pull heat from the row below, subtract decay, and clamp to 0
        grid[y, :] = np.clip(grid[y + 1, src_x] - decay, 0, 36)


def main(
    num_frames: int = 0,
    fps: int = 30,
    width: int = 80,
    height: int = 50,
    save: str | None = None,
    downscale: int = 1,
):
    """Animate the classic Doom Fire effect."""
    np.random.seed(42)
    # mp.image uses half-block characters (▀), so each character contains 2 vertical pixels.
    # If we want `height` lines in the terminal, we need an internal grid of `height * 2`.
    internal_height = height * 2
    grid = np.zeros((internal_height, width), dtype=np.int32)
    
    # Ignite the bottom row (set to maximum heat: 36)
    grid[-1, :] = 36
    
    animation = mp.animate(
        fps=fps,
        record=save is not None,
        stop_on_interrupt=True,
    )
    
    with animation as anim:
        frame = 0
        while num_frames == 0 or frame < num_frames:
            do_fire_step(grid)
            
            # Fast vectorized mapping from heat integers [0..36] to RGB
            rgb = DOOM_PALETTE[grid]
            
            # mp.image naturally takes an RGB array of shape (H, W, 3)
            plot = mp.image(rgb)
            
            anim.update(plot)
            frame += 1

    if save:
        anim.frames.savegif(save, bgcolor="black", downscale=downscale)


if __name__ == "__main__":
    tyro.cli(main)
