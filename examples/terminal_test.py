"""
Does your terminal render matthewplotlib correctly?

Exercises every escape sequence the library can emit, in the situations where
they are hardest, and says what each stage should look like so that you can
judge it. See `docs/compatibility.md` for what is being tested and why.

The stages, in the order they run:

1. Colour, which is the only part of the vocabulary that genuinely varies
   between terminals. A smooth ramp shows 24-bit colour; visible bands mean
   your terminal is approximating.
2. Redrawing in place, which is cursor movement and nothing else.
3. Resizing, which adds the erases: whole rows when a plot shrinks, and written
   spaces for the columns it gives up when it narrows.
4. The right margin, where a plot exactly as wide as the terminal leaves the
   cursor on the margin with a wrap pending. This is the one case where
   terminals genuinely disagree, so it is the one worth watching closely.

If a stage misbehaves, please open an issue saying which one, along with your
terminal, its version, and `echo $TERM`.

By Claude Opus 5.
"""

import os
import sys

import tyro
import numpy as np

import matthewplotlib as mp


def terminal_width() -> int | None:
    """How many columns the attached terminal has, or None if there isn't one.

    `shutil.get_terminal_size` is the wrong tool here, for the same reason
    `matthewplotlib.animations` avoids it: its job is to paper over the absence
    of a terminal by returning a fallback, and a fallback quietly mistaken for a
    measurement is exactly what would make stage 4 test nothing while looking
    like it had. Asking the file descriptor directly tells the two apart.
    """
    try:
        return os.get_terminal_size(sys.stdout.fileno()).columns
    except (AttributeError, OSError, ValueError):
        return None


def banner(n: int, title: str, expect: str) -> None:
    """Announce a stage, and say what passing looks like."""
    print()
    print(str(mp.text(f"{n}. {title}", fgcolor="white")))
    print(str(mp.text(f"   expect: {expect}", fgcolor="cyan")))


def ramp(width: int, height: int, phase: float) -> mp.plot:
    """A smooth two-dimensional colour ramp, which banding shows up in."""
    ys, xs = np.mgrid[0:height, 0:width]
    return mp.image(
        (xs / max(width - 1, 1) + ys / max(height - 1, 1) + phase) % 1.0,
        colormap=mp.viridis,
    )


def stage_colour(width: int) -> None:
    banner(
        1, "Colour",
        "a smooth ramp, no stripes, then coloured words",
    )
    print(str(ramp(width, 6, 0.0)))
    print(str(
        mp.text("foreground", fgcolor="red")
        + mp.text(" default ")
        + mp.text("background", bgcolor="blue")
    ))


def stage_redraw(width: int, frames: int, fps: float) -> None:
    banner(
        2, "Redrawing in place",
        "one plot updating in place, not marching down",
    )
    with mp.animate(fps=fps) as anim:
        for frame in range(frames):
            anim.update(mp.border(ramp(width - 2, 6, frame / frames)))


def stage_resize(width: int, frames: int, fps: float) -> None:
    banner(
        3, "Growing and shrinking",
        "no leftovers to the right as it narrows, no smear",
    )
    with mp.animate(fps=fps) as anim:
        for frame in range(frames):
            # a full cycle of the sine, so it both grows and shrinks
            t = frame / frames * 2 * np.pi
            w = int(width * (0.55 + 0.45 * np.sin(t)))
            h = 4 + int(3 * (1 + np.cos(t)))
            anim.update(mp.border(ramp(max(w - 2, 1), h, 0.0)))


def stage_margin(width: int, frames: int, fps: float) -> None:
    banner(
        4, "The right margin",
        f"{width} columns wide, no drift or smear at the edge",
    )
    with mp.animate(fps=fps) as anim:
        for frame in range(frames):
            anim.update(ramp(width, 6, frame / frames))


def main(
    width: int | None = None,
    fps: float = 20,
    frames: int = 24,
):
    """
    Check that this terminal renders matthewplotlib correctly.

    Inputs:

    * width: optional int.
      Columns to draw into. Defaults to the width of the attached terminal,
      which is what stage 4 needs in order to test anything: it puts a plot
      against the right margin, and a plot narrower than the screen never
      reaches one. Pass a value to override.
    * fps: float.
      Frame rate for the animated stages.
    * frames: int.
      Frames per animated stage.
    """
    print(str(mp.text(
        "matthewplotlib terminal test", fgcolor="white",
    )))
    if width is None:
        width = terminal_width()
        if width is None:
            width = 64
            print(str(mp.text(
                f"stdout is not a terminal, so its width cannot be measured; "
                f"drawing {width} columns wide. Stage 4 will not test the "
                f"right margin.",
                fgcolor="yellow",
            )))
    stage_colour(width)
    stage_redraw(width, frames, fps)
    stage_resize(width, frames, fps)
    stage_margin(width, frames, fps)
    print()
    print(str(mp.text(
        "Done. See docs/compatibility.md.",
        fgcolor="cyan",
    )))


if __name__ == "__main__":
    try:
        tyro.cli(main)
    except KeyboardInterrupt:
        print()
