"""
Animations: a sequence of plots, and the terminal session that shows them.

There are two things in this module and they are two halves of one loop.

* `tstack` is an animation as a **value**: a sequence of plots with a frame
  rate. It has no terminal and no clock, and like any other plot expression it
  composes -- slice it, map a combinator over its frames, save it as a gif.

* `animate` is an animation as a **session**: a context manager that owns the
  terminal for the duration of a `with` block. Inside it, `anim.update(plot)`
  writes one frame, the previous-frame bookkeeping of `print(plot - prev)`
  disappears, frames are paced, and `anim.print(...)` can log a line without
  corrupting the plot.

Each produces the other, so there is one code path rather than two:

```
# push a live animation into the terminal, keeping the frames
with mp.animate(fps=20, record=True) as anim:
    while running:
        anim.update(compute_frame())
anim.frames.savegif("out.gif")

# pull a finished animation back out of a value
mp.tstack(*frames).play(fps=20)
```

Note that neither is required. Animation is a loop of `print(plot - prev)` and
that stays fully supported; `animate` is there to own the parts of it that are
about terminals rather than about plots.
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np

from PIL import Image

from typing import Any, Callable, Iterator, Literal, Self, Sequence, overload

from matthewplotlib.colors import ColorLike
from matthewplotlib.plots import plot




# # #
# ANIMATIONS AS VALUES


# A gif counts its frame delays in hundredths of a second, so this is the
# shortest one it can express. Asking for less rounds to zero, which is not
# "very fast" but "unspecified": viewers variously play it flat out or
# substitute a tenth of a second. Clamping keeps the fastest gif there is.
_GIF_MIN_DELAY_MS = 10.0


class tstack:
    """
    Temporally stack plots into an animation.

    The third stacking operation, alongside `hstack` (`+`) and `vstack` (`/`):
    where those lay plots out across the screen, this one lays them out in
    time. The result is a value, not an action -- nothing is printed until you
    `play` it or `savegif` it.

    Inputs:

    * *plots : plot | tstack.
        The frames, in order. Any `tstack` among them is spliced in frame by
        frame, so `tstack(a, b)` concatenates two animations and
        `tstack(a, still)` appends a frame to one.
    * fps : optional float.
        The rate this animation is meant to play at, used as the default by
        both `play` and `savegif`. An upper bound rather than a promise: see
        `animate`. Defaults to the frame rate of the first `tstack` among the
        inputs, so concatenating and slicing preserve it, or to 12.0 if there
        is none.
    * durations : optional sequence of float.
        How long each frame was on screen, in milliseconds, if this animation
        was recorded from a live run. Supplied by `animate.frames`, which is
        the only thing that can know it, and read back by
        `savegif(fps="achieved")`. Must have one entry per frame.

    Examples:

    ```
    # build one frame at a time
    a = mp.tstack(*[
        mp.image(field(t)) for t in np.linspace(0, 1, 60)
    ], fps=30)

    # 60 frames, 20 rows, 40 columns
    len(a), a.height, a.width

    # every frame, in a titled border
    a = a.map(lambda p: mp.border(p, title=" diffusion "))

    # the second half, backwards
    a[len(a)//2:][::-1].play()
    ```

    Notes:

    * Frames do not have to be the same size. `savegif` pads them to the
      largest, aligned at the top left, and the terminal renders each frame
      correctly -- but a plot that changes size visibly jitters, so it is
      usually worth pinning whatever makes it move (axis ranges, label widths).
    """
    def __init__(
        self,
        *plots: plot | tstack,
        fps: float | None = None,
        durations: Sequence[float] | None = None,
    ):
        if fps is not None and fps <= 0:
            raise ValueError(f"fps must be positive, not {fps!r}")

        frames: list[plot] = []
        inherited: float | None = None
        for item in plots:
            if isinstance(item, tstack):
                frames.extend(item.plots)
                if inherited is None:
                    inherited = item.fps
            else:
                frames.append(item)
        self.plots: tuple[plot, ...] = tuple(frames)
        if fps is not None:
            self.fps: float = fps
        elif inherited is not None:
            self.fps = inherited
        else:
            self.fps = 12.0

        if durations is not None and len(durations) != len(self.plots):
            raise ValueError(
                f"got {len(durations)} durations for {len(self.plots)} frames"
            )
        self.durations: tuple[float, ...] | None = (
            None if durations is None else tuple(durations)
        )


    @property
    def height(self) -> int:
        """
        Number of character rows in the tallest frame (0 if there are none).
        """
        return max((p.height for p in self.plots), default=0)


    @property
    def width(self) -> int:
        """
        Number of character columns in the widest frame (0 if there are none).
        """
        return max((p.width for p in self.plots), default=0)


    def __len__(self) -> int:
        """
        Number of frames.
        """
        return len(self.plots)


    def __iter__(self) -> Iterator[plot]:
        """
        Iterate over the frames, in order.
        """
        return iter(self.plots)


    @overload
    def __getitem__(self, index: int) -> plot: ...

    @overload
    def __getitem__(self, index: slice) -> tstack: ...

    def __getitem__(self, index: int | slice) -> plot | tstack:
        """
        Index a single frame, or slice out a shorter animation.

        An integer gives the plot at that frame. A slice gives a new `tstack`,
        keeping the frame rate and the matching stretch of the recorded
        durations, so `a[100:200]` and `a[::-1]` both still save honestly.
        """
        if isinstance(index, slice):
            return tstack(
                *self.plots[index],
                fps=self.fps,
                durations=(
                    None if self.durations is None else self.durations[index]
                ),
            )
        return self.plots[index]


    def map(self, f: Callable[[plot], plot]) -> tstack:
        """
        Apply a function to every frame, giving a new animation.

        This is how an animation composes: any of the combinators in
        `matthewplotlib.plots` can be lifted over the time axis by passing it
        through here, which is usually how a static furnishing gets wrapped
        around a moving interior.

        ```
        a.map(lambda p: mp.border(p, title=" gen 0 "))   # a border on each
        a.map(lambda p: p + legend)                      # a panel beside each
        ```

        The frame rate and recorded durations carry over unchanged, since
        neither depends on what the frames contain.

        For combining two animations frame by frame there is no method: zip
        them and rebuild, as in
        `mp.tstack(*[x + y for x, y in zip(a, b)])`.
        """
        return tstack(
            *[f(p) for p in self.plots],
            fps=self.fps,
            durations=self.durations,
        )


    def play(
        self,
        fps: float | None = None,
        loop: bool = False,
        stop_on_interrupt: bool = True,
    ) -> None:
        """
        Print the frames to the terminal, in order, one frame at a time.

        A thin wrapper around `animate`: this pushes the animation's own frames
        through a session, so playing a finished animation and driving a live
        one go down exactly the same path.

        Inputs:

        * fps : optional float.
            Frame rate to play at, defaulting to the animation's own `fps`.
        * loop : bool (default False).
            If true, start again from the first frame instead of returning,
            forever. Interrupt to stop.
        * stop_on_interrupt : bool (default True).
            Whether Ctrl-C ends playback quietly rather than raising
            `KeyboardInterrupt`. Unlike `animate`, this defaults to true: there
            is no caller loop here whose control flow could be taken over, and
            interrupting playback is the only way to end `loop=True`.
        """
        if not self.plots:
            return
        with animate(
            fps=self.fps if fps is None else fps,
            stop_on_interrupt=stop_on_interrupt,
        ) as anim:
            while True:
                for p in self.plots:
                    anim.update(p)
                if not loop:
                    break


    def savegif(
        self,
        filename: str,
        fps: float | Literal["achieved"] | None = None,
        upscale: int = 1,
        downscale: int = 1,
        bgcolor: ColorLike | None = None,
        repeat: bool = True,
    ) -> None:
        """
        Render the frames and save them as an animated gif.

        Inputs:

        * filename : str.
            Where to save the gif. Should usually include a '.gif' extension.
        * fps : optional float or the string "achieved".
            Frame rate to encode. By default the animation's own `fps`, which
            for a recording is the rate that was *asked* for. Pass a number to
            override it, or "achieved" to use the durations actually measured
            while recording -- faithful right down to reproducing a frame that
            stalled, which is honest but rarely what a showcase gif wants.
            Requires `durations`, so only a recorded animation can use it.
        * upscale : int (>=1, default 1).
            Represent each pixel with a square of side-length `upscale` pixels.
        * downscale : int (>=1, default 1).
            Keep every `downscale`th pixel. Does not need to evenly divide the
            image height or width (think slice(0, height or width, downscale)).
            Applied after upscaling.
        * bgcolor : optional ColorLike.
            Default background colour. If none, a transparent background is
            used.
        * repeat : bool (default True).
            If true (default), the gif loops indefinitely. If false, the gif
            only plays once.

        Notes:

        * Frames of different sizes are aligned at the top left corner and
          padded with transparent pixels on the bottom and right. For different
          padding, compose the frames with `blank` blocks first.
        * A gif stores its frame delays in hundredths of a second, so the rate
          in the file is quantised to 10ms steps: 12 fps asks for 83ms, gets
          80ms, and plays at 12.5. Delays are clamped to 10ms, the shortest a
          gif can express, so above 100 fps the file stops getting faster --
          and above roughly 50 fps the number is fiction anyway, since many
          viewers refuse delays under 20ms.
        """
        if not self.plots:
            raise ValueError("cannot save a gif of an animation with no frames")

        # decide the frame delay(s), in milliseconds
        duration: float | list[float]
        if isinstance(fps, str):
            if fps != "achieved":
                raise ValueError(
                    f"fps should be a number or 'achieved', not {fps!r}"
                )
            if self.durations is None:
                raise ValueError(
                    "fps='achieved' needs the frame timings from a live run, "
                    "and this animation has none. Record one with "
                    "mp.animate(record=True), or pass a number of frames per "
                    "second."
                )
            duration = [max(d, _GIF_MIN_DELAY_MS) for d in self.durations]
        else:
            rate = self.fps if fps is None else fps
            if rate <= 0:
                raise ValueError(f"fps must be positive, not {rate!r}")
            duration = max(1000 / rate, _GIF_MIN_DELAY_MS)

        # render plots as image arrays
        frames = [
            p.renderimg(
                upscale=upscale,
                downscale=downscale,
                bgcolor=bgcolor,
            ) for p in self.plots
        ]

        # pad them to u8[height, width, RGBA]
        h = max(frame.shape[0] for frame in frames)
        w = max(frame.shape[1] for frame in frames)
        frames_uniform = [
            np.pad(
                frame,
                pad_width=((0,h-frame.shape[0]),(0,w-frame.shape[1]),(0,0)),
                mode='constant',
                constant_values=0,
            ) for frame in frames
        ]

        # convert to PIL images
        images = [Image.fromarray(frame) for frame in frames_uniform]

        # save
        loop = 0 if repeat else None  # 0 = loop forever, None = play once
        images[0].save(
            filename,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=loop,
        )


    def __repr__(self):
        # the frames themselves are deliberately left out: an animation is
        # routinely hundreds of plots long, and each of those reprs its own
        # children in turn
        return (
            f"tstack(frames={len(self.plots)}, height={self.height}, "
            f"width={self.width}, fps={self.fps})"
        )


# # #
# ANIMATIONS AS TERMINAL SESSIONS


def _terminal_rows() -> int | None:
    """
    How many rows the attached terminal has, or None if stdout is not one.

    `shutil.get_terminal_size` is the wrong tool for this: its job is to paper
    over the absence of a terminal by returning a fallback size, and a fallback
    is exactly what must not be mistaken for a measurement. Asking the file
    descriptor directly tells the two apart.

    Used only to decide whether a warning is warranted. Nothing in this library
    changes *what* it writes based on whether a terminal is attached.
    """
    try:
        return os.get_terminal_size(sys.stdout.fileno()).lines
    except (AttributeError, OSError, ValueError):
        return None


class animate:
    """
    A context manager that owns the terminal while an animation runs.

    Inside the block, each call to `update` writes one frame:

    ```
    with mp.animate(fps=20) as anim:
        while running:
            anim.update(mp.axes(...))
    ```

    which is the `print(plot - prev)` loop with the parts that are about
    terminals rather than about plots taken over: the `prev` sentinel and the
    assignment that maintains it, the frame clock, and separating whatever comes
    next on the screen from the plot on the way out.

    Everything beyond that first job is opt-in, because a context manager takes
    over the caller's control flow and this library would rather be a library
    than a framework. The plain loop remains fully supported and is what the
    quickstart teaches.

    Inputs:

    * fps : optional float.
        If given, cap the frame rate: `update` sleeps off whatever is left of
        the previous frame's budget before writing. An upper bound, not a
        guarantee -- a frame that takes longer than `1/fps` to compute simply
        takes longer, and the clock picks up from there rather than trying to
        catch up. If omitted, `update` returns as soon as it has written.
    * record : bool (default False).
        If true, keep every frame, readable afterwards as `anim.frames`. Opt-in
        because holding an entire animation in memory is a real cost and a
        `while True` loop would never stop paying it.
    * stop_on_interrupt : bool (default False).
        If true, Ctrl-C ends the animation quietly: the `with` block exits and
        the statements after it still run, so a recording survives to be saved.
        Off by default because swallowing `KeyboardInterrupt` is precisely the
        kind of control flow a library should not take without being asked.

    Attributes, readable during the block and after it:

    * frames : tstack.
        The recorded animation, with the frame timings attached. Raises unless
        `record=True` was passed.
    * achieved_fps : float | None.
        The frame rate actually managed so far, or None before the second
        frame. Worth printing after a run that felt sluggish: asking for 20 and
        getting 6 is not otherwise visible.

    Notes:

    * A plot has to fit the terminal with a row to spare for the newline that
      `print` appends, so at most `rows - 1` of it will render. If the first
      frame does not fit, and stdout is a terminal that can be measured, this
      warns once.
    * The cursor is left visible. It sits in the plot and blinks there, which is
      a fair trade for never leaving a terminal with an invisible cursor.
    """
    def __init__(
        self,
        fps: float | None = None,
        record: bool = False,
        stop_on_interrupt: bool = False,
    ):
        if fps is not None and fps <= 0:
            raise ValueError(f"fps must be positive, not {fps!r}")
        self.fps = fps
        self.stop_on_interrupt = stop_on_interrupt

        self._recording: list[plot] | None = [] if record else None
        self._times: list[float] = []       # write times, seconds, if recording
        self._prev: plot | None = None
        self._deadline: float | None = None
        self._count = 0
        self._first: float | None = None
        self._last: float | None = None
        self._warned = False


    def __enter__(self: Self) -> Self:
        return self


    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if exc_type is None:
            return False
        # A completed frame ends with the newline `print` appends, so the cursor
        # is already on a fresh line below the plot and there is nothing to
        # tidy. But an exception can arrive mid-frame, and the ^C a terminal
        # echoes lands wherever the cursor stopped, so separate whatever comes
        # next -- a traceback, or the shell prompt -- from the plot.
        if self._prev is not None:
            print()
        return self.stop_on_interrupt and issubclass(exc_type, KeyboardInterrupt)


    def update(self, plot: plot) -> str:
        """
        Write one frame, and return the string that was written.

        The first call renders the whole plot; each later call repaints only the
        cells that differ from the frame before it, which is the same
        `plot.updatestr(prev)` the `-` operator gives, with `prev` looked after
        here.

        If the session has an `fps`, this is also where it waits: the sleep
        happens *before* the write, so the time the caller spent computing this
        frame counts towards the frame's budget instead of being added to it.
        Four of the six animated examples got that wrong with a flat
        `sleep(1/fps)` before this existed.

        The return value is the frame's exact bytes, which is what makes the
        cost of differential rendering measurable from user code -- see
        `examples/life.py`, which plots it.
        """
        # pace: sleep off the remainder of the previous frame's budget
        if self._deadline is not None:
            delay = self._deadline - time.perf_counter()
            if delay > 0:
                time.sleep(delay)

        self._check_fits(plot)

        # write, in exactly one print
        update = plot.updatestr(self._prev)
        print(update)
        self._prev = plot

        # book-keeping: timings, recording, and the next deadline
        now = time.perf_counter()
        self._count += 1
        if self._first is None:
            self._first = now
        self._last = now
        if self._recording is not None:
            self._recording.append(plot)
            self._times.append(now)
        if self.fps is not None:
            period = 1 / self.fps
            self._deadline = (
                now + period if self._deadline is None
                else max(now, self._deadline + period)
            )

        return update


    def print(self, *args: object, **kwargs: Any) -> None:
        """
        Print a line above the animation, without corrupting it.

        A bare `print` from inside an animated loop lands in the middle of the
        plot, which leaves debugging an animated program a choice between not
        printing and not animating. This routes the line instead: the plot is
        erased, the message takes the row the plot's first row was on, and the
        plot is redrawn one row lower.

        ```
        with mp.animate(fps=20) as anim:
            for step in range(1000):
                anim.update(vis(params))
                if step % 100 == 0:
                    anim.print(f"step {step}: loss {loss:.4f}")
        ```

        So messages pile up above the plot and scroll off the top of the screen
        in the usual way, with the plot pinned below them. Takes the same
        arguments as the builtin `print`, except that `end` must stay a newline:
        the redraw has to start at the beginning of a line.

        Costs a full repaint, rather than the differential redraw a frame gets.
        Printing happens at human speed, so this is not worth optimising.
        """
        # `print` here is the builtin: a class attribute is not in scope inside
        # its own method body.
        if self._prev is None:
            print(*args, **kwargs)      # no plot on screen to protect yet
            return
        print(-self._prev)              # erase, and step above the plot
        print(*args, **kwargs)          # the message takes that row
        print(self._prev)               # redraw, one row lower


    @property
    def achieved_fps(self) -> float | None:
        """
        Frames per second actually achieved so far, or None before frame two.
        """
        if self._count < 2 or self._first is None or self._last is None:
            return None
        span = self._last - self._first
        if span <= 0:
            return None
        return (self._count - 1) / span


    @property
    def frames(self) -> tstack:
        """
        The recorded frames, as a `tstack`, with their measured durations.

        The animation's `fps` is the rate that was requested, so
        `anim.frames.savegif(...)` writes a gif at the intended speed by
        default, and `savegif(fps="achieved")` writes one at the speed the run
        actually managed.

        Raises ValueError unless the session was created with `record=True`.
        """
        if self._recording is None:
            raise ValueError(
                "this animation kept no frames: pass record=True to "
                "mp.animate() to have it hold on to them"
            )
        return tstack(
            *self._recording,
            fps=12.0 if self.fps is None else self.fps,
            durations=self._durations(),
        )


    def _durations(self) -> list[float] | None:
        """
        How long each recorded frame was on screen, in milliseconds.

        The gap between consecutive writes. Nothing replaced the final frame, so
        how long it was up is unknowable and it inherits the gap before it.
        Returns None if fewer than two frames were recorded, when there is no
        gap to measure.
        """
        if len(self._times) < 2:
            return None
        gaps = [1000 * (b - a) for a, b in zip(self._times, self._times[1:])]
        return gaps + [gaps[-1]]


    def _check_fits(self, plot: plot) -> None:
        """
        Warn, once per session, if the plot cannot fit the terminal.
        """
        if self._warned:
            return
        rows = _terminal_rows()
        if rows is None or plot.height <= rows - 1:
            return
        self._warned = True
        warnings.warn(
            f"animating a plot {plot.height} rows tall in a terminal only "
            f"{rows} rows high: a frame needs one spare row for the newline "
            f"`print` appends, so at most {rows - 1} rows will render and the "
            f"plot will tear. Make the plot shorter, or the window taller.",
            stacklevel=3,
        )


    def __repr__(self):
        return (
            f"animate(fps={self.fps}, record={self._recording is not None}, "
            f"stop_on_interrupt={self.stop_on_interrupt})"
        )
