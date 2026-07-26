"""
The example snapshot machinery: run an example, and record what it put on a
terminal.

An example is run the way a user runs it -- as a subprocess, through `tyro`,
from its `__main__` block -- but with stdout redirected to a recorder that
keeps each `print` separately, `time.sleep` stubbed out, and (for the one
example that reads live data) `psutil` replaced by a fixed source. The recorded
prints are then replayed one at a time into a tmux pane through `tests/tmux.py`
and the screen is captured after each, so an animation is snapshotted frame by
frame rather than only at the end. That matters: at `--num-frames 5` the last
frame of `mandelbrot.py` is a solid black rectangle, and a last-screen-only
snapshot of it would assert nothing.

A snapshot pins four things per frame, each of which catches regressions the
others do not (see `notes/example-snapshot-tests.md` for the mutation study):

* the **text** of every cell -- glyph and layout regressions;
* the **colour** of every cell -- colormap regressions, and erases that paint
  in the wrong background (an emulator that erases to default cannot see this);
* the **byte count** of the print -- the only signal that separates saying
  something efficiently from saying it at all, and so the only one that notices
  when differential rendering stops differentiating;
* the **cursor** and the lines scrolled -- the frame-to-frame contract that
  every subsequent diff is written against.

Plus, for the examples that save one, a digest of the image's decoded pixels,
which covers the disjoint `to_rgba_array` path through the pixel font.

Goldens live in `tests/goldens/`, one file per example, in a format meant to be
read: see `dumps`. Regenerate them with `make goldens`, which says what changed
before it writes.

Usage:

    python -m tests.examples --update [example ...]   rewrite goldens
    python -m tests.examples --show example           print a golden in colour
    python -m tests.examples --diff example           golden against a fresh run
    python -m tests.examples --sizes [example ...]    the terminal each needs
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import io
import json
import re
import runpy
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path
from typing import Iterator, Sequence

from tests.tmux import BLANK, Row, Screen, Terminal


ROOT = Path(__file__).parent.parent
EXAMPLES_DIR = ROOT / "examples"
GOLDENS_DIR = Path(__file__).parent / "goldens"


# # #
# THE EXAMPLES, AND THE TERMINAL EACH IS SNAPSHOTTED IN


@dataclasses.dataclass(frozen=True)
class Example:
    """One example script, and how to run and snapshot it.

    The terminal size is part of the test, not a convenience. A pane wider than
    the plot never exercises the deferred wrap at the final column, which is
    where the cursor bookkeeping is hardest and where the bug that prompted the
    tmux backend lived: with a roomy pane that bug is invisible to every layer
    except the byte count. So each example gets the terminal it is written for,
    one row taller than its output to hold the newline `print` appends.
    """
    script: str
    height: int
    width: int
    args: tuple[str, ...] = ()
    saves: str | None = None    # "png" or "gif", if the example saves an image

    @property
    def name(self) -> str:
        return self.script.removesuffix(".py")

    @property
    def path(self) -> Path:
        return EXAMPLES_DIR / self.script

    @property
    def golden(self) -> Path:
        return GOLDENS_DIR / f"{self.name}.txt"


EXAMPLES: tuple[Example, ...] = (
    Example("calendar_heatmap.py",      25, 64, saves="png"),
    Example("colormaps.py",             64, 72, saves="png"),
    Example("dashboard.py",             15, 56, saves="gif",
            args=("--num-frames", "5")),
    Example("demo.py",                  62, 80, saves="png"),
    Example("functions.py",             31, 74, saves="png"),
    Example("hilbert_curve.py",         33, 64, saves="png"),
    Example("image.py",                 33, 96, saves="png"),
    Example("jointplot.py",             35, 64, saves="png"),
    Example("life.py",                  26, 74, saves="gif",
            args=("--num-frames", "5")),
    Example("lissajous.py",             46, 75, saves="png"),
    Example("mandelbrot.py",            43, 80, saves="gif",
            args=("--num-frames", "5")),
    Example("quickstart1.py",           14, 81, saves="png"),
    Example("quickstart2.py",           14, 81, saves="gif",
            args=("--num-frames", "5")),
    Example("quickstart3.py",           14, 81, saves="gif",
            args=("--num-frames", "5")),
    Example("scatter.py",               24, 46, saves="png"),
    # --log-every is turned down so that a five step run still exercises
    # `anim.print`. The pane is three rows taller than the 23 the plot needs:
    # one for each logged line, and one for the row `clearstr` cannot step onto
    # the first time, when the plot is still at the top of the screen.
    Example("teacher_student.py",       27, 46, saves="gif",
            args=("--num-steps", "5", "--log-every", "2")),
    Example("teapot.py",                21, 80, saves="gif",
            args=("--num-frames", "5")),
    Example("time_series_histogram.py", 40, 80, saves="png"),
    Example("voronoi.py",               21, 70, saves="png"),
)


def find(name: str) -> Example:
    """Look an example up by name, with or without the `.py`."""
    wanted = name.removesuffix(".py")
    for example in EXAMPLES:
        if example.name == wanted:
            return example
    raise KeyError(f"no example named {name!r}")


# # #
# RUNNING AN EXAMPLE (IN THE CHILD PROCESS)


class Recorder(io.TextIOBase):
    """A stdout that keeps every write, so prints can be told apart.

    `print(x)` writes the payload and then the newline as two separate calls,
    and no example passes `end=` or touches `sys.stdout`, so a write of exactly
    "\\n" marks the end of a print. That is the only structure in the stream:
    the payloads themselves are full of newlines.
    """

    def __init__(self) -> None:
        self.frames: list[str] = []
        self._pending: list[str] = []

    def write(self, s: str) -> int:
        self._pending.append(s)
        if s == "\n":
            self.frames.append("".join(self._pending))
            self._pending = []
        return len(s)

    def finish(self) -> list[str]:
        if self._pending:
            self.frames.append("".join(self._pending))
            self._pending = []
        return self.frames


def fake_psutil(seed: int = 0, cores: int = 4) -> types.ModuleType:
    """A psutil whose readings are fixed, so `dashboard.py` can have a golden.

    The real one reports live load on however many cores the host has, which
    makes the example's content differ between runs and its width differ
    between machines -- the only example of the eighteen that is not already
    reproducible. The numbers below are arbitrary but deterministic, and move
    enough to give the history plot a shape.
    """
    import numpy as np

    rng = np.random.RandomState(seed)
    module = types.ModuleType("psutil")

    def cpu_percent(interval=None, percpu=False):
        if percpu:
            return [round(float(v), 1) for v in rng.uniform(0, 100, cores)]
        return round(float(rng.uniform(5, 95)), 1)

    def virtual_memory():
        return types.SimpleNamespace(
            total=16 * 1024**3,
            available=8 * 1024**3,
            percent=round(float(rng.uniform(30, 70)), 1),
        )

    module.cpu_percent = cpu_percent                # type: ignore[attr-defined]
    module.virtual_memory = virtual_memory          # type: ignore[attr-defined]
    module.cpu_count = lambda logical=True: cores   # type: ignore[attr-defined]
    return module


def run_and_record(example: Example, save: Path | None) -> list[str]:
    """Run the example here, as `python examples/x.py ...` does, and record it.

    Only called in the child process. `runpy` with `run_name="__main__"` runs
    the script's own `__main__` block, so it goes through its own `tyro.cli`
    call and the command line interface stays under test.
    """
    argv = [str(example.path), *example.args]
    if save is not None:
        argv += ["--save", str(save)]

    sys.modules["psutil"] = fake_psutil()
    real_sleep, time.sleep = time.sleep, lambda seconds: None
    recorder = Recorder()
    try:
        sys.argv = argv
        with contextlib.redirect_stdout(recorder):  # type: ignore[arg-type]
            runpy.run_path(str(example.path), run_name="__main__")
    finally:
        time.sleep = real_sleep
    return recorder.finish()


# # #
# RUNNING AN EXAMPLE (FROM THE TEST PROCESS)


@dataclasses.dataclass(frozen=True)
class ImageDigest:
    """A saved image, as its decoded pixels rather than its file bytes.

    PNG is lossless so the two agree, but a GIF's palette is chosen by Pillow's
    quantiser, which is free to change between versions without a pixel moving.
    """
    frames: int
    height: int
    width: int
    digest: str

    @staticmethod
    def of(path: Path) -> ImageDigest:
        import numpy as np
        from PIL import Image, ImageSequence

        arrays = [
            np.asarray(frame.convert("RGBA"))
            for frame in ImageSequence.Iterator(Image.open(path))
        ]
        sha = hashlib.sha256()
        for array in arrays:
            sha.update(array.tobytes())
        return ImageDigest(
            frames=len(arrays),
            height=arrays[0].shape[0],
            width=arrays[0].shape[1],
            digest=sha.hexdigest()[:16],
        )

    def __str__(self) -> str:
        return (f"{self.frames} frames, {self.height}x{self.width}, "
                f"sha256:{self.digest}")


@dataclasses.dataclass(frozen=True)
class Capture:
    """What an example printed, and the image it saved."""
    frames: tuple[str, ...]
    image: ImageDigest | None


def capture(example: Example) -> Capture:
    """Run one example in a fresh interpreter and collect what it produced.

    A subprocess per example, rather than importing them all into the test
    process: it keeps the `tyro` CLI and the `__main__` block under test, stops
    one example's module state from reaching the next, and contains a crash.
    """
    with tempfile.TemporaryDirectory() as tmp:
        result = Path(tmp) / "frames.json"
        save = Path(tmp) / f"{example.name}.{example.saves}" if example.saves else None
        command = [sys.executable, "-m", "tests.examples",
                   "--capture", example.name, "--result", str(result)]
        if save is not None:
            command += ["--save", str(save)]
        process = subprocess.run(
            command, capture_output=True, text=True, timeout=300, cwd=ROOT,
        )
        if process.returncode != 0:
            raise RuntimeError(
                f"{example.script} failed (exit {process.returncode}):\n"
                + process.stderr
            )
        frames = json.loads(result.read_text())
        if save is None:
            image = None
        elif not save.exists():
            raise RuntimeError(f"{example.script} saved no {save.name}")
        else:
            image = ImageDigest.of(save)
    return Capture(frames=tuple(frames), image=image)


# # #
# SNAPSHOTTING


@dataclasses.dataclass(frozen=True)
class Frame:
    """One print, and the screen it left behind."""
    nbytes: int
    cursor: tuple[int, int]
    scrolled: int
    cells: tuple[Row, ...]      # cropped to the content's bounding box

    @property
    def height(self) -> int:
        return len(self.cells)

    @property
    def width(self) -> int:
        return len(self.cells[0]) if self.cells else 0


@dataclasses.dataclass(frozen=True)
class Snapshot:
    """Everything a golden pins down about one example."""
    example: Example
    frames: tuple[Frame, ...]
    image: ImageDigest | None


def crop(screen: Screen) -> tuple[Row, ...]:
    """Trim the blank margin, so a golden records content and not padding.

    A blank cell painted in a colour is not blank, and survives the crop: that
    is how an erase in the wrong background colour stays visible.
    """
    rows = [i for i, row in enumerate(screen.cells) if any(c != BLANK for c in row)]
    if not rows:
        return ()
    height = rows[-1] + 1
    cols = [
        j for j in range(len(screen.cells[0]))
        if any(screen.cells[i][j] != BLANK for i in range(height))
    ]
    if not cols:
        return ()
    return tuple(row[:cols[-1] + 1] for row in screen.cells[:height])


def snapshot(example: Example, captured: Capture) -> Snapshot:
    """Replay a capture into a real terminal, one print at a time."""
    terminal = Terminal(example.height, example.width)
    frames = []
    for payload in captured.frames:
        terminal.feed(payload)
        screen = terminal.screen()
        frames.append(Frame(
            nbytes=len(payload),
            cursor=screen.cursor,
            scrolled=screen.scrolled,
            cells=crop(screen),
        ))
    return Snapshot(example=example, frames=tuple(frames), image=captured.image)


def take(example: Example) -> Snapshot:
    """Run an example and snapshot it: the whole pipeline, for one example."""
    return snapshot(example, capture(example))


# # #
# THE GOLDEN FILE FORMAT


# Symbols for the colour layer. The terminal's own default pair is always ".",
# so a blank background reads as blank; the rest are assigned in the order they
# first appear. Two characters per cell once a screen has more distinct pairs
# than there are symbols (`colormaps.py` has 2102 of them).
DEFAULT_SYMBOL = "."
SYMBOLS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

INDENT = "  "

FRAME_LINE = re.compile(
    r"frame (?P<i>\d+): (?P<nbytes>\d+) bytes, (?P<h>\d+)x(?P<w>\d+), "
    r"cursor (?P<row>\d+),(?P<col>\d+), scrolled (?P<scrolled>\d+)$"
)


def _hex(colour: tuple[int, int, int] | None) -> str:
    return "default" if colour is None else "#%02x%02x%02x" % colour


def _unhex(text: str) -> tuple[int, int, int] | None:
    if text == "default":
        return None
    return (int(text[1:3], 16), int(text[3:5], 16), int(text[5:7], 16))


def palette_of(cells: Sequence[Row]) -> list[tuple[tuple, str]]:
    """The distinct (fg, bg) pairs, in order of first appearance, with symbols."""
    order: list[tuple] = []
    seen: set[tuple] = set()
    for row in cells:
        for _, fg, bg in row:
            if (fg, bg) not in seen:
                seen.add((fg, bg))
                order.append((fg, bg))
    default = (None, None) in seen
    if default:
        order.remove((None, None))
        order.insert(0, (None, None))
    span = 1 if len(order) - default <= len(SYMBOLS) else 2
    out, n = [], 0
    for pair in order:
        if pair == (None, None):
            out.append((pair, DEFAULT_SYMBOL * span))
        else:
            out.append((pair, SYMBOLS[n] if span == 1 else
                        SYMBOLS[n // len(SYMBOLS)] + SYMBOLS[n % len(SYMBOLS)]))
            n += 1
    return out


def dumps(snap: Snapshot) -> str:
    """Serialise a snapshot into its golden file.

    One block per frame: a header line carrying everything scalar, then the
    palette, then the text layer, then the colour layer, the last two aligned
    cell for cell so they can be read against each other. Headers start at
    column zero and grid rows are indented, which is the whole of the grammar;
    the header says how many rows to expect, so a row of spaces that some
    editor has stripped still parses.
    """
    example = snap.example
    out = [
        f"example: {example.script}",
        f"args: {' '.join(example.args) if example.args else '(none)'}",
        f"terminal: {example.height}x{example.width}",
    ]
    if snap.image is not None:
        out.append(f"image: {snap.image}")
    out += [
        "",
        "# Regenerate with `make goldens`. One frame per print; the text and",
        "# colour grids are the terminal screen, cropped to its content.",
    ]

    for i, frame in enumerate(snap.frames):
        out += [
            "",
            f"frame {i}: {frame.nbytes} bytes, {frame.height}x{frame.width}, "
            f"cursor {frame.cursor[0]},{frame.cursor[1]}, "
            f"scrolled {frame.scrolled}",
        ]
        if not frame.cells:
            continue
        palette = palette_of(frame.cells)
        symbol = dict(palette)
        out.append("palette:")
        out += [f"{INDENT}{s} {_hex(fg)} {_hex(bg)}" for (fg, bg), s in palette]
        out.append("text:")
        out += [(INDENT + "".join(c for c, _, _ in row)).rstrip()
                for row in frame.cells]
        out.append("colour:")
        out += [INDENT + "".join(symbol[(fg, bg)] for _, fg, bg in row)
                for row in frame.cells]
    return "\n".join(out) + "\n"


class GoldenError(Exception):
    """A golden file that cannot be read."""


def loads(text: str) -> tuple[dict[str, str], tuple[Frame, ...]]:
    """Parse a golden file back into its header and its frames.

    Parsed rather than compared as text, so that a mismatch can name the cell
    that moved instead of the line.
    """
    header: dict[str, str] = {}
    frames: list[Frame] = []
    lines = text.split("\n")
    i = 0

    def section(name: str, count: int) -> list[str]:
        """Read `count` indented rows under a `name:` heading."""
        nonlocal i
        if i >= len(lines) or lines[i] != f"{name}:":
            raise GoldenError(f"line {i + 1}: expected {name!r}, got {lines[i]!r}")
        i += 1
        rows = [line[len(INDENT):] for line in lines[i:i + count]]
        if len(rows) < count:
            raise GoldenError(f"line {i + 1}: {name} block ends early")
        i += count
        return rows

    while i < len(lines):
        line = lines[i]
        if not line or line.startswith("#"):
            i += 1
            continue
        if not line.startswith("frame "):
            key, _, value = line.partition(":")
            header[key] = value.strip()
            i += 1
            continue

        match = FRAME_LINE.match(line)
        if match is None:
            raise GoldenError(f"line {i + 1}: malformed frame header {line!r}")
        f = match.groupdict()
        height, width = int(f["h"]), int(f["w"])
        i += 1
        cells: tuple[Row, ...] = ()
        if height:
            symbol: dict[str, tuple] = {}
            span = 1
            while i < len(lines) and lines[i] == "palette:":
                i += 1
                while i < len(lines) and lines[i].startswith(INDENT):
                    s, fg, bg = lines[i][len(INDENT):].split()
                    symbol[s], span = (_unhex(fg), _unhex(bg)), len(s)
                    i += 1
            text_rows = [row.ljust(width) for row in section("text", height)]
            colour_rows = section("colour", height)
            try:
                cells = tuple(
                    tuple(
                        (text_rows[r][c],
                         *symbol[colour_rows[r][c * span:(c + 1) * span]])
                        for c in range(width)
                    )
                    for r in range(height)
                )
            except KeyError as e:
                raise GoldenError(f"frame {f['i']}: unknown colour symbol {e}")
        frames.append(Frame(
            nbytes=int(f["nbytes"]),
            cursor=(int(f["row"]), int(f["col"])),
            scrolled=int(f["scrolled"]),
            cells=cells,
        ))
    return header, tuple(frames)


# # #
# COMPARING


def differences(snap: Snapshot, golden: str) -> list[str]:
    """Everything about a snapshot that its golden did not predict.

    Readable one-liners: the point of storing the whole colour layer rather
    than a digest of it is that this can name the cell.
    """
    header, expected = loads(golden)
    out = []
    size = f"{snap.example.height}x{snap.example.width}"
    if header.get("terminal") != size:
        out.append(f"terminal: golden {header.get('terminal')}, table {size}")
    if len(expected) != len(snap.frames):
        out.append(f"prints: golden {len(expected)}, now {len(snap.frames)}")
    want_image = header.get("image")
    have_image = str(snap.image) if snap.image is not None else None
    if want_image != have_image:
        out.append(f"image: golden {want_image}, now {have_image}")

    for i, (want, have) in enumerate(zip(expected, snap.frames)):
        if want.nbytes != have.nbytes:
            delta = have.nbytes - want.nbytes
            out.append(f"frame {i}: {want.nbytes} bytes -> {have.nbytes} "
                       f"({delta:+d}, {delta / want.nbytes:+.1%})")
        if want.cursor != have.cursor:
            out.append(f"frame {i}: cursor {want.cursor} -> {have.cursor}")
        if want.scrolled != have.scrolled:
            out.append(
                f"frame {i}: scrolled {want.scrolled} -> {have.scrolled}"
                + (" (the output no longer fits its terminal)"
                   if have.scrolled > want.scrolled else "")
            )
        if (want.height, want.width) != (have.height, have.width):
            out.append(f"frame {i}: screen {want.height}x{want.width} -> "
                       f"{have.height}x{have.width}")
            continue
        out += cell_differences(i, want, have)
    return out


def cell_differences(i: int, want: Frame, have: Frame, limit: int = 6) -> list[str]:
    """Which cells moved, up to a few of each kind."""
    out = []
    glyphs = colours = 0
    for r in range(want.height):
        for c in range(want.width):
            a, b = want.cells[r][c], have.cells[r][c]
            if a == b:
                continue
            if a[0] != b[0]:
                glyphs += 1
                if glyphs <= limit:
                    out.append(f"frame {i} cell {r},{c}: "
                               f"glyph {a[0]!r} -> {b[0]!r}")
            else:
                colours += 1
                if colours <= limit:
                    out.append(f"frame {i} cell {r},{c}: colour "
                               f"{_hex(a[1])}/{_hex(a[2])} -> "
                               f"{_hex(b[1])}/{_hex(b[2])}")
    if glyphs > limit:
        out.append(f"frame {i}: ... and {glyphs - limit} more glyph changes")
    if colours > limit:
        out.append(f"frame {i}: ... and {colours - limit} more colour changes")
    return out


# # #
# LOOKING AT A SNAPSHOT


def to_ansi(cells: Sequence[Row]) -> str:
    """Turn captured cells back into what the terminal was showing."""
    out = []
    for row in cells:
        fg: tuple[int, int, int] | None = None
        bg: tuple[int, int, int] | None = None
        for char, f, b in row:
            if f != fg:
                out.append("\x1b[39m" if f is None else "\x1b[38;2;%d;%d;%dm" % f)
                fg = f
            if b != bg:
                out.append("\x1b[49m" if b is None else "\x1b[48;2;%d;%d;%dm" % b)
                bg = b
            out.append(char)
        out.append("\x1b[0m\n")
    return "".join(out)


def render(script: str, i: int, frame: Frame) -> str:
    """One frame, captioned, as the terminal showed it."""
    return (f"\x1b[1m{script} frame {i}: {frame.nbytes} bytes, "
            f"{frame.height}x{frame.width}\x1b[0m\n" + to_ansi(frame.cells))


def render_golden(example: Example) -> Iterator[str]:
    """Each frame of a golden, as the terminal showed it."""
    _, frames = loads(example.golden.read_text())
    for i, frame in enumerate(frames):
        yield render(example.script, i, frame)


# # #
# COMMAND LINE


def _update(names: list[str]) -> int:
    GOLDENS_DIR.mkdir(exist_ok=True)
    chosen = [find(n) for n in names] if names else list(EXAMPLES)
    written = 0
    for example in chosen:
        snap = take(example)
        new = dumps(snap)
        if example.golden.exists():
            old = example.golden.read_text()
            if old == new:
                print(f"{example.script}: unchanged")
                continue
            print(f"\x1b[1m{example.script}: changed\x1b[0m")
            for line in differences(snap, old):
                print(f"  {line}")
        else:
            print(f"\x1b[1m{example.script}: new golden\x1b[0m")
        scrolled = max(f.scrolled for f in snap.frames) if snap.frames else 0
        if scrolled:
            print(f"  \x1b[33mwarning: {scrolled} line(s) scrolled off the top; "
                  f"give it a taller terminal in EXAMPLES\x1b[0m")
        example.golden.write_text(new)
        written += 1
    print(f"{written} golden(s) written to {GOLDENS_DIR}")
    return 0


ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\][^\x07]*\x07")
MOTION = re.compile(r"\x1b\[[0-9;]*[^m0-9;]")


def _sizes(names: list[str]) -> int:
    """Report the terminal each example needs, for filling in EXAMPLES.

    Height is measured, not guessed: an example is replayed into a pane far
    taller than it needs, and the row the cursor reaches is how many rows its
    output really occupies. Guessing from the ink is what put `teapot.py` in a
    19-row terminal when its plot is 20 rows tall and merely draws nothing in
    the top row.

    Width cannot be measured the same way -- a diff only mentions the cells it
    repaints, and stripping its cursor moves runs the glyphs of a whole frame
    into one long line -- so the widest line of a full redraw is offered as a
    suggestion instead. A payload is a full redraw when every escape in it is
    an SGR; anything that moves the cursor is a diff. This still over-reports
    for an example that prints something other than a plot (`demo.py` prints a
    1233-character `repr` that is meant to wrap), so it is a prompt for a
    human, not an answer.
    """
    chosen = [find(n) for n in names] if names else list(EXAMPLES)
    print(f"{'example':26} {'in table':>10} {'needs':>10} {'widest line':>12}")
    for example in chosen:
        captured = capture(example)
        terminal = Terminal(200, example.width)
        rows = 0
        for payload in captured.frames:
            terminal.feed(payload)
            rows = max(rows, terminal.screen().cursor[0])
        widest = max(
            (len(line) for payload in captured.frames
             if not MOTION.search(payload)
             for line in ANSI.sub("", payload).split("\n")),
            default=0,
        )
        table = f"{example.height}x{example.width}"
        needs = f"{rows + 1}x{example.width}"
        flag = "" if table == needs else "  <-- change"
        print(f"{example.script:26} {table:>10} {needs:>10} {widest:>12}{flag}")
    return 0


def _show(name: str) -> int:
    for block in render_golden(find(name)):
        print(block)
    return 0


def _diff(name: str) -> int:
    example = find(name)
    if not example.golden.exists():
        print(f"{example.script}: no golden yet; run `make goldens`")
        return 1
    snap = take(example)
    lines = differences(snap, example.golden.read_text())
    if not lines:
        print(f"{example.script}: matches its golden")
        return 0
    print(f"\x1b[1m{example.script}: {len(lines)} difference(s)\x1b[0m")
    for line in lines:
        print(f"  {line}")
    print("\n\x1b[1m--- golden ---\x1b[0m")
    for block in render_golden(example):
        print(block)
    print("\x1b[1m--- now ---\x1b[0m")
    for i, frame in enumerate(snap.frames):
        print(render(example.script, i, frame))
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Snapshot the examples, and inspect the snapshots."
    )
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument("--update", nargs="*", metavar="EXAMPLE",
                      help="rewrite goldens (all of them, or just the named)")
    what.add_argument("--show", metavar="EXAMPLE",
                      help="print a golden as the terminal showed it")
    what.add_argument("--diff", metavar="EXAMPLE",
                      help="compare a fresh run against its golden")
    what.add_argument("--sizes", nargs="*", metavar="EXAMPLE",
                      help="report the terminal each example needs")
    what.add_argument("--capture", metavar="EXAMPLE", help=argparse.SUPPRESS)
    parser.add_argument("--result", metavar="PATH", help=argparse.SUPPRESS)
    parser.add_argument("--save", metavar="PATH", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.capture is not None:
        # the child process: run one example and hand back what it printed
        frames = run_and_record(
            find(args.capture),
            Path(args.save) if args.save else None,
        )
        Path(args.result).write_text(json.dumps(frames))
        return 0
    if args.sizes is not None:
        return _sizes(args.sizes)
    if args.show is not None:
        return _show(args.show)
    if args.diff is not None:
        return _diff(args.diff)
    return _update(args.update)


if __name__ == "__main__":
    sys.exit(main())
