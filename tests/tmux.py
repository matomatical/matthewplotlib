"""
A real terminal for the tests: a tmux pane you can feed bytes to and read back.

How it works:

* One tmux server, on a socket private to this process, holding one session
  whose single pane runs `sleep` to keep a pty open. It is created on first use
  and killed when the process exits.
* Feeding means writing bytes to the pane's slave tty (`#{pane_tty}`), exactly
  as a program printing to a terminal does: the line discipline turns "\\n" into
  CR LF, and tmux's emulator interprets the rest. Nothing parses what we feed.
* Reading back means `capture-pane -p -e -N`, which round-trips 24-bit colour,
  plus `display-message` for the cursor and for the number of lines that have
  scrolled off into the pane's history.
* Between terminals the pane is reset and resized in one command: `send-keys
  -R` (clear the screen, home the cursor, drop the attributes), `clear-history`
  (so the scroll count starts from zero) and `resize-window` (which a detached
  session honours because we set `window-size manual`).

Sequencing: a tty write returns before tmux has read it, so every read first
appends an OSC 2 (set window title) sentinel and waits for `#{pane_title}` to
catch it up. tmux applies pty bytes in order, so an arrived title proves
everything written before it has been applied. OSC 2 touches the title and
nothing else -- not the screen, not the cursor, not the last-column flag
(`TestTheHarness` pins that down).

Cost: a tmux write is free, but each read is a tmux client invocation, ~4 ms.
So feed as much as a test needs, then read once.

Design notes, and what a real terminal catches that an emulator did not, are in
`notes/terminal-test-backend.md`.
"""

import atexit
import os
import shutil
import subprocess
import time
from typing import Self, Sequence


# # #
# TMUX


_TMUX = shutil.which("tmux")

if _TMUX is None:
    raise RuntimeError(
        "tmux is not installed. It is a development dependency: without it "
        "nothing checks what the library's escape sequences do to a terminal. "
        "Install tmux and run the tests again (see CONTRIBUTING.md)."
    )

TMUX: str = _TMUX


# # #
# THE SHARED PANE


# The pane needs some program to hold its tty open. If this process is killed
# before the exit hook runs, this bounds how long the stray server survives.
HOLD_COMMAND = "sleep 3600"

# Sequence read back with the screen, to prove the screen is up to date.
SENTINEL = "\x1b]2;matthewplotlib-{}\x07"


class _Pane:
    """The one tmux pane the tests share, and the plumbing to talk to it.

    Not used directly: `Terminal` claims it, sizes it and clears it.
    """

    def __init__(self) -> None:
        self.socket = f"matthewplotlib-tests-{os.getpid()}"
        self.generation = 0     # bumped per claim, to catch use of a stale Terminal
        self.sequence = 0       # sentinel counter
        self._tmux(
            "-f", "/dev/null",              # ignore the developer's tmux.conf
            "new-session", "-d", "-x", "80", "-y", "24", HOLD_COMMAND,
            ";", "set-option", "-g", "window-size", "manual",
        )
        atexit.register(self.kill)
        tty, self.socket_path = self._tmux(
            "display-message", "-p", "#{pane_tty}\n#{socket_path}"
        ).split()
        self.tty = open(tty, "wb", buffering=0)

    def _tmux(self, *args: str) -> str:
        result = subprocess.run(
            [TMUX, "-L", self.socket, *args],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"tmux {' '.join(args)!r} failed: {result.stderr.strip()!r}. "
                "The test pane may have died."
            )
        return result.stdout

    def kill(self) -> None:
        self._tmux("kill-server")
        # killing the server does not unlink its socket, and this one is never
        # coming back: the name has our pid in it
        try:
            os.unlink(self.socket_path)
        except OSError:
            pass

    def claim(self, height: int, width: int) -> int:
        """Clear and resize the pane, and return the new generation number."""
        self._tmux(
            "send-keys", "-R",                  # clear screen, home cursor, reset SGR
            ";", "clear-history",               # so scrolled lines count from zero
            ";", "resize-window", "-x", str(width), "-y", str(height),
        )
        self.generation += 1
        return self.generation

    def write(self, payload: str) -> None:
        self.tty.write(payload.encode())

    def read(self, height: int, width: int) -> "Screen":
        """Wait for everything written so far to be applied, then snapshot."""
        self.sequence += 1
        expected = f"matthewplotlib-{self.sequence}"
        self.write(SENTINEL.format(self.sequence))
        deadline = time.monotonic() + 10
        while True:
            output = self._tmux(
                "display-message", "-p", "#{pane_title}|#{cursor_y}|#{cursor_x}|"
                                         "#{history_size}|#{pane_dead}",
                ";", "capture-pane", "-p", "-e", "-N",
            )
            status, capture = output.split("\n", 1)
            title, row, col, history, dead = status.split("|")
            if dead == "1":
                raise RuntimeError("the test pane died")
            if title == expected:
                return Screen.parse(
                    capture,
                    height=height,
                    width=width,
                    cursor=(int(row), int(col)),
                    scrolled=int(history),
                )
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"tmux never applied what we wrote: title {title!r}, "
                    f"wanted {expected!r}"
                )
            time.sleep(0.001)


_pane: _Pane | None = None


def _shared_pane() -> _Pane:
    global _pane
    if _pane is None:
        _pane = _Pane()
    return _pane


# # #
# TERMINAL


class Terminal:
    """A real terminal of a given size, cleared, ready to be fed.

    Only one is live at a time -- they all share a single tmux pane -- so
    constructing a second one retires the first, and touching the retired one
    raises rather than quietly writing to the wrong screen. Read the screen you
    want before opening the next terminal.
    """

    def __init__(self, height: int, width: int) -> None:
        self.height = height
        self.width = width
        self.pane = _shared_pane()
        self.generation = self.pane.claim(height, width)

    def _check(self) -> None:
        if self.generation != self.pane.generation:
            raise RuntimeError(
                "this Terminal has been retired by a later one; capture its "
                "screen before opening another"
            )

    def feed(self, payload: str) -> Self:
        """Write bytes to the terminal, as a program printing to it would."""
        self._check()
        self.pane.write(payload)
        return self

    def print(self, payload: str) -> Self:
        """Feed a payload the way the library intends: as `print` would.

        That is, plainly, so that the trailing newline `print` appends supplies
        the one each of the library's sequences is shaped to expect.
        """
        return self.feed(payload + "\n")

    def screen(self) -> "Screen":
        """Snapshot the screen: cells, cursor, and lines scrolled away."""
        self._check()
        return self.pane.read(self.height, self.width)


# # #
# SCREEN


# A cell is (character, foreground, background), each colour an RGB triple or
# None for the terminal's default.
Cell = tuple[str, tuple[int, int, int] | None, tuple[int, int, int] | None]
Row = tuple[Cell, ...]

# An untouched cell, which is what a screen is full of to begin with.
BLANK: Cell = (" ", None, None)


class Screen:
    """What a terminal is showing: a grid of cells, the cursor, the scroll count.

    A value, not a handle -- snapshot two of them and compare.
    """

    def __init__(
        self,
        cells: Sequence[Row],
        cursor: tuple[int, int],
        scrolled: int,
    ) -> None:
        self.cells = tuple(cells)
        # Row and column of the cursor. Note that after a glyph lands in the
        # final column the column reads as the width, not width - 1: the wrap is
        # deferred, and the terminal reports the position it would wrap to.
        self.cursor = cursor
        # Lines that have scrolled off the top into the pane's history.
        self.scrolled = scrolled

    @classmethod
    def parse(
        cls,
        capture: str,
        height: int,
        width: int,
        cursor: tuple[int, int],
        scrolled: int,
    ) -> "Screen":
        """Parse `capture-pane -p -e -N` output into cells.

        The capture is one continuous stream: colour set on one row carries into
        the next unless the capture says otherwise, so the SGR state is *not*
        reset per line. Rows are emitted only as far as tmux tracks them as
        used, and every cell past that point is a default cell.
        """
        fg: tuple[int, int, int] | None = None
        bg: tuple[int, int, int] | None = None
        rows = []
        for line in capture.split("\n")[:height]:
            cells: list[Cell] = []
            i = 0
            while i < len(line):
                if line[i] == "\x1b":
                    assert line[i + 1] == "[", f"unexpected escape in {line!r}"
                    end = line.index("m", i)
                    codes = [int(c) for c in line[i + 2:end].split(";") if c] or [0]
                    k = 0
                    while k < len(codes):
                        code = codes[k]
                        if code == 0:
                            fg = bg = None
                            k += 1
                        elif code == 39:
                            fg = None
                            k += 1
                        elif code == 49:
                            bg = None
                            k += 1
                        elif code == 38 and codes[k + 1] == 2:
                            fg = (codes[k + 2], codes[k + 3], codes[k + 4])
                            k += 5
                        elif code == 48 and codes[k + 1] == 2:
                            bg = (codes[k + 2], codes[k + 3], codes[k + 4])
                            k += 5
                        else:
                            raise AssertionError(
                                f"the terminal reported SGR {code}, which the "
                                f"library does not emit: {line!r}"
                            )
                    i = end + 1
                else:
                    cells.append((line[i], fg, bg))
                    i += 1
            cells.extend([BLANK] * (width - len(cells)))
            rows.append(tuple(cells[:width]))
        rows.extend([(BLANK,) * width] * (height - len(rows)))
        return cls(rows, cursor=cursor, scrolled=scrolled)

    def region(self, height: int, width: int) -> tuple[Row, ...]:
        """The top-left `height` by `width` cells, for comparing two screens."""
        return tuple(row[:width] for row in self.cells[:height])

    def line(self, row: int) -> str:
        """One row as plain text, trailing blanks stripped."""
        return "".join(c for c, _, _ in self.cells[row]).rstrip()

    def text(self) -> str:
        """The whole screen as plain text, for a failure message."""
        return "\n".join(self.line(r) for r in range(len(self.cells)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Screen):
            return NotImplemented
        return (
            self.cells == other.cells
            and self.cursor == other.cursor
            and self.scrolled == other.scrolled
        )

    def __repr__(self) -> str:
        return (
            f"<Screen {len(self.cells)}x{len(self.cells[0]) if self.cells else 0} "
            f"cursor={self.cursor} scrolled={self.scrolled}\n{self.text()}\n>"
        )
