"""
What the escape sequences we emit do to a real terminal.

Everything here drives an actual terminal -- a tmux pane, see `tests/tmux.py`,
which needs tmux installed. What is asserted is the screen: which glyph and
which 24-bit colour ended up in each cell, where the cursor was left, and how
many lines scrolled away.

The string-level claims about the same sequences, which need no terminal, are in
`test_core.py`.
"""

import io
import contextlib

import numpy as np
import pytest

import matthewplotlib as mp
from matthewplotlib.core import CharArray, unicode_image
from tests.tmux import BLANK, Row, Screen, Terminal


# # #
# HELPERS


def _rand_chars(rng, h: int, w: int) -> CharArray:
    """A realistic CharArray: a random half-block image (every cell coloured)."""
    img = rng.integers(0, 256, size=(2 * h, w, 3), dtype=np.uint8)
    return unicode_image(img)


def _cells(ca: CharArray) -> tuple[Row, ...]:
    """The cells a CharArray describes, read off the array itself.

    The reference a rendering is measured against when what is being checked is
    the rendering: it involves no escape sequence and no terminal.
    """
    def colour(on, rgb) -> tuple[int, int, int] | None:
        return (int(rgb[0]), int(rgb[1]), int(rgb[2])) if on else None

    return tuple(
        tuple(
            (
                chr(ca.codes[i, j]),
                colour(ca.fg[i, j], ca.fg_rgb[i, j]),
                colour(ca.bg[i, j], ca.bg_rgb[i, j]),
            )
            for j in range(ca.width)
        )
        for i in range(ca.height)
    )


def _screen_after(height: int, width: int, *payloads: str) -> Screen:
    """Feed payloads into a fresh terminal of this size, and snapshot it."""
    term = Terminal(height, width)
    for payload in payloads:
        term.feed(payload)
    return term.screen()


def _screen_after_diff(prev, new, slack=2, rows_below=3) -> Screen:
    """Emulate print(prev), then print(new.to_ansi_diff_str(prev)).

    Both are printed the way the library intends -- plainly, so `print` supplies
    the trailing newline each sequence is shaped to expect.

    `slack` is the number of spare columns beyond the plot's width; pass 0 to put
    the plot flush against the right edge of the screen. `rows_below` is the
    spare rows beneath it; pass 1 for the tightest layout that still animates.
    """
    term = Terminal(prev.height + rows_below, prev.width + slack)
    term.print(prev.to_ansi_str())                    # print(prev)
    term.print(new.to_ansi_diff_str(prev))            # print(new - prev)
    return term.screen()


def _screen_after_resize(prev, new, footer=None, slack=2, rows_below=4) -> Screen:
    """print(prev), optionally a footer line beneath it, then print(new - prev).

    The footer is written on the row the cursor already occupies -- one below
    prev -- and the carriage return puts the cursor back where the contract
    wants it. It must fit on one screen row, or it would wrap and leave the
    cursor somewhere the contract does not expect.
    """
    UH = max(prev.height, new.height)
    UW = max(prev.width, new.width)
    assert footer is None or len(footer) <= UW + slack
    term = Terminal(UH + rows_below, UW + slack)
    term.print(prev.to_ansi_str())
    if footer is not None:
        term.feed(footer).feed("\r")
    term.print(new.to_ansi_diff_str(prev))
    return term.screen()


def _difference(got: tuple[Row, ...], want: tuple[Row, ...]) -> str:
    """Where two regions first differ, for a failure message."""
    for i, (got_row, want_row) in enumerate(zip(got, want)):
        for j, (g, w) in enumerate(zip(got_row, want_row)):
            if g != w:
                return f"cell ({i}, {j}) is {g}, expected {w}"
    if len(got) != len(want) or any(len(g) != len(w) for g, w in zip(got, want)):
        return f"different shapes: {len(got)} rows against {len(want)}"
    return "no cell differs"


# # #
# THE HARNESS ITSELF


class TestTheHarness:
    """The terminal behaviours the library's sequences are built on, and the two
    the harness itself needs: that reading the screen does not disturb it, and
    that colour survives the round trip.

    This is tmux's behaviour, which these tests make the de facto specification,
    so it is also where the support matrix starts.
    """

    def test_glyphs_land_where_they_are_written(self):
        screen = _screen_after(3, 4, "ab\ncd")
        assert [screen.line(r) for r in range(3)] == ["ab", "cd", ""]
        assert screen.cursor == (1, 2)

    def test_a_newline_also_returns_to_column_zero(self):
        """As it does for a program printing to a terminal: the line discipline
        turns "\\n" into CR LF. Every sequence here is shaped for `print`."""
        assert _screen_after(3, 4, "ab\n").cursor == (1, 0)

    def test_24_bit_colour_survives_the_round_trip(self):
        screen = _screen_after(
            1, 2, "\x1b[38;2;12;34;56m\x1b[48;2;200;100;50mA\x1b[0mB"
        )
        assert screen.cells[0] == (
            ("A", (12, 34, 56), (200, 100, 50)),
            ("B", None, None),
        )

    def test_wrap_is_deferred_at_the_right_margin(self):
        """The behaviour that hid the cursor bug this work started from.

        Writing the final column leaves the cursor on the margin with a wrap
        pending rather than moving it past the edge -- and the terminal reports
        the column it would wrap to, which is the width, not width - 1.
        """
        term = Terminal(3, 4).feed("....")
        assert term.screen().cursor == (0, 4)
        term.feed("X")                     # only now does the line actually wrap
        screen = term.screen()
        assert [screen.line(r) for r in range(3)] == ["....", "X", ""]

    def test_a_carriage_return_cancels_a_deferred_wrap(self):
        """How the renderer gets out of the pending-wrap state.

        `goto_col` stops trusting its column arithmetic once a glyph fills the
        plot's last column, and recovers with a carriage return. That only works
        if the return resolves the pending wrap rather than leaving it armed for
        the next glyph -- which would land the glyph a row below where the
        renderer thinks it is. This is the assumption the audit traded CHA for,
        so it is the one worth pinning.
        """
        term = Terminal(3, 4).feed("....")     # margin reached, wrap pending
        term.feed("\rX")                       # return, then paint
        screen = term.screen()
        assert [screen.line(r) for r in range(3)] == ["X...", "", ""]
        assert screen.cursor == (0, 1)

    def test_reading_the_screen_leaves_the_wrap_pending(self):
        """Reading appends a set-title sentinel to synchronise with tmux. If that
        disturbed the last-column flag, the harness would hide the very bug it
        exists to catch, so: fill the margin, read, then move and paint from
        there -- an absolute column move must still land where it says.
        """
        term = Terminal(3, 4).feed("....")
        term.screen()                      # writes the sentinel
        term.feed("\x1b[3G!")              # CHA to column 3, counted from 1
        assert term.screen().line(0) == "..!."

    def test_cursor_moves_clamp_at_the_edges(self):
        """Which is why appending rows uses newlines and not cursor-down: at the
        bottom margin, cursor-down stops where a line feed scrolls."""
        assert _screen_after(3, 4, "\x1b[9A").cursor == (0, 0)     # up
        assert _screen_after(3, 4, "\x1b[9B").cursor == (2, 0)     # down
        assert _screen_after(3, 4, "\x1b[9C").cursor == (0, 3)     # forward
        assert _screen_after(3, 4, "\x1b[9D").cursor == (0, 0)     # back
        assert _screen_after(3, 4, "\x1b[9E").cursor == (2, 0)     # next line
        assert _screen_after(3, 4, "\x1b[9B").scrolled == 0        # no scrolling

    def test_a_line_feed_at_the_bottom_scrolls(self):
        screen = _screen_after(3, 4, "a\nb\nc\nd\n")
        assert screen.scrolled == 2
        assert [screen.line(r) for r in range(3)] == ["c", "d", ""]

    def test_blanking_paints_the_active_background(self):
        """Why `reset_colour` runs before every blanking in `to_ansi_diff_str`.

        Both ways the renderer blanks a cell carry the active background with
        them: a written space always does, and an erase does on any terminal
        with background-colour erase. A model that erased to default -- as the
        emulator this replaced did -- cannot see the second.
        """
        # a written space, which is how trailing columns are cleared
        spaces = _screen_after(1, 4, "\x1b[48;2;7;7;7mAB  ")
        assert spaces.cells[0][2] == (" ", None, (7, 7, 7))
        spaces_reset = _screen_after(1, 4, "\x1b[48;2;7;7;7mAB\x1b[0m  ")
        assert spaces_reset.cells[0][2] == BLANK
        # erase-in-line, which is how whole lost rows are cleared
        erased = _screen_after(2, 4, "\x1b[48;2;7;7;7mAB\n\x1b[2K")
        assert erased.cells[1][0] == (" ", None, (7, 7, 7))
        erased_reset = _screen_after(2, 4, "\x1b[48;2;7;7;7mAB\n\x1b[0m\x1b[2K")
        assert erased_reset.cells[1][0] == BLANK

    def test_each_terminal_starts_from_a_clean_screen(self):
        _screen_after(2, 4, "\x1b[48;2;7;7;7mdirt\nmore dirt")
        screen = _screen_after(2, 4)
        assert screen.cells == ((BLANK,) * 4,) * 2
        assert screen.cursor == (0, 0)
        assert screen.scrolled == 0

    def test_a_retired_terminal_raises(self):
        """They share one pane, so a stale handle would write to a live screen."""
        term = Terminal(2, 2)
        Terminal(2, 2)
        with pytest.raises(RuntimeError):
            term.feed("x")


# # #
# FULL REDRAW


class TestFullRedraw:
    """`to_ansi_str` must put on the screen exactly what the CharArray says.

    Measured against the array itself rather than against another sequence, so
    this is the one comparison in the file that does not assume the renderer is
    right about anything -- and, incidentally, the check that the harness reads
    colour back correctly.
    """

    def test_a_fully_coloured_plot_arrives_cell_for_cell(self):
        ca = _rand_chars(np.random.default_rng(3), 4, 7)
        screen = _screen_after(ca.height + 2, ca.width + 3, ca.to_ansi_str())
        got, want = screen.region(ca.height, ca.width), _cells(ca)
        assert got == want, _difference(got, want)

    def test_a_coloured_blank_at_the_right_edge_arrives(self):
        """A cell can be blank and still carry a background colour. Flush against
        the right margin it is also the case a screen reader is most likely to
        drop, so it is worth pinning: nothing about it is trimmed away.
        """
        ca = CharArray.from_size(1, 4, bgcolor="red")
        screen = _screen_after(2, 4, ca.to_ansi_str())
        got, want = screen.region(1, 4), _cells(ca)
        assert got == want, _difference(got, want)


# # #
# DIFFERENTIAL RENDERING


class TestCharArrayDiffStr:
    def test_no_change_still_holds_the_cursor(self):
        """An unchanged frame must not return "": printed, that would add a
        newline and walk the cursor a row further down every frame."""
        rng = np.random.default_rng(0)
        ca = _rand_chars(rng, 4, 6)
        screen = _screen_after_diff(ca, ca)
        ref = _screen_after(ca.height + 3, ca.width + 2, ca.to_ansi_str())
        assert screen.region(ca.height, ca.width) == ref.region(ca.height, ca.width)
        assert screen.cursor == (ca.height, 0)

    def test_cursor_returns_below_plot(self):
        rng = np.random.default_rng(3)
        prev = _rand_chars(rng, 5, 7)
        new = _rand_chars(rng, 5, 7)
        assert _screen_after_diff(prev, new).cursor == (prev.height, 0)

    def test_color_only_change(self):
        rng = np.random.default_rng(4)
        prev = _rand_chars(rng, 3, 5)
        new = unicode_image(rng.integers(0, 256, (6, 5, 3), dtype=np.uint8))
        new.codes = prev.codes.copy()  # same glyphs, different colours
        screen = _screen_after_diff(prev, new)
        ref = _screen_after(new.height + 3, new.width + 2, new.to_ansi_str())
        got, want = screen.region(new.height, new.width), ref.region(new.height, new.width)
        assert got == want, _difference(got, want)

    def test_diff_matches_full_redraw_random(self):
        """The strong one: a diff must leave the screen identical to a fresh
        redraw of `new`, across many random partial changes."""
        rng = np.random.default_rng(2024)
        for _ in range(40):
            h = int(rng.integers(1, 9))
            w = int(rng.integers(1, 14))
            base = rng.integers(0, 256, (2 * h, w, 3), dtype=np.uint8)
            after = base.copy()
            mask = rng.random((2 * h, w)) < rng.uniform(0.0, 0.6)
            after[mask] = rng.integers(0, 256, (int(mask.sum()), 3), dtype=np.uint8)
            prev = unicode_image(base)
            new = unicode_image(after)
            screen = _screen_after_diff(prev, new)
            ref = _screen_after(new.height + 3, new.width + 2, new.to_ansi_str())
            got = screen.region(new.height, new.width)
            want = ref.region(new.height, new.width)
            assert got == want, f"{(h, w)}: {_difference(got, want)}"
            assert screen.cursor == (new.height, 0)

    def test_diff_at_right_edge_of_screen(self):
        """A plot exactly as wide as the screen still diffs correctly.

        Writing the final column leaves the terminal's cursor on that column
        with a wrap pending, not one past it, so a renderer tracking the column
        by counting glyphs is off by one from there on.
        """
        # minimal case: change the last column of row 0, then an interior cell
        # of row 1, so the second cell needs a move made from the edge.
        base = np.zeros((4, 6, 3), dtype=np.uint8)
        after = base.copy()
        after[0, 5] = (1, 2, 3)     # row 0, final column
        after[3, 3] = (4, 5, 6)     # row 1, column 3
        prev, new = unicode_image(base), unicode_image(after)
        screen = _screen_after_diff(prev, new, slack=0)
        ref = _screen_after(new.height + 3, new.width, new.to_ansi_str())
        got = screen.region(new.height, new.width)
        want = ref.region(new.height, new.width)
        assert got == want, _difference(got, want)
        assert screen.cursor == (new.height, 0)

    def test_diff_at_right_edge_of_screen_random(self):
        rng = np.random.default_rng(99)
        for _ in range(40):
            h = int(rng.integers(1, 9))
            w = int(rng.integers(2, 14))
            base = rng.integers(0, 256, (2 * h, w, 3), dtype=np.uint8)
            after = base.copy()
            mask = rng.random((2 * h, w)) < rng.uniform(0.0, 0.6)
            after[mask] = rng.integers(0, 256, (int(mask.sum()), 3), dtype=np.uint8)
            prev, new = unicode_image(base), unicode_image(after)
            screen = _screen_after_diff(prev, new, slack=0)
            ref = _screen_after(new.height + 3, new.width, new.to_ansi_str())
            got = screen.region(new.height, new.width)
            want = ref.region(new.height, new.width)
            assert got == want, f"{(h, w)}: {_difference(got, want)}"
            assert screen.cursor == (new.height, 0)

    def test_diff_at_bottom_edge_of_screen(self):
        """One spare row below the plot is enough -- the diff needs no more.

        That row is where the newline `print` appends goes, so a plot of height
        R-1 on an R-row screen animates without ever scrolling. (A plot of the
        full screen height cannot animate at all, by either path: printing H rows
        plus a newline into H rows must scroll, and the plot's top row is then
        lost off the screen.)
        """
        rng = np.random.default_rng(7)
        base = rng.integers(0, 256, (6, 9, 3), dtype=np.uint8)
        after = base.copy()
        after[rng.random((6, 9)) < 0.4] = 200
        prev, new = unicode_image(base), unicode_image(after)
        screen = _screen_after_diff(prev, new, rows_below=1)
        ref = _screen_after(new.height + 1, new.width + 2, new.to_ansi_str())
        got = screen.region(new.height, new.width)
        want = ref.region(new.height, new.width)
        assert got == want, _difference(got, want)
        assert screen.cursor == (new.height, 0)
        assert screen.scrolled == 0


# # #
# RESIZING


class TestCharArrayDiffStrResize:
    """A resize is still a diff: the overlap is compared, the rest painted or
    erased. Afterwards the screen must look exactly as a fresh render of `new`
    does -- i.e. the new plot, and blanks everywhere the old one reached."""

    SIZES = [
        ((4, 8), (4, 8), "same"),
        ((4, 8), (6, 8), "taller"),
        ((6, 8), (4, 8), "shorter"),
        ((4, 8), (4, 12), "wider"),
        ((4, 12), (4, 8), "narrower"),
        ((4, 8), (6, 12), "taller and wider"),
        ((6, 12), (4, 8), "shorter and narrower"),
        ((6, 8), (4, 12), "shorter and wider"),
        ((4, 12), (6, 8), "taller and narrower"),
        ((1, 1), (5, 9), "from a single cell"),
        ((5, 9), (1, 1), "down to a single cell"),
    ]

    @pytest.mark.parametrize("pshape,nshape,label", SIZES)
    def test_resize_matches_a_fresh_render(self, pshape, nshape, label):
        rng = np.random.default_rng(abs(hash(label)) % 2**32)
        prev = _rand_chars(rng, *pshape)
        new = _rand_chars(rng, *nshape)
        screen = _screen_after_resize(prev, new)
        UH = max(prev.height, new.height)
        UW = max(prev.width, new.width)
        ref = _screen_after(UH + 4, UW + 2, new.to_ansi_str())
        got, want = screen.region(UH, UW), ref.region(UH, UW)
        assert got == want, f"{label}: {_difference(got, want)}"
        assert screen.cursor == (new.height, 0), label
        assert screen.scrolled == 0, label

    def test_resize_random(self):
        rng = np.random.default_rng(4242)
        for _ in range(120):
            ph, pw = int(rng.integers(1, 7)), int(rng.integers(1, 11))
            nh, nw = int(rng.integers(1, 7)), int(rng.integers(1, 11))
            prev, new = _rand_chars(rng, ph, pw), _rand_chars(rng, nh, nw)
            screen = _screen_after_resize(prev, new)
            UH, UW = max(ph, nh), max(pw, nw)
            ref = _screen_after(UH + 4, UW + 2, new.to_ansi_str())
            got, want = screen.region(UH, UW), ref.region(UH, UW)
            assert got == want, f"{(ph, pw, nh, nw)}: {_difference(got, want)}"
            assert screen.cursor == (nh, 0), (ph, pw, nh, nw)

    def test_shrinking_leaves_content_below_alone(self):
        """Losing rows erases exactly those rows -- a gap, not a bulldozer."""
        rng = np.random.default_rng(5)
        prev, new = _rand_chars(rng, 6, 8), _rand_chars(rng, 3, 8)
        screen = _screen_after_resize(prev, new, footer="below")
        assert [screen.line(r) for r in range(3, 6)] == ["", "", ""]   # the gap
        assert screen.line(6) == "below"                              # untouched

    def test_growing_taller_overwrites_content_below(self):
        """The documented cost of growth: those rows have to be written into."""
        rng = np.random.default_rng(6)
        prev, new = _rand_chars(rng, 3, 8), _rand_chars(rng, 5, 8)
        screen = _screen_after_resize(prev, new, footer="below")
        assert screen.line(3) != "below"      # row 3 now belongs to the plot
        ref = _screen_after(9, 10, new.to_ansi_str())
        got, want = screen.region(5, 8), ref.region(5, 8)
        assert got == want, _difference(got, want)

    def test_growing_taller_scrolls_at_the_bottom_of_the_screen(self):
        """Appended rows use newlines, so they scroll rather than clamp.

        Cursor-down would have stopped at the bottom margin and painted the
        appended rows on top of each other.
        """
        rng = np.random.default_rng(8)
        prev, new = _rand_chars(rng, 3, 8), _rand_chars(rng, 5, 8)
        # screen fits prev plus the row its trailing newline needs, and no more
        rows = prev.height + 1
        term = Terminal(rows, new.width + 2)
        term.print(prev.to_ansi_str())
        term.print(new.to_ansi_diff_str(prev))
        screen = term.screen()
        assert screen.scrolled == 2            # once per appended row past the end
        assert screen.cursor == (rows - 1, 0)
        # the screen now shows the plot's tail: its top rows scrolled away
        ref = _screen_after(new.height, new.width + 2, new.to_ansi_str())
        visible = rows - 1                     # the bottom row is the cursor's
        offset = new.height - visible
        for r in range(visible):
            assert screen.cells[r][:new.width] == ref.cells[r + offset][:new.width]

    def test_narrowing_erases_only_the_lost_columns(self):
        """Trailing columns are blanked with spaces, which reach exactly as far
        as they are written and no further.

        And blanked to default, not to whatever colour was last in effect: a
        space carries the active background (see `TestTheHarness`).
        """
        rng = np.random.default_rng(9)
        prev, new = _rand_chars(rng, 3, 10), _rand_chars(rng, 3, 6)
        term = Terminal(8, 14)
        # mark the columns to the right of both plots, then home the cursor again
        term.feed("".join(f"\x1b[{r + 1};13H|" for r in range(3)) + "\x1b[H")
        term.print(prev.to_ansi_str())
        term.print(new.to_ansi_diff_str(prev))
        screen = term.screen()
        for r in range(3):
            assert screen.cells[r][6:12] == (BLANK,) * 6
            assert screen.cells[r][12] == ("|", None, None)


# # #
# ANIMATION LOOPS


class TestAnimationLoop:
    def test_animation_loop_is_one_uniform_print(self):
        """Replay the loop documented on `plot.__sub__`, verbatim.

        Every frame -- including the seed, where prev is None -- is the single
        statement `print(frame - prev)`, with no `end=""` anywhere.
        """
        rng = np.random.default_rng(11)
        frames = [mp.image(rng.random((6, 9))) for _ in range(5)]
        term = Terminal(frames[0].height + 3, frames[0].width + 2)

        prev = None
        screen = None
        for frame in frames:
            term.print(frame - prev)               # print(frame - prev)
            screen = term.screen()
            assert screen.cursor == (frame.height, 0)
            prev = frame

        assert screen is not None
        last = frames[-1]
        ref = _screen_after(last.height + 3, last.width + 2, last.renderstr())
        got = screen.region(last.height, last.width)
        want = ref.region(last.height, last.width)
        assert got == want, _difference(got, want)
        assert screen.scrolled == 0

    def test_animation_loop_holds_still_when_nothing_changes(self):
        """A repeated frame must not creep down the screen."""
        p = mp.image(np.random.default_rng(12).random((6, 9)))
        term = Terminal(p.height + 3, p.width + 2)
        prev = None
        for _ in range(5):
            term.print(p - prev)
            prev = p
        screen = term.screen()
        assert screen.cursor == (p.height, 0)
        assert screen.scrolled == 0

    def test_clear_and_redraw_loop_is_stable(self):
        """print(-plot) then print(new) must redraw in place, frame after frame."""
        frames = [mp.border(mp.text(f"F{n}")) for n in range(1, 5)]
        term = Terminal(10, 30)
        term.feed("\n\n")                          # start partway down
        term.print(frames[0].renderstr())
        occupied = lambda s: [r for r in range(10) if s.line(r)]
        screen = term.screen()
        rows = [occupied(screen)]
        for f in frames[1:]:
            term.print(-f)                         # print(-plot)
            term.print(f.renderstr())              # print(plot)
            screen = term.screen()
            rows.append(occupied(screen))
        assert all(r == rows[0] for r in rows), rows
        assert screen.scrolled == 0


# # #
# THE ANIMATION SESSION


def _session(term, block) -> Screen:
    """Run `block(anim)` inside an `mp.animate` session, on `term`.

    The session writes with `print`, so its output is captured and then replayed
    into the terminal one print at a time -- the same route
    `tests/examples.py` takes, and for the same reason: it keeps the session
    under test rather than a special-cased copy of it that writes somewhere else.
    """
    prints: list[str] = []
    pending: list[str] = []

    class Recorder(io.TextIOBase):
        def write(self, s: str) -> int:
            if s == "\n":
                prints.append("".join(pending))
                pending.clear()
            else:
                pending.append(s)
            return len(s)

    with contextlib.redirect_stdout(Recorder()):    # type: ignore[arg-type]
        with mp.animate() as anim:
            block(anim)
    for payload in prints:
        term.print(payload)
    return term.screen()


class TestAnimateSession:
    def test_the_session_draws_what_the_bare_loop_draws(self):
        """`anim.update(f)` must put exactly on screen what `print(f - prev)` does."""
        rng = np.random.default_rng(21)
        frames = [mp.image(rng.random((6, 9))) for _ in range(5)]
        h, w = frames[0].height, frames[0].width

        term = Terminal(h + 3, w + 2)
        screen = _session(term, lambda anim: [anim.update(f) for f in frames])
        got = screen.region(h, w)

        ref = _screen_after(h + 3, w + 2, frames[-1].renderstr())
        want = ref.region(h, w)
        assert got == want, _difference(got, want)
        assert screen.cursor == (h, 0)
        assert screen.scrolled == 0

    def test_a_printed_line_lands_above_the_plot(self):
        """The message takes the plot's first row, and the plot moves down one."""
        p = mp.border(mp.text("frame"))

        term = Terminal(12, 30)
        term.print("$ python examples/train.py")

        def block(anim):
            anim.update(p)
            anim.print("step 100: loss 0.5")

        screen = _session(term, block)

        assert screen.line(0) == "$ python examples/train.py"
        assert screen.line(1) == "step 100: loss 0.5"
        # the plot, intact, one row lower than it started
        ref = _screen_after(12, 30, p.renderstr())
        got = tuple(row[:p.width] for row in screen.cells[2:2 + p.height])
        want = ref.region(p.height, p.width)
        assert got == want, _difference(got, want)
        assert screen.cursor == (2 + p.height, 0)

    def test_printed_lines_stack_up_above_the_plot(self):
        """Each message pushes the plot down one row, with the log above it."""
        p = mp.border(mp.text("frame"))

        term = Terminal(14, 30)
        term.print("$ prompt")

        def block(anim):
            anim.update(p)
            for i in range(3):
                anim.print(f"line {i}")

        screen = _session(term, block)

        assert screen.line(0) == "$ prompt"
        assert [screen.line(r) for r in (1, 2, 3)] == ["line 0", "line 1", "line 2"]
        ref = _screen_after(14, 30, p.renderstr())
        got = tuple(row[:p.width] for row in screen.cells[4:4 + p.height])
        want = ref.region(p.height, p.width)
        assert got == want, _difference(got, want)

    def test_the_next_frame_still_diffs_after_a_printed_line(self):
        """Printing moves the plot down the screen, which must not break the diff.

        `to_ansi_diff_str` states its cursor contract relative to the cursor, not
        absolutely on the screen, so the frame on screen stays a valid
        predecessor even though it is no longer where it was drawn.
        """
        a = mp.border(mp.text("frame A"))
        b = mp.border(mp.text("frame B"))

        term = Terminal(12, 30)
        term.print("$ prompt")

        def block(anim):
            anim.update(a)
            anim.print("a message")
            anim.update(b)

        screen = _session(term, block)

        assert screen.line(1) == "a message"
        ref = _screen_after(12, 30, b.renderstr())
        got = tuple(row[:b.width] for row in screen.cells[2:2 + b.height])
        want = ref.region(b.height, b.width)
        assert got == want, _difference(got, want)

    def test_printing_from_the_top_row_costs_one_extra_row(self):
        """A plot drawn at the very top of the screen has nowhere to step above.

        `clearstr` steps a row above the plot so the newline `print` appends
        lands where the plot began. On row 0 that move clamps, so the message
        takes row 1 and the plot lands two rows down rather than one. Documented
        on `plot.clearstr`, and harmless: it happens once.
        """
        p = mp.border(mp.text("frame"))

        term = Terminal(12, 30)

        def block(anim):
            anim.update(p)
            anim.print("a message")

        screen = _session(term, block)

        assert screen.line(0) == ""
        assert screen.line(1) == "a message"
        ref = _screen_after(12, 30, p.renderstr())
        got = tuple(row[:p.width] for row in screen.cells[2:2 + p.height])
        want = ref.region(p.height, p.width)
        assert got == want, _difference(got, want)

    def test_printing_at_the_bottom_of_the_screen_scrolls_the_log_away(self):
        """With no spare rows the message scrolls off the top, plot intact."""
        p = mp.border(mp.text("frame"))
        # exactly enough for the plot plus the newline print appends
        term = Terminal(p.height + 1, 30)

        def block(anim):
            anim.update(p)
            anim.print("this will scroll away")

        screen = _session(term, block)

        ref = _screen_after(p.height + 1, 30, p.renderstr())
        got = screen.region(p.height, p.width)
        want = ref.region(p.height, p.width)
        assert got == want, _difference(got, want)
        assert screen.scrolled > 0

    def test_a_message_before_the_first_frame_is_an_ordinary_print(self):
        term = Terminal(8, 30)

        screen = _session(term, lambda anim: anim.print("nothing drawn yet"))

        assert screen.line(0) == "nothing drawn yet"
        assert screen.cursor == (1, 0)


# # #
# CLEARING


class TestPlotClearStr:
    def test_clearstr_preserves_the_row_above_the_plot(self):
        """The clear erases the plot, then steps above it -- it must not erase
        the row it steps onto, which is typically the shell's command line."""
        p = mp.border(mp.text("frame"))
        term = Terminal(10, 30)
        term.print("$ python examples/wave.py")
        term.print(p.renderstr())
        term.print(-p)                             # print(-plot)
        assert term.screen().line(0) == "$ python examples/wave.py"

    def test_clearstr_erases_only_the_plots_own_rows(self):
        """Content below the plot must survive the clear."""
        p = mp.border(mp.text("frame"))
        term = Terminal(12, 30)
        term.print("$ prompt")
        term.print(p.renderstr())
        term.feed("footer text").feed("\r")   # below the plot, back to column 0
        term.print(-p)                        # print(-plot)
        screen = term.screen()
        assert screen.line(0) == "$ prompt"   # the row stepped onto, not erased
        assert [screen.line(r) for r in range(1, 1 + p.height)] == [""] * p.height
        assert screen.line(1 + p.height) == "footer text"

    def test_clearstr_of_an_empty_plot_is_a_no_op(self):
        z = mp.blank(height=0, width=0)
        term = Terminal(6, 12)
        term.print("keep me")
        before = term.screen()
        term.print(-z)                        # print(-plot) of an empty plot
        assert term.screen() == before        # cells, cursor and scroll count
