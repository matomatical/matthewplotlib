"""Unit tests for animations: the `tstack` value and the `animate` session.

What `animate`'s escape sequences do to a real terminal is tested in
`tests/test_terminal.py`. These tests are about the strings, the frame
book-keeping and the clock.
"""

import io
import os
import time
import warnings
import contextlib

import numpy as np
import pytest
from PIL import Image

from matthewplotlib.animations import animate, tstack, _terminal_rows
from matthewplotlib.plots import blank, border, image, plot, text


# # #
# HELPERS


class Writes(io.TextIOBase):
    """A stdout that keeps each `print` separately, without its newline.

    `print(x)` writes the payload and then the newline as two separate calls, so
    a write of exactly "\\n" ends a print. Same trick as the `Recorder` in
    `tests/examples.py`, kept separate so these tests do not depend on the
    example harness.
    """

    def __init__(self, fd: int | None = None) -> None:
        self.prints: list[str] = []
        self._pending: list[str] = []
        self._fd = fd

    def write(self, s: str) -> int:
        if s == "\n":
            self.prints.append("".join(self._pending))
            self._pending = []
        else:
            self._pending.append(s)
        return len(s)

    def fileno(self) -> int:
        # io.IOBase.fileno raises for objects with no descriptor, which is what
        # `_terminal_rows` reads as "not a terminal". Tests that want the height
        # check to engage pass one.
        if self._fd is None:
            return super().fileno()
        return self._fd


class Clock:
    """A stand-in for the parts of `time` that `animate` uses.

    Sleeping advances the clock instead of the wall, so the pacing tests assert
    exact numbers and cost nothing to run. `work` is the same thing from the
    caller's side: time passing while a frame is computed.
    """

    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def perf_counter(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds

    def work(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(time, "perf_counter", c.perf_counter)
    monkeypatch.setattr(time, "sleep", c.sleep)
    return c


def frames(n: int, height: int = 2, width: int = 4) -> list[plot]:
    """`n` distinguishable frames of a fixed size."""
    return [
        image(np.full((2 * height, width), i / n), colormap=None)
        for i in range(n)
    ]


@contextlib.contextmanager
def capture(fd: int | None = None):
    """Run a block with stdout replaced by a `Writes`."""
    writes = Writes(fd=fd)
    with contextlib.redirect_stdout(writes):    # type: ignore[arg-type]
        yield writes


def interrupt_on_sleep(n: int):
    """A `time.sleep` that raises KeyboardInterrupt on its nth call.

    Where a Ctrl-C usually lands: an animation waiting out a frame's budget is
    what the keypress interrupts, not the compute between frames.
    """
    calls = 0

    def sleep(_seconds: float) -> None:
        nonlocal calls
        calls += 1
        if calls == n:
            raise KeyboardInterrupt

    return sleep


# # #
# tstack: CONSTRUCTION


class TestTStackConstruction:
    def test_frames_are_kept_in_order(self):
        ps = frames(4)
        a = tstack(*ps)

        assert len(a) == 4
        assert list(a) == ps
        assert a[2] is ps[2]

    def test_size_is_the_largest_frame(self):
        a = tstack(blank(height=2, width=9), blank(height=5, width=3))

        assert a.height == 5
        assert a.width == 9

    def test_empty_is_allowed_and_measures_zero(self):
        # so that slicing is total: a[5:5] has to produce something
        a = tstack()

        assert len(a) == 0
        assert (a.height, a.width) == (0, 0)
        assert list(a) == []

    def test_nested_animations_are_spliced_in(self):
        a = tstack(*frames(3))
        b = tstack(*frames(2))

        both = tstack(a, b)

        assert len(both) == 5
        assert list(both) == list(a) + list(b)

    def test_a_still_frame_can_be_appended(self):
        a = tstack(*frames(3))
        end = text("done")

        assert list(tstack(a, end)) == list(a) + [end]

    def test_frame_rate_defaults_to_twelve(self):
        assert tstack(*frames(2)).fps == 12.0

    def test_frame_rate_is_inherited_from_the_first_animation(self):
        # otherwise concatenating two 20fps animations quietly gives 12fps
        fast = tstack(*frames(2), fps=20)
        slow = tstack(*frames(2), fps=3)

        assert tstack(fast, slow).fps == 20
        assert tstack(slow, fast).fps == 3
        assert tstack(fast, slow, fps=7).fps == 7

    def test_a_frame_rate_must_be_positive(self):
        with pytest.raises(ValueError, match="fps must be positive"):
            tstack(*frames(2), fps=0)

    def test_durations_must_match_the_frame_count(self):
        with pytest.raises(ValueError, match="2 durations for 3 frames"):
            tstack(*frames(3), durations=[10.0, 20.0])

    def test_repr_omits_the_frames(self):
        # an animation is routinely hundreds of plots long and each of those
        # reprs its own children in turn
        r = repr(tstack(*frames(200), fps=25))

        assert r == "tstack(frames=200, height=2, width=4, fps=25)"


# # #
# tstack: INDEXING, SLICING AND MAPPING


class TestTStackSlicing:
    def test_an_integer_gives_one_frame(self):
        ps = frames(4)

        assert tstack(*ps)[1] is ps[1]
        assert tstack(*ps)[-1] is ps[-1]

    def test_a_slice_gives_an_animation(self):
        ps = frames(6)

        half = tstack(*ps)[3:]

        assert isinstance(half, tstack)
        assert list(half) == ps[3:]

    def test_a_slice_can_reverse(self):
        ps = frames(4)

        assert list(tstack(*ps)[::-1]) == ps[::-1]

    def test_a_slice_keeps_the_frame_rate(self):
        assert tstack(*frames(6), fps=30)[::2].fps == 30

    def test_a_slice_takes_the_matching_durations(self):
        a = tstack(*frames(4), durations=[10.0, 20.0, 30.0, 40.0])

        assert a[1:3].durations == (20.0, 30.0)
        assert a[::-1].durations == (40.0, 30.0, 20.0, 10.0)

    def test_slicing_an_animation_without_durations_keeps_none(self):
        assert tstack(*frames(4))[1:].durations is None


class TestTStackMap:
    def test_map_applies_to_every_frame(self):
        a = tstack(*frames(3, height=2, width=4))

        bordered = a.map(lambda p: border(p))

        assert len(bordered) == 3
        assert bordered.height == a.height + 2
        assert bordered.width == a.width + 2

    def test_map_keeps_the_rate_and_the_durations(self):
        # neither depends on what the frames contain
        a = tstack(*frames(2), fps=25, durations=[40.0, 40.0])

        mapped = a.map(lambda p: border(p))

        assert mapped.fps == 25
        assert mapped.durations == (40.0, 40.0)

    def test_map_does_not_mutate_the_original(self):
        a = tstack(*frames(2))
        before = list(a)

        a.map(lambda p: border(p))

        assert list(a) == before


# # #
# tstack: PLAYING


class TestPlay:
    def test_every_frame_is_written_once(self):
        ps = frames(4)

        with capture() as out:
            tstack(*ps).play()

        assert len(out.prints) == 4
        assert out.prints[0] == ps[0].renderstr()
        assert out.prints[1] == ps[1] - ps[0]

    def test_playing_goes_through_the_same_path_as_a_live_loop(self):
        # pull is implemented as push, so the bytes have to agree exactly
        ps = frames(4)

        with capture() as pulled:
            tstack(*ps).play()
        with capture() as pushed, animate() as anim:
            for p in ps:
                anim.update(p)

        assert pulled.prints == pushed.prints

    def test_the_animations_own_rate_paces_it(self, clock):
        with capture():
            tstack(*frames(3), fps=25).play()

        assert clock.slept == [pytest.approx(0.040), pytest.approx(0.040)]

    def test_an_explicit_rate_overrides_it(self, clock):
        with capture():
            tstack(*frames(2), fps=25).play(fps=10)

        assert clock.slept == [pytest.approx(0.100)]

    def test_looping_repeats_until_interrupted(self, monkeypatch):
        # the only way out of loop=True, which is why play catches it by default
        ps = frames(2)

        monkeypatch.setattr(time, "sleep", interrupt_on_sleep(3))

        with capture() as out:
            tstack(*ps, fps=100).play(loop=True)

        # the first frame does not wait, so the third sleep falls in the second
        # round: the third print being the first frame again is the loop going
        # round rather than returning after one pass
        assert out.prints[:3] == [
            ps[0].renderstr(), ps[1] - ps[0], ps[0] - ps[1],
        ]
        assert out.prints[-1] == ""     # the blank line __exit__ leaves behind

    def test_an_interrupt_can_be_allowed_out(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", interrupt_on_sleep(1))

        with pytest.raises(KeyboardInterrupt):
            with capture():
                tstack(*frames(3), fps=100).play(stop_on_interrupt=False)

    def test_playing_nothing_is_not_an_error(self):
        # and in particular loop=True must not spin forever on no frames
        with capture() as out:
            tstack().play(loop=True)

        assert out.prints == []


# # #
# tstack: SAVING A GIF


def durations_of(path) -> list[int]:
    """The frame delays a gif file actually carries, in milliseconds."""
    gif = Image.open(path)
    out = []
    for i in range(gif.n_frames):
        gif.seek(i)
        out.append(gif.info["duration"])
    return out


class TestSaveGif:
    def test_every_frame_is_written(self, tmp_path):
        out = tmp_path / "a.gif"

        tstack(*frames(5)).savegif(str(out))

        assert Image.open(out).n_frames == 5

    def test_the_delay_comes_from_the_frame_rate(self, tmp_path):
        out = tmp_path / "a.gif"

        tstack(*frames(3), fps=20).savegif(str(out))

        assert durations_of(out) == [50, 50, 50]

    def test_an_explicit_rate_overrides_the_animation(self, tmp_path):
        out = tmp_path / "a.gif"

        tstack(*frames(3), fps=20).savegif(str(out), fps=10)

        assert durations_of(out) == [100, 100, 100]

    def test_achieved_uses_the_recorded_durations(self, tmp_path):
        out = tmp_path / "a.gif"
        a = tstack(*frames(3), fps=20, durations=[100.0, 250.0, 250.0])

        a.savegif(str(out), fps="achieved")

        assert durations_of(out) == [100, 250, 250]

    def test_delays_are_clamped_to_what_a_gif_can_express(self, tmp_path):
        # a gif counts delays in hundredths of a second, so anything under 10ms
        # rounds to zero, which viewers read as unspecified rather than as fast
        out = tmp_path / "a.gif"

        tstack(*frames(2), fps=500).savegif(str(out))

        assert durations_of(out) == [10, 10]

    def test_achieved_needs_a_recording(self, tmp_path):
        a = tstack(*frames(3), fps=20)

        with pytest.raises(ValueError, match="needs the frame timings"):
            a.savegif(str(tmp_path / "a.gif"), fps="achieved")

    def test_an_unknown_string_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="number or 'achieved'"):
            tstack(*frames(2)).savegif(
                str(tmp_path / "a.gif"), fps="fast",     # type: ignore[arg-type]
            )

    def test_an_empty_animation_cannot_be_saved(self, tmp_path):
        with pytest.raises(ValueError, match="no frames"):
            tstack().savegif(str(tmp_path / "a.gif"))

    def test_frames_of_different_sizes_are_padded_to_the_largest(self, tmp_path):
        out = tmp_path / "a.gif"

        tstack(blank(height=1, width=2), blank(height=3, width=5)).savegif(
            str(out)
        )

        gif = Image.open(out)
        assert gif.size == (5 * 8, 3 * 16)   # the glyph cell is 8x16 pixels

    def test_repeat_controls_the_loop_flag(self, tmp_path):
        looping = tmp_path / "loop.gif"
        once = tmp_path / "once.gif"

        tstack(*frames(2)).savegif(str(looping), repeat=True)
        tstack(*frames(2)).savegif(str(once), repeat=False)

        assert Image.open(looping).info.get("loop") == 0
        assert "loop" not in Image.open(once).info


# # #
# animate: WRITING FRAMES


class TestUpdate:
    def test_the_first_frame_is_a_full_render(self):
        ps = frames(2)

        with capture() as out, animate() as anim:
            written = anim.update(ps[0])

        assert written == ps[0].renderstr()
        assert out.prints == [written]

    def test_later_frames_are_diffs_against_their_predecessor(self):
        ps = frames(3)

        with capture(), animate() as anim:
            anim.update(ps[0])
            second = anim.update(ps[1])
            third = anim.update(ps[2])

        assert second == ps[1] - ps[0]
        assert third == ps[2] - ps[1]

    def test_a_frame_is_written_in_exactly_one_print(self):
        # the example snapshots tell frames apart by counting prints, and a
        # frame split over two would land a stray newline inside the plot
        ps = frames(4)

        with capture() as out, animate() as anim:
            for p in ps:
                anim.update(p)

        assert len(out.prints) == 4

    def test_update_returns_what_it_wrote(self):
        # examples/life.py plots len() of this to show what a redraw costs
        ps = frames(3)

        with capture() as out, animate() as anim:
            written = [anim.update(p) for p in ps]

        assert written == out.prints

    def test_a_session_can_be_reused_as_its_own_previous_frame(self):
        # an unchanged frame still writes something: a bare cursor move, so the
        # newline print appends lands where the contract promises
        p = frames(1)[0]

        with capture(), animate() as anim:
            anim.update(p)
            again = anim.update(p)

        assert again == p - p
        assert again != ""


# # #
# animate: RECORDING


class TestRecording:
    def test_frames_are_refused_without_record(self):
        with capture(), animate() as anim:
            anim.update(frames(1)[0])

        with pytest.raises(ValueError, match="kept no frames"):
            anim.frames

    def test_recorded_frames_come_back_as_an_animation(self):
        ps = frames(4)

        with capture(), animate(record=True) as anim:
            for p in ps:
                anim.update(p)

        assert isinstance(anim.frames, tstack)
        assert list(anim.frames) == ps

    def test_the_recording_carries_the_requested_rate(self):
        # so that anim.frames.savegif() defaults to the intended speed
        with capture(), animate(fps=20, record=True) as anim:
            for p in frames(2):
                anim.update(p)

        assert anim.frames.fps == 20

    def test_a_recording_without_a_rate_falls_back_to_the_default(self):
        with capture(), animate(record=True) as anim:
            for p in frames(2):
                anim.update(p)

        assert anim.frames.fps == 12.0

    def test_durations_are_the_gaps_between_writes(self, clock):
        with capture(), animate(record=True) as anim:
            anim.update(frames(1)[0])
            clock.work(0.100)
            anim.update(frames(1)[0])
            clock.work(0.250)
            anim.update(frames(1)[0])

        # the last frame was never replaced, so how long it was up is unknowable
        # and it inherits the gap before it
        assert anim.frames.durations == pytest.approx((100.0, 250.0, 250.0))

    def test_a_single_frame_has_no_measurable_duration(self):
        with capture(), animate(record=True) as anim:
            anim.update(frames(1)[0])

        assert anim.frames.durations is None

    def test_a_recording_round_trips_to_an_honest_gif(self, clock, tmp_path):
        out = tmp_path / "a.gif"

        with capture(), animate(fps=20, record=True) as anim:
            for p in frames(3):
                anim.update(p)
                clock.work(0.200)       # five times slower than requested

        anim.frames.savegif(str(out))
        assert durations_of(out) == [50, 50, 50]            # as requested
        anim.frames.savegif(str(out), fps="achieved")
        assert durations_of(out) == [200, 200, 200]         # as achieved


# # #
# animate: THE CLOCK


class TestPacing:
    def test_without_a_rate_nothing_sleeps(self, clock):
        with capture(), animate() as anim:
            for p in frames(5):
                anim.update(p)

        assert clock.slept == []

    def test_the_first_frame_is_not_delayed(self, clock):
        with capture(), animate(fps=10) as anim:
            anim.update(frames(1)[0])

        assert clock.slept == []

    def test_later_frames_wait_out_the_remaining_budget(self, clock):
        with capture(), animate(fps=10) as anim:      # a 100ms budget
            anim.update(frames(1)[0])
            clock.work(0.030)                         # 30ms computing
            anim.update(frames(1)[0])

        assert clock.slept == [pytest.approx(0.070)]

    def test_the_sleep_comes_before_the_write(self, clock):
        # the caller's compute has to count towards the frame's budget rather
        # than being added to it: a flat sleep(1/fps) after each write is what
        # four of the six animated examples used to do, and it runs slow by
        # exactly the compute time
        with capture(), animate(fps=10) as anim:
            start = clock.now
            for _ in range(4):
                clock.work(0.030)
                anim.update(frames(1)[0])

        assert clock.now - start == pytest.approx(0.330)   # 30 + 3 * 100
        assert anim.achieved_fps == pytest.approx(10.0)

    def test_an_overrun_does_not_sleep(self, clock):
        with capture(), animate(fps=10) as anim:
            anim.update(frames(1)[0])
            clock.work(0.250)                         # 150ms over budget
            anim.update(frames(1)[0])

        assert clock.slept == []

    def test_an_overrun_is_not_paid_back_by_the_next_frame(self, clock):
        # the deadline picks up from now rather than trying to catch up, so one
        # slow frame does not turn into a burst of fast ones
        with capture(), animate(fps=10) as anim:
            anim.update(frames(1)[0])
            clock.work(0.250)
            anim.update(frames(1)[0])
            clock.work(0.010)
            anim.update(frames(1)[0])

        assert clock.slept == [pytest.approx(0.090)]

    def test_drift_does_not_accumulate(self, clock):
        # each frame is scheduled from the last deadline, not from when the last
        # write happened, so a millisecond of overshoot per frame does not add up
        with capture(), animate(fps=10) as anim:
            start = clock.now
            for _ in range(11):
                clock.work(0.001)
                anim.update(frames(1)[0])

        assert clock.now - start == pytest.approx(1.001)   # 1ms + 10 * 100ms

    def test_a_rate_must_be_positive(self):
        with pytest.raises(ValueError, match="fps must be positive"):
            animate(fps=0)

    def test_the_clock_really_sleeps(self):
        # the tests above run on a fake clock; this one checks that the thing
        # being faked is really called
        start = time.perf_counter()
        with capture(), animate(fps=50) as anim:
            for p in frames(3):
                anim.update(p)
        elapsed = time.perf_counter() - start

        assert elapsed >= 2 * 0.020 * 0.9


class TestAchievedFps:
    def test_unknown_before_the_second_frame(self):
        with capture(), animate() as anim:
            assert anim.achieved_fps is None
            anim.update(frames(1)[0])
            assert anim.achieved_fps is None

    def test_measured_from_the_gaps_between_writes(self, clock):
        with capture(), animate() as anim:
            anim.update(frames(1)[0])
            clock.work(0.100)
            anim.update(frames(1)[0])
            clock.work(0.100)
            anim.update(frames(1)[0])

        assert anim.achieved_fps == pytest.approx(10.0)

    def test_reports_the_rate_achieved_not_the_one_asked_for(self, clock):
        with capture(), animate(fps=100) as anim:
            for _ in range(3):
                clock.work(0.200)       # far too slow to keep up
                anim.update(frames(1)[0])

        assert anim.achieved_fps == pytest.approx(5.0)


# # #
# animate: PRINTING FROM INSIDE AN ANIMATION


class TestPrint:
    def test_a_message_is_three_prints(self):
        # erase the plot, put the message on its top row, redraw a row lower
        p = frames(1)[0]

        with capture() as out, animate() as anim:
            anim.update(p)
            anim.print("hello")

        assert out.prints[1:] == [p.clearstr(), "hello", p.renderstr()]

    def test_the_message_is_formatted_like_a_builtin_print(self):
        p = frames(1)[0]

        with capture() as out, animate() as anim:
            anim.update(p)
            anim.print("step", 7, sep="=")

        assert out.prints[2] == "step=7"

    def test_before_the_first_frame_it_is_just_a_print(self):
        with capture() as out, animate() as anim:
            anim.print("nothing on screen yet")

        assert out.prints == ["nothing on screen yet"]

    def test_the_next_frame_still_diffs_against_the_last_one(self):
        # the plot moved down the screen, but to_ansi_diff_str states its cursor
        # contract relative to the cursor rather than absolutely, so the frame
        # on screen is still a valid predecessor
        ps = frames(2)

        with capture() as out, animate() as anim:
            anim.update(ps[0])
            anim.print("a message")
            after = anim.update(ps[1])

        assert after == ps[1] - ps[0]
        assert out.prints[-1] == after


# # #
# animate: LEAVING THE BLOCK


class TestExit:
    def test_a_clean_exit_writes_nothing_extra(self):
        # a frame ends with the newline print appends, so the cursor is already
        # on a fresh line below the plot
        with capture() as out, animate() as anim:
            anim.update(frames(1)[0])

        assert len(out.prints) == 1

    def test_an_exception_is_separated_from_the_plot(self):
        with pytest.raises(RuntimeError):
            with capture() as out, animate() as anim:
                anim.update(frames(1)[0])
                raise RuntimeError("boom")

        assert out.prints[-1] == ""

    def test_nothing_is_tidied_if_no_frame_was_drawn(self):
        with pytest.raises(RuntimeError):
            with capture() as out, animate():
                raise RuntimeError("boom")

        assert out.prints == []

    def test_an_interrupt_propagates_by_default(self):
        with pytest.raises(KeyboardInterrupt):
            with capture(), animate() as anim:
                anim.update(frames(1)[0])
                raise KeyboardInterrupt

    def test_stop_on_interrupt_ends_the_animation_quietly(self):
        with capture(), animate(stop_on_interrupt=True) as anim:
            anim.update(frames(1)[0])
            raise KeyboardInterrupt

        # the statements after the block still run, which is what lets an
        # interrupted recording still be saved
        assert anim.achieved_fps is None

    def test_stop_on_interrupt_does_not_swallow_other_exceptions(self):
        with pytest.raises(RuntimeError):
            with capture(), animate(stop_on_interrupt=True) as anim:
                anim.update(frames(1)[0])
                raise RuntimeError("boom")


# # #
# animate: THE HEIGHT CHECK


class TestHeightCheck:
    def test_no_terminal_means_no_warning(self, monkeypatch):
        def no_terminal(_fd):
            raise OSError("no attached terminal")

        monkeypatch.setattr(os, "get_terminal_size", no_terminal)

        with warnings_are_errors(), capture(fd=1), animate() as anim:
            anim.update(blank(height=100, width=4))

    def test_a_plot_that_fits_is_not_warned_about(self, monkeypatch):
        monkeypatch.setattr(os, "get_terminal_size", terminal_of(rows=25))

        with warnings_are_errors(), capture(fd=1), animate() as anim:
            anim.update(blank(height=24, width=4))

    def test_a_plot_too_tall_to_animate_warns(self, monkeypatch):
        # 24 rows of plot in a 24 row terminal leaves nowhere for the newline
        monkeypatch.setattr(os, "get_terminal_size", terminal_of(rows=24))

        with capture(fd=1):
            with pytest.warns(UserWarning, match="24 rows tall"):
                with animate() as anim:
                    anim.update(blank(height=24, width=4))

    def test_the_warning_names_what_will_render(self, monkeypatch):
        monkeypatch.setattr(os, "get_terminal_size", terminal_of(rows=10))

        with capture(fd=1):
            with pytest.warns(UserWarning, match="at most 9 rows will render"):
                with animate() as anim:
                    anim.update(blank(height=40, width=4))

    def test_it_warns_once_and_not_once_a_frame(self, monkeypatch):
        monkeypatch.setattr(os, "get_terminal_size", terminal_of(rows=10))

        with capture(fd=1):
            with pytest.warns(UserWarning) as caught:
                with animate() as anim:
                    for _ in range(20):
                        anim.update(blank(height=40, width=4))

        assert len(caught) == 1


def terminal_of(rows: int):
    """A fake `os.get_terminal_size` reporting a terminal of a given height."""
    return lambda _fd: os.terminal_size((80, rows))


@contextlib.contextmanager
def warnings_are_errors():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


# # #
# _terminal_rows


class TestTerminalRows:
    def test_reads_the_height_from_the_descriptor(self, monkeypatch):
        monkeypatch.setattr(os, "get_terminal_size", terminal_of(rows=42))

        with capture(fd=1):
            assert _terminal_rows() == 42

    def test_a_redirected_stdout_has_no_height(self):
        # deliberately not a fallback: `shutil.get_terminal_size` would answer
        # 24 here, and a fallback must not be mistaken for a measurement
        with capture():
            assert _terminal_rows() is None

    def test_a_pipe_has_no_height(self, monkeypatch):
        def no_terminal(_fd):
            raise OSError("not a terminal")

        monkeypatch.setattr(os, "get_terminal_size", no_terminal)

        with capture(fd=1):
            assert _terminal_rows() is None
