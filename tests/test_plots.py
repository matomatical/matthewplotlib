"""Unit tests for plot construction and arrangement."""

import os

import numpy as np
import pytest

import matthewplotlib as mp

from matthewplotlib.plots import (
    axes,
    border,
    dstack2,
    line,
    line3,
    scatter,
    text,
    wrap,
)


# # #
# text


class TestTextControls:
    def test_cr_and_lf_are_line_breaks(self):
        plot = text("one\r\ntwo\nthree\rfour")

        assert plot.chars.to_plain_str() == "one  \ntwo  \nthree\nfour "

    @pytest.mark.parametrize("control", ["\0", "\t", "\x1b", "\x7f", "\x85"])
    def test_other_controls_are_rejected(self, control):
        with pytest.raises(ValueError, match=f"U\\+{ord(control):04X}"):
            text(f"before{control}after")

    def test_a_border_title_cannot_add_a_terminal_sequence(self):
        with pytest.raises(ValueError, match="U\\+001B"):
            border(text("plot"), title="\x1b[1mbold")

    def test_an_axis_label_cannot_add_a_terminal_sequence(self):
        with pytest.raises(ValueError, match="U\\+001B"):
            axes(_narrow_ticks_scatter(), xlabel="\x1b]52;c;SGVsbG8=\x07")


# # #
# wrap


class TestWrap:
    def test_automatic_columns_fall_back_without_terminal(self, monkeypatch):
        def no_terminal(_fd):
            raise OSError("no attached terminal")

        monkeypatch.delenv("COLUMNS", raising=False)
        monkeypatch.delenv("LINES", raising=False)
        monkeypatch.setattr(os, "get_terminal_size", no_terminal)

        plot = wrap(text("a"), text("b"), text("c"))

        assert plot.height == 1
        assert plot.width == 80
        assert plot.chars.to_plain_str().startswith("abc")


# # #
# axes


def _narrow_ticks_scatter():
    """A scatter whose y ticks format to a single character ("0" and "7")."""
    return scatter(
        np.array([[1.0, 1.0], [9.0, 6.0]]),
        width=20,
        height=5,
        xrange=(0, 10),
        yrange=(0, 7),
    )


def _interior_rows(plot):
    """The rows between the top and bottom rules, which the border encloses."""
    return plot.chars.to_plain_str().splitlines()[1:-2]


class TestAxesYLabelGutter:
    # The ylabel is painted at column L-1-ypad, where L is the width of the y
    # tick gutter. When the ticks are narrower than ypad+1 that index goes
    # negative and numpy wraps it onto the right-hand border, silently
    # replacing it. There is no error and no clue other than the missing
    # border, so each case below asserts the border survived.

    def test_ylabel_does_not_overwrite_right_border(self):
        plot = axes(
            _narrow_ticks_scatter(),
            ylabel="yLABEL",
            xfmt="{x:.0f}",
            yfmt="{y:.0f}",
        )

        for row in _interior_rows(plot):
            assert row.endswith("│")

    def test_absent_ylabel_does_not_blank_right_border(self):
        # An empty ylabel is centered into a field of spaces, which is truthy,
        # so it used to be painted too -- wiping the border with blanks.
        plot = axes(
            _narrow_ticks_scatter(),
            xfmt="{x:.0f}",
            yfmt="{y:.0f}",
        )

        for row in _interior_rows(plot):
            assert row.endswith("│")

    def test_ylabel_lands_in_the_gutter(self):
        plot = axes(
            _narrow_ticks_scatter(),
            ylabel="yLABEL",
            xfmt="{x:.0f}",
            yfmt="{y:.0f}",
        )

        gutter = "".join(row[0] for row in _interior_rows(plot))
        assert gutter == "yLABE"

    def test_ypad_widens_the_gutter(self):
        narrow = axes(_narrow_ticks_scatter(), ylabel="y", yfmt="{y:.0f}")
        padded = axes(
            _narrow_ticks_scatter(), ylabel="y", ypad=3, yfmt="{y:.0f}"
        )

        assert padded.width == narrow.width + 2
        for row in _interior_rows(padded):
            assert row.endswith("│")

    def test_wide_ticks_are_not_widened_further(self):
        # The gutter only grows when the ticks are too narrow to hold the
        # ylabel column; two-character ticks already are.
        labelled = axes(_narrow_ticks_scatter(), ylabel="y", yfmt="{y:2.0f}")
        plain = axes(_narrow_ticks_scatter(), yfmt="{y:2.0f}")

        assert labelled.width == plain.width


# # #
# line


def drawn_cells(plot):
    """Which character cells of a plot have anything in them."""
    return plot.chars.isnonblank()


class TestLine:
    def test_size_in_characters(self):
        plot = line((np.arange(10), np.arange(10)), width=20, height=5)
        assert plot.width == 20
        assert plot.height == 5

    def test_range_from_the_data(self):
        plot = line((np.array([1.0, 3.0]), np.array([-2.0, 6.0])))
        assert plot.xrange == (1.0, 3.0)
        assert plot.yrange == (-2.0, 6.0)

    def test_explicit_range_is_kept(self):
        plot = line(
            (np.array([1.0, 3.0]), np.array([1.0, 3.0])),
            xrange=(0.0, 10.0),
            yrange=(-5.0, 5.0),
        )
        assert plot.xrange == (0.0, 10.0)
        assert plot.yrange == (-5.0, 5.0)

    def test_a_range_is_found_for_a_constant_series(self):
        """A constant series reaches no distance, so it is given room around
        itself rather than dividing by a zero-width range."""
        plot = line((np.arange(10), np.ones(10)), width=10, height=3)
        assert plot.yrange == (0.5, 1.5)
        assert drawn_cells(plot).any()

    def test_the_ends_of_the_data_reach_the_ends_of_the_plot(self):
        plot = line((np.arange(10), np.arange(10)), width=10, height=3)
        cells = drawn_cells(plot)
        assert cells[:, 0].any(), "nothing drawn in the first column"
        assert cells[:, -1].any(), "nothing drawn in the last column"
        assert cells[0, :].any(), "nothing drawn in the top row"
        assert cells[-1, :].any(), "nothing drawn in the bottom row"

    def test_the_dots_are_connected(self):
        """Two distant points joined by a line fill the columns between them,
        which is the whole difference from a scatter of the same data."""
        data = (np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        as_line = drawn_cells(line(data, width=20, height=5))
        as_scatter = drawn_cells(scatter(data, width=20, height=5))
        assert as_line.sum() > as_scatter.sum()
        assert as_line[:, 10].any(), "the middle of the line is missing"

    def test_a_gap_breaks_the_line(self):
        xs = np.linspace(0.0, 1.0, 21)
        ys = np.zeros(21)
        whole = drawn_cells(line((xs, ys), width=21, height=2))
        ys[10] = np.nan
        broken = drawn_cells(line((xs, ys), width=21, height=2))
        assert broken.sum() < whole.sum()
        assert not broken[:, 10].any(), "the gap was drawn through"

    def test_separate_series_are_not_joined(self):
        """One series through two points draws a line between them. The same
        two points as a series each draw nothing, because a single point has
        nothing to be connected to."""
        far = (np.array([0.0]), np.array([0.0]))
        apart = (np.array([1.0]), np.array([1.0]))
        together = (np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        assert not drawn_cells(line(far, apart, width=20, height=5)).any()
        assert drawn_cells(line(together, width=20, height=5)).any()

    def test_series_are_drawn_in_a_shared_range(self):
        plot = line(
            (np.array([0.0, 1.0]), np.array([0.0, 1.0])),
            (np.array([2.0, 3.0]), np.array([2.0, 3.0])),
        )
        assert plot.xrange == (0.0, 3.0)
        assert plot.yrange == (0.0, 3.0)
        assert plot.num_strokes == 2

    def test_thickness_widens_the_line(self):
        data = (np.arange(20), np.arange(20))
        thin = drawn_cells(line(data, width=20, height=10, thickness=1.0))
        thick = drawn_cells(line(data, width=20, height=10, thickness=4.0))
        assert thick.sum() > thin.sum()

    def test_colors_come_out_along_the_line(self):
        plot = line(
            (np.array([0.0, 1.0]), np.array([0.0, 0.0]), "red"),
            width=10,
            height=1,
        )
        painted = plot.chars.fg_rgb[plot.chars.fg]
        assert len(painted)
        assert np.array_equal(
            painted,
            np.full_like(painted, np.array([255, 0, 0], dtype=np.uint8)),
        )

    def test_points_outside_an_explicit_range_are_clipped_away(self):
        plot = line(
            (np.array([-10.0, 10.0]), np.array([0.0, 0.0])),
            xrange=(0.0, 1.0),
            yrange=(-1.0, 1.0),
            width=10,
            height=3,
        )
        # the line crosses the whole view along its middle row, and the
        # parts of it beyond the range are simply not there
        cells = drawn_cells(plot)
        assert cells[1].all()
        assert cells.sum() == cells[1].sum()

    def test_a_series_with_no_points_at_all(self):
        plot = line((np.array([]), np.array([])), width=8, height=2)
        assert plot.width == 8
        assert not drawn_cells(plot).any()

    def test_takes_an_axis_series(self):
        plot = line(mp.xaxis(0, 1, 10), width=10, height=3)
        assert drawn_cells(plot).any()

    def test_repr_reports_the_data(self):
        plot = line((np.arange(5), np.arange(5)), width=8, height=2)
        assert repr(plot) == (
            "line(height=2, width=8, thickness=1.0, "
            "data=<4 segments, 1 strokes on [0.00,4.00]x[0.00,4.00]>)"
        )

    def test_fits_inside_axes(self):
        plot = axes(line((np.arange(5), np.arange(5)), width=8, height=2))
        # the border adds a row above and below plus a row of x labels, and a
        # column each side plus a gutter as wide as the y labels ("0.0")
        assert plot.height == 2 + 3
        assert plot.width == 8 + 2 + 3

    def test_covers_the_scatter_of_the_same_points(self):
        """Both map the data onto the dots the same way, so a line passes
        through every dot a scatter of its points marks."""
        data = (np.array([0.0, 0.4, 1.0]), np.array([0.0, 0.9, 0.3]))
        as_scatter = drawn_cells(scatter(data, width=20, height=6))
        as_line = drawn_cells(line(data, width=20, height=6))
        assert not (as_scatter & ~as_line).any()

    def test_layers_with_a_scatter_of_the_same_range(self):
        data = (np.arange(5), np.arange(5))
        plot = dstack2(
            scatter(data, width=10, height=3),
            line(data, width=10, height=3),
        )
        assert plot.xrange == (0, 4)
        assert drawn_cells(plot).any()


# # #
# line3


class TestLine3:
    def test_size_in_characters(self):
        wire = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        plot = line3(wire, width=24, height=6)
        assert plot.width == 24
        assert plot.height == 6

    def test_a_wire_across_the_view_is_drawn(self):
        wire = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
        plot = line3(wire, width=20, height=5)
        assert drawn_cells(plot).any()
        assert plot.num_segments == 1

    def test_a_3d_scatter_is_not_a_scatter(self):
        """Its coordinates are projected, so `axes` would label them as though
        they were the data's own."""
        from matthewplotlib.plots import scatter3
        cloud = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
        assert not isinstance(scatter3(cloud, width=8, height=4), scatter)

    def test_a_wire_behind_the_camera_is_not_drawn(self):
        wire = np.array([[0.0, 0.0, 3.0], [0.5, 0.0, 5.0]])
        plot = line3(wire, width=20, height=5)
        assert not drawn_cells(plot).any()
        assert plot.num_segments == 0

    def test_a_wire_reaching_behind_the_camera_is_still_drawn(self):
        wire = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 4.0]])
        plot = line3(wire, width=20, height=5)
        assert plot.num_segments == 1
        assert drawn_cells(plot).any()

    def test_a_gap_separates_two_wires(self):
        gap = np.array([np.nan, np.nan, np.nan])
        wires = np.array([
            [-0.5, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            gap,
            [-0.5, -0.5, 0.0],
            [0.5, -0.5, 0.0],
        ])
        plot = line3(wires, width=20, height=5)
        assert plot.num_segments == 2

    def test_thickness_widens_the_wire(self):
        wire = np.array([[-0.5, -0.5, 0.0], [0.5, 0.5, 0.0]])
        thin = drawn_cells(line3(wire, width=20, height=5, thickness=1.0))
        thick = drawn_cells(line3(wire, width=20, height=5, thickness=4.0))
        assert thick.sum() > thin.sum()

    def test_separate_series_are_not_joined(self):
        here = np.array([[-0.5, 0.5, 0.0]])
        there = np.array([[0.5, -0.5, 0.0]])
        plot = line3(here, there, width=20, height=5)
        assert plot.num_segments == 0
        assert not drawn_cells(plot).any()

    def test_a_wire_with_no_points_at_all(self):
        plot = line3(np.zeros((0, 3)), width=8, height=2)
        assert plot.num_segments == 0
        assert not drawn_cells(plot).any()

    def test_repr_reports_the_segments(self):
        wire = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
        plot = line3(wire, width=8, height=2)
        assert repr(plot) == (
            "line3(height=2, width=8, thickness=1.0, "
            "data=<1 segments drawn>)"
        )
