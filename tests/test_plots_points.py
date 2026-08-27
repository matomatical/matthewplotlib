"""Unit tests for the braille line plots, flat and projected."""

import numpy as np
import pytest

import matthewplotlib as mp

from matthewplotlib.plots import (
    axes,
    dstack2,
    image,
    line,
    line3,
    scatter,
)

from tests.plots_common import drawn_cells


# # #
# line


class TestLine:
    def test_size_in_characters(self):
        plot = line((np.arange(10), np.arange(10)), width=20, height=5)
        assert plot.width == 20
        assert plot.height == 5

    def test_range_from_the_data(self):
        plot = line((np.array([1.0, 3.0]), np.array([-2.0, 6.0])))
        assert plot.window.xrange == (1.0, 3.0)
        assert plot.window.yrange == (-2.0, 6.0)

    def test_explicit_range_is_kept(self):
        plot = line(
            (np.array([1.0, 3.0]), np.array([1.0, 3.0])),
            xrange=(0.0, 10.0),
            yrange=(-5.0, 5.0),
        )
        assert plot.window.xrange == (0.0, 10.0)
        assert plot.window.yrange == (-5.0, 5.0)

    def test_a_range_is_found_for_a_constant_series(self):
        """A constant series reaches no distance, so it is given room around
        itself rather than dividing by a zero-width range."""
        plot = line((np.arange(10), np.ones(10)), width=10, height=3)
        assert plot.window.yrange == (0.5, 1.5)
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
        assert plot.window.xrange == (0.0, 3.0)
        assert plot.window.yrange == (0.0, 3.0)
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
            "line(<4 segments, 1 strokes>, thickness=1.0, "
            "window(x=[0.00,4.00], y=[0.00,4.00], 8x2 cells))"
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
        assert plot.window.xrange == (0, 4)
        assert drawn_cells(plot).any()

    def test_layers_must_cover_the_same_data(self):
        with pytest.raises(ValueError, match="cannot overlay"):
            dstack2(
                scatter((np.arange(5), np.arange(5)), width=10, height=3),
                scatter((np.arange(5), np.arange(5)), width=10, height=3,
                        yrange=(0.0, 10.0)),
            )

    def test_layers_must_cover_it_in_the_same_number_of_cells(self):
        """Equal ranges rendered at different sizes put a coordinate in
        different cells, and a rendered plot cannot be resampled."""
        data = (np.arange(5), np.arange(5))
        with pytest.raises(ValueError, match="cannot overlay"):
            dstack2(
                scatter(data, width=10, height=3),
                scatter(data, width=20, height=3),
            )

    def test_a_plot_without_coordinates_cannot_be_overlaid(self):
        with pytest.raises(ValueError, match="no coordinates"):
            dstack2(image(np.zeros((4, 4))))

    def test_there_must_be_something_to_overlay(self):
        with pytest.raises(ValueError, match="no plots"):
            dstack2()


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
