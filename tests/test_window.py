import numpy as np
import pytest

from matthewplotlib.window import window


# # #
# dots


class TestDots:
    def test_the_limits_land_on_the_centres_of_the_outermost_dots(self):
        """A point at the extreme of its data should be drawn, rather than
        landing on the boundary between the plot and the outside."""
        w = window(xrange=(0.0, 1.0), yrange=(0.0, 1.0), width=7, height=5)
        corners = np.array([[0.0, 1.0], [1.0, 0.0]])

        assert np.allclose(w.dots(corners), [[0.5, 0.5], [19.5, 13.5]])

    def test_a_dots_integer_part_selects_it(self):
        w = window(xrange=(0.0, 1.0), yrange=(0.0, 1.0), width=2, height=1)
        dots = w.dots(np.array([[0.0, 1.0]]))

        assert np.floor(dots).tolist() == [[0.0, 0.0]]

    def test_reversing_a_range_mirrors_the_dots(self):
        """col -> 2*width - col, which is the exact mirror index in a grid of
        2*width dot columns."""
        points = np.array([[0.2, 0.3], [0.9, 0.1], [0.5, 0.5]])
        forward = window(xrange=(0.0, 1.0), yrange=(0.0, 1.0), width=7, height=5)
        reverse = window(xrange=(1.0, 0.0), yrange=(0.0, 1.0), width=7, height=5)

        assert np.allclose(
            reverse.dots(points)[:, 1], 2 * 7 - forward.dots(points)[:, 1]
        )

    def test_placing_points_needs_both_coordinates(self):
        w = window(xrange=None, yrange=(0.0, 1.0), width=4, height=4)

        with pytest.raises(ValueError, match="no coordinates"):
            w.dots(np.zeros((1, 2)))


# # #
# the grid


class TestPixelGrid:
    def test_the_edges_tile_the_range(self):
        w = window(xrange=(0.0, 1.0), yrange=(0.0, 1.0), width=4, height=1)
        xedges, yedges = w.pixel_edges()

        assert np.allclose(xedges, [0.0, 0.25, 0.5, 0.75, 1.0])
        assert np.allclose(yedges, [0.0, 0.5, 1.0])

    def test_the_centres_are_the_midpoints_of_the_edges(self):
        w = window(xrange=(0.0, 1.0), yrange=(0.0, 1.0), width=4, height=1)
        X, _Y = w.pixel_centres()

        assert np.allclose(X[0], [0.125, 0.375, 0.625, 0.875])

    def test_row_zero_is_the_top_of_the_window(self):
        w = window(xrange=(0.0, 1.0), yrange=(0.0, 1.0), width=2, height=1)
        _X, Y = w.pixel_centres()

        assert Y[0, 0] > Y[-1, 0]

    def test_the_edges_run_in_screen_order(self):
        """From the left edge of the window to the right, which descends
        numerically wherever the range does."""
        w = window(xrange=(1.0, 0.0), yrange=(0.0, 1.0), width=4, height=1)
        xedges, _yedges = w.pixel_edges()

        assert np.allclose(xedges, [1.0, 0.75, 0.5, 0.25, 0.0])

    def test_reversing_a_range_mirrors_the_centres(self):
        forward = window(xrange=(0.0, 1.0), yrange=(0.0, 1.0), width=4, height=1)
        reverse = window(xrange=(1.0, 0.0), yrange=(0.0, 1.0), width=4, height=1)

        assert np.allclose(reverse.pixel_centres()[0], forward.pixel_centres()[0][:, ::-1])

    def test_laying_out_a_grid_needs_both_coordinates(self):
        w = window(xrange=(0.0, 1.0), yrange=None, width=4, height=4)

        with pytest.raises(ValueError, match="no coordinates"):
            w.pixel_edges()


# # #
# the value itself


class TestWindow:
    def test_a_window_reports_the_coordinates_it_carries(self):
        both = window(xrange=(0.0, 1.0), yrange=(-2.0, 3.0), width=7, height=5)
        one = window(xrange=None, yrange=(0.0, 1.0), width=1, height=12)
        neither = window(xrange=None, yrange=None, width=14, height=7)

        assert repr(both) == "window(x=[0.00,1.00], y=[-2.00,3.00], 7x5 cells)"
        assert repr(one) == "window(y=[0.00,1.00], 1x12 cells)"
        assert repr(neither) == "window(14x7 cells)"

    def test_a_window_must_cover_at_least_one_cell(self):
        with pytest.raises(ValueError, match="at least one cell"):
            window(xrange=(0.0, 1.0), yrange=(0.0, 1.0), width=0, height=4)

    def test_a_range_must_cover_an_interval(self):
        """Otherwise every coordinate maps to the same place, by dividing by a
        zero-width range."""
        with pytest.raises(ValueError, match="covers no interval"):
            window(xrange=(1.0, 1.0), yrange=(0.0, 1.0), width=4, height=4)

    def test_windows_of_the_same_shape_are_equal(self):
        one = window(xrange=(0.0, 1.0), yrange=(0.0, 1.0), width=4, height=4)
        same = window(xrange=(0.0, 1.0), yrange=(0.0, 1.0), width=4, height=4)
        wider = window(xrange=(0.0, 1.0), yrange=(0.0, 1.0), width=8, height=4)

        assert one == same
        assert one != wider
