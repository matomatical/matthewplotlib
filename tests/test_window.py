import numpy as np
import pytest

from matthewplotlib.scales import scale, logscale
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
# sample points


class TestSamplePoints:
    def test_the_points_are_the_pixel_centres(self):
        w = window(xrange=(0.0, 1.0), yrange=(0.0, 1.0), width=2, height=1)
        X, Y = w.pixel_centres()

        points = w.sample_points()

        assert np.allclose(points[:, 0], X.ravel())
        assert np.allclose(points[:, 1], Y.ravel())

    def test_the_top_row_of_pixels_comes_first(self):
        """So that reshaping the values back to [2 * height, width] puts them
        where they came from."""
        w = window(xrange=(0.0, 1.0), yrange=(0.0, 1.0), width=3, height=2)

        points = w.sample_points()

        assert points.shape == (2 * 2 * 3, 2)
        assert points[0, 1] > points[-1, 1]

    def test_endpoints_reach_the_ends_of_both_ranges(self):
        """The four corner pixels then stand for the four corner combinations
        of the ranges, so the pixels reach half a pixel outside them."""
        w = window(xrange=(0.0, 1.0), yrange=(0.0, 1.0), width=3, height=1)

        points = w.sample_points(endpoints=True)

        assert np.allclose(points[0], [0.0, 1.0])
        assert np.allclose(points[-1], [1.0, 0.0])

    def test_without_endpoints_the_pixels_stay_inside_the_ranges(self):
        w = window(xrange=(0.0, 1.0), yrange=(0.0, 1.0), width=3, height=1)

        points = w.sample_points()

        assert points[:, 0].min() > 0.0
        assert points[:, 0].max() < 1.0

    def test_sampling_needs_both_coordinates(self):
        w = window(xrange=(0.0, 1.0), yrange=None, width=4, height=4)

        with pytest.raises(ValueError, match="no coordinates"):
            w.sample_points()


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


# # #
# scales on the axes


class TestScaledAxes:
    def test_a_plain_pair_is_kept_as_a_linear_scale(self):
        w = window(xrange=(0.0, 1.0), yrange=(0.0, 1.0), width=4, height=4)

        assert w.xrange == scale(0.0, 1.0)
        assert tuple(w.xrange) == (0.0, 1.0)

    def test_both_spellings_build_equal_windows(self):
        from_pair = window(
            xrange=(0.0, 1.0), yrange=(0.0, 1.0), width=4, height=4,
        )
        from_scale = window(
            xrange=scale(0.0, 1.0), yrange=(0.0, 1.0), width=4, height=4,
        )

        assert from_pair == from_scale

    def test_windows_differing_only_in_spacing_are_not_equal(self):
        """Which is what lets `dstack2` refuse to overlay a log plot on a
        linear one covering the same interval."""
        linear = window(
            xrange=(1.0, 100.0), yrange=(0.0, 1.0), width=4, height=4,
        )
        logged = window(
            xrange=logscale(1.0, 100.0), yrange=(0.0, 1.0), width=4, height=4,
        )

        assert linear != logged

    def test_a_partial_scale_is_refused(self):
        with pytest.raises(ValueError, match="missing endpoint"):
            window(
                xrange=logscale(), yrange=(0.0, 1.0), width=4, height=4,
            )

    def test_the_repr_names_a_nonlinear_spacing(self):
        w = window(
            xrange=logscale(1.0, 100.0),
            yrange=(0.0, 1.0),
            width=7,
            height=5,
        )

        assert repr(w) \
            == "window(x=logscale[1.00,100.00], y=[0.00,1.00], 7x5 cells)"

    def test_dots_place_equal_factors_at_equal_steps(self):
        """A log axis is linear in the transformed coordinate, so the dots of
        [1, 10, 100] on a log axis are the dots of [0, 1, 2] on a linear
        one."""
        logged = window(
            xrange=logscale(1.0, 100.0), yrange=(0.0, 1.0), width=7, height=5,
        )
        linear = window(
            xrange=(0.0, 2.0), yrange=(0.0, 1.0), width=7, height=5,
        )
        factors = np.array([[1.0, 0.5], [10.0, 0.5], [100.0, 0.5]])
        steps = np.array([[0.0, 0.5], [1.0, 0.5], [2.0, 0.5]])

        assert np.allclose(logged.dots(factors), linear.dots(steps))

    def test_a_point_the_scale_cannot_place_is_not_a_number(self):
        """A negative value has no position on a log axis, so the point comes
        out unplaceable---the drawing routines cull it---rather than raising
        or warning."""
        w = window(
            xrange=logscale(1.0, 100.0), yrange=(0.0, 1.0), width=7, height=5,
        )

        dots = w.dots(np.array([[-5.0, 0.5], [10.0, 0.5]]))

        assert np.isnan(dots[0, 1])
        assert np.isfinite(dots[1]).all()

    def test_the_edges_tile_the_scale(self):
        w = window(
            xrange=logscale(1.0, 16.0), yrange=(0.0, 1.0), width=4, height=1,
        )
        xedges, _yedges = w.pixel_edges()

        assert np.allclose(xedges, [1.0, 2.0, 4.0, 8.0, 16.0])

    def test_the_outermost_edges_are_the_limits_exactly(self):
        """Not the transform round trip of them, which floating point can
        land a whisker away, putting a data point at the limit outside its
        own bin."""
        w = window(
            xrange=logscale(0.1, 10.0), yrange=(0.0, 1.0), width=4, height=1,
        )
        xedges, _yedges = w.pixel_edges()

        assert xedges[0] == 0.1
        assert xedges[-1] == 10.0

    def test_the_centres_are_the_midpoints_along_the_scale(self):
        """On a log axis, the geometric rather than arithmetic mean of the
        pixel's edges: the centre of the pixel on the screen."""
        w = window(
            xrange=logscale(1.0, 16.0), yrange=(0.0, 1.0), width=4, height=1,
        )
        X, _Y = w.pixel_centres()

        assert np.allclose(X[0], [2 ** 0.5, 2 ** 1.5, 2 ** 2.5, 2 ** 3.5])

    def test_endpoint_samples_reach_the_ends_of_the_scale_exactly(self):
        w = window(
            xrange=logscale(0.1, 10.0),
            yrange=logscale(0.1, 10.0),
            width=3,
            height=1,
        )

        points = w.sample_points(endpoints=True)

        assert points[0].tolist() == [0.1, 10.0]
        assert points[-1].tolist() == [10.0, 0.1]

    def test_a_scale_without_an_inverse_can_place_but_not_sample(self):
        """Only the sampling methods ask what value a position stands for, so
        a custom scale that left `inverse` unwritten works everywhere else
        and is refused there, rather than silently sampling wrong points
        through the base class's identity."""
        class forwardscale(scale):
            def transform(self, values):
                return np.exp(np.asarray(values, dtype=float))

        w = window(
            xrange=forwardscale(0.0, 1.0),
            yrange=(0.0, 1.0),
            width=4,
            height=1,
        )

        assert np.isfinite(w.dots(np.array([[0.5, 0.5]]))).all()
        with pytest.raises(NotImplementedError, match="override inverse"):
            w.pixel_edges()
