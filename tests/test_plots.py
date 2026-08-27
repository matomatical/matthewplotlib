"""Behaviour shared across the plot families.

Every plot that measures its values against an interval settles on that
interval the same way, and every plot given a descending range mirrors the
same picture. Each family's own behaviour is tested in the test_plots_*
module named after it.
"""

import numpy as np
import pytest

import matthewplotlib as mp

from matthewplotlib.plots import (
    axes,
    boxes,
    candles,
    colorbar,
    function2,
    heatmap,
    histogram2,
    line,
    scatter,
    vfunction2,
)

from tests.plots_common import drawn_cells


def _channels_within(one, other, tolerance):
    """Whether two arrays of colour channels agree to within a few bytes."""
    difference = one.astype(int) - other.astype(int)
    return int(np.abs(difference).max()) <= tolerance


class TestDescendingRanges:
    """A range runs from the value at the low end of the screen axis to the
    value at the high end, so giving one descending inverts that axis. That
    mirrors the picture and changes nothing else: the same data still lands in
    the same bins, and the labels still name the ends they are written at."""

    def test_a_point_at_the_low_end_moves_to_the_right(self):
        point = np.array([[0.0, 0.5]])
        forward = scatter(point, width=4, height=1, xrange=(0.0, 1.0))
        reverse = scatter(point, width=4, height=1, xrange=(1.0, 0.0))

        assert drawn_cells(forward)[:, 0].any()
        assert not drawn_cells(forward)[:, -1].any()
        assert drawn_cells(reverse)[:, -1].any()
        assert not drawn_cells(reverse)[:, 0].any()

    def test_a_line_follows_its_range_around(self):
        series = (np.array([0.0, 1.0]), np.array([0.0, 0.0]))
        forward = line(series, width=6, height=1, xrange=(0.0, 1.0))
        reverse = line(series, width=6, height=1, xrange=(1.0, 0.0))

        assert np.array_equal(
            drawn_cells(reverse), drawn_cells(forward)[:, ::-1]
        )

    def test_a_grid_plot_is_mirrored_across_its_columns(self):
        """Within a byte: the sample coordinates mirror exactly, but the
        midpoints of a descending linspace are not bit for bit the reverse of
        an ascending one, and a last bit of difference can round a colour
        channel the other way."""
        def ramp(xy):
            return xy[:, 0]

        forward = function2(ramp, xrange=(0.0, 1.0), yrange=(0.0, 1.0),
                            width=6, height=2)
        reverse = function2(ramp, xrange=(1.0, 0.0), yrange=(0.0, 1.0),
                            width=6, height=2)

        assert _channels_within(
            reverse.chars.fg_rgb, forward.chars.fg_rgb[:, ::-1], 1
        )

    def test_a_grid_plot_is_mirrored_across_its_rows(self):
        """A cell holds two pixels, an upper in its foreground and a lower in
        its background, so turning the picture over swaps them as well as
        reversing the rows."""
        def ramp(xy):
            return xy[:, 1]

        forward = function2(ramp, xrange=(0.0, 1.0), yrange=(0.0, 1.0),
                            width=6, height=2)
        reverse = function2(ramp, xrange=(0.0, 1.0), yrange=(1.0, 0.0),
                            width=6, height=2)

        assert _channels_within(reverse.chars.fg_rgb, forward.chars.bg_rgb[::-1], 1)
        assert _channels_within(reverse.chars.bg_rgb, forward.chars.fg_rgb[::-1], 1)

    def test_a_histogram_bins_the_same_data_either_way_round(self):
        """numpy needs its bin edges ascending, so the counts are turned back
        around afterwards rather than the data being binned differently."""
        rng = np.random.default_rng(0)
        x, y = rng.random(200), rng.random(200)
        counts = {"x": x, "y": y, "width": 8, "height": 2}

        forward = histogram2(**counts, xrange=(0.0, 1.0), yrange=(0.0, 1.0))
        across = histogram2(**counts, xrange=(1.0, 0.0), yrange=(0.0, 1.0))
        over = histogram2(**counts, xrange=(0.0, 1.0), yrange=(1.0, 0.0))

        assert np.array_equal(across.chars.fg_rgb, forward.chars.fg_rgb[:, ::-1])
        assert np.array_equal(over.chars.fg_rgb, forward.chars.bg_rgb[::-1])
        assert np.array_equal(over.chars.bg_rgb, forward.chars.fg_rgb[::-1])

    def test_axes_label_the_ends_the_range_names(self):
        data = np.array([[0.0, 0.0], [1.0, 1.0]])
        plot = axes(scatter(data, width=10, height=2, yrange=(1.0, 0.0)))
        rows = plot.chars.to_plain_str().splitlines()

        assert rows[0].startswith("0.0")
        assert rows[-2].startswith("1.0")


class TestValueRanges:
    """Every plot that measures its values against an interval settles on that
    interval the same way, whether the interval becomes a colour scale or a
    length along the screen."""

    FLAT = pytest.mark.parametrize("draw", [
        pytest.param(lambda r: mp.bars([1.0, 2.0], vrange=r), id="bars"),
        pytest.param(lambda r: mp.columns([1.0, 2.0], vrange=r), id="columns"),
        pytest.param(lambda r: heatmap([[1.0, 2.0]], vrange=r), id="heatmap"),
        pytest.param(lambda r: boxes([[1.0, 2.0]], vrange=r), id="boxes"),
        pytest.param(
            lambda r: candles([1.0], [2.0], [0.0], [1.5], vrange=r),
            id="candles",
        ),
        pytest.param(
            lambda r: vfunction2(
                lambda xy: xy,
                xrange=(-1.0, 1.0),
                yrange=(-1.0, 1.0),
                width=2,
                height=1,
                vrange=r,
            ),
            id="vfunction2",
        ),
    ])

    @FLAT
    def test_an_interval_the_caller_wrote_and_covering_nothing_is_refused(
        self, draw,
    ):
        """There is no reading of it to act on: nothing can be measured
        against an interval with no extent."""
        with pytest.raises(ValueError, match="covers no interval"):
            draw((2.0, 2.0))

    @FLAT
    def test_the_scale_it_settled_on_unpacks_as_a_pair_of_floats(self, draw):
        vmin, vmax = draw((0.0, 4.0)).vrange
        assert (type(vmin), type(vmax)) == (float, float)

    def test_a_value_that_is_not_a_number_is_left_out_of_a_bar_chart(self):
        """It used to poison the largest value, and so every bar: one missing
        measurement drew the whole chart full."""
        chart = mp.bars([1.0, float("nan"), 3.0], width=6)

        assert chart.vrange == mp.scale(0.0, 3.0)
        assert chart.chars.to_plain_str().split("\n") == [
            "██    ",
            "      ",
            "██████",
        ]

    def test_bars_measure_from_zero_rather_than_the_lowest_value(self):
        """A bar's length is read against a baseline, so a chart of equal
        values is a row of full bars rather than a row of empty ones."""
        assert mp.bars([5.0, 5.0]).vrange == mp.scale(0.0, 5.0)
        assert mp.columns([5.0, 5.0]).vrange == mp.scale(0.0, 5.0)

    def test_a_scale_spaces_the_bar_lengths_within_the_interval(self):
        chart = mp.bars(
            [1.0, 4.0],
            width=4,
            vrange=mp.powscale(0.0, 4.0, exponent=0.5),
        )

        assert chart.vrange == mp.powscale(0.0, 4.0, exponent=0.5)
        assert chart.chars.to_plain_str().split("\n") == [
            "██  ",
            "████",
        ]

    def test_an_inferred_bar_interval_starts_at_zero_which_log_cannot(self):
        """Measuring from zero is what makes a bar's length readable on its
        own, so the inference is kept even for a scale that cannot cover
        zero, and the error says to give the endpoint instead."""
        with pytest.raises(ValueError, match="give one explicitly"):
            mp.bars([1.0, 1000.0], vrange=mp.logscale())

    def test_a_scale_covering_nothing_has_no_colorbar_to_draw(self):
        """A plot whose values are all the same settles on an interval with no
        extent, which is one colour and no axis to label it along."""
        flat = heatmap([[5.0, 5.0]])

        assert flat.vrange == mp.scale(5.0, 5.0)
        with pytest.raises(ValueError, match="no scale to draw a colorbar"):
            colorbar(flat)
