"""Unit tests for plot construction and arrangement."""

import datetime
import os

import numpy as np
import pytest

import matthewplotlib as mp

from matthewplotlib.plots import (
    axes,
    border,
    boxes,
    calendar,
    candles,
    weeks,
    cfunction2,
    dstack2,
    function2,
    histogram2,
    image,
    line,
    line3,
    scatter,
    table,
    text,
    vfunction2,
    wrap,
)


class RecordingVectorColormap:
    """A vector colormap that retains the field it received."""

    def __init__(self):
        self.input = None

    def __call__(self, values):
        self.input = np.array(values, copy=True)
        return np.zeros((*np.shape(values)[:2], 3), dtype=np.uint8)


class RecordingColormap:
    """A greyscale colormap that retains the scalar grid it received."""

    def __init__(self):
        self.input = None

    def __call__(self, values):
        self.input = np.array(values, copy=True)
        return np.repeat(
            (255 * values[..., np.newaxis]).astype(np.uint8),
            3,
            axis=-1,
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


class TestAxesSides:
    def test_a_plot_with_both_coordinates_is_framed(self):
        plot = axes(_narrow_ticks_scatter())

        assert plot.sides == ("rule", "rule", "label", "label")

    def test_a_plot_with_one_coordinate_is_labelled_on_one_side_only(self):
        """A colorbar is a strip, and a frame is for a 2d canvas, so the
        sides belonging to the missing coordinate are dropped entirely."""
        bar = image(np.linspace(0, 1, 8)[:, None], yrange=(0.0, 1.0))

        assert axes(bar).sides == ("crop", "crop", "crop", "label")

    def test_labelling_one_side_rules_the_opposite_one(self):
        plot = axes(_narrow_ticks_scatter(), north="label")

        assert plot.sides == ("label", "rule", "rule", "label")

    def test_both_sides_of_an_axis_can_be_labelled_on_request(self):
        plot = axes(_narrow_ticks_scatter(), north="label", south="label")

        assert plot.sides == ("label", "rule", "label", "label")

    def test_a_side_cannot_be_labelled_without_its_coordinate(self):
        bar = image(np.linspace(0, 1, 8)[:, None], yrange=(0.0, 1.0))

        with pytest.raises(ValueError, match="no x coordinate"):
            axes(bar, south="label")

    def test_a_plot_with_no_coordinates_cannot_be_given_axes(self):
        with pytest.raises(ValueError, match="no coordinates"):
            axes(image(np.zeros((4, 4))))

    def test_a_cropped_side_costs_nothing(self):
        framed = axes(_narrow_ticks_scatter())
        bare = axes(
            _narrow_ticks_scatter(),
            north="crop", east="crop", south="crop", west="crop",
        )
        inner = _narrow_ticks_scatter()

        assert (bare.height, bare.width) == (inner.height, inner.width)
        assert framed.width > bare.width

    def test_a_padded_side_holds_its_space_without_drawing(self):
        plot = axes(
            _narrow_ticks_scatter(),
            north="pad", east="crop", south="crop", west="crop",
        )

        assert plot.height == _narrow_ticks_scatter().height + 1
        assert plot.chars.to_plain_str().splitlines()[0].strip() == ""

    def test_a_title_goes_into_a_ruled_north_side(self):
        plot = axes(_narrow_ticks_scatter(), title="hi")

        assert "hi" in plot.chars.to_plain_str().splitlines()[0]
        assert plot.height == axes(_narrow_ticks_scatter()).height

    def test_a_title_takes_its_own_row_when_north_is_not_ruled(self):
        plain = axes(_narrow_ticks_scatter(), north="crop")
        titled = axes(_narrow_ticks_scatter(), north="crop", title="hi")

        assert titled.height == plain.height + 1
        assert "hi" in titled.chars.to_plain_str().splitlines()[0]


class TestAxesLabelsThatDoNotFit:
    def _bottom_row(self, width, **kwargs):
        data = np.array([[0.0, 0.0], [1.0, 1.0]])
        plot = axes(scatter(data, width=width, height=2), **kwargs)
        return plot.chars.to_plain_str().splitlines()[-1]

    def test_the_limits_spread_into_the_gutter_before_giving_up(self):
        """The label row is as wide as the whole plot, and the columns under
        the y gutter are blank, so they are used before anything is lost."""
        assert self._bottom_row(2) == "0.0 1.0"

    def test_limits_with_no_room_at_all_are_hashed_out(self):
        assert self._bottom_row(2, xfmt="{x:.3f}") == "#######"

    def test_hashing_does_not_widen_the_plot(self):
        data = np.array([[0.0, 0.0], [1.0, 1.0]])
        narrow = axes(scatter(data, width=2, height=2), xfmt="{x:.3f}")
        plain = axes(scatter(data, width=2, height=2), xfmt="{x:.1f}")

        assert narrow.width == plain.width

    def test_an_axis_name_survives_a_narrow_plot(self):
        with_name = self._bottom_row(30, xlabel="time")

        assert "time" in with_name

    def test_a_narrow_plot_with_an_axis_name_does_not_raise(self):
        # the name is squeezed to whatever is left, but the limits survive it
        row = self._bottom_row(2, xlabel="time")

        assert row.startswith("0.0") and row.endswith("1.0")


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
# heatmaps


class TestImage:
    @pytest.mark.parametrize("channels", [1, 2, 4])
    def test_an_rgb_image_must_have_three_channels(self, channels):
        with pytest.raises(ValueError, match="RGB colors of shape"):
            mp.image(np.zeros((4, 5, channels), dtype=np.uint8))

    def test_a_colormap_may_reduce_feature_vectors_to_rgb(self):
        features = np.zeros((4, 5, 7))

        plot = mp.image(
            features,
            colormap=lambda values: np.zeros((*values.shape[:2], 3)),
        )

        assert (plot.height, plot.width) == (2, 5)


class TestFunction2:
    def test_values_outside_zrange_saturate_before_colormapping(self):
        colormap = RecordingColormap()

        function2(
            lambda xy: xy[:, 0],
            xrange=(-1.0, 2.0),
            yrange=(0.0, 1.0),
            width=2,
            height=1,
            zrange=(0.0, 1.0),
            colormap=colormap,
            endpoints=True,
        )

        assert np.array_equal(colormap.input, [[0.0, 1.0], [0.0, 1.0]])

    def test_squares_are_sampled_at_their_centres(self):
        """The squares tile the ranges, and each shows the value of the
        function at the centre of the square it stands for."""
        sampled = []

        def record(xy):
            sampled.append(xy)
            return xy[:, 0]

        function2(record, xrange=(0.0, 1.0), yrange=(0.0, 2.0), width=4, height=1)

        assert np.allclose(np.unique(sampled[0][:, 0]), [0.125, 0.375, 0.625, 0.875])
        assert np.allclose(np.unique(sampled[0][:, 1]), [0.5, 1.5])

    def test_endpoints_reaches_the_ends_of_both_ranges(self):
        sampled = []

        def record(xy):
            sampled.append(xy)
            return xy[:, 0]

        function2(
            record,
            xrange=(0.0, 1.0),
            yrange=(0.0, 2.0),
            width=4,
            height=1,
            endpoints=True,
        )

        assert np.allclose(np.unique(sampled[0][:, 0]), [0.0, 1 / 3, 2 / 3, 1.0])
        assert np.allclose(np.unique(sampled[0][:, 1]), [0.0, 2.0])


class TestVFunction2:
    def test_the_field_is_scaled_into_the_unit_disc(self):
        colormap = RecordingVectorColormap()

        vfunction2(
            lambda xy: np.stack([xy[:, 0], np.zeros(len(xy))], axis=-1),
            xrange=(0.0, 4.0),
            yrange=(0.0, 1.0),
            width=2,
            height=1,
            colormap=colormap,
        )

        # x is sampled at 1 and 3, and the largest magnitude becomes one
        assert np.allclose(colormap.input[..., 0], [[1 / 3, 1.0], [1 / 3, 1.0]])

    def test_magnitudes_outside_vrange_saturate(self):
        colormap = RecordingVectorColormap()

        vfunction2(
            lambda xy: np.stack([xy[:, 0], np.zeros(len(xy))], axis=-1),
            xrange=(0.0, 4.0),
            yrange=(0.0, 1.0),
            width=2,
            height=1,
            vrange=(0.0, 2.0),
            colormap=colormap,
        )

        assert np.allclose(colormap.input[..., 0], [[0.5, 1.0], [0.5, 1.0]])

    def test_scaling_keeps_the_direction(self):
        colormap = RecordingVectorColormap()

        vfunction2(
            lambda xy: np.full((len(xy), 2), [3.0, 4.0]),
            xrange=(0.0, 1.0),
            yrange=(0.0, 1.0),
            width=1,
            height=1,
            vrange=(0.0, 10.0),
            colormap=colormap,
        )

        # magnitude five out of ten, along the same 3:4 diagonal
        assert np.allclose(colormap.input, [[[0.3, 0.4]], [[0.3, 0.4]]])

    def test_a_field_of_zeroes_does_not_divide_by_zero(self):
        with np.errstate(all="raise"):
            plot = vfunction2(
                lambda xy: np.zeros_like(xy),
                xrange=(0.0, 1.0),
                yrange=(0.0, 1.0),
                width=2,
                height=1,
            )
        assert plot.width == 2

    def test_a_field_that_returns_the_wrong_shape_is_rejected(self):
        with pytest.raises(ValueError, match="one .u, v. vector per point"):
            vfunction2(
                lambda xy: xy[:, 0],
                xrange=(0.0, 1.0),
                yrange=(0.0, 1.0),
                width=2,
                height=1,
            )


class TestCFunction2:
    def test_the_values_reach_the_colormap_unnormalised(self):
        """A domain colouring reads the modulus on an absolute scale."""
        seen = []

        def colormap(values):
            seen.append(np.array(values, copy=True))
            return np.zeros((*np.shape(values), 3), dtype=np.uint8)

        cfunction2(
            lambda z: 10 * z,
            xrange=(0.0, 2.0),
            yrange=(0.0, 1.0),
            width=2,
            height=1,
            colormap=colormap,
        )

        assert np.allclose(seen[0].real, [[5.0, 15.0], [5.0, 15.0]])

    def test_the_function_is_given_complex_numbers(self):
        sampled = []

        def record(z):
            sampled.append(np.array(z, copy=True))
            return z

        cfunction2(record, (0.0, 2.0), (0.0, 2.0), width=2, height=1)

        assert np.iscomplexobj(sampled[0])
        assert np.allclose(np.unique(sampled[0].real), [0.5, 1.5])
        assert np.allclose(np.unique(sampled[0].imag), [0.5, 1.5])

    def test_a_function_that_returns_the_wrong_shape_is_rejected(self):
        with pytest.raises(ValueError, match="one value per point"):
            cfunction2(
                lambda z: np.stack([z.real, z.imag], axis=-1),
                xrange=(0.0, 1.0),
                yrange=(0.0, 1.0),
                width=2,
                height=1,
            )

    def test_a_pole_at_the_origin_is_not_sampled_on(self):
        """Centre sampling never lands on a round number."""
        with np.errstate(all="raise"):
            cfunction2(lambda z: 1 / z, (-1.0, 1.0), (-1.0, 1.0), 4, 2)


class TestHistogram2:
    def test_counts_above_max_count_saturate_before_colormapping(self):
        colormap = RecordingColormap()

        histogram2(
            x=[0.0, 0.0, 0.0],
            y=[0.0, 0.0, 0.0],
            width=1,
            height=1,
            xrange=(-1.0, 1.0),
            yrange=(-1.0, 1.0),
            max_count=2,
            colormap=colormap,
        )

        assert colormap.input.max() == 1.0
        assert np.count_nonzero(colormap.input == 1.0) == 1

    def test_an_inferred_zero_maximum_produces_a_blank_heatmap(self):
        colormap = RecordingColormap()

        histogram2(
            x=[2.0],
            y=[2.0],
            width=1,
            height=1,
            xrange=(0.0, 1.0),
            yrange=(0.0, 1.0),
            colormap=colormap,
        )

        assert np.array_equal(colormap.input, np.zeros((2, 1)))

    @pytest.mark.parametrize("max_count", [0, -1])
    def test_an_explicit_maximum_must_be_positive(self, max_count):
        with pytest.raises(ValueError, match="max_count must be positive"):
            histogram2(
                x=[0.0],
                y=[0.0],
                width=1,
                height=1,
                xrange=(-1.0, 1.0),
                yrange=(-1.0, 1.0),
                max_count=max_count,
            )


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


# # #
# calendar


def a_month(month=1, year=2025, value=1.0):
    """Every day of one month, all with the same value."""
    day = datetime.date(year, month, 1)
    days = {}
    while day.month == month:
        days[day] = value
        day += datetime.timedelta(days=1)
    return days


class TestCalendar:
    def test_a_month_is_a_caption_a_header_and_its_weeks(self):
        plot = calendar(a_month(), cols=1, month_spacing=0)
        # January 2025 runs Wednesday to Friday, over five Monday-start weeks.
        assert plot.height == 1 + 1 + 5
        assert plot.width == 7 * 2

    def test_day_width_scales_the_block(self):
        plot = calendar(a_month(), cols=1, month_spacing=0, day_width=3)
        assert plot.width == 7 * 3

    def test_months_are_wrapped_into_a_grid(self):
        days = a_month(1) | a_month(2) | a_month(3) | a_month(4)
        plot = calendar(days, cols=2, month_spacing=0)
        assert plot.width == 2 * 7 * 2

    def test_the_spacing_is_not_left_on_the_far_edges(self):
        """A gap between the months should not become a margin around them."""
        days = a_month(1) | a_month(2)
        snug = calendar(days, cols=2, month_spacing=0)
        spaced = calendar(days, cols=2, month_spacing=1)
        # One gap between the two months, and none past the second.
        assert snug.width == 2 * 7 * 2
        assert spaced.width == 2 * 7 * 2 + 1 * 2
        # February 2025 ends on a Friday, so its last column has days in it.
        assert drawn_cells(spaced)[:, -1].any()

    def test_a_grid_is_no_wider_than_the_months_in_it(self):
        """Asking for four columns and giving it one month should not leave
        three columns of blank beside it."""
        for cols in (4, None):
            plot = calendar(a_month(), cols=cols, month_spacing=0)
            assert plot.width == 7 * 2

    def test_a_day_with_no_value_is_blank(self):
        days = a_month()
        del days[datetime.date(2025, 1, 15)]
        plot = calendar(days, cols=1, month_spacing=0)
        # The 15th is a Wednesday, in the third full week of the month.
        row, column = 2 + 2, 2 * 2
        assert not drawn_cells(plot)[row, column]

    def test_a_day_whose_value_is_not_finite_is_blank(self):
        days = a_month()
        days[datetime.date(2025, 1, 15)] = float("nan")
        plot = calendar(days, cols=1, month_spacing=0)
        assert not drawn_cells(plot)[2 + 2, 2 * 2]
        assert plot.num_days == len(days) - 1

    def test_a_day_whose_value_is_zero_is_drawn(self):
        """Distinctly from a day with no value at all."""
        days = a_month(value=0.0)
        plot = calendar(days, vrange=(0.0, 1.0), cols=1, month_spacing=0)
        assert drawn_cells(plot)[2 + 2, 2 * 2]
        assert plot.num_days == len(days)

    def test_days_outside_the_daterange_are_left_out(self):
        days = a_month(1) | a_month(2)
        plot = calendar(
            days,
            daterange=("2025-01-01", "2025-01-31"),
            cols=1,
            month_spacing=0,
        )
        assert plot.num_days == 31

    def test_the_daterange_can_reach_past_the_data(self):
        plot = calendar(
            a_month(1),
            daterange=("2025-01-01", "2025-02-28"),
            cols=2,
            month_spacing=0,
        )
        assert plot.width == 2 * 7 * 2

    def test_the_value_scale_spans_the_data_by_default(self):
        plot = calendar({"2025-01-01": 3.0, "2025-01-02": 9.0}, cols=1)
        assert (plot.vmin, plot.vmax) == (3.0, 9.0)

    def test_a_single_number_scales_up_from_zero(self):
        plot = calendar({"2025-01-01": 3.0}, vrange=10.0, cols=1)
        assert (plot.vmin, plot.vmax) == (0.0, 10.0)

    def test_the_value_scale_ignores_the_days_with_no_value(self):
        days = {"2025-01-01": 3.0, "2025-01-02": float("nan")}
        plot = calendar(days, cols=1)
        assert (plot.vmin, plot.vmax) == (3.0, 3.0)

    def test_the_first_weekday_shifts_the_columns(self):
        """The 1st of January 2025 is a Wednesday, so it lands in the third
        column of a Monday-start week and the fourth of a Sunday-start one."""
        days = a_month()
        monday = calendar(days, cols=1, month_spacing=0)
        sunday = calendar(days, cols=1, month_spacing=0, first_weekday=6)
        week = 2
        assert drawn_cells(monday)[week, 2 * 2]
        assert not drawn_cells(monday)[week, 1 * 2]
        assert drawn_cells(sunday)[week, 3 * 2]
        assert not drawn_cells(sunday)[week, 2 * 2]

    def test_the_labels_can_be_left_off(self):
        plot = calendar(
            a_month(),
            cols=1,
            month_spacing=0,
            month_labels=False,
            weekday_labels=False,
        )
        assert plot.height == 5

    def test_a_narrow_month_is_captioned_in_a_shorter_spelling(self):
        plot = calendar(a_month(), cols=1, month_spacing=0, day_width=1)
        caption = "".join(chr(c) for c in plot.chars.codes[0])
        assert caption == "Jan  25"

    def test_a_wide_month_is_captioned_in_full(self):
        plot = calendar(a_month(), cols=1, month_spacing=0)
        caption = "".join(chr(c) for c in plot.chars.codes[0])
        assert caption == "January   2025"

    def test_the_years_line_up_across_the_months(self):
        """Whichever months are drawn, and however long their names are."""
        days = a_month(5) | a_month(9)
        plot = calendar(days, cols=1, month_spacing=0)
        rows = ["".join(chr(c) for c in row) for row in plot.chars.codes]
        captions = [row for row in rows if "2025" in row]
        # May through September, since the months between them are filled in.
        assert len(captions) == 5
        assert captions[0] == "May       2025"
        assert captions[-1] == "September 2025"
        assert len({len(caption) for caption in captions}) == 1

    def test_it_needs_something_to_draw(self):
        with pytest.raises(ValueError, match="at least one"):
            calendar({})

    def test_the_daterange_has_to_run_forwards(self):
        with pytest.raises(ValueError, match="ends"):
            calendar(a_month(), daterange=("2025-02-01", "2025-01-01"))

    def test_a_day_has_to_have_a_width(self):
        with pytest.raises(ValueError, match="day_width"):
            calendar(a_month(), day_width=0)

    def test_the_spacing_cannot_be_negative(self):
        with pytest.raises(ValueError, match="month_spacing"):
            calendar(a_month(), month_spacing=-1)

    def test_the_week_has_to_start_on_a_weekday(self):
        with pytest.raises(ValueError, match="first_weekday"):
            calendar(a_month(), first_weekday=7)

# # #
# weeks


def a_year(year=2025, value=1.0):
    """Every day of one year, all with the same value."""
    day = datetime.date(year, 1, 1)
    days = {}
    while day.year == year:
        days[day] = value
        day += datetime.timedelta(days=1)
    return days


def rows_of(plot):
    """Each row of a plot as a string, for reading its captions back."""
    return ["".join(chr(code) for code in row) for row in plot.chars.codes]


class TestWeeks:
    def test_a_strip_is_two_captions_and_the_seven_weekdays(self):
        plot = weeks(a_year())
        # 2025 starts on a Wednesday, so its 365 days touch 53 Monday-weeks.
        assert plot.height == 2 + 7
        assert plot.width == 2 + 53 * 2
        assert plot.num_weeks == 53

    def test_day_width_scales_the_strip(self):
        plot = weeks(a_year(), day_width=3)
        assert plot.width == 2 + 53 * 3

    def test_the_strip_starts_at_the_top_of_its_first_week(self):
        """So that a weekday keeps to one row. The 1st of January 2025 is a
        Wednesday, so the Monday and Tuesday above it are blank."""
        plot = weeks(a_year())
        assert not drawn_cells(plot)[2 + 0, 2]
        assert not drawn_cells(plot)[2 + 1, 2]
        assert drawn_cells(plot)[2 + 2, 2]

    def test_the_weekdays_are_named_down_the_gutter(self):
        plot = weeks(a_year())
        assert [chr(plot.chars.codes[2 + i, 0]) for i in range(7)] == list(
            "MTWtFSs"
        )

    def test_the_first_weekday_rotates_the_rows(self):
        plot = weeks(a_year(), first_weekday=6)
        assert [chr(plot.chars.codes[2 + i, 0]) for i in range(7)] == list(
            "sMTWtFS"
        )

    def test_the_months_are_captioned(self):
        captions = " ".join(rows_of(weeks(a_year()))[:2])
        for month in ("Jan", "Feb", "Jun", "Dec"):
            assert month in captions

    def test_the_year_is_captioned_once(self):
        assert rows_of(weeks(a_year()))[0].count("2025") == 1

    def test_a_wide_strip_wraps_into_bands(self):
        plot = weeks(a_year(), width=80)
        assert plot.width == 80
        # Two bands of nine rows, with a blank row between them.
        assert plot.height == 9 + 1 + 9
        assert not drawn_cells(plot)[9, :].any()

    def test_every_band_names_its_year(self):
        """A band should be readable without looking back at the one above."""
        rows = rows_of(weeks(a_year(), width=80))
        assert "2025" in rows[0]
        assert "2025" in rows[10]

    def test_a_span_of_two_years_names_both(self):
        days = a_year(2024) | a_year(2025)
        captions = rows_of(weeks(days))[0]
        assert "2024" in captions
        assert "2025" in captions

    def test_a_caption_that_will_not_fit_is_dropped_not_truncated(self):
        months = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]
        plot = weeks(a_year(), width=20)
        # Only the caption rows, so that the weekday initials in the gutter of
        # each day row are not mistaken for clipped month names.
        drawn = []
        for row, codes in zip(rows_of(plot), plot.chars.codes):
            if (codes == ord("\u2588")).any():
                continue
            drawn.extend(run for run in row.split() if run.isalpha())
        # Narrow bands leave some months no room, but never half a name.
        assert drawn
        assert all(run in months for run in drawn)
        assert len(drawn) < 12

    def test_the_captions_can_be_left_off(self):
        assert weeks(a_year(), year_labels=False).height == 1 + 7
        assert weeks(a_year(), month_labels=False).height == 1 + 7
        assert weeks(
            a_year(), year_labels=False, month_labels=False
        ).height == 7

    def test_the_gutter_can_be_left_off(self):
        assert weeks(a_year(), weekday_labels=False).width == 53 * 2

    def test_a_day_with_no_value_is_blank(self):
        days = a_year()
        del days[datetime.date(2025, 1, 8)]
        plot = weeks(days)
        # The 8th is the Wednesday of the second week.
        assert not drawn_cells(plot)[2 + 2, 2 + 2]

    def test_a_day_whose_value_is_not_finite_is_blank(self):
        days = a_year()
        days[datetime.date(2025, 1, 8)] = float("nan")
        plot = weeks(days)
        assert not drawn_cells(plot)[2 + 2, 2 + 2]
        assert plot.num_days == 364

    def test_a_day_whose_value_is_zero_is_drawn(self):
        plot = weeks(a_year(value=0.0), vrange=(0.0, 1.0))
        assert drawn_cells(plot)[2 + 2, 2]
        assert plot.num_days == 365

    def test_days_outside_the_daterange_are_left_out(self):
        plot = weeks(a_year(), daterange=("2025-01-01", "2025-01-31"))
        assert plot.num_days == 31

    def test_the_value_scale_spans_the_data_by_default(self):
        plot = weeks({"2025-01-01": 3.0, "2025-01-02": 9.0})
        assert (plot.vmin, plot.vmax) == (3.0, 9.0)

    def test_it_draws_the_same_days_as_a_calendar_would(self):
        """The two share the front end that decides which days get a colour."""
        days = a_year()
        days[datetime.date(2025, 5, 5)] = float("nan")
        del days[datetime.date(2025, 6, 6)]
        strip = weeks(days, daterange=("2025-02-01", "2025-11-30"))
        grid = calendar(days, daterange=("2025-02-01", "2025-11-30"))
        assert strip.num_days == grid.num_days
        assert (strip.vmin, strip.vmax) == (grid.vmin, grid.vmax)

    def test_it_needs_something_to_draw(self):
        with pytest.raises(ValueError, match="at least one"):
            weeks({})

    def test_the_daterange_has_to_run_forwards(self):
        with pytest.raises(ValueError, match="ends"):
            weeks(a_year(), daterange=("2025-02-01", "2025-01-01"))

    def test_a_day_has_to_have_a_width(self):
        with pytest.raises(ValueError, match="day_width"):
            weeks(a_year(), day_width=0)

    def test_the_week_has_to_start_on_a_weekday(self):
        with pytest.raises(ValueError, match="first_weekday"):
            weeks(a_year(), first_weekday=7)

    def test_the_width_has_to_fit_the_gutter_and_a_week(self):
        with pytest.raises(ValueError, match="width must leave room"):
            weeks(a_year(), width=3)

    def test_the_narrowest_width_is_one_week_per_band(self):
        plot = weeks(a_year(), width=4)
        assert plot.width == 4
        assert plot.height == 53 * 9 + 52


# # #
# candles


class TestCandles:
    """A candlestick chart draws one candle per period, coloured by whether the
    period closed above or below where it opened. It carries its value range as
    a vertical coordinate and no horizontal one, since its candles are a
    sequence of periods rather than a measured axis."""

    OPENS = np.array([10.0, 11.0, 12.0, 11.5])
    HIGHS = np.array([11.5, 12.5, 12.5, 13.0])
    LOWS = np.array([9.5, 10.5, 11.0, 11.0])
    CLOSES = np.array([11.0, 12.0, 11.5, 12.5])

    def chart(self, **kwargs):
        return candles(
            opens=self.OPENS,
            highs=self.HIGHS,
            lows=self.LOWS,
            closes=self.CLOSES,
            **kwargs,
        )

    def test_the_value_range_covers_every_candle(self):
        plot = self.chart()
        assert plot.window.yrange == (9.5, 13.0)

    def test_there_is_no_horizontal_coordinate(self):
        assert self.chart().window.xrange is None

    def test_axes_labels_the_value_axis_and_leaves_the_rest_alone(self):
        lines = axes(self.chart(height=4)).chars.to_plain_str().splitlines()
        # a gutter of labels down the left, one rule beside it, and no row of
        # labels underneath
        assert len(lines) == 4
        assert [line[:5] for line in lines] == ["13.0┐", "    │", "    │", " 9.5┘"]

    def test_the_rectangle_is_one_column_per_candle_by_default(self):
        plot = self.chart(height=5)
        assert (plot.height, plot.width) == (5, 4)

    def test_bodies_and_spacing_widen_the_rectangle(self):
        plot = self.chart(height=5, body_width=3, spacing=1)
        assert plot.width == 4 * 4 - 1

    def test_a_rising_candle_takes_the_rising_colour(self):
        plot = self.chart(height=4, rising=(1.0, 0.0, 0.0))
        column = plot.chars.fg_rgb[:, 0]
        drawn = column[plot.chars.fg[:, 0]]
        assert (drawn == np.array([255, 0, 0])).all(axis=-1).any()

    def test_a_falling_candle_takes_the_falling_colour(self):
        # the third period opened at 12.0 and closed at 11.5
        plot = self.chart(height=4, falling=(0.0, 0.0, 255))
        column = plot.chars.fg_rgb[:, 2]
        drawn = column[plot.chars.fg[:, 2]]
        assert (drawn == np.array([0, 0, 255])).all(axis=-1).any()

    def test_a_candle_that_closed_where_it_opened_counts_as_rising(self):
        plot = candles(
            opens=np.array([1.0]),
            highs=np.array([2.0]),
            lows=np.array([0.0]),
            closes=np.array([1.0]),
            height=4,
            rising=(1.0, 0.0, 0.0),
            falling=(0.0, 0.0, 1.0),
        )
        drawn = plot.chars.fg_rgb[plot.chars.fg]
        assert (drawn == np.array([255, 0, 0])).all(axis=-1).any()
        assert not (drawn == np.array([0, 0, 255])).all(axis=-1).any()

    def test_a_wick_takes_its_body_colour_by_default(self):
        plot = self.chart(height=6, rising=(1.0, 0.0, 0.0))
        # the first candle rose, so its wick is drawn in the rising colour
        wicks = np.isin(plot.chars.codes[:, 0], [ord("│"), ord("╵"), ord("╷")])
        assert wicks.any()
        assert (plot.chars.fg_rgb[wicks, 0] == np.array([255, 0, 0])).all()

    def test_a_wick_colour_applies_to_every_wick(self):
        plot = self.chart(height=6, wick=(1.0, 1.0, 1.0))
        wicks = np.isin(plot.chars.codes, [ord("│"), ord("╵"), ord("╷")])
        assert wicks.any()
        assert (plot.chars.fg_rgb[wicks] == 255).all()

    def test_the_background_is_painted_behind_the_whole_rectangle(self):
        plot = self.chart(height=5, background=(0.0, 0.0, 0.0))
        assert plot.chars.bg.all()
        assert (plot.chars.bg_rgb[~plot.chars.fg] == 0).all()

    def test_a_narrower_vrange_clips_the_candles_into_it(self):
        plot = self.chart(height=4, vrange=(12.0, 13.0))
        assert plot.window.yrange == (12.0, 13.0)
        # the first candle traded entirely below the range, so all four of its
        # values clip to the bottom and it is drawn in the bottom row alone
        assert not plot.chars.fg[:-1, 0].any()
        assert plot.chars.fg[-1, 0]

    def test_the_four_series_must_be_the_same_length(self):
        with pytest.raises(ValueError, match="same length"):
            candles(
                opens=np.zeros(3),
                highs=np.ones(3),
                lows=np.zeros(3),
                closes=np.ones(2),
            )

    def test_each_series_must_be_one_dimensional(self):
        with pytest.raises(ValueError, match="sequence of numbers"):
            candles(
                opens=np.zeros((2, 2)),
                highs=np.ones((2, 2)),
                lows=np.zeros((2, 2)),
                closes=np.ones((2, 2)),
            )

    def test_a_high_inside_the_body_is_refused(self):
        with pytest.raises(ValueError, match="opens, highs, lows, closes"):
            candles(
                opens=np.array([1.0]),
                highs=np.array([1.5]),
                lows=np.array([0.5]),
                closes=np.array([2.0]),
            )

    def test_a_low_inside_the_body_is_refused(self):
        with pytest.raises(ValueError, match="opens, highs, lows, closes"):
            candles(
                opens=np.array([1.0]),
                highs=np.array([2.0]),
                lows=np.array([1.5]),
                closes=np.array([1.2]),
            )

    def test_candles_all_at_one_value_have_no_range_to_plot_in(self):
        with pytest.raises(ValueError, match="same value"):
            candles(
                opens=np.array([1.0]),
                highs=np.array([1.0]),
                lows=np.array([1.0]),
                closes=np.array([1.0]),
            )

    def test_no_candles_and_no_range_to_infer_one_from(self):
        with pytest.raises(ValueError, match="no candles"):
            candles(
                opens=np.zeros(0),
                highs=np.zeros(0),
                lows=np.zeros(0),
                closes=np.zeros(0),
            )

    def test_a_repr_names_the_candles_and_their_range(self):
        assert repr(self.chart(height=4)) == (
            "candles(height=4, width=4, values=<4 candles on [9.50,13.00]>)"
        )

# # #
# boxes


class TestBoxesStatistics:
    """A box plot takes raw samples and works out the quartiles itself. The box
    spans the first and third, the whiskers reach the furthest sample within
    Tukey's fence, and every sample beyond it is drawn as a point."""

    # nine samples, so the quartiles land on samples rather than between them
    PLAIN = list(range(1, 10))

    def test_the_value_range_covers_every_sample(self):
        plot = boxes([self.PLAIN])
        assert plot.window.xrange == (1, 9)

    def test_a_sample_beyond_the_fence_is_an_outlier(self):
        # quartiles 3 and 7, so the fence reaches 3 - 6 to 7 + 6
        plot = boxes([[*self.PLAIN, 20.0]], length=40)
        assert "·" in plot.chars.to_plain_str()

    def test_no_sample_beyond_the_fence_means_no_outliers(self):
        assert "·" not in boxes([self.PLAIN], length=40).chars.to_plain_str()

    def test_the_whisker_stops_at_the_furthest_sample_inside_the_fence(self):
        # the whisker reaches 9, not the fence at 13, and not the outlier
        plot = boxes([[*self.PLAIN, 20.0]], length=40, whisker_iqrs=1.5)
        # 9 is where the value axis reaches 8/19 of the way along, and a cap
        # lands in the cell holding it
        cap = plot.chars.to_plain_str().splitlines()[1].rindex("┤")
        assert cap == int(40 * (9 - 1) / (20 - 1))

    def test_without_a_reach_the_whiskers_take_the_extremes(self):
        plot = boxes([[*self.PLAIN, 20.0]], length=40, whisker_iqrs=None)
        assert "·" not in plot.chars.to_plain_str()
        assert plot.chars.to_plain_str().splitlines()[1].endswith("┤")

    def test_a_reach_of_zero_keeps_only_the_quartiles(self):
        plot = boxes([self.PLAIN], length=40, whisker_iqrs=0.0)
        # everything outside the box is an outlier
        assert plot.chars.to_plain_str().count("·") > 0

    def test_a_negative_reach_is_an_error(self):
        with pytest.raises(ValueError):
            boxes([self.PLAIN], whisker_iqrs=-1.0)

    def test_groups_need_not_be_the_same_length(self):
        plot = boxes([[1.0, 2.0], list(range(20))])
        assert plot.num_boxes == 2

    def test_a_two_dimensional_array_is_one_group_per_row(self):
        plot = boxes(np.arange(12.0).reshape(3, 4))
        assert plot.num_boxes == 3

    def test_a_single_group_of_one_sample_is_allowed(self):
        plot = boxes([[1.0]], vrange=(0.0, 2.0))
        assert plot.num_boxes == 1

    def test_a_bare_sequence_of_numbers_is_rejected(self):
        # boxes takes a group per box, so a flat list would silently become one
        # box per number
        with pytest.raises(ValueError, match="single number"):
            boxes([1.0, 2.0, 3.0])

    def test_an_empty_group_is_rejected(self):
        with pytest.raises(ValueError, match="no samples"):
            boxes([[1.0, 2.0], []])

    def test_no_groups_at_all_is_rejected(self):
        with pytest.raises(ValueError, match="at least one group"):
            boxes([])

    def test_samples_all_at_one_value_need_a_range(self):
        with pytest.raises(ValueError, match="same value"):
            boxes([[3.0, 3.0, 3.0]])
        assert boxes([[3.0, 3.0]], vrange=(0.0, 6.0)).num_boxes == 1


class TestBoxesLayout:
    """A box plot carries its value range on the axis its boxes lie along and
    no coordinate on the other, since its groups are a list of categories
    rather than a measured axis."""

    DATA = [list(range(10)), list(range(5, 20))]

    def test_a_flat_plot_is_as_wide_as_its_value_axis(self):
        plot = boxes(self.DATA, length=25, box_thickness=3, box_spacing=1)
        assert (plot.height, plot.width) == (2 * 4 - 1, 25)

    def test_a_standing_plot_turns_the_rectangle_around(self):
        plot = boxes(
            self.DATA, length=25, box_thickness=3, box_spacing=1,
            box_direction="vertical",
        )
        assert (plot.height, plot.width) == (25, 2 * 4 - 1)

    def test_a_flat_plot_carries_its_range_horizontally(self):
        plot = boxes(self.DATA)
        assert plot.window.xrange == (0, 19)
        assert plot.window.yrange is None

    def test_a_standing_plot_carries_its_range_vertically(self):
        plot = boxes(self.DATA, box_direction="vertical")
        assert plot.window.yrange == (0, 19)
        assert plot.window.xrange is None

    def test_axes_labels_the_value_axis_and_leaves_the_rest_alone(self):
        lines = axes(boxes(self.DATA, length=20)).chars.to_plain_str()
        # the range's ends are labelled along the bottom, under a single rule
        assert "0" in lines.splitlines()[-1]
        assert "19" in lines.splitlines()[-1]

    def test_a_narrower_range_clips_the_boxes_into_it(self):
        plot = boxes(self.DATA, vrange=(5.0, 10.0), length=20)
        assert plot.window.xrange == (5.0, 10.0)

    def test_a_point_outside_the_range_is_dropped(self):
        # one far outlier, and a range that excludes it
        data = [[*range(1, 10), 20.0]]
        assert "·" in boxes(data, length=40).chars.to_plain_str()
        inside = boxes(data, length=40, vrange=(1.0, 9.0))
        assert "·" not in inside.chars.to_plain_str()

    def test_the_repr_names_the_groups_and_the_range(self):
        assert repr(boxes(self.DATA)) == (
            "boxes(height=7, width=30, data=<2 groups on [0.00,19.00]>)"
        )


class TestBoxesStyle:
    """A box plot is outlined by default and filled on request, and takes its
    colours per box or all at once."""

    DATA = [list(range(10)), list(range(5, 20))]

    def test_outlined_by_default(self):
        assert "┌" in boxes(self.DATA).chars.to_plain_str()

    def test_a_filled_plot_uses_blocks_instead(self):
        drawn = boxes(self.DATA, filled=True).chars.to_plain_str()
        assert "┌" not in drawn
        assert "█" in drawn

    def test_an_outlined_plot_leaves_the_terminal_background_showing(self):
        assert not boxes(self.DATA).chars.bg.any()

    def test_a_filled_plot_paints_its_whole_rectangle(self):
        assert boxes(self.DATA, filled=True).chars.bg.all()

    def test_an_outlined_plot_takes_the_terminal_foreground(self):
        assert not boxes(self.DATA).chars.fg.any()

    def test_one_colour_covers_every_box(self):
        plot = boxes(self.DATA, color="red")
        painted = plot.chars.fg_rgb[plot.chars.fg]
        assert (painted == np.array([255, 0, 0])).all()

    def test_a_colour_for_each_box(self):
        plot = boxes(
            self.DATA, colors=["red", "blue"], box_thickness=3,
            box_spacing=1,
        )
        assert plot.chars.fg_rgb[0][plot.chars.fg[0]].tolist()[0] == [255, 0, 0]
        assert plot.chars.fg_rgb[4][plot.chars.fg[4]].tolist()[0] == [0, 0, 255]

    def test_the_wrong_number_of_colours_is_an_error(self):
        with pytest.raises(ValueError, match="2 groups but 3 colors"):
            boxes(self.DATA, colors=["red", "green", "blue"])

    def test_a_median_can_be_left_off(self):
        assert "┬" in boxes(self.DATA, length=30).chars.to_plain_str()
        assert "┬" not in boxes(
            self.DATA, length=30, median=False,
        ).chars.to_plain_str()

    def test_caps_can_be_left_off(self):
        assert "╷" in boxes(self.DATA, length=30).chars.to_plain_str()
        assert "╷" not in boxes(
            self.DATA, length=30, caps=False,
        ).chars.to_plain_str()

    # samples whose quartiles are 3 and 7 and whose median is 5, so that a
    # range of 3 to 7 gives one box filling the axis with its median mid-way
    ONE_CELL = dict(
        data=[list(range(1, 10))], vrange=(3.0, 7.0), length=1,
        filled=True, box_thickness=1,
    )

    def test_a_filled_median_is_light_flat_and_heavy_standing(self):
        # each is the weight that matches the eighth blocks the median lands on
        # at the edges of a cell, which differ between the two orientations
        flat = boxes(**self.ONE_CELL)
        standing = boxes(**self.ONE_CELL, box_direction="vertical")
        assert flat.chars.to_plain_str() == "│"
        assert standing.chars.to_plain_str() == "━"

    def test_the_median_weight_can_be_chosen(self):
        plot = boxes(**self.ONE_CELL, median_style=mp.LineStyle.HEAVY)
        assert plot.chars.to_plain_str() == "┃"

    def test_an_outlined_box_needs_three_cells_across_it(self):
        with pytest.raises(ValueError, match="at least 3"):
            boxes(self.DATA, box_thickness=2)

    def test_a_filled_box_can_be_one_cell_across(self):
        plot = boxes(self.DATA, filled=True, box_thickness=1, box_spacing=0)
        assert plot.height == 2


# # #
# table


RUNS = [
    {"run": "baseline", "lr": 1e-3, "acc": 0.873},
    {"run": "wider", "lr": 3e-4, "acc": 0.902},
]


def _rows(plot):
    """The lines of a table, as plain text."""
    return plot.chars.to_plain_str().splitlines()


class TestTableData:
    def test_a_list_of_dicts_takes_its_columns_from_the_keys(self):
        plot = table(RUNS)

        assert plot.headers == ["run", "lr", "acc"]
        assert plot.num_rows == 2
        assert _rows(plot)[1].split() == ["run", "lr", "acc"]

    def test_a_dict_of_lists_takes_its_columns_from_the_keys(self):
        plot = table({"run": ["baseline", "wider"], "acc": [0.873, 0.902]})

        assert plot.headers == ["run", "acc"]
        assert plot.num_rows == 2

    def test_a_2d_array_has_no_headers_of_its_own(self):
        plot = table(np.arange(6).reshape(2, 3))

        assert plot.headers is None
        assert plot.num_rows == 2
        assert plot.num_columns == 3

    def test_a_2d_array_can_be_given_headers(self):
        plot = table(np.arange(6).reshape(2, 3), headers=["a", "b", "c"])

        assert plot.headers == ["a", "b", "c"]

    def test_headers_pick_out_and_order_the_columns(self):
        plot = table(RUNS, headers=["acc", "run"])

        assert plot.headers == ["acc", "run"]
        assert _rows(plot)[3].split() == ["0.873", "baseline"]

    def test_headers_given_as_a_mapping_rename_as_they_pick(self):
        plot = table(RUNS, headers={"acc": "accuracy"})

        assert plot.headers == ["accuracy"]

    def test_a_row_missing_a_key_leaves_that_cell_blank(self):
        plot = table([{"a": 1, "b": 2}, {"a": 3}])

        assert _rows(plot)[4].split() == ["3"]

    def test_a_short_column_runs_out_into_blanks(self):
        plot = table({"a": [1, 2, 3], "b": [4]})

        assert plot.num_rows == 3
        assert _rows(plot)[5].split() == ["3"]

    def test_a_value_of_none_is_blank_however_it_is_formatted(self):
        plot = table([[None]], formats=".3f")

        assert _rows(plot)[1].strip() == ""


class TestTableFormatting:
    def test_a_float_is_shown_to_four_significant_figures(self):
        plot = table([[1 / 3]])

        assert _rows(plot)[1].strip() == "0.3333"

    def test_a_format_spec_applies_to_every_column(self):
        plot = table([[1.5, 2.5]], formats=".2f")

        assert _rows(plot)[1].split() == ["1.50", "2.50"]

    def test_a_template_with_a_field_in_it_is_filled_in(self):
        plot = table([[0.873]], formats="{:.1%}")

        assert _rows(plot)[1].strip() == "87.3%"

    def test_a_callable_formats_a_value_however_it_likes(self):
        plot = table([["ab"]], formats=lambda value: value.upper())

        assert _rows(plot)[1].strip() == "AB"

    def test_a_format_can_be_given_per_column_by_name(self):
        plot = table(RUNS, formats={"acc": "{:.1%}"})

        assert _rows(plot)[3].split() == ["baseline", "0.001", "87.3%"]

    def test_a_column_is_cut_to_the_widest_it_may_be(self):
        plot = table([["abcdefgh"]], max_col_width=4)

        assert _rows(plot)[1].strip() == "abc…"

    def test_a_cell_with_newlines_in_it_grows_its_row(self):
        plot = table([["one\ntwo", "x"]], toprule="skip", bottomrule="skip")

        assert _rows(plot) == ["one  x", "two   "]


class TestTableAlignment:
    def test_a_column_of_numbers_is_aligned_right_by_default(self):
        plot = table([[1], [100]], toprule="skip", bottomrule="skip")

        assert _rows(plot) == ["  1", "100"]

    def test_a_column_of_anything_else_is_aligned_left(self):
        plot = table([["a"], ["bcd"]], toprule="skip", bottomrule="skip")

        assert _rows(plot) == ["a  ", "bcd"]

    def test_a_column_of_numbers_with_a_gap_is_still_aligned_right(self):
        plot = table([[1], [None], [100]], toprule="skip", bottomrule="skip")

        assert _rows(plot) == ["  1", "   ", "100"]

    def test_an_alignment_applies_to_every_column(self):
        plot = table(
            [["a"], ["bcd"]],
            aligns="center",
            toprule="skip",
            bottomrule="skip",
        )

        assert _rows(plot) == [" a ", "bcd"]

    def test_a_header_follows_the_alignment_of_its_column(self):
        plot = table([[1000]], headers=["n"], toprule="skip", midrule="skip")

        assert _rows(plot)[0] == "   n"


class TestTableRules:
    def test_the_default_is_a_double_rule_under_the_header(self):
        plot = table(RUNS)
        top, _header, mid, _first, _second, bottom = _rows(plot)

        assert set(top) == {"─"}
        assert set(mid) == {"═"}
        assert set(bottom) == {"─"}

    def test_a_table_with_no_header_row_has_no_midrule(self):
        plot = table([[1], [2]])

        assert len(_rows(plot)) == 2 + 2

    def test_every_rule_can_be_drawn_at_once(self):
        plot = table(
            [[1, 2]],
            headers=["a", "b"],
            leftrule="single",
            colrule="single",
            rightrule="single",
        )

        assert _rows(plot) == ["┌───┬───┐", "│ a │ b │", "╞═══╪═══╡",
                               "│ 1 │ 2 │", "└───┴───┘"]

    def test_a_blank_rule_holds_its_row_open(self):
        ruled = table([[1], [2]], rowrule="single")
        blank = table([[1], [2]], rowrule="blank")

        assert _rows(ruled)[2] == "─"
        assert _rows(blank)[2] == " "

    def test_a_skipped_rule_takes_no_space(self):
        plot = table([[1]], toprule="skip", bottomrule="skip")

        assert plot.height == 1

    def test_an_index_gets_a_column_of_its_own(self):
        plot = table([[1], [2]], index=["a", "b"], index_name="i",
                     headers=["n"], indexrule="single")

        assert _rows(plot)[1] == "i │ n"

    def test_a_rule_has_to_be_one_of_the_four_weights(self):
        with pytest.raises(ValueError, match="not 'dotted'"):
            table([[1]], toprule="dotted")

    def test_a_midrule_needs_a_header_to_go_under(self):
        with pytest.raises(ValueError, match="no header row"):
            table([[1]], midrule="single")

    def test_an_indexrule_needs_an_index_to_go_beside(self):
        with pytest.raises(ValueError, match="no index"):
            table([[1]], indexrule="single")


class TestTablePadding:
    def test_a_cell_is_held_away_from_the_rules_beside_it(self):
        plot = table([[1, 2]], leftrule="single", rightrule="single",
                     colrule="single", toprule="skip", bottomrule="skip")

        assert _rows(plot) == ["│ 1 │ 2 │"]

    def test_an_edge_with_no_rule_on_it_is_not_padded(self):
        plot = table([[1, 2]], toprule="skip", bottomrule="skip")

        assert _rows(plot) == ["1  2"]

    def test_the_padding_can_be_widened(self):
        plot = table([[1, 2]], cell_padding=2, toprule="skip",
                     bottomrule="skip")

        assert _rows(plot) == ["1    2"]

    def test_the_padding_cannot_be_negative(self):
        with pytest.raises(ValueError, match="cell_padding|-1 columns"):
            table([[1]], cell_padding=-1)


class TestTableColors:
    def test_the_body_can_be_colored_cell_by_cell(self):
        plot = table(
            [[1, 2]],
            bgcolors=np.array([[[255, 0, 0], [0, 0, 255]]]),
            toprule="skip",
            bottomrule="skip",
        )

        assert tuple(plot.chars.bg_rgb[0, 0]) == (255, 0, 0)
        assert tuple(plot.chars.bg_rgb[0, -1]) == (0, 0, 255)

    def test_a_colormap_shades_the_body_by_its_values(self):
        colormap = RecordingColormap()
        table([[0.25, 0.75]], bgcolors=[[0.25, 0.75]], colormap=colormap)

        assert colormap.input.tolist() == [[0.25, 0.75]]

    def test_the_header_and_the_index_take_the_header_color(self):
        plot = table(
            [[1]],
            headers=["n"],
            index=["r"],
            header_color="red",
            color="green",
        )

        # the header row, then the index label on the row below the midrule
        assert tuple(plot.chars.fg_rgb[1, 0]) == (255, 0, 0)
        assert tuple(plot.chars.fg_rgb[3, 0]) == (255, 0, 0)
        assert tuple(plot.chars.fg_rgb[3, -1]) == (0, 255, 0)

    def test_the_rules_follow_the_table_color_unless_told_otherwise(self):
        plot = table([[1]], color="green")
        recolored = table([[1]], color="green", rule_color="red")

        assert tuple(plot.chars.fg_rgb[0, 0]) == (0, 255, 0)
        assert tuple(recolored.chars.fg_rgb[0, 0]) == (255, 0, 0)


class TestTableRejections:
    def test_it_needs_something_to_tabulate(self):
        with pytest.raises(ValueError, match="at least one column"):
            table([])

    def test_the_rows_all_have_to_be_the_same_length(self):
        with pytest.raises(ValueError, match="row 1 has 1 values"):
            table([[1, 2], [3]])

    def test_a_row_has_to_be_a_row_of_values(self):
        with pytest.raises(ValueError, match="not a row of values"):
            table([1, 2, 3])

    def test_a_header_has_to_name_a_column_the_data_has(self):
        with pytest.raises(ValueError, match="no column 'nope'"):
            table(RUNS, headers=["nope"])

    def test_a_format_has_to_name_a_column_the_table_has(self):
        with pytest.raises(ValueError, match="not a column"):
            table(RUNS, formats={"nope": ".2f"})

    def test_a_format_per_column_needs_one_for_each(self):
        with pytest.raises(ValueError, match="2 entries for 3 columns"):
            table(RUNS, formats=[".2f", ".2f"])

    def test_a_format_per_column_name_needs_headers(self):
        with pytest.raises(ValueError, match="no headers"):
            table([[1]], formats={"a": ".2f"})

    def test_a_column_can_only_be_aligned_three_ways(self):
        with pytest.raises(ValueError, match="cannot align"):
            table([[1]], aligns="middle")

    def test_the_index_has_to_have_a_label_per_row(self):
        with pytest.raises(ValueError, match="2 index labels for 1 rows"):
            table([[1]], index=["a", "b"])

    def test_a_2d_table_has_no_column_names_to_rename(self):
        with pytest.raises(ValueError, match="names none"):
            table([[1]], headers={"a": "b"})

    def test_a_cell_cannot_smuggle_in_control_characters(self):
        with pytest.raises(ValueError, match="control characters"):
            table([["a\x1b[31mb"]])
