"""Unit tests for the plots that draw values as marks with length:
candlesticks and box plots."""

import numpy as np
import pytest

import matthewplotlib as mp

from matthewplotlib.plots import (
    axes,
    boxes,
    candles,
)


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
        assert plot.window.yrange == mp.scale(9.5, 13.0)

    def test_there_is_no_horizontal_coordinate(self):
        assert self.chart().window.xrange is None

    def test_axes_labels_the_value_axis_and_leaves_the_rest_alone(self):
        lines = axes(self.chart(length=4)).chars.to_plain_str().splitlines()
        # a gutter of labels down the left, one rule beside it, and no row of
        # labels underneath
        assert len(lines) == 4
        assert [line[:5] for line in lines] == ["13.0┐", "    │", "    │", " 9.5┘"]

    def test_the_rectangle_is_one_column_per_candle_by_default(self):
        plot = self.chart(length=5)
        assert (plot.height, plot.width) == (5, 4)

    def test_bodies_and_spacing_widen_the_rectangle(self):
        plot = self.chart(length=5, body_thickness=3, spacing=1)
        assert plot.width == 4 * 4 - 1

    def test_a_rising_candle_takes_the_rising_colour(self):
        plot = self.chart(length=4, rising=(1.0, 0.0, 0.0))
        column = plot.chars.fg_rgb[:, 0]
        drawn = column[plot.chars.fg[:, 0]]
        assert (drawn == np.array([255, 0, 0])).all(axis=-1).any()

    def test_a_falling_candle_takes_the_falling_colour(self):
        # the third period opened at 12.0 and closed at 11.5
        plot = self.chart(length=4, falling=(0.0, 0.0, 255))
        column = plot.chars.fg_rgb[:, 2]
        drawn = column[plot.chars.fg[:, 2]]
        assert (drawn == np.array([0, 0, 255])).all(axis=-1).any()

    def test_a_candle_that_closed_where_it_opened_counts_as_rising(self):
        plot = candles(
            opens=np.array([1.0]),
            highs=np.array([2.0]),
            lows=np.array([0.0]),
            closes=np.array([1.0]),
            length=4,
            rising=(1.0, 0.0, 0.0),
            falling=(0.0, 0.0, 1.0),
        )
        drawn = plot.chars.fg_rgb[plot.chars.fg]
        assert (drawn == np.array([255, 0, 0])).all(axis=-1).any()
        assert not (drawn == np.array([0, 0, 255])).all(axis=-1).any()

    def test_a_wick_takes_its_body_colour_by_default(self):
        plot = self.chart(length=6, rising=(1.0, 0.0, 0.0))
        # the first candle rose, so its wick is drawn in the rising colour
        wicks = np.isin(plot.chars.codes[:, 0], [ord("│"), ord("╵"), ord("╷")])
        assert wicks.any()
        assert (plot.chars.fg_rgb[wicks, 0] == np.array([255, 0, 0])).all()

    def test_a_wick_colour_applies_to_every_wick(self):
        plot = self.chart(length=6, wick=(1.0, 1.0, 1.0))
        wicks = np.isin(plot.chars.codes, [ord("│"), ord("╵"), ord("╷")])
        assert wicks.any()
        assert (plot.chars.fg_rgb[wicks] == 255).all()

    def test_the_background_is_painted_behind_the_whole_rectangle(self):
        plot = self.chart(length=5, background=(0.0, 0.0, 0.0))
        assert plot.chars.bg.all()
        assert (plot.chars.bg_rgb[~plot.chars.fg] == 0).all()

    def test_a_narrower_vrange_clips_the_candles_into_it(self):
        plot = self.chart(length=4, vrange=(12.0, 13.0))
        assert plot.window.yrange == mp.scale(12.0, 13.0)
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

    def test_candles_all_at_one_value_are_given_room(self):
        """A flat inferred interval widens around its value, as it does for
        the other coordinate plots, so the hairline candle sits mid-plot."""
        plot = candles(
            opens=np.array([1.0]),
            highs=np.array([1.0]),
            lows=np.array([1.0]),
            closes=np.array([1.0]),
        )
        assert plot.window.yrange == mp.scale(0.5, 1.5)

    def test_a_value_that_is_not_a_number_is_refused(self):
        """A period with an unknown high has no candle to draw, and the
        ordering check passes it silently, every comparison being false."""
        with pytest.raises(ValueError, match="high of nan"):
            candles(
                opens=np.array([1.0]),
                highs=np.array([float("nan")]),
                lows=np.array([0.0]),
                closes=np.array([1.5]),
            )

    def test_no_candles_and_no_range_to_infer_one_from(self):
        with pytest.raises(ValueError, match="no candles"):
            candles(
                opens=np.zeros(0),
                highs=np.zeros(0),
                lows=np.zeros(0),
                closes=np.zeros(0),
            )

    def test_a_repr_names_the_candles_and_their_window(self):
        assert repr(self.chart(length=4)) == (
            "candles(<4 candles>, window(y=[9.50,13.00], 4x4 cells))"
        )

    def test_the_scale_it_settled_on_is_kept_and_unpacks_as_a_pair(self):
        assert self.chart().vrange == mp.scale(9.5, 13.0)
        assert self.chart(vrange=(0.0, 20.0)).vrange == mp.scale(0.0, 20.0)
        vmin, vmax = self.chart().vrange
        assert (vmin, vmax) == (9.5, 13.0)

    def test_the_value_axis_takes_a_scale_into_its_window(self):
        plot = self.chart(vrange=mp.logscale(1.0, 20.0))
        assert plot.window.yrange == mp.logscale(1.0, 20.0)
        assert plot.vrange == mp.logscale(1.0, 20.0)


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
        assert plot.window.xrange == mp.scale(1, 9)

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

    def test_samples_all_at_one_value_are_given_room(self):
        """A flat inferred interval widens around its value, as it does for
        the other coordinate plots, so the collapsed box sits mid-plot."""
        plot = boxes([[3.0, 3.0, 3.0]])
        assert plot.window.xrange == mp.scale(2.5, 3.5)
        assert boxes([[3.0, 3.0]], vrange=(0.0, 6.0)).num_boxes == 1

    def test_a_sample_that_is_not_finite_is_left_out_of_the_summary(self):
        """It is a measurement that was not made, so it neither shifts the
        quartiles nor counts as a point beyond the whiskers."""
        samples = [1.0, 2.0, 3.0, 4.0, 5.0]

        assert boxes([samples]).vrange == boxes([[
            *samples, float("nan"), float("inf"),
        ]]).vrange

    def test_a_group_of_nothing_finite_is_rejected(self):
        with pytest.raises(ValueError, match="no finite samples"):
            boxes([[float("nan"), float("nan")]])


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
        assert plot.window.xrange == mp.scale(0, 19)
        assert plot.window.yrange is None

    def test_a_standing_plot_carries_its_range_vertically(self):
        plot = boxes(self.DATA, box_direction="vertical")
        assert plot.window.yrange == mp.scale(0, 19)
        assert plot.window.xrange is None

    def test_axes_labels_the_value_axis_and_leaves_the_rest_alone(self):
        lines = axes(boxes(self.DATA, length=20)).chars.to_plain_str()
        # the range's ends are labelled along the bottom, under a single rule
        assert "0" in lines.splitlines()[-1]
        assert "19" in lines.splitlines()[-1]

    def test_a_narrower_range_clips_the_boxes_into_it(self):
        plot = boxes(self.DATA, vrange=(5.0, 10.0), length=20)
        assert plot.window.xrange == mp.scale(5.0, 10.0)

    def test_a_point_outside_the_range_is_dropped(self):
        # one far outlier, and a range that excludes it
        data = [[*range(1, 10), 20.0]]
        assert "·" in boxes(data, length=40).chars.to_plain_str()
        inside = boxes(data, length=40, vrange=(1.0, 9.0))
        assert "·" not in inside.chars.to_plain_str()

    def test_the_repr_names_the_groups_and_the_window(self):
        assert repr(boxes(self.DATA)) == (
            "boxes(<2 groups>, window(x=[0.00,19.00], 30x7 cells))"
        )

    def test_the_scale_it_settled_on_is_kept_and_unpacks_as_a_pair(self):
        assert boxes(self.DATA).vrange == mp.scale(0.0, 19.0)
        assert boxes(self.DATA, vrange=(0.0, 20.0)).vrange \
            == mp.scale(0.0, 20.0)
        vmin, vmax = boxes(self.DATA).vrange
        assert (vmin, vmax) == (0.0, 19.0)

    def test_the_value_axis_takes_a_scale_into_its_window(self):
        plot = boxes([[1.0, 2.0, 4.0, 8.0]], vrange=mp.logscale())
        assert plot.window.xrange == mp.logscale(1.0, 8.0)
        assert plot.vrange == mp.logscale(1.0, 8.0)

    def test_an_outlier_the_scale_cannot_place_is_dropped(self):
        """On a log axis a negative sample has no position at all, so a
        negative outlier is dropped the way one outside the interval is,
        rather than being an error."""
        data = [[*range(1, 10), -20.0]]
        plot = boxes(data, length=40, vrange=mp.logscale(1.0, 9.0))
        assert "·" not in plot.chars.to_plain_str()


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
