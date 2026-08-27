"""Unit tests for the plots that dress other plots: text, axes, and
colorbars."""

import numpy as np
import pytest

import matthewplotlib as mp

from matthewplotlib.plots import (
    axes,
    border,
    colorbar,
    heatmap,
    image,
    scatter,
    text,
)

from tests.plots_common import RecordingColormap


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


class TestTextSizeAndAlignment:
    def test_by_default_it_is_the_size_of_its_text(self):
        assert (text("ab\ncde").height, text("ab\ncde").width) == (2, 3)

    def test_a_width_and_height_are_minimums(self):
        assert (text("ab", height=3, width=6).height,
                text("ab", height=3, width=6).width) == (3, 6)
        assert (text("abcdef", height=1, width=2).height,
                text("abcdef", height=1, width=2).width) == (1, 6)

    @pytest.mark.parametrize("align, drawn", [
        ("left", "ab    "),
        ("center", "  ab  "),
        ("right", "    ab"),
    ])
    def test_alignment_places_each_line_in_the_width(self, align, drawn):
        assert text("ab", width=6, align=align).chars.to_plain_str() == drawn

    def test_alignment_has_no_room_to_act_without_a_width(self):
        for align in ("left", "center", "right"):
            assert text("ab", align=align).chars.to_plain_str() == "ab"

    def test_the_empty_string_is_a_plot_of_no_rows(self):
        """It has no lines in it, where `"\\n"` has one empty line, and the two
        stay distinct rather than both becoming one blank row."""
        empty = text("")

        assert (empty.height, empty.width) == (0, 0)
        assert (text("\n").height, text("\n").width) == (1, 0)

    def test_a_plot_of_no_rows_still_composes_and_renders(self):
        assert str(text("")) == ""
        assert (text("") + text("x")).chars.to_plain_str() == "x"


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
# colorbar


class TestColorbar:
    def test_it_takes_its_interval_from_a_plot(self):
        heat = heatmap([[0.0, 20.0]], colormap=mp.viridis)

        bar = colorbar(heat, colormap=mp.viridis)

        assert bar.vrange == (0.0, 20.0)
        assert bar.window.yrange == (0.0, 20.0)

    def test_an_interval_needs_no_plot(self):
        bar = colorbar((-2.0, 2.0), colormap=mp.viridis)

        assert bar.vrange == (-2.0, 2.0)

    def test_the_colormap_is_never_read_off_the_plot(self):
        """Naming it at the bar is the whole of the contract, so that a plot's
        colormap is never quietly assumed to be one a gradient can stand for.
        """
        heat = heatmap([[0.0, 1.0]], colormap=mp.viridis)

        grey = colorbar(heat, length=1)
        viridis = colorbar(heat, colormap=mp.viridis, length=1)

        assert grey.chars.fg_rgb[0, 0].tolist() == [255, 255, 255]
        assert viridis.chars.fg_rgb[0, 0].tolist() == [253, 231, 36]

    @pytest.mark.parametrize(
        "direction, xrange, yrange, first, last",
        [
            ("up", None, (0.0, 10.0), 1.0, 0.0),
            ("down", None, (10.0, 0.0), 0.0, 1.0),
            ("right", (0.0, 10.0), None, 0.0, 1.0),
            ("left", (10.0, 0.0), None, 1.0, 0.0),
        ],
    )
    def test_each_direction_places_the_interval_its_own_way(
        self, direction, xrange, yrange, first, last,
    ):
        """The coordinate runs from the low end of the screen axis to the high
        end, and the ramp is built in screen order to match it: the first row
        or the leftmost column first."""
        colormap = RecordingColormap()

        bar = colorbar(
            (0.0, 10.0),
            colormap=colormap,
            direction=direction,
            length=3,
        )

        assert (bar.window.xrange, bar.window.yrange) == (xrange, yrange)
        ramp = colormap.input.reshape(-1) if xrange is None \
          else colormap.input[0]
        assert (ramp[0], ramp[-1]) == (first, last)

    @pytest.mark.parametrize(
        "direction, height, width",
        [
            ("up", 4, 2), ("down", 4, 2), ("left", 2, 4), ("right", 2, 4),
        ],
    )
    def test_length_runs_along_the_scale_and_thickness_across_it(
        self, direction, height, width,
    ):
        bar = colorbar((0.0, 1.0), direction=direction, length=4, thickness=2)

        assert (bar.height, bar.width) == (height, width)

    def test_a_vertical_bar_has_twice_the_gradient_resolution(self):
        """A character cell holds two half-block pixels vertically and one
        horizontally, so the same length buys twice the steps."""
        vertical, horizontal = RecordingColormap(), RecordingColormap()

        colorbar((0.0, 1.0), colormap=vertical, direction="up", length=6)
        colorbar((0.0, 1.0), colormap=horizontal, direction="right", length=6)

        assert vertical.input.shape[0] == 2 * horizontal.input.shape[1]

    def test_a_plot_that_kept_no_interval_is_refused(self):
        with pytest.raises(ValueError, match="no interval"):
            colorbar(image([[0.0, 1.0]]))

    def test_any_plot_that_kept_one_lends_it(self):
        """Including the plots that measure something other than a colour by
        it: what the colours mean is settled where the bar is drawn."""
        field = mp.vfunction2(
            lambda xy: xy,
            xrange=(-1.0, 1.0),
            yrange=(-1.0, 1.0),
            width=4,
            height=1,
        )

        assert colorbar(field, colormap=mp.reds).vrange == field.vrange
        assert colorbar(mp.bars([1.0, 2.0])).vrange == (0.0, 2.0)

    def test_an_unknown_direction_is_refused(self):
        with pytest.raises(ValueError, match="up, down, left or right"):
            colorbar((0.0, 1.0), direction="north")

    def test_it_is_labelled_along_the_one_side_that_means_anything(self):
        assert axes(colorbar((0.0, 1.0))).sides \
            == ("crop", "crop", "crop", "label")
        assert axes(colorbar((0.0, 1.0), direction="right")).sides \
            == ("crop", "crop", "label", "crop")
