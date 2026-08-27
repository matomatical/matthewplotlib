"""Unit tests for the arrangement plots: wrapping, centring, and cropping."""

import os

import pytest

from matthewplotlib.plots import (
    center,
    crop,
    text,
    wrap,
)


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
# center


class TestCenter:
    def square(self):
        return text("ab\ncd")

    def test_a_target_the_plot_already_fills_leaves_it_alone(self):
        plot = center(self.square(), height=1, width=1)

        assert (plot.height, plot.width) == (2, 2)
        assert plot.chars.to_plain_str() == "ab\ncd"

    def test_no_target_at_all_leaves_the_plot_alone(self):
        plot = center(self.square())

        assert (plot.height, plot.width) == (2, 2)
        assert plot.chars.to_plain_str() == "ab\ncd"

    def test_an_even_surplus_is_split_evenly(self):
        plot = center(self.square(), height=4, width=6)

        assert (plot.height, plot.width) == (4, 6)
        assert plot.chars.to_plain_str() == "      \n  ab  \n  cd  \n      "

    def test_an_odd_surplus_leaves_the_extra_below_and_right(self):
        plot = center(self.square(), height=3, width=5)

        assert (plot.height, plot.width) == (3, 5)
        assert plot.chars.to_plain_str() == " ab  \n cd  \n     "

    def test_each_direction_is_padded_independently(self):
        wide = center(self.square(), width=6)
        tall = center(self.square(), height=4)

        assert (wide.height, wide.width) == (2, 6)
        assert (tall.height, tall.width) == (4, 2)

    def test_the_padding_is_blank_rather_than_coloured(self):
        plot = center(text("ab", bgcolor="red"), height=3, width=4)

        assert plot.chars.bg.sum() == 2
        assert plot.chars.bg[1, 1:3].all()


# # #
# crop


class TestCrop:
    def grid(self, height=4, width=8):
        """A plot whose every cell is a distinct letter, so cuts are visible."""
        letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        rows = [letters[r * width:(r + 1) * width] for r in range(height)]
        return text("\n".join(rows))

    def test_a_plot_that_already_fits_is_left_alone(self):
        plot = crop(self.grid(), height=10, width=20)

        assert (plot.height, plot.width) == (4, 8)
        assert plot.chars.to_plain_str() == self.grid().chars.to_plain_str()

    def test_a_size_the_plot_exactly_meets_is_not_marked(self):
        plot = crop(self.grid(), height=4, width=8)

        assert plot.chars.to_plain_str() == self.grid().chars.to_plain_str()

    def test_cropping_both_directions_marks_both_edges(self):
        plot = crop(self.grid(), height=3, width=5)

        assert (plot.height, plot.width) == (3, 5)
        assert plot.chars.to_plain_str() == "abcd#\nijkl#\n#####"

    def test_cropping_the_width_leaves_the_height_whole(self):
        plot = crop(self.grid(), height=10, width=5)

        assert (plot.height, plot.width) == (4, 5)
        assert plot.chars.to_plain_str() == "abcd#\nijkl#\nqrst#\nyzAB#"

    def test_cropping_the_height_leaves_the_width_whole(self):
        plot = crop(self.grid(), height=2, width=20)

        assert (plot.height, plot.width) == (2, 8)
        assert plot.chars.to_plain_str() == "abcdefgh\n########"

    def test_the_marked_edge_costs_a_row_of_content(self):
        # the last of the three rows goes to the marker, so cropping four rows
        # to three shows two of them
        plot = crop(self.grid(), height=3, width=20)

        assert plot.chars.to_plain_str() == "abcdefgh\nijklmnop\n########"

    def test_no_room_for_content_leaves_the_marker_alone(self):
        assert crop(self.grid(), height=1, width=5).chars.to_plain_str() == "#####"
        assert crop(self.grid(), height=3, width=1).chars.to_plain_str() == "#\n#\n#"
        assert crop(self.grid(), height=1, width=1).chars.to_plain_str() == "#"

    def test_the_marker_glyph_is_configurable(self):
        plot = crop(self.grid(), height=3, width=5, marker="\u2026")

        assert plot.chars.to_plain_str() == "abcd\u2026\nijkl\u2026\n\u2026\u2026\u2026\u2026\u2026"

    def test_no_marker_keeps_the_whole_rectangle_of_content(self):
        plot = crop(self.grid(), height=3, width=5, marker=None)

        assert (plot.height, plot.width) == (3, 5)
        assert plot.chars.to_plain_str() == "abcde\nijklm\nqrstu"

    def test_no_marker_can_crop_to_a_single_cell(self):
        plot = crop(self.grid(), height=1, width=1, marker=None)

        assert plot.chars.to_plain_str() == "a"

    def test_the_markers_take_the_colors_and_the_content_keeps_its_own(self):
        plot = crop(
            text("abcd\nefgh\nijkl", fgcolor="red"),
            height=2,
            width=3,
            fgcolor="blue",
        )

        assert plot.chars.to_plain_str() == "ab#\n###"
        assert plot.chars.fg.all()
        assert plot.chars.fg_rgb[0, 0].tolist() == [255, 0, 0]
        assert plot.chars.fg_rgb[0, 2].tolist() == [0, 0, 255]
        assert plot.chars.fg_rgb[1, 0].tolist() == [0, 0, 255]

    def test_unmarked_edges_leave_the_markers_uncoloured(self):
        plot = crop(self.grid(), height=3, width=5)

        # the content is uncoloured to begin with, so nothing should be
        assert not plot.chars.fg.any()
        assert not plot.chars.bg.any()

    def test_the_marker_can_be_given_a_background(self):
        plot = crop(self.grid(), height=3, width=5, bgcolor="red")

        assert plot.chars.bg[:, 4].all()
        assert plot.chars.bg[2, :].all()
        assert not plot.chars.bg[0, 0]

    def test_cropping_does_not_write_into_the_plot_it_cropped(self):
        # a crop of a character array is a view of it, so the markers have to
        # land on a copy or they reach back into the plot being cropped
        source = self.grid()
        before = source.chars.to_plain_str()

        crop(source, height=3, width=5)
        crop(source, height=2, width=20)
        crop(source, height=1, width=1)

        assert source.chars.to_plain_str() == before

    def test_a_defaulted_size_measures_the_terminal(self, monkeypatch):
        monkeypatch.setattr(
            os,
            "get_terminal_size",
            lambda _fd: os.terminal_size((6, 10)),
        )

        plot = crop(self.grid(height=30, width=30))

        # a row short of the terminal's height, so that it can animate in it,
        # and the terminal's full width, which does not wrap
        assert (plot.height, plot.width) == (9, 6)

    def test_only_the_defaulted_direction_is_measured(self, monkeypatch):
        monkeypatch.setattr(
            os,
            "get_terminal_size",
            lambda _fd: os.terminal_size((6, 10)),
        )

        assert crop(self.grid(height=30, width=30), height=3).width == 6
        assert crop(self.grid(height=30, width=30), width=3).height == 9

    def test_a_one_row_terminal_still_leaves_room_for_the_marker(
        self,
        monkeypatch,
    ):
        monkeypatch.setattr(
            os,
            "get_terminal_size",
            lambda _fd: os.terminal_size((6, 1)),
        )

        assert crop(self.grid()).height == 1

    def test_a_defaulted_size_needs_a_terminal_to_measure(self, monkeypatch):
        def no_terminal(_fd):
            raise OSError("no attached terminal")

        monkeypatch.setattr(os, "get_terminal_size", no_terminal)

        with pytest.raises(ValueError, match="no terminal to measure"):
            crop(self.grid())

    def test_a_size_given_in_full_needs_no_terminal(self, monkeypatch):
        def no_terminal(_fd):
            raise OSError("no attached terminal")

        monkeypatch.setattr(os, "get_terminal_size", no_terminal)

        assert crop(self.grid(), height=3, width=5).height == 3

    def test_a_size_has_to_be_positive(self):
        with pytest.raises(ValueError, match="height must be positive"):
            crop(self.grid(), height=0, width=5)
        with pytest.raises(ValueError, match="width must be positive"):
            crop(self.grid(), height=5, width=-1)

    def test_a_marker_has_to_be_a_single_character(self):
        with pytest.raises(ValueError, match="single character"):
            crop(self.grid(), height=3, width=5, marker="##")
        with pytest.raises(ValueError, match="single character"):
            crop(self.grid(), height=3, width=5, marker="")

    def test_a_marker_cannot_smuggle_in_control_characters(self):
        with pytest.raises(ValueError, match="control characters"):
            crop(self.grid(), height=3, width=5, marker="\x1b")

    def test_a_plot_of_no_rows_crops_to_no_rows(self):
        plot = crop(text(""), height=3, width=5)

        assert (plot.height, plot.width) == (0, 0)

    def test_repr_reports_the_size_and_the_marker(self):
        plot = crop(text("abcd"), height=1, width=3)

        assert repr(plot) == (
            "crop(height=1, width=3, marker='#', "
            "plot=text(height=1, width=4, text='abcd'))"
        )
