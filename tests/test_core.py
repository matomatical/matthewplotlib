import re

import numpy as np
import pytest

import matthewplotlib as mp
from matthewplotlib.core import (
    CharArray,
    BoxStyle,
    ords,
    unicode_bar,
    unicode_col,
    unicode_box,
    unicode_braille_array,
    unicode_image,
)


# # #
# ords


class TestOrds:
    def test_single_char(self):
        assert ords("A") == [65]

    def test_string(self):
        assert ords("abc") == [97, 98, 99]

    def test_unicode(self):
        assert ords("█") == [ord("█")]

    def test_empty(self):
        assert ords("") == []


# # #
# CharArray properties


class TestCharArrayProperties:
    def test_height_and_width(self):
        ca = CharArray.from_size(5, 10)
        assert ca.height == 5
        assert ca.width == 10

    def test_isblank_all_spaces(self):
        ca = CharArray.from_size(2, 3)
        assert np.all(ca.isblank())

    def test_isnonblank_with_content(self):
        ca = CharArray.from_size(2, 3)
        ca.codes[0, 0] = ord("X")
        assert ca.isnonblank()[0, 0]
        assert not ca.isnonblank()[0, 1]

    def test_isblank_with_bgcolor(self):
        ca = CharArray.from_size(1, 1, bgcolor="red")
        assert not ca.isblank()[0, 0]

    def test_isnonblank_with_bgcolor(self):
        ca = CharArray.from_size(1, 1, bgcolor="red")
        assert ca.isnonblank()[0, 0]


# # #
# CharArray.pad


class TestCharArrayPad:
    def test_pad_increases_size(self):
        ca = CharArray.from_size(2, 3)
        padded = ca.pad(above=1, below=1, left=2, right=2)
        assert padded.height == 4
        assert padded.width == 7

    def test_pad_preserves_content(self):
        ca = CharArray.from_size(1, 1)
        ca.codes[0, 0] = ord("X")
        padded = ca.pad(above=1, left=1)
        assert padded.codes[1, 1] == ord("X")

    def test_pad_zero_is_identity(self):
        ca = CharArray.from_size(2, 3)
        ca.codes[0, 0] = ord("A")
        padded = ca.pad()
        assert padded.height == 2
        assert padded.width == 3
        assert padded.codes[0, 0] == ord("A")


# # #
# CharArray.to_plain_str


class TestCharArrayPlainStr:
    def test_simple_text(self):
        ca = CharArray.from_size(1, 5)
        for i, c in enumerate("hello"):
            ca.codes[0, i] = ord(c)
        assert ca.to_plain_str() == "hello"

    def test_multiline(self):
        ca = CharArray.from_size(2, 3)
        for i, c in enumerate("abc"):
            ca.codes[0, i] = ord(c)
        for i, c in enumerate("def"):
            ca.codes[1, i] = ord(c)
        assert ca.to_plain_str() == "abc\ndef"


# # #
# CharArray.to_ansi_str


class TestCharArrayANSIStr:
    def test_no_colors_produces_no_escape_codes(self):
        ca = CharArray.from_size(1, 3)
        for i, c in enumerate("abc"):
            ca.codes[0, i] = ord(c)
        result = ca.to_ansi_str()
        assert result == "abc"
        assert "\x1b" not in result

    def test_fg_color_emits_code_and_reset(self):
        ca = CharArray.from_size(1, 2, fgcolor="red")
        ca.codes[0, 0] = ord("a")
        ca.codes[0, 1] = ord("b")
        result = ca.to_ansi_str()
        # should start with fg escape
        assert result.startswith("\x1b[38;2;255;0;0m")
        # should end with reset
        assert result.endswith("\x1b[0m")
        # characters should be present
        assert "ab" in result

    def test_fg_color_merged_across_same_color(self):
        """Same fg color on consecutive chars should only emit once."""
        ca = CharArray.from_size(1, 3, fgcolor="red")
        for i, c in enumerate("abc"):
            ca.codes[0, i] = ord(c)
        result = ca.to_ansi_str()
        # the fg escape code should appear exactly once
        assert result.count("\x1b[38;2;255;0;0m") == 1

    def test_fg_color_change_emits_new_code(self):
        """Different fg colors should emit separate codes."""
        ca = CharArray.from_size(1, 2)
        ca.codes[0, 0] = ord("a")
        ca.codes[0, 1] = ord("b")
        # first char: red
        ca.fg[0, 0] = True
        ca.fg_rgb[0, 0] = [255, 0, 0]
        # second char: blue
        ca.fg[0, 1] = True
        ca.fg_rgb[0, 1] = [0, 0, 255]
        result = ca.to_ansi_str()
        assert "\x1b[38;2;255;0;0m" in result
        assert "\x1b[38;2;0;0;255m" in result

    def test_reset_at_end_of_colored_line(self):
        """Each line with colors should end with \\x1b[0m reset."""
        ca = CharArray.from_size(2, 1, fgcolor="green")
        ca.codes[0, 0] = ord("a")
        ca.codes[1, 0] = ord("b")
        result = ca.to_ansi_str()
        lines = result.split("\n")
        assert len(lines) == 2
        for line in lines:
            assert line.endswith("\x1b[0m")

    def test_fg_reset_when_color_removed(self):
        """When fg goes from colored to uncolored, a fg reset (39) is emitted."""
        ca = CharArray.from_size(1, 2)
        ca.codes[0, 0] = ord("a")
        ca.codes[0, 1] = ord("b")
        ca.fg[0, 0] = True
        ca.fg_rgb[0, 0] = [255, 0, 0]
        ca.fg[0, 1] = False
        result = ca.to_ansi_str()
        # should contain fg set, then fg reset (39)
        assert "\x1b[38;2;255;0;0m" in result
        assert "\x1b[39m" in result

    def test_bg_color_emits_bg_code(self):
        ca = CharArray.from_size(1, 1, bgcolor="blue")
        ca.codes[0, 0] = ord("x")
        result = ca.to_ansi_str()
        assert "\x1b[48;2;0;0;255m" in result
        assert result.endswith("\x1b[0m")

    def test_no_reset_for_uncolored_lines(self):
        """A multiline CharArray with no colors should have no resets."""
        ca = CharArray.from_size(2, 2)
        for i, c in enumerate("ab"):
            ca.codes[0, i] = ord(c)
        for i, c in enumerate("cd"):
            ca.codes[1, i] = ord(c)
        result = ca.to_ansi_str()
        assert result == "ab\ncd"
        assert "\x1b" not in result

    def test_colors_re_emitted_after_newline_reset(self):
        """After newline reset, the same color must be re-emitted."""
        ca = CharArray.from_size(2, 1, fgcolor="red")
        ca.codes[0, 0] = ord("a")
        ca.codes[1, 0] = ord("b")
        result = ca.to_ansi_str()
        # the fg code should appear twice: once per line
        assert result.count("\x1b[38;2;255;0;0m") == 2


# # #
# unicode_braille_array


class TestUnicodeBrailleArray:
    def test_docstring_example(self):
        """Test the example from the unicode_braille_array docstring."""
        dots = np.array([
            [1,0, 0,1, 0,1, 1,1, 1,0, 1,0, 0,0, 0,1, 0,0, 0,0, 0,1, 1,0],
            [1,0, 0,1, 0,1, 0,0, 0,0, 1,0, 0,0, 0,1, 0,0, 0,0, 1,0, 0,1],
            [1,0, 0,1, 0,1, 0,0, 0,0, 1,0, 0,0, 0,1, 0,0, 0,0, 1,0, 0,1],
            [1,0, 0,1, 0,1, 0,0, 0,0, 1,0, 0,0, 0,1, 0,0, 0,0, 1,0, 0,1],
            [1,1, 1,1, 0,1, 1,1, 1,0, 1,0, 0,0, 0,1, 0,0, 0,0, 1,0, 0,1],
            [1,0, 0,1, 0,1, 0,0, 0,0, 1,0, 0,0, 0,1, 0,0, 0,0, 1,0, 0,1],
            [1,0, 0,1, 0,1, 0,0, 0,0, 1,0, 0,0, 0,1, 0,0, 0,0, 1,0, 0,1],
            [1,0, 0,1, 0,1, 1,1, 1,0, 1,1, 1,1, 0,1, 1,1, 1,0, 0,1, 1,0],
        ])
        ca = unicode_braille_array(dots)
        assert ca.height == 2
        assert ca.width == 12
        result = ca.to_plain_str()
        # empty cells (all-zero 4x2 blocks) become spaces, not braille blank
        assert result == (
            "⡇⢸⢸⠉⠁⡇ ⢸  ⡎⢱\n"
            "⡏⢹⢸⣉⡁⣇⣀⢸⣀⡀⢇⡸"
        )

    def test_all_zeros_gives_spaces(self):
        dots = np.zeros((4, 2), dtype=int)
        ca = unicode_braille_array(dots)
        assert ca.height == 1
        assert ca.width == 1
        assert ca.to_plain_str() == " "

    def test_all_ones_gives_full_braille(self):
        dots = np.ones((4, 2), dtype=int)
        ca = unicode_braille_array(dots)
        assert ca.height == 1
        assert ca.width == 1
        # all 8 dots = 0xFF, char = 0x2800 + 0xFF = ⣿
        assert ca.to_plain_str() == "⣿"

    def test_padding_odd_dimensions(self):
        """Non-multiple dimensions should be padded to 4×2 cells."""
        dots = np.ones((3, 3), dtype=int)
        ca = unicode_braille_array(dots)
        assert ca.height == 1  # ceil(3/4) = 1
        assert ca.width == 2   # ceil(3/2) = 2

    def test_single_dot(self):
        dots = np.zeros((4, 2), dtype=int)
        dots[0, 0] = 1  # dot 1
        ca = unicode_braille_array(dots)
        # dot 1 = bit 0 = 0x01, char = 0x2800 + 1 = ⠁
        assert ca.to_plain_str() == "⠁"

    def test_hello_text(self):
        """Braille rendering of 'HELLO' spelled out as a dot pattern."""
        dots = np.array([
            [0,1,0,1,1,0,1,1,0,1,0,0,1,1,0,1,1,0,1,1,0,1,1,0],
            [1,1,0,0,1,0,1,1,0,1,1,0,1,0,0,1,1,0,1,1,0,1,1,0],
            [1,1,0,1,1,0,0,1,0,1,1,0,1,1,0,1,0,0,1,1,0,1,1,0],
            [1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,0,1,0,1,0,0],
            [1,0,0,1,0,1,1,1,1,0,1,0,0,0,0,1,0,0,0,0,0,1,1,0],
            [1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1],
            [1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1],
            [1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1],
            [1,1,1,1,0,1,1,1,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1],
            [1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1],
            [1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1],
            [1,0,0,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,0,1,1,0],
        ]).astype(bool)
        ca = unicode_braille_array(dots)
        assert ca.height == 3
        assert ca.width == 12
        assert ca.to_plain_str() == (
            "⣾⢨⡇⣻⢸⡆⣯⢸⡃⢿⢸⠇\n"
            "⡇⢸⢸⠉⠁⡇ ⢸  ⡎⢱\n"
            "⡏⢹⢸⣉⡁⣇⣀⢸⣀⡀⢇⡸"
        )


# # #
# unicode_bar


class TestUnicodeBar:
    def test_full_bar(self):
        assert unicode_bar(1.0, 5).to_plain_str() == "█████"

    def test_empty_bar(self):
        assert unicode_bar(0.0, 5).to_plain_str() == "     "

    def test_half_bar(self):
        assert unicode_bar(0.5, 10).to_plain_str() == "█████     "

    def test_docstring_example_0625(self):
        """Docstring example: unicode_bar(0.625, 10)"""
        assert unicode_bar(0.625, 10).to_plain_str() == "██████▎   "

    def test_dimensions(self):
        ca = unicode_bar(0.5, 8, height=3)
        assert ca.height == 3
        assert ca.width == 8

    def test_multi_row_bar_is_uniform(self):
        """All rows of a multi-row bar should be identical."""
        ca = unicode_bar(0.5, 6, height=3)
        result = ca.to_plain_str()
        lines = result.split("\n")
        assert len(lines) == 3
        assert lines[0] == lines[1] == lines[2]

    def test_clamp_above_one(self):
        assert unicode_bar(1.5, 4).to_plain_str() == "████"

    def test_clamp_below_zero(self):
        assert unicode_bar(-0.5, 4).to_plain_str() == "    "

    def test_width_one_full(self):
        assert unicode_bar(1.0, 1).to_plain_str() == "█"

    def test_width_one_empty(self):
        assert unicode_bar(0.0, 1).to_plain_str() == " "

    @pytest.mark.parametrize("eighths, char", [
        (1, "▏"), (2, "▎"), (3, "▍"), (4, "▌"),
        (5, "▋"), (6, "▊"), (7, "▉"),
    ])
    def test_partial_block_characters(self, eighths, char):
        """Each 1/8 fraction of a single-width bar uses correct partial."""
        result = unicode_bar(eighths / 8, 1).to_plain_str()
        assert result == char


# # #
# unicode_col


class TestUnicodeCol:
    def test_full_col(self):
        assert unicode_col(1.0, 3).to_plain_str() == "█\n█\n█"

    def test_empty_col(self):
        assert unicode_col(0.0, 3).to_plain_str() == " \n \n "

    def test_half_col_docstring_example(self):
        """Docstring example: unicode_col(0.5, 3)"""
        assert unicode_col(0.5, 3).to_plain_str() == " \n▄\n█"

    def test_dimensions(self):
        ca = unicode_col(0.5, 5, width=2)
        assert ca.height == 5
        assert ca.width == 2

    def test_multi_col_is_uniform(self):
        """All columns of a multi-width col should be identical."""
        ca = unicode_col(0.5, 4, width=3)
        for i in range(ca.height):
            assert ca.codes[i, 0] == ca.codes[i, 1] == ca.codes[i, 2]

    def test_clamp_above_one(self):
        assert unicode_col(1.5, 3).to_plain_str() == "█\n█\n█"

    def test_clamp_below_zero(self):
        assert unicode_col(-0.5, 3).to_plain_str() == " \n \n "

    def test_height_one_full(self):
        assert unicode_col(1.0, 1).to_plain_str() == "█"

    def test_height_one_empty(self):
        assert unicode_col(0.0, 1).to_plain_str() == " "

    def test_partial_block_one_eighth(self):
        """1/8 of height 1 → ▁"""
        assert unicode_col(1/8, 1).to_plain_str() == "▁"

    def test_partial_block_one_quarter(self):
        assert unicode_col(2/8, 1).to_plain_str() == "▂"

    def test_partial_block_three_eighths(self):
        assert unicode_col(3/8, 1).to_plain_str() == "▃"

    def test_partial_block_half(self):
        assert unicode_col(4/8, 1).to_plain_str() == "▄"

    def test_partial_block_five_eighths(self):
        assert unicode_col(5/8, 1).to_plain_str() == "▅"

    def test_partial_block_three_quarters(self):
        assert unicode_col(6/8, 1).to_plain_str() == "▆"

    def test_partial_block_seven_eighths(self):
        assert unicode_col(7/8, 1).to_plain_str() == "▇"

    def test_grows_from_bottom(self):
        """Column should grow upward: bottom rows fill first."""
        result = unicode_col(0.75, 4).to_plain_str()
        assert result == " \n█\n█\n█"

    def test_partial_at_top_of_filled_region(self):
        """3/8 of height 4 = 12 eighths = 1 full + 4/8 partial."""
        result = unicode_col(3/8, 4).to_plain_str()
        assert result == " \n \n▄\n█"


# # #
# unicode_box


class TestUnicodeBox:
    def test_light_box_snapshot(self):
        inner = CharArray.from_size(1, 3)
        for i, c in enumerate("abc"):
            inner.codes[0, i] = ord(c)
        boxed = unicode_box(inner, BoxStyle.LIGHT)
        assert boxed.to_plain_str() == (
            "┌───┐\n"
            "│abc│\n"
            "└───┘"
        )

    def test_dimensions(self):
        inner = CharArray.from_size(2, 4)
        boxed = unicode_box(inner, BoxStyle.LIGHT)
        assert boxed.height == 4  # 2 + 2
        assert boxed.width == 6   # 4 + 2

    def test_title(self):
        inner = CharArray.from_size(1, 6)
        boxed = unicode_box(inner, BoxStyle.LIGHT, title="hi")
        top_row = boxed.to_plain_str().split("\n")[0]
        assert "hi" in top_row

    def test_title_truncated_to_inner_width(self):
        inner = CharArray.from_size(1, 3)
        boxed = unicode_box(inner, BoxStyle.LIGHT, title="toolong")
        top_row = boxed.to_plain_str().split("\n")[0]
        # title should be truncated to inner width (3)
        assert len(top_row) == 5  # 3 inner + 2 border

    def test_heavy_box_snapshot(self):
        inner = CharArray.from_size(1, 2)
        inner.codes[0, 0] = ord("X")
        inner.codes[0, 1] = ord("Y")
        boxed = unicode_box(inner, BoxStyle.HEAVY)
        assert boxed.to_plain_str() == (
            "┏━━┓\n"
            "┃XY┃\n"
            "┗━━┛"
        )

    def test_preserves_inner_content(self):
        inner = CharArray.from_size(2, 2)
        inner.codes[0, 0] = ord("a")
        inner.codes[0, 1] = ord("b")
        inner.codes[1, 0] = ord("c")
        inner.codes[1, 1] = ord("d")
        boxed = unicode_box(inner, BoxStyle.LIGHT)
        result = boxed.to_plain_str()
        lines = result.split("\n")
        assert lines[1] == "│ab│"
        assert lines[2] == "│cd│"


# # #
# unicode_image


class TestUnicodeImage:
    def test_even_height_dimensions(self):
        """A 4×3 image should produce a 2×3 CharArray."""
        img = np.zeros((4, 3, 3), dtype=np.uint8)
        ca = unicode_image(img)
        assert ca.height == 2
        assert ca.width == 3

    def test_odd_height_dimensions(self):
        """A 3×2 image should produce a 2×2 CharArray (ceil(3/2)=2)."""
        img = np.zeros((3, 2, 3), dtype=np.uint8)
        ca = unicode_image(img)
        assert ca.height == 2
        assert ca.width == 2

    def test_single_pixel(self):
        """A 1×1 image should produce a 1×1 CharArray."""
        img = np.array([[[255, 0, 0]]], dtype=np.uint8)
        ca = unicode_image(img)
        assert ca.height == 1
        assert ca.width == 1

    def test_uses_upper_half_block(self):
        """All characters should be ▀ (upper half block)."""
        img = np.zeros((2, 3, 3), dtype=np.uint8)
        ca = unicode_image(img)
        assert np.all(ca.codes == ord("▀"))

    def test_top_pixel_is_fg_bottom_is_bg(self):
        """For a 2×1 image, top pixel should be fg and bottom pixel bg."""
        red = [255, 0, 0]
        blue = [0, 0, 255]
        img = np.array([[red], [blue]], dtype=np.uint8)
        ca = unicode_image(img)
        assert ca.height == 1
        assert ca.width == 1
        assert ca.fg[0, 0]
        assert ca.bg[0, 0]
        assert np.array_equal(ca.fg_rgb[0, 0], red)
        assert np.array_equal(ca.bg_rgb[0, 0], blue)

    def test_not_transposed(self):
        """A 2×4 image (tall×wide) should produce 1×4, not 4×1."""
        img = np.zeros((2, 4, 3), dtype=np.uint8)
        ca = unicode_image(img)
        assert ca.height == 1
        assert ca.width == 4

    def test_not_transposed_tall(self):
        """A 6×2 image should produce 3×2, not 2×3."""
        img = np.zeros((6, 2, 3), dtype=np.uint8)
        ca = unicode_image(img)
        assert ca.height == 3
        assert ca.width == 2

    def test_odd_height_last_row_bg_disabled(self):
        """For odd-height images, the bottom bg of the last row should be off."""
        img = np.ones((3, 2, 3), dtype=np.uint8) * 128
        ca = unicode_image(img)
        assert ca.height == 2
        # first row: both fg and bg should be on
        assert ca.bg[0, 0]
        assert ca.bg[0, 1]
        # last row: bg should be off (padded row)
        assert not ca.bg[1, 0]
        assert not ca.bg[1, 1]
        # fg of last row should still be on (real pixel)
        assert ca.fg[1, 0]
        assert ca.fg[1, 1]

    def test_even_height_all_bg_enabled(self):
        """For even-height images, all bg should be enabled."""
        img = np.ones((4, 2, 3), dtype=np.uint8) * 128
        ca = unicode_image(img)
        assert ca.height == 2
        assert np.all(ca.bg)

    def test_color_mapping_multirow(self):
        """Verify correct pixel-to-cell mapping across multiple rows."""
        # 4×1 image, each pixel a different color
        img = np.array([
            [[255, 0, 0]],     # row 0 → fg of cell (0,0)
            [[0, 255, 0]],     # row 1 → bg of cell (0,0)
            [[0, 0, 255]],     # row 2 → fg of cell (1,0)
            [[255, 255, 0]],   # row 3 → bg of cell (1,0)
        ], dtype=np.uint8)
        ca = unicode_image(img)
        assert ca.height == 2
        assert np.array_equal(ca.fg_rgb[0, 0], [255, 0, 0])
        assert np.array_equal(ca.bg_rgb[0, 0], [0, 255, 0])
        assert np.array_equal(ca.fg_rgb[1, 0], [0, 0, 255])
        assert np.array_equal(ca.bg_rgb[1, 0], [255, 255, 0])


# # #
# CharArray.to_ansi_diff_str (differential rendering)
#
# What these sequences do to a screen is tested against a real terminal, in
# `test_terminal.py`. Here are the claims about the string itself.


class TestCharArrayDiffStr:
    def test_different_shapes_are_diffed_not_rejected(self):
        a = CharArray.from_size(3, 4, fgcolor="red")
        b = CharArray.from_size(3, 5, fgcolor="blue")
        assert b.to_ansi_diff_str(a)          # no exception; see the resize tests

    def test_empty_plot_raises(self):
        """updatestr owns the empty cases, so the array-level call rejects them
        rather than silently emitting a sequence for a plot with no rows."""
        a = CharArray.from_size(3, 4)
        empty = CharArray.from_size(height=0, width=0)
        with pytest.raises(ValueError):
            a.to_ansi_diff_str(empty)
        with pytest.raises(ValueError):
            empty.to_ansi_diff_str(a)

    def test_diff_repaints_only_changed_cell(self):
        rng = np.random.default_rng(2)
        prev = unicode_image(rng.integers(0, 256, (8, 8, 3), dtype=np.uint8))
        new = unicode_image(rng.integers(0, 256, (8, 8, 3), dtype=np.uint8))
        # force exactly one differing cell
        new.codes = prev.codes.copy()
        new.fg = prev.fg.copy()
        new.fg_rgb = prev.fg_rgb.copy()
        new.bg = prev.bg.copy()
        new.bg_rgb = prev.bg_rgb.copy()
        new.fg_rgb[1, 5] = (prev.fg_rgb[1, 5].astype(int) + 40) % 256
        diff = new.to_ansi_diff_str(prev)
        # the diff carries a single glyph, far smaller than a full redraw
        assert diff.count("▀") == 1
        assert len(diff) < len(new.to_ansi_str())


class TestPlotUpdateStr:
    def test_sub_operator_matches_updatestr(self):
        a = mp.image(np.random.default_rng(5).random((6, 8)))
        b = mp.image(np.random.default_rng(6).random((6, 8)))
        assert (b - a) == b.updatestr(a)

    def test_updatestr_of_none_renders_the_whole_plot(self):
        """The seed frame of an animation: nothing on screen to diff against."""
        p = mp.image(np.random.default_rng(9).random((6, 8)))
        assert p.updatestr(None) == p.renderstr()
        assert (p - None) == p.renderstr()

    def test_updatestr_handles_an_empty_previous_plot(self):
        p = mp.image(np.random.default_rng(7).random((6, 8)))
        empty = mp.blank(height=0, width=0)
        assert p.updatestr(empty) == p.renderstr()   # nothing on screen to diff
        assert empty.updatestr(p) == p.clearstr()    # nothing left to show


class TestPlotClearStr:
    def test_clearstr_of_an_empty_plot_is_a_bare_cursor_up(self):
        """Nothing to erase, so the sequence is only the step above the plot --
        and never `CSI 0 A`, which a terminal reads as `CSI 1 A`."""
        z = mp.blank(height=0, width=0)
        assert z.clearstr() == "\x1b[1A"


# # #
# THE EMITTED VOCABULARY


# The complete set of escape sequences the library is allowed to send, by final
# byte. Everything here is in the VT100 core, which is what keeps
# `pages/compatibility.md` short; the only sequences that are not are the SGR
# colours, and those degrade to a nearby colour rather than corrupting a screen.
ALLOWED_FINALS = {
    "A": "CUU, cursor up",
    "B": "CUD, cursor down",
    "C": "CUF, cursor forward",
    "D": "CUB, cursor back",
    "K": "EL, erase in line",
    "m": "SGR, select graphic rendition",
}

# Retired by the escape-sequence audit, and not to be reintroduced without a
# corresponding row on the compatibility page. See the reasoning in
# notes/closed/escape-vocabulary.md.
RETIRED_FINALS = {
    "E": "CNL, cursor next line (now a carriage return and a cursor down)",
    "G": "CHA, absolute column (now a carriage return and a cursor forward)",
    "X": "ECH, erase character (now written spaces)",
}

CSI = re.compile(r"\x1b\[([0-9;]*)([A-Za-z])")


def _sequences(emitted: str) -> list[tuple[str, str]]:
    """Every escape sequence in `emitted`, as (final byte, parameters).

    Also insists that every escape in the string is a CSI sequence of this
    plain shape. A regex that only looks for the sequences we know about would
    let an OSC, a two-byte escape or a private-mode sequence through unread.
    """
    found = [(m.group(2), m.group(1)) for m in CSI.finditer(emitted)]
    assert emitted.count("\x1b") == len(found), (
        f"{emitted!r} contains an escape that is not a plain CSI sequence"
    )
    return found


def _sgr_is_allowed(params: str) -> bool:
    """SGR is only ever a reset, a default fg/bg, or a 24-bit fg/bg colour.

    Several of those can be merged into one sequence, so the parameters are
    read as a stream of items rather than matched whole.
    """
    codes = [int(p) for p in params.split(";")] if params else [0]
    i = 0
    while i < len(codes):
        if codes[i] in (0, 39, 49):
            i += 1
        elif (
            codes[i] in (38, 48)
            and codes[i + 1:i + 2] == [2]
            and len(codes) - i >= 5
        ):
            i += 5
        else:
            return False
    return True


def _vocabulary_scenarios() -> list[tuple[str, str]]:
    """One emitted string per path through the renderer that produces any."""
    rng = np.random.default_rng(11)

    def chars(h: int, w: int) -> CharArray:
        """Every cell coloured, foreground and background."""
        return unicode_image(rng.integers(0, 256, (2 * h, w, 3), dtype=np.uint8))

    base = chars(4, 6)
    one_cell = CharArray(
        codes=base.codes.copy(),
        fg=base.fg.copy(),
        fg_rgb=base.fg_rgb.copy(),
        bg=base.bg.copy(),
        bg_rgb=base.bg_rgb.copy(),
    )
    one_cell.codes[1, 2] = ord("@")

    return [
        ("render, coloured", base.to_ansi_str()),
        ("render, uncoloured", CharArray.from_size(3, 4).to_ansi_str()),
        ("diff, every cell", chars(4, 6).to_ansi_diff_str(base)),
        ("diff, one cell", one_cell.to_ansi_diff_str(base)),
        ("diff, nothing changed", base.to_ansi_diff_str(base)),
        ("diff, wider", chars(4, 9).to_ansi_diff_str(base)),
        ("diff, narrower", chars(4, 3).to_ansi_diff_str(base)),
        ("diff, taller", chars(7, 6).to_ansi_diff_str(base)),
        ("diff, shorter", chars(2, 6).to_ansi_diff_str(base)),
        ("diff, narrower and shorter", chars(2, 3).to_ansi_diff_str(base)),
        ("diff, wider and taller", chars(7, 9).to_ansi_diff_str(base)),
        ("clearstr", mp.blank(height=3, width=4).clearstr()),
        ("clearstr, empty plot", mp.blank(height=0, width=0).clearstr()),
    ]


VOCABULARY_SCENARIOS = _vocabulary_scenarios()
VOCABULARY_IDS = [label for label, _ in VOCABULARY_SCENARIOS]
VOCABULARY_STRINGS = [emitted for _, emitted in VOCABULARY_SCENARIOS]


class TestEmittedVocabulary:
    """What the library is allowed to say to a terminal.

    `pages/compatibility.md` documents this set, and what each member of it
    needs from a terminal. This is the same claim, executable: a sequence
    appearing in the renderer that the page does not cover fails here, so the
    page cannot go quietly out of date.
    """

    @pytest.mark.parametrize("emitted", VOCABULARY_STRINGS, ids=VOCABULARY_IDS)
    def test_only_documented_sequences_are_emitted(self, emitted: str):
        for final, params in _sequences(emitted):
            assert final in ALLOWED_FINALS, (
                f"CSI {params}{final} is outside the documented vocabulary"
                f" ({RETIRED_FINALS.get(final, 'never emitted before')})"
            )
            if final == "m":
                assert _sgr_is_allowed(params), f"SGR {params} is not a colour"
            elif final == "K":
                assert params == "2", "erase-in-line is only ever the whole line"

    @pytest.mark.parametrize("emitted", VOCABULARY_STRINGS, ids=VOCABULARY_IDS)
    def test_no_cursor_move_asks_to_move_zero(self, emitted: str):
        """`CSI 0 B` moves one row, not none: a zero count reads as one.

        So a renderer that computes a distance of zero has to emit nothing at
        all, rather than the sequence with a zero in it -- an off-by-one that
        would only show up on the frames where a plot happens not to move.

        An *omitted* count is a different thing and is fine. It also means one,
        by the same default that terminfo records for xterm as `cuu1=\\E[A`, and
        the renderer uses the short form where the distance is always one.
        """
        for final, params in _sequences(emitted):
            if final in ("A", "B", "C", "D"):
                assert params != "0", (
                    f"CSI 0 {final} moves one, not none; emit nothing instead"
                )
                assert params == "" or int(params) >= 1

    def test_the_retired_sequences_stay_retired(self):
        """Named separately from the subset check above, so that widening
        `ALLOWED_FINALS` cannot quietly bring one of these back with it.
        """
        for label, emitted in VOCABULARY_SCENARIOS:
            for final, params in _sequences(emitted):
                assert final not in RETIRED_FINALS, (
                    f"{label} emits CSI {params}{final}, retired by the audit:"
                    f" {RETIRED_FINALS[final]}"
                )
