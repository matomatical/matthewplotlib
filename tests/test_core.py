import numpy as np
import pytest

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
