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


class _Term:
    """A tiny ANSI screen emulator -- just enough to verify diff rendering.

    Applies the escape sequences our renderer emits (cursor moves, SGR colour,
    printable glyphs, erase-to-end) to a grid of (char, fg, bg) cells, so a test
    can check that a diff transforms the screen exactly like a full redraw.
    """

    def __init__(self, rows, cols):
        self.rows, self.cols = rows, cols
        self.grid = [[(" ", None, None) for _ in range(cols)] for _ in range(rows)]
        self.r = self.c = 0
        self.fg = None
        self.bg = None

    def feed(self, s):
        i = 0
        while i < len(s):
            ch = s[i]
            if ch == "\x1b":
                assert s[i + 1] == "["
                j = i + 2
                while s[j] not in "ABCDEFGHJKmf":
                    j += 1
                self._csi(s[i + 2:j], s[j])
                i = j + 1
            elif ch == "\n":
                self.r += 1
                self.c = 0
                i += 1
            elif ch == "\r":
                self.c = 0
                i += 1
            else:
                self.grid[self.r][self.c] = (ch, self.fg, self.bg)
                self.c += 1
                i += 1
        return self

    def _csi(self, params, letter):
        if letter == "m":
            codes = [int(x) for x in params.split(";")] if params else [0]
            k = 0
            while k < len(codes):
                x = codes[k]
                if x == 0:
                    self.fg = self.bg = None
                    k += 1
                elif x == 39:
                    self.fg = None
                    k += 1
                elif x == 49:
                    self.bg = None
                    k += 1
                elif x == 38 and codes[k + 1] == 2:
                    self.fg = tuple(codes[k + 2:k + 5])
                    k += 5
                elif x == 48 and codes[k + 1] == 2:
                    self.bg = tuple(codes[k + 2:k + 5])
                    k += 5
                else:
                    k += 1
            return
        n = int(params) if params else 1
        if letter == "A":
            self.r -= n
        elif letter == "B":
            self.r += n
        elif letter == "C":
            self.c += n
        elif letter == "D":
            self.c -= n
        elif letter == "E":
            self.r += n
            self.c = 0
        elif letter == "J":  # erase from cursor to end of screen
            for cc in range(self.c, self.cols):
                self.grid[self.r][cc] = (" ", None, None)
            for rr in range(self.r + 1, self.rows):
                for cc in range(self.cols):
                    self.grid[rr][cc] = (" ", None, None)

    def region(self, h, w):
        return [row[:w] for row in self.grid[:h]]


def _rand_chars(rng, h, w):
    """A realistic CharArray: a random half-block image (every cell coloured)."""
    img = rng.integers(0, 256, size=(2 * h, w, 3), dtype=np.uint8)
    return unicode_image(img)


def _screen_after_diff(prev, new):
    """Emulate printing `prev`, then applying new.to_ansi_diff_str(prev)."""
    H, W = prev.height, prev.width
    term = _Term(H + 3, W + 2)
    term.feed(prev.to_ansi_str()).feed("\n")   # as if print(prev) had run
    term.feed(new.to_ansi_diff_str(prev))
    return term


class TestCharArrayDiffStr:
    def test_no_change_is_empty(self):
        rng = np.random.default_rng(0)
        ca = _rand_chars(rng, 4, 6)
        assert ca.to_ansi_diff_str(ca) == ""

    def test_shape_mismatch_raises(self):
        rng = np.random.default_rng(1)
        a = _rand_chars(rng, 3, 4)
        b = _rand_chars(rng, 3, 5)
        with pytest.raises(ValueError):
            b.to_ansi_diff_str(a)

    def test_diff_repaints_only_changed_cell(self):
        rng = np.random.default_rng(2)
        prev = _rand_chars(rng, 4, 8)
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

    def test_cursor_returns_below_plot(self):
        rng = np.random.default_rng(3)
        prev = _rand_chars(rng, 5, 7)
        new = _rand_chars(rng, 5, 7)
        term = _screen_after_diff(prev, new)
        assert (term.r, term.c) == (prev.height, 0)

    def test_color_only_change(self):
        rng = np.random.default_rng(4)
        prev = _rand_chars(rng, 3, 5)
        new = unicode_image(rng.integers(0, 256, (6, 5, 3), dtype=np.uint8))
        new.codes = prev.codes.copy()  # same glyphs, different colours
        term = _screen_after_diff(prev, new)
        ref = _Term(new.height + 3, new.width + 2).feed(new.to_ansi_str())
        assert term.region(new.height, new.width) == ref.region(new.height, new.width)

    def test_diff_matches_full_redraw_random(self):
        """The strong one: a diff must leave the screen identical to a fresh
        redraw of `new`, across many random partial changes."""
        rng = np.random.default_rng(2024)
        for _ in range(40):
            h = int(rng.integers(1, 9))
            w = int(rng.integers(1, 14))
            base = rng.integers(0, 256, (2 * h, w, 3), dtype=np.uint8)
            after = base.copy()
            mask = rng.random((2 * h, w)) < rng.uniform(0.0, 0.6)
            after[mask] = rng.integers(0, 256, (int(mask.sum()), 3), dtype=np.uint8)
            prev = unicode_image(base)
            new = unicode_image(after)
            term = _screen_after_diff(prev, new)
            ref = _Term(new.height + 3, new.width + 2).feed(new.to_ansi_str())
            assert term.region(new.height, new.width) == ref.region(new.height, new.width)
            assert (term.r, term.c) == (new.height, 0)


class TestPlotUpdateStr:
    def test_sub_operator_matches_updatestr(self):
        a = mp.image(np.random.default_rng(5).random((6, 8)))
        b = mp.image(np.random.default_rng(6).random((6, 8)))
        assert (b - a) == b.updatestr(a)

    def test_updatestr_falls_back_on_size_change(self):
        # different sizes -> clear + full redraw; emulate and check the screen
        prev = mp.image(np.random.default_rng(7).random((8, 10)))   # 4 rows
        new = mp.image(np.random.default_rng(8).random((4, 6)))     # 2 rows
        H = max(prev.height, new.height)
        term = _Term(H + 3, max(prev.width, new.width) + 2)
        term.feed(prev.renderstr()).feed("\n")     # as if print(prev) had run
        term.feed(new.updatestr(prev))
        ref = _Term(H + 3, new.width + 2).feed(new.renderstr())
        assert term.region(new.height, new.width) == ref.region(new.height, new.width)
        # the old plot's extra rows must have been cleared
        assert term.grid[prev.height - 1][0] == (" ", None, None)
