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

    Models *deferred wrap* (VT/xterm behaviour): writing the glyph in the final
    column leaves the cursor on that column with a wrap flag set, rather than
    moving it past the edge. The flag is cleared by any cursor movement, and
    resolved into an actual line-wrap by the next glyph. Cursor moves clamp at
    the screen edges. Both matter only when a plot is exactly as wide as the
    screen -- which is when a renderer's own column bookkeeping can drift out of
    step with the terminal's.
    """

    def __init__(self, rows, cols):
        self.rows, self.cols = rows, cols
        self.grid = [[(" ", None, None) for _ in range(cols)] for _ in range(rows)]
        self.r = self.c = 0
        self.wrap_pending = False
        self.scrolled = 0
        self.fg = None
        self.bg = None

    def feed(self, s):
        i = 0
        while i < len(s):
            ch = s[i]
            if ch == "\x1b":
                assert s[i + 1] == "["
                j = i + 2
                while s[j] not in "ABCDEFGHJKXmf":
                    j += 1
                self._csi(s[i + 2:j], s[j])
                i = j + 1
            elif ch == "\n":
                self._linefeed()
                self.c = 0
                i += 1
            elif ch == "\r":
                self.c = 0
                self.wrap_pending = False
                i += 1
            else:
                if self.wrap_pending:
                    self._linefeed()
                    self.c = 0
                self.grid[self.r][self.c] = (ch, self.fg, self.bg)
                if self.c == self.cols - 1:
                    self.wrap_pending = True
                else:
                    self.c += 1
                i += 1
        return self

    def _linefeed(self):
        """Advance a line, scrolling the screen if already on the last one.

        A line feed is the only thing here that scrolls: cursor-down and
        cursor-next-line clamp at the bottom margin instead (which is what lets
        a diff sequence stay put where a full redraw would scroll).
        """
        self.wrap_pending = False
        if self.r == self.rows - 1:
            self.grid.pop(0)
            self.grid.append([(" ", None, None) for _ in range(self.cols)])
            self.scrolled += 1
        else:
            self.r += 1

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
        self.wrap_pending = False  # any cursor movement clears the wrap flag
        if letter == "A":
            self.r = max(0, self.r - n)
        elif letter == "B":
            self.r = min(self.rows - 1, self.r + n)
        elif letter == "C":
            self.c = min(self.cols - 1, self.c + n)
        elif letter == "D":
            self.c = max(0, self.c - n)
        elif letter == "G":  # absolute column (1-indexed)
            self.c = min(self.cols - 1, max(0, n - 1))
        elif letter == "E":
            self.r = min(self.rows - 1, self.r + n)
            self.c = 0
        elif letter == "J":  # erase from cursor to end of screen
            for cc in range(self.c, self.cols):
                self.grid[self.r][cc] = (" ", None, None)
            for rr in range(self.r + 1, self.rows):
                for cc in range(self.cols):
                    self.grid[rr][cc] = (" ", None, None)
        elif letter == "K":  # erase in line: 0 to end, 1 to start, 2 all
            n = int(params) if params else 0
            lo = 0 if n in (1, 2) else self.c
            hi = self.cols if n in (0, 2) else self.c + 1
            for cc in range(lo, hi):
                self.grid[self.r][cc] = (" ", None, None)
        elif letter == "X":  # erase n characters from the cursor, cursor unmoved
            for cc in range(self.c, min(self.cols, self.c + n)):
                self.grid[self.r][cc] = (" ", None, None)

    def region(self, h, w):
        return [row[:w] for row in self.grid[:h]]


def _rand_chars(rng, h, w):
    """A realistic CharArray: a random half-block image (every cell coloured)."""
    img = rng.integers(0, 256, size=(2 * h, w, 3), dtype=np.uint8)
    return unicode_image(img)


def _screen_after_diff(prev, new, slack=2, rows_below=3):
    """Emulate print(prev), then print(new.to_ansi_diff_str(prev)).

    Both are printed the way the library intends -- plainly, so `print` supplies
    the trailing newline each sequence is shaped to expect.

    `slack` is the number of spare columns beyond the plot's width; pass 0 to put
    the plot flush against the right edge of the screen. `rows_below` is the
    spare rows beneath it; pass 1 for the tightest layout that still animates.
    """
    H, W = prev.height, prev.width
    term = _Term(H + rows_below, W + slack)
    term.feed(prev.to_ansi_str()).feed("\n")            # print(prev)
    term.feed(new.to_ansi_diff_str(prev)).feed("\n")    # print(new - prev)
    return term


class TestCharArrayDiffStr:
    def test_no_change_still_holds_the_cursor(self):
        """An unchanged frame must not return "": printed, that would add a
        newline and walk the cursor a row further down every frame."""
        rng = np.random.default_rng(0)
        ca = _rand_chars(rng, 4, 6)
        term = _screen_after_diff(ca, ca)
        ref = _Term(ca.height + 3, ca.width + 2).feed(ca.to_ansi_str())
        assert term.region(ca.height, ca.width) == ref.region(ca.height, ca.width)
        assert (term.r, term.c) == (ca.height, 0)

    def test_different_shapes_are_diffed_not_rejected(self):
        rng = np.random.default_rng(1)
        a = _rand_chars(rng, 3, 4)
        b = _rand_chars(rng, 3, 5)
        assert b.to_ansi_diff_str(a)          # no exception; see the resize tests

    def test_empty_plot_raises(self):
        """updatestr owns the empty cases, so the array-level call rejects them
        rather than silently emitting a sequence for a plot with no rows."""
        rng = np.random.default_rng(1)
        a = _rand_chars(rng, 3, 4)
        empty = CharArray.from_size(height=0, width=0)
        with pytest.raises(ValueError):
            a.to_ansi_diff_str(empty)
        with pytest.raises(ValueError):
            empty.to_ansi_diff_str(a)

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

    def test_diff_at_right_edge_of_screen(self):
        """A plot exactly as wide as the screen still diffs correctly.

        Writing the final column leaves the terminal's cursor on that column
        with a wrap pending, not one past it, so a renderer tracking the column
        by counting glyphs is off by one from there on.
        """
        # minimal case: change the last column of row 0, then an interior cell
        # of row 1, so the second cell needs a move made from the edge.
        base = np.zeros((4, 6, 3), dtype=np.uint8)
        after = base.copy()
        after[0, 5] = (1, 2, 3)     # row 0, final column
        after[3, 3] = (4, 5, 6)     # row 1, column 3
        prev, new = unicode_image(base), unicode_image(after)
        term = _screen_after_diff(prev, new, slack=0)
        ref = _Term(new.height + 3, new.width).feed(new.to_ansi_str())
        assert term.region(new.height, new.width) == ref.region(new.height, new.width)
        assert (term.r, term.c) == (new.height, 0)

    def test_diff_at_right_edge_of_screen_random(self):
        rng = np.random.default_rng(99)
        for _ in range(40):
            h = int(rng.integers(1, 9))
            w = int(rng.integers(2, 14))
            base = rng.integers(0, 256, (2 * h, w, 3), dtype=np.uint8)
            after = base.copy()
            mask = rng.random((2 * h, w)) < rng.uniform(0.0, 0.6)
            after[mask] = rng.integers(0, 256, (int(mask.sum()), 3), dtype=np.uint8)
            prev, new = unicode_image(base), unicode_image(after)
            term = _screen_after_diff(prev, new, slack=0)
            ref = _Term(new.height + 3, new.width).feed(new.to_ansi_str())
            assert term.region(new.height, new.width) == ref.region(new.height, new.width)
            assert (term.r, term.c) == (new.height, 0)

    def test_diff_at_bottom_edge_of_screen(self):
        """One spare row below the plot is enough -- the diff needs no more.

        That row is where the newline `print` appends goes, so a plot of height
        R-1 on an R-row screen animates without ever scrolling. (A plot of the
        full screen height cannot animate at all, by either path: printing H rows
        plus a newline into H rows must scroll, and the plot's top row is then
        lost off the screen.)
        """
        rng = np.random.default_rng(7)
        base = rng.integers(0, 256, (6, 9, 3), dtype=np.uint8)
        after = base.copy()
        after[rng.random((6, 9)) < 0.4] = 200
        prev, new = unicode_image(base), unicode_image(after)
        term = _screen_after_diff(prev, new, rows_below=1)
        ref = _Term(new.height + 1, new.width + 2).feed(new.to_ansi_str())
        assert term.region(new.height, new.width) == ref.region(new.height, new.width)
        assert (term.r, term.c) == (new.height, 0)
        assert term.scrolled == 0


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

    def test_animation_loop_is_one_uniform_print(self):
        """Replay the loop documented on `plot.__sub__`, verbatim.

        Every frame -- including the seed, where prev is None -- is the single
        statement `print(frame - prev)`, with no `end=""` anywhere.
        """
        rng = np.random.default_rng(11)
        frames = [mp.image(rng.random((6, 9))) for _ in range(5)]
        term = _Term(frames[0].height + 3, frames[0].width + 2)

        prev = None
        for frame in frames:
            term.feed(frame - prev).feed("\n")     # print(frame - prev)
            assert (term.r, term.c) == (frame.height, 0)
            prev = frame

        last = frames[-1]
        ref = _Term(last.height + 3, last.width + 2).feed(last.renderstr())
        assert term.region(last.height, last.width) == ref.region(last.height, last.width)
        assert term.scrolled == 0

    def test_animation_loop_holds_still_when_nothing_changes(self):
        """A repeated frame must not creep down the screen."""
        p = mp.image(np.random.default_rng(12).random((6, 9)))
        term = _Term(p.height + 3, p.width + 2)
        prev = None
        for _ in range(5):
            term.feed(p - prev).feed("\n")
            prev = p
        assert (term.r, term.c) == (p.height, 0)
        assert term.scrolled == 0


def _screen_after_resize(prev, new, footer=None, slack=2, rows_below=4):
    """print(prev), optionally a footer line beneath it, then print(new - prev).

    The footer is written on the row the cursor already occupies -- one below
    prev -- and the carriage return puts the cursor back where the contract
    wants it. It must fit on one screen row, or it would wrap and leave the
    cursor somewhere the contract does not expect.
    """
    UH = max(prev.height, new.height)
    UW = max(prev.width, new.width)
    assert footer is None or len(footer) <= UW + slack
    term = _Term(UH + rows_below, UW + slack)
    term.feed(prev.to_ansi_str()).feed("\n")
    if footer is not None:
        term.feed(footer).feed("\r")
    term.feed(new.to_ansi_diff_str(prev)).feed("\n")
    return term


class TestCharArrayDiffStrResize:
    """A resize is still a diff: the overlap is compared, the rest painted or
    erased. Afterwards the screen must look exactly as a fresh render of `new`
    does -- i.e. the new plot, and blanks everywhere the old one reached."""

    SIZES = [
        ((4, 8), (4, 8), "same"),
        ((4, 8), (6, 8), "taller"),
        ((6, 8), (4, 8), "shorter"),
        ((4, 8), (4, 12), "wider"),
        ((4, 12), (4, 8), "narrower"),
        ((4, 8), (6, 12), "taller and wider"),
        ((6, 12), (4, 8), "shorter and narrower"),
        ((6, 8), (4, 12), "shorter and wider"),
        ((4, 12), (6, 8), "taller and narrower"),
        ((1, 1), (5, 9), "from a single cell"),
        ((5, 9), (1, 1), "down to a single cell"),
    ]

    @pytest.mark.parametrize("pshape,nshape,label", SIZES)
    def test_resize_matches_a_fresh_render(self, pshape, nshape, label):
        rng = np.random.default_rng(abs(hash(label)) % 2**32)
        prev = _rand_chars(rng, *pshape)
        new = _rand_chars(rng, *nshape)
        term = _screen_after_resize(prev, new)
        UH = max(prev.height, new.height)
        UW = max(prev.width, new.width)
        ref = _Term(UH + 4, UW + 2).feed(new.to_ansi_str())
        assert term.region(UH, UW) == ref.region(UH, UW), label
        assert (term.r, term.c) == (new.height, 0), label
        assert term.scrolled == 0, label

    def test_resize_random(self):
        rng = np.random.default_rng(4242)
        for _ in range(120):
            ph, pw = int(rng.integers(1, 7)), int(rng.integers(1, 11))
            nh, nw = int(rng.integers(1, 7)), int(rng.integers(1, 11))
            prev, new = _rand_chars(rng, ph, pw), _rand_chars(rng, nh, nw)
            term = _screen_after_resize(prev, new)
            UH, UW = max(ph, nh), max(pw, nw)
            ref = _Term(UH + 4, UW + 2).feed(new.to_ansi_str())
            assert term.region(UH, UW) == ref.region(UH, UW), (ph, pw, nh, nw)
            assert (term.r, term.c) == (nh, 0), (ph, pw, nh, nw)

    def test_shrinking_leaves_content_below_alone(self):
        """Losing rows erases exactly those rows -- a gap, not a bulldozer."""
        rng = np.random.default_rng(5)
        prev, new = _rand_chars(rng, 6, 8), _rand_chars(rng, 3, 8)
        term = _screen_after_resize(prev, new, footer="below")
        line = lambda r: "".join(c for c, _, _ in term.grid[r]).rstrip()
        assert [line(r) for r in range(3, 6)] == ["", "", ""]   # the gap
        assert line(6) == "below"                              # untouched

    def test_growing_taller_overwrites_content_below(self):
        """The documented cost of growth: those rows have to be written into."""
        rng = np.random.default_rng(6)
        prev, new = _rand_chars(rng, 3, 8), _rand_chars(rng, 5, 8)
        term = _screen_after_resize(prev, new, footer="below")
        line = lambda r: "".join(c for c, _, _ in term.grid[r]).rstrip()
        assert line(3) != "below"             # row 3 now belongs to the plot
        ref = _Term(9, 10).feed(new.to_ansi_str())
        assert term.region(5, 8) == ref.region(5, 8)

    def test_growing_taller_scrolls_at_the_bottom_of_the_screen(self):
        """Appended rows use newlines, so they scroll rather than clamp.

        Cursor-down would have stopped at the bottom margin and painted the
        appended rows on top of each other.
        """
        rng = np.random.default_rng(8)
        prev, new = _rand_chars(rng, 3, 8), _rand_chars(rng, 5, 8)
        # screen fits prev plus the row its trailing newline needs, and no more
        term = _Term(prev.height + 1, new.width + 2)
        term.feed(prev.to_ansi_str()).feed("\n")
        term.feed(new.to_ansi_diff_str(prev)).feed("\n")
        assert term.scrolled == 2             # once per appended row past the end
        assert (term.r, term.c) == (term.rows - 1, 0)
        # the screen now shows the plot's tail: its top rows scrolled away
        ref = _Term(new.height, new.width + 2).feed(new.to_ansi_str())
        visible = term.rows - 1               # the bottom row is the cursor's
        offset = new.height - visible
        for r in range(visible):
            assert term.grid[r][:new.width] == ref.grid[r + offset][:new.width]

    def test_narrowing_erases_only_the_lost_columns(self):
        """Trailing columns are erased with ECH, which cannot reach past them."""
        rng = np.random.default_rng(9)
        prev, new = _rand_chars(rng, 3, 10), _rand_chars(rng, 3, 6)
        term = _Term(8, 14)
        term.feed(prev.to_ansi_str()).feed("\n")
        # mark the columns to the right of both plots
        for r in range(3):
            term.grid[r][12] = ("|", None, None)
        term.feed(new.to_ansi_diff_str(prev)).feed("\n")
        for r in range(3):
            assert [c for c, _, _ in term.grid[r][6:12]] == [" "] * 6
            assert term.grid[r][12] == ("|", None, None)


class TestPlotClearStr:
    def test_clearstr_preserves_the_row_above_the_plot(self):
        """The clear erases the plot, then steps above it -- it must not erase
        the row it steps onto, which is typically the shell's command line."""
        p = mp.border(mp.text("frame"))
        term = _Term(10, 30)
        term.feed("$ python examples/wave.py").feed("\n")
        term.feed(p.renderstr()).feed("\n")
        term.feed(-p).feed("\n")                   # print(-plot)
        assert "".join(c for c, _, _ in term.grid[0]).rstrip() \
            == "$ python examples/wave.py"

    def test_clearstr_erases_only_the_plots_own_rows(self):
        """Content below the plot must survive the clear."""
        p = mp.border(mp.text("frame"))
        term = _Term(12, 30)
        term.feed("$ prompt").feed("\n")
        term.feed(p.renderstr()).feed("\n")
        term.feed("footer text").feed("\r")   # below the plot, back to column 0
        term.feed(-p).feed("\n")              # print(-plot)
        line = lambda r: "".join(c for c, _, _ in term.grid[r]).rstrip()
        assert line(0) == "$ prompt"          # the row stepped onto, not erased
        assert [line(r) for r in range(1, 1 + p.height)] == [""] * p.height
        assert line(1 + p.height) == "footer text"

    def test_clearstr_of_an_empty_plot_is_a_no_op(self):
        z = mp.blank(height=0, width=0)
        assert z.clearstr() == "\x1b[1A"      # CSI 0 A would mean CSI 1 A
        term = _Term(6, 12)
        term.feed("keep me").feed("\n")
        before = [row[:] for row in term.grid], (term.r, term.c)
        term.feed(-z).feed("\n")              # print(-plot) of an empty plot
        assert ([row[:] for row in term.grid], (term.r, term.c)) == before

    def test_clear_and_redraw_loop_is_stable(self):
        """print(-plot) then print(new) must redraw in place, frame after frame."""
        frames = [mp.border(mp.text(f"F{n}")) for n in range(1, 5)]
        term = _Term(10, 30)
        term.feed("\n\n")                          # start partway down
        term.feed(frames[0].renderstr()).feed("\n")
        rows = lambda: [r for r, row in enumerate(term.grid)
                        if any(c != " " for c, _, _ in row)]
        occupied = [rows()]
        for f in frames[1:]:
            term.feed(-f).feed("\n")               # print(-plot)
            term.feed(f.renderstr()).feed("\n")    # print(plot)
            occupied.append(rows())
        assert all(o == occupied[0] for o in occupied), occupied
        assert term.scrolled == 0
