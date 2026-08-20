"""
The character grid underneath every plot, and the glyphs that fill it.

A plot is ultimately a rectangle of coloured unicode characters. This module
provides that rectangle, along with the routines that turn numeric data into
characters dense enough to draw with. The plot types in
`matthewplotlib.plots` are built on top of it.

The grid:

* `CharArray`: A grid of unicode codepoints with optional foreground and
  background colors. Supports composition (stacking, layering, padding),
  rendering to ANSI strings---including differential updates that repaint only
  the cells that changed between frames---and rendering to images and animated
  gifs using an embedded pixel font.

Drawing characters, each packing several data points into one character cell:

* `unicode_braille_array`: Boolean matrices to braille characters, at 2 by 4
  dots per cell.
* `rasterise_points`, `rasterise_segments`, and the `unicode_braille_points`
  and `unicode_braille_segments` that draw with them: points or line segments,
  given in dots, to a grid of dots or straight to braille characters.
* `unicode_bar` and `unicode_col`: Values to horizontal or vertical bars, using
  partial block characters for eighth-of-a-cell resolution.
* `unicode_image`: Images to half-block characters, at 1 by 2 pixels per cell.
* `unicode_box` and `BoxStyle`: Box-drawing borders, optionally titled.
"""

from __future__ import annotations

import enum
import dataclasses

import numpy as np
import einops

from typing import Self, Callable, Sequence
from numpy.typing import NDArray

from matthewplotlib.unscii16 import bitmaps
from matthewplotlib.colors import ColorLike, parse_color


# # # 
# COLOURED CHARACTER ARRAY


@dataclasses.dataclass
class CharArray:
    """
    A grid of possibly-coloured characters comprising a plot. For internal use.

    Fields:

    * codes: uint32[h,w].
        Unicode code point for the character.
    * fg: bool[h,w].
        Whether to use a custom foreground color.
    * fg_rgb: uint8[h,w,3].
        (If fg) RGB for custom foreground color.
    * bg: bool[h,w].
        Whether to use a custom background color.
    * bg_rgb: uint8[h,w,3].
        (If bg) RGB for custom background color.
    """
    codes: NDArray  # uint32[h,w]
    fg: NDArray     # bool[h,w]
    fg_rgb: NDArray # uint8[h,w,3]
    bg: NDArray     # bool[h,w]
    bg_rgb: NDArray # uint8[h,w,3]


    @property
    def height(self: Self) -> int:
        h, _w = self.codes.shape
        return h
    

    @property
    def width(self: Self) -> int:
        _h, w = self.codes.shape
        return w


    @staticmethod
    def from_codes(
        codes: NDArray, # uint32[h,w]
        fgcolor: ColorLike | None,
        bgcolor: ColorLike | None,
    ) -> CharArray:
        # foreground
        fgcolor_ = parse_color(fgcolor)
        if fgcolor_ is None:
            fg = np.zeros_like(codes, dtype=bool)
            fg_rgb = np.zeros((*codes.shape, 3), dtype=np.uint8)
        else:
            fg = np.ones_like(codes, dtype=bool)
            fg_rgb = np.full(
                (*codes.shape, 3),
                fgcolor_,
                dtype=np.uint8,
            )
        # background
        bgcolor_ = parse_color(bgcolor)
        if bgcolor_ is None:
            bg = np.zeros_like(codes, dtype=bool)
            bg_rgb = np.zeros((*codes.shape, 3), dtype=np.uint8)
        else:
            bg = np.ones_like(codes, dtype=bool)
            bg_rgb = np.full(
                (*codes.shape, 3),
                bgcolor_,
                dtype=np.uint8,
            )
        # construct chars
        return CharArray(
            codes=codes,
            fg=fg,
            fg_rgb=fg_rgb,
            bg=bg,
            bg_rgb=bg_rgb,
        )

    
    @staticmethod
    def from_size(
        height: int,
        width: int,
        fgcolor: ColorLike | None = None,
        bgcolor: ColorLike | None = None,
    ) -> CharArray:
        codes = np.full(
            (height, width),
            ord(" "),
            dtype=np.uint32,
        )
        return CharArray.from_codes(
            codes=codes,
            fgcolor=fgcolor,
            bgcolor=bgcolor,
        )


    def pad(
        self: Self,
        above: int = 0,
        below: int = 0,
        left: int = 0,
        right: int = 0,
        fgcolor: ColorLike | None = None,
        bgcolor: ColorLike | None = None,
    ) -> CharArray:
        height = above + self.height + below
        width = left + self.width + right
        padded = CharArray.from_size(
            height=height,
            width=width,
            fgcolor=fgcolor,
            bgcolor=bgcolor,
        )
        padded.codes[above:height-below,left:width-right] = self.codes
        padded.fg[above:height-below,left:width-right] = self.fg
        padded.fg_rgb[above:height-below,left:width-right] = self.fg_rgb
        padded.bg[above:height-below,left:width-right] = self.bg
        padded.bg_rgb[above:height-below,left:width-right] = self.bg_rgb
        return padded

    
    @staticmethod
    def map(
        f: Callable[[list[NDArray]], NDArray],
        charss: list[CharArray],
    ) -> CharArray:
        return CharArray(
            codes=f([chars.codes for chars in charss]),
            fg=f([chars.fg for chars in charss]),
            fg_rgb=f([chars.fg_rgb for chars in charss]),
            bg=f([chars.bg for chars in charss]),
            bg_rgb=f([chars.bg_rgb for chars in charss]),
        )


    def isblank(self: Self) -> NDArray: # bool[h,w]
        """
        True where the character has no visible content.
        """
        return (self.codes == ord(" ")) & (~self.bg)


    def isnonblank(self: Self) -> NDArray: # bool[h,w]
        """
        True where the character has visible content.
        """
        return (self.codes != ord(" ")) | (self.bg)


    def to_ansi_str(self: Self) -> str:
        """
        Render a CharArray as a sequence of characters and ANSI control codes
        (merging codes where possible).
        """
        s: list[str] = []
        current_fg: None | NDArray = None
        current_bg: None | NDArray = None
        for i in range(self.height):
            # line
            for j in range(self.width):
                # next character
                ansi_controls = []
                # manage fg
                fg = self.fg_rgb[i,j] if self.fg[i,j] else None
                if fg is None and current_fg is not None:
                    ansi_controls.append(39) # reset fg
                elif fg is not None and np.any(fg != current_fg):
                    ansi_controls.extend([38, 2, *fg]) # set fg
                current_fg = fg
                # manage bg
                bg = self.bg_rgb[i,j] if self.bg[i,j] else None
                if bg is None and current_bg is not None:
                    ansi_controls.append(49) # reset bg
                elif bg is not None and np.any(bg != current_bg):
                    ansi_controls.extend([48, 2, *bg]) # set bg
                current_bg = bg
                if ansi_controls:
                    s.append(f"\x1b[{';'.join(map(str, ansi_controls))}m")
                s.append(chr(self.codes[i,j]))
            # end of line
            if current_fg is not None or current_bg is not None:
                s.append("\x1b[0m")
                current_fg = None
                current_bg = None
            if i < self.height - 1:
                s.append("\n")
        return "".join(s)


    def to_ansi_diff_str(self: Self, prev: CharArray) -> str:
        """
        Render the minimal ANSI sequence that updates a terminal already
        showing `prev` so that it shows `self` instead, repainting only the
        cells that differ.

        This is the differential counterpart to `to_ansi_str`: where redrawing
        a whole frame re-emits every cell, this jumps the cursor to just the
        changed cells. For an animation whose frames are mostly stable, that is
        dramatically fewer bytes down the wire.

        Cursor contract (the sequence is shaped for a plain `print`):

        * On entry, the cursor is assumed to be at column 0 on the line
          immediately below where `prev` was rendered -- exactly where it sits
          after `print`ing `prev`.
        * On exit, the cursor is left at column 0 on the *last row of the plot*,
          so the newline `print` appends carries it to the line below, ready for
          the next frame to diff against this one in the same way.

        In other words the sequence is incomplete on its own: it expects a
        trailing newline, just as `to_ansi_str` does. Write it with
        `print(...)`, not `print(..., end="")`.

        Assumes `prev` was rendered starting at column 0 and that all glyphs are
        single-width (the standard animated-plot layout).

        The two need not be the same size, and only the difference is sent in
        that case too. Cells outside the region `prev` covered are always
        painted, since nothing is on screen there to keep. Rows and columns
        `prev` covered but `self` does not are erased. Growing taller overwrites
        the rows immediately below the plot: they have to be written into, and a
        string cannot push them out of the way without knowing where it sits on
        the screen. Growing shorter leaves those rows blank rather than closing
        the gap, for the same reason.

        Note that an unchanged frame does not return "": it returns a bare
        cursor-up, so that the newline `print` appends still lands the cursor
        where the contract promises rather than a row lower.
        """
        H, W = self.height, self.width
        PH, PW = prev.height, prev.width
        if H == 0 or PH == 0:
            raise ValueError(
                "to_ansi_diff_str needs a row in each plot; plot.updatestr "
                "handles the empty cases"
            )
        oh, ow = min(H, PH), min(W, PW)

        # Which cells of self need painting? Within the region prev also covered,
        # those that differ from it. Outside that region, all of them: either the
        # screen is blank there (self is wider) or the row is not on screen yet
        # (self is taller).
        changed = np.ones((H, W), dtype=bool)
        if ow:
            sl = (slice(None, oh), slice(None, ow))
            fg_changed = (self.fg[sl] != prev.fg[sl]) | (
                self.fg[sl] & np.any(self.fg_rgb[sl] != prev.fg_rgb[sl], axis=-1)
            )
            bg_changed = (self.bg[sl] != prev.bg[sl]) | (
                self.bg[sl] & np.any(self.bg_rgb[sl] != prev.bg_rgb[sl], axis=-1)
            )
            changed[sl] = (
                (self.codes[sl] != prev.codes[sl]) | fg_changed | bg_changed
            )

        trailing = max(0, PW - W)    # columns prev covered on each shared row
        lost_rows = max(0, PH - H)   # rows prev covered and self does not
        new_rows = max(0, H - PH)    # rows self covers and prev did not

        s: list[str] = []
        # Cursor position in plot coordinates -- shared by both plots, since they
        # start at the same corner. On entry it is one row below prev.
        # cur_col is None when the true column is unknown (see the wrap note).
        cur_row = PH
        cur_col: int | None = 0
        # SGR colour state: persists across cursor moves, so we only emit a code
        # when the colour actually changes (as in to_ansi_str, but the running
        # state now carries across the gaps we skip over).
        current_fg: None | NDArray = None
        current_bg: None | NDArray = None

        def goto_row(i: int) -> None:
            nonlocal cur_row
            if i != cur_row:
                d = i - cur_row
                s.append(f"\x1b[{abs(d)}{'B' if d > 0 else 'A'}")
                cur_row = i

        def goto_col(col: int) -> None:
            # A relative move when the column is known. When it is not -- after
            # a deferred wrap, see `paint` -- a carriage return is what resolves
            # it: it lands on column 0 whatever the terminal thinks the column
            # is, and it cancels the pending wrap, which is the part that must
            # not be left to chance. Absolute column addressing (CHA) would do
            # both in one sequence, but only on terminals that agree about the
            # wrap flag, and that is exactly what terminals disagree about.
            nonlocal cur_col
            if col == cur_col:
                return
            if col == 0 or cur_col is None:
                s.append("\r")          # one byte, and never the wrong column
                cur_col = 0
            if col != cur_col:
                d = col - cur_col
                s.append(f"\x1b[{abs(d)}{'C' if d > 0 else 'D'}")
            cur_col = col

        def reset_colour() -> None:
            # Blanking paints in the current background -- an erase does so on
            # many terminals, a written space does so on all of them -- so the
            # colour must be back to default before any blanking (and at the
            # end).
            nonlocal current_fg, current_bg
            if current_fg is not None or current_bg is not None:
                s.append("\x1b[0m")
                current_fg = current_bg = None

        def paint(i: int, col: int) -> None:
            nonlocal cur_col, current_fg, current_bg
            goto_col(col)
            controls = []
            fg = self.fg_rgb[i, col] if self.fg[i, col] else None
            if fg is None and current_fg is not None:
                controls.append(39)  # reset fg
            elif fg is not None and (
                current_fg is None or np.any(fg != current_fg)
            ):
                controls.extend([38, 2, *fg])  # set fg
            current_fg = fg
            bg = self.bg_rgb[i, col] if self.bg[i, col] else None
            if bg is None and current_bg is not None:
                controls.append(49)  # reset bg
            elif bg is not None and (
                current_bg is None or np.any(bg != current_bg)
            ):
                controls.extend([48, 2, *bg])  # set bg
            current_bg = bg
            if controls:
                s.append(f"\x1b[{';'.join(map(str, controls))}m")
            s.append(chr(self.codes[i, col]))
            # That glyph advanced the cursor one column -- unless it filled the
            # final column, in which case terminals defer the wrap: the cursor
            # stays put with a wrap flag rather than moving past the edge, so on
            # a screen exactly this wide it is not where counting glyphs says.
            # Forget the column; `goto_col` recovers with a carriage return.
            cur_col = col + 1 if col + 1 < W else None

        # repaint the rows that are already on screen
        for i in range(oh):
            cols = np.flatnonzero(changed[i])
            if not len(cols) and not trailing:
                continue
            goto_row(i)
            for j in cols:
                paint(i, int(j))
            if trailing:
                # Spaces, not an erase-character sequence: `reset_colour` has
                # already put the colour back to default, so a space paints
                # exactly what an erase would, and a space is a character every
                # terminal has an opinion about. Costs `trailing` bytes instead
                # of five, on the one frame where a plot narrows.
                reset_colour()
                goto_col(W)
                s.append(" " * trailing)
                # Those spaces moved the cursor, where an erase would not have.
                # They stop at column PW, which is on screen (prev was rendered
                # there) -- but PW may be the last column, and then the wrap is
                # deferred and the column is not what counting says. Forget it.
                cur_col = None

        # erase the rows prev covered and self does not
        if lost_rows:
            reset_colour()
            goto_row(H)
            s.append("\x1b[2K" + "\x1b[B\x1b[2K" * (lost_rows - 1))
            cur_row = H + lost_rows - 1

        # Append the rows self covers and prev did not. These need newlines
        # rather than cursor moves: cursor-down clamps at the bottom margin,
        # where a newline scrolls.
        if new_rows:
            goto_row(PH - 1)
            for i in range(PH, H):
                s.append("\n")
                cur_row, cur_col = i, 0
                for col in range(W):
                    paint(i, col)

        reset_colour()
        # Park at column 0 of the plot's last row, leaving the newline that
        # `print` appends to complete the frame. Column first, then row: a
        # carriage return costs a byte where cursor-next-line would have done
        # both at once, but it is the move no terminal can get wrong, and it
        # settles the deferred wrap before anything counts on the column again.
        goto_col(0)
        d = (H - 1) - cur_row
        if d:
            s.append(f"\x1b[{abs(d)}{'B' if d > 0 else 'A'}")
        return "".join(s)


    def to_plain_str(self: Self) -> str:
        """
        Render a CharArray as a sequence of characters without colour.
        """
        rows = [
            [chr(self.codes[i,j]) for j in range(self.width)]
            for i in range(self.height)
        ]
        return "\n".join("".join(row) for row in rows)


    def to_rgba_array(
        self: Self,
        bgcolor: ColorLike | None = None,
    ) -> np.ndarray: # uint8[height*16,width*8,4]
        """
        Convert a CharArray to an RGBA image array
        """
        # foreground color array
        fg = np.full(
            (self.height, self.width, 4),
            255,
            dtype=np.uint8,
        )
        fg[self.fg, :3] = self.fg_rgb[self.fg]
        
        # background color array
        bgcolor_ = parse_color(bgcolor)
        if bgcolor_ is None:
            bg = np.zeros(
                (self.height, self.width, 4),
                dtype=np.uint8,
            )
        else:
            bg = np.full(
                (self.height, self.width, 4),
                (*bgcolor_, 255),
                dtype=np.uint8,
            )
        bg[self.bg, :3] = self.bg_rgb[self.bg]
        bg[self.bg, 3] = 255
        
        # construct rgba array
        bits = bitmaps(self.codes)  # bool[h,w,16,8]
        tiles = np.where(
                bits[:,:,:,:,None], # bool[h,w,16,8,1]
            fg[:,:,None,None,:],    # uint8[h,w,1,1,4]
            bg[:,:,None,None,:],    # uint8[h,w,1,1,4]
        )                           # -> uint8[h,w,16,8,4]
        img = einops.rearrange(tiles, 'H W h w rgba -> (H h) (W w) rgba')
                                    # -> uint8[h*16,w*8,4]
        return img
    

    def to_bit_array(
        self: Self,
    ) -> np.ndarray: # bool[height*16,width*8]
        """
        Convert a CharArray to an bitmap image array
        """
        # construct rgba array
        tiles = bitmaps(self.codes) # bool[h,w,16,8]
        img = einops.rearrange(
            tiles,
            'H W h w -> (H h) (W w)',
        ) # -> uint8[h*16,w*8]
        return img
        

def _validate_text(
    chrs: Sequence[str],
    *,
    allow_line_breaks: bool = False,
) -> None:
    """Reject terminal control characters from text that will become glyphs."""
    allowed = "\r\n" if allow_line_breaks else ""
    controls = sorted({
        ord(char)
        for char in chrs
        if char not in allowed
        and (ord(char) < 0x20 or 0x7f <= ord(char) <= 0x9f)
    })
    if controls:
        names = ", ".join(f"U+{code:04X}" for code in controls)
        raise ValueError(
            f"text contains unsupported control characters: {names}. Raw "
            "terminal control sequences are not supported"
        )


def ords(chrs: Sequence[str]) -> list[int]:
    """
    Convert a string or list of glyphs to a list of unicode code points.

    C0 and C1 control characters are not glyphs and are rejected. In
    particular, raw ANSI formatting is not supported: terminal styling has to
    be represented in the character array so that its size and rendering stay
    well-defined.
    """
    _validate_text(chrs)
    return [ord(c) for c in chrs]


# # # 
# UNICODE BRAILLE DOT MATRIX


BRAILLE_MAP = np.array([
    [0, 3],
    [1, 4],
    [2, 5],
    [6, 7],
], dtype=np.uint8)


def unicode_braille_array(
    dots: NDArray, # bool[H, W] or int[H, W]
    dotc: NDArray | None = None, # uint8[H, W, rgb]
    dotw: NDArray | None = None, # float[H, W]
    fgcolor: ColorLike | None = None,
    bgcolor: ColorLike | None = None,
) -> CharArray: # Char[ceil(H/4), ceil(W/2)]
    """
    Turns a H by W array of dots into a h=ceil(H/4) by w=ceil(W/2) array of
    braille Unicode characters.

    Inputs:

    * dots: bool[H, W].
        Array of booleans or counts. Dots are placed where this array contains
        nonzero.
    * dotc: optional uint8[H, W, RGB].
        Array of colours to use for the fg of each dot. Where multiple dots
        are coloured within one one character, mixes the colours according to
        dotw.
    * dotw: optional float[H, W].
        Weights for combining colors when multiple dots occur in one cell. If
        not provided, combine uniformly. If dotc is not provided, this is not
        used.
    * fgcolor: optional ColorLike.
        Foreground color used for all braille characters. Overrides dotc if
        both are provided.
    * bgcolor: optional ColorLike.
        Background color used for all characters.

    Returns:

    * chars: CharArray.
        An array of Braille characters with h rows and w columns.

    An illustrated example, not including colour combination, is as follows:
    ```
    Start with an array. Assume height is divisible by 4 and width divisible by
    2, otherwise pad with 0s until that is the case.
        ____
       [1  0] 0  1  0  1  1  1  1  0  1  0  0  0  0  1  0  0  0  0  0  1  1  0
       [1  0] 0  1  0  1  0  0  0  0  1  0  0  0  0  1  0  0  0  0  1  0  0  1
     .-[1  0] 0  1  0  1  0  0  0  0  1  0  0  0  0  1  0  0  0  0  1  0  0  1
     | [1__0] 0  1  0  1  0  0  0  0  1  0  0  0  0  1  0  0  0  0  1  0  0  1
     |  1  1  1  1  0  1  1  1  1  0  1  0  0  0  0  1  0  0  0  0  1  0  0  1
     |  1  0  0  1  0  1  0  0  0  0  1  0  0  0  0  1  0  0  0  0  1  0  0  1
     |  1  0  0  1  0  1  0  0  0  0  1  0  0  0  0  1  0  0  0  0  1  0  0  1
     |  1  0  0  1  0  1  1  1  1  0  1  1  1  1  0  1  1  1  1  0  0  1  1  0
     |
     | take each 4x2 subarray and ...
     |                                                               braille
     | identify the 4x2 bits with the                                unicode
     | eight numbered braille dots:                                  start pt
     |                                                               |
     |  (dot 1) 1 0 (dot 4)     convert to                           v
     `> (dot 2) 1 0 (dot 5) -----------------> 0 b 0 1 0 0 0 1 1 1 + 0x2800 -.
        (dot 3) 1 0 (dot 6)    braille code        | | | | | | | |           |
        (dot 7) 1 0 (dot 8)                    dot 8 7 6 5 4 3 2 1           |
                                                                             |
      convert the braille code to a unicode character and collate into array |
     .-----------------------------------------------------------------------'
     |  '''
     `->⡇⢸⢸⠉⠁⡇⠀⢸⠀⠀⡎⢱  (Note: this function returns a CharArray, use
        ⡏⢹⢸⣉⡁⣇⣀⢸⣀⡀⢇⡸  .to_plain_str() to get a string.)
        '''
    ```
    """
    # process input
    dots = dots.astype(bool)
    H, W = dots.shape
    if dotc is not None and fgcolor is not None:
        dotc = None
    if dotc is not None and dotw is None:
        dotw = np.ones_like(dots, dtype=float)

    # pad to next multiple of (4, 2)
    hpad = H % 4
    wpad = W % 2
    if hpad or wpad:
        padding = ((0, 4-hpad), (0, 2-wpad))
        dots = np.pad(dots, padding, constant_values=False)
        H, W = dots.shape
        if dotc is not None:
            assert dotw is not None
            dotc = np.pad(dotc, (*padding, (0,0)), constant_values=0)
            dotw = np.pad(dotw, padding, constant_values=0.)

    # chunk it into 4x2 cells
    h, w = H // 4, W // 2
    cells = dots.reshape(h, 4, w, 2)
    
    # convert each bit in each cell into a mask and combine into code array
    masks = np.left_shift(cells, BRAILLE_MAP.reshape(1,4,1,2), dtype=np.uint32)
    codes = np.bitwise_or.reduce(masks, axis=(1,3))
    
    # convert to unicode braille codepoints (except for blanks)
    codes = np.where(
        codes > 0,
        0x2800 + codes,
        ord(" "),
    )

    # determine cell colors
    fgcolor_ = parse_color(fgcolor)
    if fgcolor_ is not None:
        fg = np.ones_like(codes, dtype=bool)
        fg_rgb = np.full(
            (*codes.shape, 3),
            fgcolor_,
            dtype=np.uint8,
        )
    elif dotc is not None:
        assert dotw is not None
        cellc = dotc.reshape(h, 4, w, 2, 3)
        cellw = dotw.reshape(h, 4, w, 2, 1)
        numer = np.sum(cellc * cellw, axis=(1,3))
        denom = np.sum(cellw, axis=(1,3))
        fg = (denom > 0)[:,:,0]
        fg_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        fg_rgb[fg] = numer[fg] / denom[fg]
        # TODO: Colormap after averaging...?
    else:
        fg = np.zeros_like(codes, dtype=bool)
        fg_rgb = np.zeros((*codes.shape, 3), dtype=np.uint8)
        
    # background colors
    bgcolor_ = parse_color(bgcolor)
    if bgcolor_ is None:
        bg = np.zeros_like(codes, dtype=bool)
        bg_rgb = np.zeros((*codes.shape, 3), dtype=np.uint8)
    else:
        bg = np.ones_like(codes, dtype=bool)
        bg_rgb = np.full((*codes.shape, 3), bgcolor_, dtype=np.uint8)

    return CharArray(
        codes=codes,
        fg=fg,
        fg_rgb=fg_rgb,
        bg=bg,
        bg_rgb=bg_rgb,
    )


# # #
# SEGMENT RASTERISATION


def disc_offsets(
    thickness: float,
) -> NDArray: # int[k, 2]
    """
    The dots covered by a disc of the given thickness centred on a dot, as
    offsets from that dot.

    A dot is included when its centre lies within `thickness / 2` of the
    centre of the disc, so a thickness of 1 covers one dot, 2 covers a plus of
    five, 3 covers a three-by-three square, and so on.

    Inputs:

    * thickness: float.
        Diameter of the disc, measured in dots.

    Returns:

    * offsets: int[k, 2].
        One `(row, col)` offset per covered dot, always including `(0, 0)`.
    """
    radius = max(thickness, 0.) / 2
    reach = int(np.floor(radius))
    span = np.arange(-reach, reach + 1)
    rows, cols = np.meshgrid(span, span, indexing="ij")
    covered = rows ** 2 + cols ** 2 <= radius ** 2
    return np.stack([rows[covered], cols[covered]], axis=1)


def rasterise_segments(
    starts: NDArray,                        # float[n, 2]
    ends: NDArray,                          # float[n, 2]
    height: int,
    width: int,
    start_colors: NDArray | None = None,    # uint8[n, 3]
    end_colors: NDArray | None = None,      # uint8[n, 3]
    thickness: float = 1.0,
) -> tuple[
    NDArray,                                # int[height, width]
    NDArray | None,                         # uint8[height, width, 3]
    NDArray | None,                         # float[height, width]
]:
    """
    Draw straight line segments onto a grid of dots.

    Coordinates are in dots, with rows increasing downwards, so that dot
    `(i, j)` covers the unit square from `(i, j)` up to but not including
    `(i+1, j+1)`, and the centre of that dot is at `(i+0.5, j+0.5)`. Whatever
    falls outside the grid is clipped away.

    Inputs:

    * starts: float[n, 2].
        One `(row, col)` coordinate per segment, where each segment begins.
    * ends: float[n, 2].
        Where each segment ends.
    * height: int.
        Number of rows of dots in the grid.
    * width: int.
        Number of columns of dots in the grid.
    * start_colors: optional uint8[n, 3].
        Color at the start of each segment. If omitted, no colors are
        computed and the color outputs are None.
    * end_colors: optional uint8[n, 3].
        Color at the end of each segment, interpolated along it. If omitted
        while `start_colors` is given, segments are a single flat color.
    * thickness: float (default 1.0).
        Width of the stroke, in dots. See `disc_offsets`.

    Returns:

    * dots: int[height, width].
        How many times each dot was covered. Zero where the dot is not part of
        any segment.
    * dotc: optional uint8[height, width, 3].
        The average of the colors covering each dot, or None if no colors were
        given.
    * dotw: optional float[height, width].
        The coverage counts again, as the weights that mix these colors
        together within a character cell, or None if no colors were given.

    The three outputs are exactly the `dots`, `dotc` and `dotw` inputs of
    `unicode_braille_array`.

    Segments with a non-finite endpoint are skipped, which is how a gap in a
    sequence of points becomes a gap in the line drawn through them.

    Notes:

    * Segments are clipped to the grid before they are drawn, so a segment may
      run arbitrarily far outside it without costing anything to skip.
    * A stroke is the segment thickened by a disc, which is to say the union of
      discs centred along it. Round caps and correctly filled joins between
      consecutive segments both follow from that, without either being a case
      to handle.
    """
    starts = np.asarray(starts, dtype=float).reshape(-1, 2)
    ends = np.asarray(ends, dtype=float).reshape(-1, 2)

    # a stroke reaches this far to either side of its segment
    offsets = disc_offsets(thickness)
    reach = int(np.abs(offsets).max())

    # skip segments that are not fully specified: this is where gaps come from.
    # their coordinates are stood down to zero as well, to keep the arithmetic
    # below free of non-finite values
    drawable = np.isfinite(starts).all(axis=1) & np.isfinite(ends).all(axis=1)
    starts = np.where(drawable[:, np.newaxis], starts, 0.)
    ends = np.where(drawable[:, np.newaxis], ends, 0.)

    # clip to the grid, plus the margin a stroke can reach in from (Liang and
    # Barsky's algorithm: intersect the parameter interval [0,1] of each
    # segment with the four half planes bounding the region)
    deltas = ends - starts
    enter = np.zeros(len(starts))
    leave = np.ones(len(starts))
    limits = ((-reach, height + reach), (-reach, width + reach))
    for axis, (low, high) in enumerate(limits):
        run = deltas[:, axis]
        start = starts[:, axis]
        for slope, offset in ((-run, start - low), (run, high - start)):
            # the constraint is `slope * t <= offset` for t in the interval
            ratio = np.divide(
                offset,
                slope,
                out=np.zeros_like(offset),
                where=(slope != 0),
            )
            enter = np.where(slope < 0, np.maximum(enter, ratio), enter)
            leave = np.where(slope > 0, np.minimum(leave, ratio), leave)
            # a segment parallel to a boundary is in or out on its own
            drawable &= (slope != 0) | (offset >= 0)
    drawable &= enter <= leave

    # trim the segments (and the colors along them) to what survived
    enter = enter[drawable, np.newaxis]
    leave = leave[drawable, np.newaxis]
    deltas = deltas[drawable]
    clipped_starts = starts[drawable] + enter * deltas
    clipped_ends = starts[drawable] + leave * deltas
    color_starts: NDArray | None = None
    color_ends: NDArray | None = None
    if start_colors is not None:
        c0 = np.asarray(start_colors, dtype=float).reshape(-1, 3)[drawable]
        if end_colors is None:
            c1 = c0
        else:
            c1 = np.asarray(end_colors, dtype=float).reshape(-1, 3)[drawable]
        color_starts = c0 + enter * (c1 - c0)
        color_ends = c0 + leave * (c1 - c0)

    # sample each segment densely enough that consecutive samples land in the
    # same dot or in adjacent ones, which is what makes the line unbroken
    deltas = clipped_ends - clipped_starts
    samples = np.ceil(np.abs(deltas).max(axis=1)).astype(int) + 1

    # ...and flatten the ragged result: one index within its own segment for
    # every sample of every segment, without padding any of them out
    segment = np.repeat(np.arange(len(samples)), samples)
    within = np.arange(samples.sum()) - np.repeat(
        np.cumsum(samples) - samples,
        samples,
    )
    t = within / np.maximum(samples - 1, 1)[segment]
    points = clipped_starts[segment] + t[:, np.newaxis] * deltas[segment]

    # the dots those samples fall in, and then the strokes around them
    dot = np.floor(points).astype(int)
    dot = (dot[:, np.newaxis, :] + offsets[np.newaxis, :, :]).reshape(-1, 2)

    # accumulate coverage, and the colors to average over it
    if color_starts is None or color_ends is None:
        return accumulate_dots(dot, None, height=height, width=width)
    colors = color_starts[segment] + t[:, np.newaxis] * (
        color_ends[segment] - color_starts[segment]
    )
    return accumulate_dots(
        dot,
        np.repeat(colors, len(offsets), axis=0),
        height=height,
        width=width,
    )


def rasterise_points(
    points: NDArray,                # float[n, 2]
    height: int,
    width: int,
    colors: NDArray | None = None,  # uint8[n, 3]
) -> tuple[
    NDArray,                        # int[height, width]
    NDArray | None,                 # uint8[height, width, 3]
    NDArray | None,                 # float[height, width]
]:
    """
    Mark the dots that a set of points falls in.

    Coordinates are in dots and mean what they mean for `rasterise_segments`,
    which also describes the three arrays this returns. Points that are not
    fully specified are skipped, as is anything outside the grid.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    specified = np.isfinite(points).all(axis=1)
    dot = np.floor(points[specified]).astype(int)
    if colors is None:
        return accumulate_dots(dot, None, height=height, width=width)
    colors = np.asarray(colors, dtype=float).reshape(-1, 3)[specified]
    return accumulate_dots(dot, colors, height=height, width=width)


def accumulate_dots(
    dot: NDArray,                   # int[n, 2]
    colors: NDArray | None,         # float[n, 3]
    height: int,
    width: int,
) -> tuple[
    NDArray,                        # int[height, width]
    NDArray | None,                 # uint8[height, width, 3]
    NDArray | None,                 # float[height, width]
]:
    """
    Count how many times each dot of a grid is covered, and average the colors
    covering it.

    Whatever falls outside the grid is dropped. The counts double as the weights
    that mix the colors of one character cell together, so they come back a
    second time as those weights, which is what `unicode_braille_array` takes.
    """
    inside = (
        (dot[:, 0] >= 0) & (dot[:, 0] < height)
        & (dot[:, 1] >= 0) & (dot[:, 1] < width)
    )
    flat = dot[inside, 0] * width + dot[inside, 1]
    dots = np.bincount(flat, minlength=height * width).reshape(height, width)
    if colors is None:
        return dots, None, None
    total = np.stack([
        np.bincount(
            flat,
            weights=colors[inside, channel],
            minlength=height * width,
        )
        for channel in range(3)
    ], axis=1).reshape(height, width, 3)
    lit = dots > 0
    dotc = np.zeros((height, width, 3), dtype=np.uint8)
    dotc[lit] = total[lit] / dots[lit, np.newaxis]
    return dots, dotc, dots.astype(float)


def unicode_braille_points(
    points: NDArray,                        # float[n, 2]
    height: int,
    width: int,
    colors: NDArray | None = None,          # uint8[n, 3]
) -> CharArray: # Char[ceil(height/4), ceil(width/2)]
    """
    Draw a set of points as a grid of braille characters.

    Coordinates are in dots, as for `rasterise_points`, which this draws with.
    """
    dots, dotc, dotw = rasterise_points(
        points=points,
        height=height,
        width=width,
        colors=colors,
    )
    return unicode_braille_array(dots=dots, dotc=dotc, dotw=dotw)


def unicode_braille_segments(
    starts: NDArray,                        # float[n, 2]
    ends: NDArray,                          # float[n, 2]
    height: int,
    width: int,
    start_colors: NDArray | None = None,    # uint8[n, 3]
    end_colors: NDArray | None = None,      # uint8[n, 3]
    thickness: float = 1.0,
) -> CharArray: # Char[ceil(height/4), ceil(width/2)]
    """
    Draw line segments as a grid of braille characters.

    Coordinates are in dots, as for `rasterise_segments`, which this draws with
    and which documents what the arguments mean.
    """
    dots, dotc, dotw = rasterise_segments(
        starts=starts,
        ends=ends,
        height=height,
        width=width,
        start_colors=start_colors,
        end_colors=end_colors,
        thickness=thickness,
    )
    return unicode_braille_array(dots=dots, dotc=dotc, dotw=dotw)


# # #
# UNICODE PARTIAL BLOCKS


PARTIAL_BLOCKS_ROW = ords([" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"])


def unicode_bar(
    proportion: float,
    width: int,
    height: int = 1,
    fgcolor: ColorLike | None = None,
    bgcolor: ColorLike | None = None,
) -> CharArray:
    """
    Generates a Unicode progress bar as a list of characters.

    This function creates a fixed-width left-to-right bar using Unicode block
    elements to represent the proportion rounded down to nearest 1/8th of a
    block.

    Inputs:

    * proportion: float.
        The fraction of the bar to fill. Should be between 0.0 and 1.0
        inclusive.
    * width: int (positive).
        The width of the full bar in characters.
    * height: int (positive, default 1).
        The number of rows that the bar takes up.
    * fgcolor: optional ColorLike.
        Foreground color used for the progress bar characters.
    * bgcolor: optional ColorLike.
        Background color used for the progress bar remainder.

    Returns:

    * chars: CharArray
        A character array representing the bar.

    Examples:

    ```pycon
    >>> unicode_bar(0.5, 10).to_plain_str()
    '█████     '
    >>> unicode_bar(0.625, 10).to_plain_str()
    '██████▎   '

    ```
    """
    # clip inputs to valid range
    proportion = max(0.0, min(1.0, proportion))

    # calculate number of filled 'eighths'
    full_eighths = int(proportion * width * 8)
    full_blocks, remainder = divmod(full_eighths, 8)

    # construct bar
    codes = np.zeros((height, width), dtype=np.uint32)
    codes[:, :full_blocks] = PARTIAL_BLOCKS_ROW[-1]
    if remainder > 0:
        codes[:, full_blocks] = PARTIAL_BLOCKS_ROW[remainder]
        codes[:, full_blocks+1:] = ord(" ")
    else:
        codes[:, full_blocks:] = ord(" ")

    return CharArray.from_codes(codes, fgcolor, bgcolor)


PARTIAL_BLOCKS_COL = ords([" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"])


def unicode_col(
    proportion: float,
    height: int,
    width: int = 1,
    fgcolor: ColorLike | None = None,
    bgcolor: ColorLike | None = None,
) -> CharArray:
    """
    Generates a Unicode progress column as a list of characters.

    This function creates a fixed-height column using Unicode block elements to
    represent a proportion rounded down to nearest 1/8th of a block. The list
    goes from the top of the bar to the bottom, but the bar grows from the
    bottom towards the top.

    Inputs:

    * proportion: float.
        The fraction of the column to fill. Should be between 0.0 and 1.0
        inclusive.
    * height: int (positive).
        The height of the full bar in characters.
    * width: int (positive, default 1).
        The number of columns that the bar takes up.
    * fgcolor: optional ColorLike.
        Foreground color used for the progress bar characters.
    * bgcolor: optional ColorLike.
        Background color used for the progress bar remainder.

    Returns:

    * chars: CharArray
        A char array representing the column.

    Examples:

    ```pycon
    >>> unicode_col(0.5, 3).to_plain_str()
    ' \\n▄\\n█'
    
    ```
    """
    # clip inputs to valid range
    proportion = max(0.0, min(1.0, proportion))

    # calculate number of filled 'eighths'
    full_eighths = int(proportion * height * 8)
    full_blocks, remainder = divmod(full_eighths, 8)

    # construct column (upside down)
    codes = np.zeros((height, width), dtype=np.uint32)
    codes[:full_blocks, :] = PARTIAL_BLOCKS_COL[-1]
    if remainder > 0:
        codes[full_blocks, :] = PARTIAL_BLOCKS_COL[remainder]
        codes[full_blocks+1:, :] = ord(" ")
    else:
        codes[full_blocks:, :] = ord(" ")
    # (flip)
    codes = codes[::-1]

    return CharArray.from_codes(codes, fgcolor, bgcolor)


# # # 
# UNICODE BOX DRAWING


class BoxStyle(str, enum.Enum):
    """
    A string enum defining preset styles for the `border` plot.

    Each style is a string of eight characters representing the border
    elements.

    Available Styles:

    * `LIGHT`:  A standard, single-line border.
    * `HEAVY`:  A thicker, bold border.
    * `DOUBLE`: A double-line border.
    * `DASHED`: A dashed single-line border.
    * `BLANK`:  An invisible border (easily add 1-width padding).
    * `ROUND`:  A single-line border with rounded corners.
    * `BUMPER`: A single-line border with corners made of blocks.
    * `BLOCK1`: A blocky border with half-width left and right walls.
    * `BLOCK2`: A uniform blocky border.
    * `TIGER1`: A stripy block border.
    * `TIGER2`: An alternative stripy block border.

    Demo:

    ```
    ┌──────┐ ┏━━━━━━┓ ╔══════╗ ┌╌╌╌╌╌╌┐ ⡤⠤⠤⠤⠤⠤⠤⢤ ╭──────╮
    │LIGHT │ ┃HEAVY ┃ ║DOUBLE║ ┊DASHED┊ ⡇DOTTED⢸ │ROUND │
    └──────┘ ┗━━━━━━┛ ╚══════╝ └╌╌╌╌╌╌┘ ⠓⠒⠒⠒⠒⠒⠒⠚ ╰──────╯
             ▛──────▜ ▛▀▀▀▀▀▀▜ █▀▀▀▀▀▀█ ▞▝▝▝▝▝▝▝ ▘▘▘▘▘▘▘▚
     BLANK   │BUMPER│ ▌BLOCK1▐ █BLOCK2█ ▖TIGER1▝ ▘TIGER2▗
             ▙──────▟ ▙▄▄▄▄▄▄▟ █▄▄▄▄▄▄█ ▖▖▖▖▖▖▖▞ ▚▗▗▗▗▗▗▗
    ```

    TODO:
    
    * It might make sense to consider borders with two characters on the left
      and right sides of the contents. Would open up new design possibilities.
    """
    LIGHT  = "┌─┐││└─┘"
    HEAVY  = "┏━┓┃┃┗━┛"
    DOUBLE = "╔═╗║║╚═╝"
    DASHED = "┌╌┐┊┊└╌┘"
    DOTTED = "⡤⠤⢤⢸⡇⠓⠒⠚"
    ROUND  = "╭─╮││╰─╯"
    BLANK  = "        "
    BUMPER = "▛─▜││▙─▟"
    BLOCK1 = "▛▀▜▐▌▙▄▟"
    BLOCK2 = "█▀████▄█"
    TIGER1 = "▞▝▝▝▖▖▖▞"
    TIGER2 = "▘▘▚▘▘▚▗▗"

    @property
    def _nw(self) -> int:
        """Northwest corner symbol."""
        return ord(self[0])

    @property
    def _n(self) -> int:
        """North edge symbol."""
        return ord(self[1])

    @property
    def _ne(self) -> int:
        """Norteast corner symbol."""
        return ord(self[2])

    @property
    def _e(self) -> int:
        """East edge symbol."""
        return ord(self[3])

    @property
    def _w(self) -> int:
        """West edge symbol."""
        return ord(self[4])

    @property
    def _sw(self) -> int:
        """Southwest corner symbol."""
        return ord(self[5])

    @property
    def _s(self) -> int:
        """South edge symbol."""
        return ord(self[6])

    @property
    def _se(self) -> int:
        """Southeast corner symbol."""
        return ord(self[7])


def unicode_box(
    chars: CharArray,
    style: BoxStyle,
    fgcolor: ColorLike | None = None,
    bgcolor: ColorLike | None = None,
    title: str = "",
) -> CharArray:
    """
    Wrap a character array in an outline of box drawing characters.
    """
    # padded codepoints
    codes = np.pad(chars.codes, 1, constant_values=0)
    # assemble box
    codes[ 0,1:-1] = style._n
    codes[-1,1:-1] = style._s
    codes[1:-1, 0] = style._w
    codes[1:-1,-1] = style._e
    codes[ 0, 0] = style._nw
    codes[ 0,-1] = style._ne
    codes[-1, 0] = style._sw
    codes[-1,-1] = style._se
    # padded foreground colours
    fg = np.pad(chars.fg, 1, constant_values=fgcolor is not None)
    fg_rgb = np.pad(
        chars.fg_rgb,
        ((1,1),(1,1),(0,0)),
        constant_values=0,
    )
    if fgcolor is not None:
        fgcolor_ = parse_color(fgcolor)
        fg_rgb[[0,-1],:] = fgcolor_
        fg_rgb[:,[0,-1]] = fgcolor_
    # padded background colours
    bg = np.pad(chars.bg, 1, constant_values=bgcolor is not None)
    bg_rgb = np.pad(
        chars.bg_rgb,
        ((1,1),(1,1),(0,0)),
        constant_values=0,
    )
    if bgcolor is not None:
        bgcolor_ = parse_color(bgcolor)
        bg_rgb[[0,-1],:] = bgcolor_
        bg_rgb[:,[0,-1]] = bgcolor_
    # assemble char array
    wrapped_chars = CharArray(
        codes=codes,
        fg=fg,
        fg_rgb=fg_rgb,
        bg=bg,
        bg_rgb=bg_rgb,
    )
    # position title
    title = title[:chars.width]
    spos = wrapped_chars.width//2-len(title)//2
    wrapped_chars.codes[0,spos:spos+len(title)] = ords(title)
    return wrapped_chars


# # #
# UNICODE RULES AND TICKS


# the four directions a character can reach out in, summed into the index of
# the character that joins them
_UP, _DOWN, _LEFT, _RIGHT = 1, 2, 4, 8


class LineStyle(str, enum.Enum):
    """
    A string enum defining the weights of line available to draw axes with.

    Each style is a string of sixteen characters, one for every combination of
    directions a character can reach out in. Index the string by the sum of 1
    for up, 2 for down, 4 for left and 8 for right to find the character that
    joins exactly those directions.

    Available Styles:

    * `LIGHT`:  Single lines meeting at square corners.
    * `HEAVY`:  Thick single lines.
    * `ROUND`:  Single lines meeting at rounded corners.
    * `DOUBLE`: Double lines. This set has no half-length stubs, so a line
      that ends without either a corner or a tick to finish it runs to the
      edge of its final cell instead of stopping halfway.

    Demo:

    ```
    ┌─┬─┐  ┏━┳━┓  ╭─┬─╮  ╔═╦═╗
    ├─┼─┤  ┣━╋━┫  ├─┼─┤  ╠═╬═╣
    └─┴─┘  ┗━┻━┛  ╰─┴─╯  ╚═╩═╝
    ```
    """
    LIGHT  = " ╵╷│╴┘┐┤╶└┌├─┴┬┼"
    HEAVY  = " ╹╻┃╸┛┓┫╺┗┏┣━┻┳╋"
    ROUND  = " ╵╷│╴╯╮┤╶╰╭├─┴┬┼"
    DOUBLE = " ║║║═╝╗╣═╚╔╠═╩╦╬"


def unicode_frame(
    chars: CharArray,
    style: LineStyle,
    cells: tuple[bool, bool, bool, bool],
    rules: tuple[bool, bool, bool, bool],
    ticks: tuple[bool, bool, bool, bool],
    title: str = "",
    fgcolor: ColorLike | None = None,
) -> CharArray:
    """
    Surround a character array with a rule along any of its four sides.

    Each side, in the order north, east, south, west, is described by three
    flags: whether it takes a cell at all, whether a line is drawn in that
    cell, and whether the ends of that line are ticked. A tick is an arm
    reaching outward from the end of a line, towards wherever a label goes.

    Every character is derived rather than chosen. A cell reaches towards each
    neighbouring cell that is also part of a rule, and outward wherever a tick
    is called for, and the resulting set of directions selects the character
    from the style. So the corner where two ruled sides meet turns, the corner
    where one of them is missing finishes, and a ticked end grows the arm that
    points at its label, without any of the three being written down.

    A rule runs the length of the array it is drawn beside, and reaches into
    the corner cell it shares with a neighbouring side only when that side is
    ruled as well, so that a side left blank stays outside the frame.

    Inputs:

    * chars : CharArray.
        The array to surround.
    * style : LineStyle.
        The weight of line to draw.
    * cells : (bool, bool, bool, bool).
        Whether the north, east, south and west sides each take a cell.
    * rules : (bool, bool, bool, bool).
        Whether a line is drawn in that cell. A side that draws one must take
        a cell.
    * ticks : (bool, bool, bool, bool).
        Whether the ends of that line are ticked. A side that ticks must draw
        a line.
    * title : str.
        Written along the north side, centred over the array and truncated to
        fit. The north side must take a cell.
    * fgcolor : optional ColorLike.
        The colour of the rules and the title. Defaults to the terminal's
        foreground colour.

    Returns:

    * framed : CharArray.
        The array, surrounded by whichever of the four sides took a cell.
    """
    for side, cell, rule, tick in zip("nesw", cells, rules, ticks):
        if rule and not cell:
            raise ValueError(f"side {side} draws a rule but takes no cell")
        if tick and not rule:
            raise ValueError(f"side {side} is ticked but draws no rule")
    if title and not cells[0]:
        raise ValueError("a title needs the north side to take a cell")

    n_cell, e_cell, s_cell, w_cell = (int(c) for c in cells)
    n_rule, e_rule, s_rule, w_rule = (int(r) for r in rules)
    n_tick, e_tick, s_tick, w_tick = ticks

    framed = chars.pad(
        above=n_cell,
        below=s_cell,
        left=w_cell,
        right=e_cell,
        fgcolor=fgcolor,
    )
    height, width = framed.height, framed.width

    # each rule runs the length of the array, reaching into a shared corner
    # only where the neighbouring side is ruled too
    columns = slice(w_cell - w_rule, w_cell + chars.width + e_rule)
    rows = slice(n_cell - n_rule, n_cell + chars.height + s_rule)
    ruled = np.zeros((height, width), dtype=bool)
    if n_rule:
        ruled[0, columns] = True
    if s_rule:
        ruled[-1, columns] = True
    if w_rule:
        ruled[rows, 0] = True
    if e_rule:
        ruled[rows, -1] = True

    # a ruled cell reaches towards each of its ruled neighbours
    arms = np.zeros((height, width), dtype=int)
    arms[1:, :] |= np.where(ruled[:-1, :], _UP, 0)
    arms[:-1, :] |= np.where(ruled[1:, :], _DOWN, 0)
    arms[:, 1:] |= np.where(ruled[:, :-1], _LEFT, 0)
    arms[:, :-1] |= np.where(ruled[:, 1:], _RIGHT, 0)
    arms[~ruled] = 0

    # and outward, at the ends of a ticked rule, towards its labels
    for tick, direction, ends in (
        (n_tick, _UP,    ((0, columns.start), (0, columns.stop - 1))),
        (s_tick, _DOWN,  ((-1, columns.start), (-1, columns.stop - 1))),
        (w_tick, _LEFT,  ((rows.start, 0), (rows.stop - 1, 0))),
        (e_tick, _RIGHT, ((rows.start, -1), (rows.stop - 1, -1))),
    ):
        if tick:
            for row, column in ends:
                arms[row, column] |= direction

    glyphs = np.array([ord(c) for c in style], dtype=np.uint32)
    framed.codes[ruled] = glyphs[arms[ruled]]

    # position title
    title = title[:chars.width]
    start = width // 2 - len(title) // 2
    framed.codes[0, start:start + len(title)] = ords(title)

    return framed


# # # 
# UNICODE HALF-BLOCK IMAGE


def unicode_image(
    image: NDArray, # uint8[h, w, rgb]
) -> CharArray:     # Char[ceil(h/2), w]
    """
    Convert an RGB image into an array of coloured Unicode half-block
    characters representing the pixels of the image.

    Inputs:

    * image: u8[h, w, rgb].
        The pixels of the image.

    Returns:

    * chars: CharArray[ceil(h/2), w].
        The array of coloured half-block characters. If the image has odd
        height, the bottom half of the final row is set to the default
        background colour.
    """
    # pad to even height
    h, _w, _3 = image.shape
    pad = (h % 2 == 1)
    if pad:
        image = np.pad(image, ((0, 1), (0, 0), (0, 0)))

    # pair pixels along vertical axis
    stacked = einops.rearrange(
        image,
        '(h fgbg) w c -> h fgbg w c',
        fgbg=2,
    )

    # construct character array
    H, _2, W, _3 = stacked.shape
    chars = CharArray(
        codes=np.full((H, W), ord("▀"), dtype=np.uint32),
        fg=np.ones((H, W), dtype=bool),
        fg_rgb=stacked[:,0,:,:],
        bg=np.ones((H, W), dtype=bool),
        bg_rgb=stacked[:,1,:,:],
    )

    # remove final row if necessary
    if pad:
        chars.bg[-1,:] = False

    return chars

