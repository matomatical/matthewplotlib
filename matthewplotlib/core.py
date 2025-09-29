from __future__ import annotations

import enum
import dataclasses

import numpy as np
import einops

from typing import Self, Callable
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
        return self.codes == ord(" ") & ~self.bg


    def isnonblank(self: Self) -> NDArray: # bool[h,w]
        """
        True where the character has visible content.
        """
        return self.codes != ord(" ") | self.bg


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
        

def ords(chrs):
    """
    Convert a string or list of characters to a list of unicode code points.
    """
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

    ```
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

    ```
    >>> unicode_col(0.5, 3).to_plain_str()
    ' \n▄\n█'
    
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
    * `LIGHTX`: A light border with axis ticks.
    * `HEAVYX`: A heavy border with axis ticks.
    * `LOWERX`: A partial border with axis ticks.

    Demo:

    ```
    ┌──────┐ ┏━━━━━━┓ ╔══════╗ ┌╌╌╌╌╌╌┐ ⡤⠤⠤⠤⠤⠤⠤⢤ ╭──────╮
    │LIGHT │ ┃HEAVY ┃ ║DOUBLE║ ┊DASHED┊ ⡇DOTTED⢸ │ROUND │
    └──────┘ ┗━━━━━━┛ ╚══════╝ └╌╌╌╌╌╌┘ ⠓⠒⠒⠒⠒⠒⠒⠚ ╰──────╯
             ▛──────▜ ▛▀▀▀▀▀▀▜ █▀▀▀▀▀▀█ ▞▝▝▝▝▝▝▝ ▘▘▘▘▘▘▘▚
     BLANK   │BUMPER│ ▌BLOCK1▐ █BLOCK2█ ▖TIGER1▝ ▘TIGER2▗
             ▙──────▟ ▙▄▄▄▄▄▄▟ █▄▄▄▄▄▄█ ▖▖▖▖▖▖▖▞ ▚▗▗▗▗▗▗▗
    ┬──────┐ ┲━━━━━━┓ ╷        
    │LIGHTX│ ┃HEAVYX┃ │LOWERX  
    ┼──────┤ ╄━━━━━━┩ ┼──────╴ 
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
    LIGHTX = "┬─┐││┼─┤"
    HEAVYX = "┲━┓┃┃╄━┩"
    LOWERX = "╷   │┼─╴"

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


