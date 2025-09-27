import enum
import dataclasses

import numpy as np
import einops

from typing import Self
from numpy.typing import NDArray, ArrayLike

from matthewplotlib.unscii16 import bitmaps


type ColorLike = (
    str
    | NDArray # float[3] (0 to 1) or uint8[3] (0 to 255)
    | tuple[int, int, int]
    | tuple[float, float, float]
    | Color
)


type Color = NDArray # uint8[3]


@dataclasses.dataclass(frozen=True)
class CharArray:
    """
    A grid of possibly-coloured characters comprising a plot. For internal use.

    Fields:

    * c: uint32[h,w].
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
    c: NDArray      # uint32[h,w]
    fg: NDArray     # bool[h,w]
    fg_rgb: NDArray # uint8[h,w,3]
    bg: NDArray     # bool[h,w]
    bg_rgb: NDArray # uint8[h,w,3]


    @property
    def height(self: Self) -> int:
        h, _w = self.c.shape
        return h
    

    @property
    def width(self: Self) -> int:
        _h, w = self.c.shape
        return w


    @staticmethod
    def from_codes(
        codes: NDArray, # uint32[h,w]
        fgcolor: ColorLike | None,
        bgcolor: ColorLike | None,
    ) -> CharArray:
        if fgcolor is None:
            fg = np.zeros_like(codes, dtype=bool)
            fg_rgb = np.zeros((*codes.shape, 3), dtype=np.uint8)
        else:
            fg = np.ones_like(codes, dtype=bool)
            fg_rgb = np.full(
                (*codes.shape, 3),
                parse_color(fgcolor),
                dtype=np.uint8,
            )
        if bgcolor is None:
            bg = np.zeros_like(codes, dtype=bool)
            bg_rgb = np.zeros((*codes.shape, 3), dtype=np.uint8)
        else:
            bg = np.ones_like(codes, dtype=bool)
            bg_rgb = np.full(
                (*codes.shape, 3),
                parse_color(bgcolor),
                dtype=np.uint8,
            )
        return CharArray(
            c=codes,
            fg=fg,
            fg_rgb=fg_rgb,
            bg=bg,
            bg_rgb=bg_rgb,
        )


    def isblank(self: Self) -> NDArray: # bool[h,w]
        """
        True where the character has no visible content.
        """
        return (self.c == ord(" ")) & (~self.bg)


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
                s.append(chr(self.c[i,j]))
            # end of line
            if i < self.height - 1:
                s.append("\n")
        # end of grid
        if current_fg is not None or current_bg is not None:
            s.append("\x1b[0m")
        return "".join(s)
    

    def to_plain_str(self: Self) -> str:
        """
        Render a CharArray as a sequence of characters without colour.
        """
        rows = [
            [chr(self.c[i,j]) for j in range(self.width)]
            for i in range(self.height)
        ]
        return "\n".join("".join(row) for row in rows)


    def to_rgba_array(
        self: Self,
        bgcolor: NDArray | None = None, # optional uint8[4] (RGBA)
    ) -> np.ndarray: # uint8[height*16,width*8,4]
        """
        Convert a CharArray to an RGBA image array
        """
        # foreground color array
        fg = np.full((self.height, self.width, 4), 255, dtype=np.uint8)
        fg[self.fg, :3] = self.fg_rgb[self.fg]
        
        # background color array
        if bgcolor is None:
            bg = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        else:
            bg = np.full((self.height, self.width, 4), bgcolor, dtype=np.uint8)
        bg[self.bg, :3] = self.bg_rgb[self.bg]
        bg[self.bg, 3] = 255
        
        # construct rgba array
        bits = bitmaps(self.c)       # bool[h,w,16,8]
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
        tiles = bitmaps(self.c) # bool[h,w,16,8]
        img = einops.rearrange(
            tiles,
            'H W h w -> (H h) (W w)',
        ) # -> uint8[h*16,w*8]
        return img


# # # 
# Color handling


def parse_color(color: ColorLike | None) -> Color | None:
    """
    Accept and standardise RGB triples in any of the following 'color like'
    formats:

    1. **Named colours:** The following strings are recognised and translated
       to RGB triples: `"black"`, `"red"`, `"green"`, `"blue"`, `"cyan"`,
       `"magenta"`, `"yellow"`, `"white"`.

    2. **Hexadecimal:** A hexadecimal string like ``"#ff0000"`` specifying the
       RGB values in the usual manner.

    3. **Short hexadecimal:** A three-character hexadecimal string like
       `"#f00"`, where `"#RGB"` is equivalent to `"#RRGGBB"` in the usual
       hexadecimal format.

    4. **Integer triple:** An array or tuple of three integers in the range 0
       to 255, converted directly to an RGB triple.

    5. **Float triple:** An array or tuple of three floats in the range 0.0 to
       1.0, converted to an RGB triple by multiplying by 255 and rounding down
       to the nearest integer.

    (Arrays or tuples with mixed integers and floats are promoted by NumPy to
    become float triples.)
    """
    if color is None:
        return None

    if isinstance(color, str):
        if color.startswith("#") and len(color) == 4:
            return np.array((
                17*int(color[1], base=16),
                17*int(color[2], base=16),
                17*int(color[3], base=16),
            ), dtype=np.uint8)
        if color.startswith("#") and len(color) == 7:
            return np.array((
                int(color[1:3], base=16),
                int(color[3:5], base=16),
                int(color[5:7], base=16),
            ), dtype=np.uint8)
        if color.lower() in NAMED_COLORS:
            return NAMED_COLORS[color.lower()]

    elif isinstance(color, (np.ndarray, tuple)):
        color_ = np.asarray(color)
        if color_.shape == (3,):
            if np.issubdtype(color_.dtype, np.floating):
                return (255*np.clip(color_, 0., 1.)).astype(np.uint8)
            if np.issubdtype(color_.dtype, np.integer):
                return np.clip(color_, 0, 255).astype(np.uint8)
    
    raise ValueError(f"invalid color {color!r}")


NAMED_COLORS = {
    "black":    np.array((  0,   0,   0), dtype=np.uint8),
    "red":      np.array((255,   0,   0), dtype=np.uint8),
    "green":    np.array((  0, 255,   0), dtype=np.uint8),
    "blue":     np.array((  0,   0, 255), dtype=np.uint8),
    "cyan":     np.array((  0, 255, 255), dtype=np.uint8),
    "magenta":  np.array((255,   0, 255), dtype=np.uint8),
    "yellow":   np.array((255, 255,   0), dtype=np.uint8),
    "white":    np.array((255, 255, 255), dtype=np.uint8),
}
# # # 
# Core characters


# # # 
# UNICODE BRAILLE DOT MATRIX


BRAILLE_MAP = np.array([
    [0, 3],
    [1, 4],
    [2, 5],
    [6, 7],
], dtype=np.uint8)


def unicode_braille_array(
    dots: NDArray, # bool[4h, 2w]
    fgcolor: ColorLike | None = None,
    bgcolor: ColorLike | None = None,
) -> CharArray: # Char[h, w]
    """
    Turns a HxW array of booleans into a (H//4)x(W//2) array of braille
    binary codes.

    Inputs:

    * dots: bool[4h, 2w].
        Array of booleans, height divisible by 4 and width divisible by 2.
    * fgcolor: optional Color.
        Foreground color used for braille characters.
    * bgcolor: optional Color.
        Background color used for all characters.

    Returns:

    * chars: CharArray.
        An array of Braille characters with H rows and W columns.

    An illustrated example is as follows:
    ```
    Start with an array with height divisible by 4, width divisible by 2:
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
        ⡏⢹⢸⣉⡁⣇⣀⢸⣀⡀⢇⡸  .to_ansi_str to get a string.)
        '''
    ```
    """
    # process input
    dots_ = np.asarray(dots, dtype=bool)
    H, W = dots_.shape
    h, w = H // 4, W // 2
    
    # create a view that chunks it into 4x2 cells
    cells = dots_.reshape(h, 4, w, 2)
    
    # convert each bit in each cell into a mask and combine into code array
    masks = np.left_shift(cells, BRAILLE_MAP.reshape(1,4,1,2), dtype=np.uint32)
    codes = np.bitwise_or.reduce(masks, axis=(1,3))
    
    # convert to unicode braille codepoints (except for blanks)
    codes = np.where(
        codes > 0,
        0x2800 + codes,
        ord(" "),
    )
    
    return CharArray.from_codes(codes, fgcolor, bgcolor)


# # # 
# UNICODE PARTIAL BLOCKS


PARTIAL_BLOCKS_ROW = [
    ord(c) for c in [" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"]
]


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
    * fgcolor: optional Color.
        Foreground color used for the progress bar characters.
    * bgcolor: optional Color.
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


PARTIAL_BLOCKS_COL = [
    ord(c) for c in [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
]


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
) -> CharArray:
    """
    Wrap a character array in an outline of box drawing characters.
    """
    # padded codepoints
    codes = np.pad(chars.c, 1, constant_values=0)
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
        c=codes,
        fg=fg,
        fg_rgb=fg_rgb,
        bg=bg,
        bg_rgb=bg_rgb,
    )
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
        c=np.full((H, W), ord("▀"), dtype=np.uint32),
        fg=np.ones((H, W), dtype=bool),
        fg_rgb=stacked[:,0,:,:],
        bg=np.ones((H, W), dtype=bool),
        bg_rgb=stacked[:,1,:,:],
    )

    # remove final row if necessary
    if pad:
        chars.bg[-1,:] = False

    return chars


# # # 
# 3D projection


def project3(
    xyz: np.ndarray, # float[n, 3]
    camera_position: np.ndarray = np.array([0., 0., 2.]), # float[3]
    camera_target: np.ndarray = np.zeros(3), # float[3]
    scene_up: np.ndarray = np.array([0.,1.,0.]), # float[3]
    fov_degrees: float = 90.0,
) -> tuple[
    np.ndarray, # float[n, 2]
    np.ndarray, # bool[n]
]:
    """
    Project a 3d point cloud into two dimensions based on a given camera
    configuration.

    Inputs:

    * xyz: float[n, 3].
        The points to project, with columns corresponding to X, Y, and Z.
    * camera_position: float[3] (default: [0. 0. 2.]).
        The position at which the camera is placed. The default is positioned
        along the positive Z axis.
    * camera_target: float[3] (default: [0. 0. 0.]).
        The position towards which the camera is facing. Should be distinct
        from camera position. The default is that the camera is facing towards
        the origin.
    * scene_up: float[3] (default: [0. 1. 0.]).
        The unit vector designating the 'up' direction for the scene. The
        default is the positive Y direction. Should not have the same direction
        as camera_target - camera_position.
    * fov_degrees: float (default 90).
        Field of view. Points within a cone (or frustum) of this angle leaving
        the camera are projected into the unit disk (or the square [-1,1]^2).

    Returns:

    * xy: float[n, 2].
        Projected points.
    * valid: bool[n].
        Mask indicating which of the points are in front of the camera.

    Notes:

    * The combined effect of the defaults is that the camera is looking down
      the Z axis towards the origin from the positive direction, with the X
      axis extending towards the right and the Y axis extending upwards, with
      the field of view ensuring that points within the cube [-1,1]^3 are
      projected into the square [-1,1]^2.
    * The valid mask only considers whether points are in front of the camera.
      A more comprehensive frustum clipping approach is not supported.
    
    Internal notes:

    * This implementation uses a coordinate system for the camera where X and Y
      point left and up respectively and Z points towards the object ahead of
      the camera (an alternative convention is for Z to point behind the
      camera).
    """
    n, _3 = xyz.shape

    # compute view matrix
    V_z = camera_target - camera_position
    V_z /= np.linalg.norm(V_z)
    V_x = np.cross(V_z, scene_up)
    V_x /= np.linalg.norm(V_x)
    V_y = np.cross(V_x, V_z)
    V = np.array([V_x, V_y, V_z]).T

    # transform points to camera coordinate system
    xyz_ = (xyz - camera_position) @ V
    
    # mask for valid points
    valid = xyz_[:, 2] > 0.
    
    # perspective projection
    xy = np.zeros((n, 2))
    np.divide(
        xyz_[:, :2],
        xyz_[:, 2, np.newaxis],
        out=xy,
        where=valid[:, np.newaxis],
    )

    # scale fov to within [-1,1]^2
    focal_length = 1 / np.tan(np.radians(fov_degrees) / 2)
    xy *= focal_length

    return xy, valid
