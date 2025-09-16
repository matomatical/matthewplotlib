"""
A collection of classes and types used internally.

Constants:

* `BLANK : Char`

    A colourless space character used for padding.
"""

import enum
import dataclasses
import numpy as np

from typing import Self
from numpy.typing import ArrayLike

from matthewplotlib.colors import Color
from matthewplotlib.unscii16 import bitmap


# # # 
# Core characters


@dataclasses.dataclass(frozen=True)
class Char:
    """
    A single possibly-coloured character. Plots are assembled from characters
    like these.
    """
    c: str = " "
    fg: Color | None = None
    bg: Color | None = None
    

    def isblank(self: Self) -> bool:
        """
        True if the character has no visible content.
        """
        return bool(self.c.isspace() and self.bg is None)

    
    @property
    def bg_(self: Self) -> Color | None:
        """
        The 'effective' background color of this Char.

        Usually, this is just the background color, except in the special case
        where c happens to be '█', in which case return the foreground color.
        """
        if self.c == "█":
            return self.fg
        else:
            return self.bg


    def to_ansi_str(self: Self) -> str:
        """
        If necessary, wrap a Char in ANSI control codes that switch the color into
        the given fg and bg colors; plus a control code to switch back to default
        mode.
        """
        ansi_controls = []
        if self.fg is not None:
            ansi_controls.extend([38, 2, *self.fg])
        if self.bg is not None:
            ansi_controls.extend([48, 2, *self.bg])
        if ansi_controls:
            return f"\x1b[{";".join(map(str, ansi_controls))}m{self.c}\x1b[0m"
        else:
            return self.c


    def to_rgba_array(
        self: Self,
        bgcolor: Color | None = None,
    ) -> np.ndarray: # uint8[16,8,4]
        """
        Convert a Char to a small RGBA image patch, with the specified foreground
        color (or white) and background color (or a transparent background).
        """
        # decide foreground color
        if self.fg is not None:
            fg = np.array([*self.fg, 255], dtype=np.uint8)
        else:
            fg = np.array([255, 255, 255, 255], dtype=np.uint8)
        
        # decide background color
        if self.bg is not None:
            bg = np.array([*self.bg, 255], dtype=np.uint8)
        elif bgcolor is not None:
            bg = np.array([*bgcolor, 255], dtype=np.uint8)
        else:
            bg = np.array([0, 0, 0, 0], dtype=np.uint8)

        # construct rgb array
        bits = bitmap(self.c)       # bool[16,8]
        img = np.where(
            bits[..., np.newaxis],  # bool[16,8,1]
            fg,                     # uint8[3]
            bg,                     # uint8[3]
        )                           # -> uint8[16,8,3]
        return img


BLANK = Char(c=" ", fg=None, bg=None)


# # # 
# UNICODE BRAILLE DOT MATRIX


BRAILLE_MAP = np.array([
    [0, 3],
    [1, 4],
    [2, 5],
    [6, 7],
], dtype=np.uint8)


def braille_encode(
    a: ArrayLike,   # bool[4h, 2w]
) -> np.ndarray:    # -> uint16[h, w]
    """
    Turns a HxW array of booleans into a (H//4)x(W//2) array of braille
    binary codes.

    Inputs:

    * a: bool[4h, 2w].
        Array of booleans, height divisible by 4 and width divisible by 2.

    Returns:

    * bits: uint16[h, w].
        An array of braille unicode code points. The unicode characters will
        have a dot in the corresponding places where `a` is True.

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
     `->⡇⢸⢸⠉⠁⡇⠀⢸⠀⠀⡎⢱  (Note: this function returns codepoints, use `chr()`
        ⡏⢹⢸⣉⡁⣇⣀⢸⣀⡀⢇⡸  to convert these into braille characters for printing.)
        '''
    ```
    """
    # process input
    array = np.asarray(a, dtype=bool)
    H, W = array.shape
    h, w = H // 4, W // 2
    
    # create a view that chunks it into 4x2 cells
    cells = array.reshape(h, 4, w, 2)
    
    # convert each bit in each cell into a mask and combine into code array
    masks = np.left_shift(cells, BRAILLE_MAP.reshape(1,4,1,2), dtype=np.uint16)
    codes = np.bitwise_or.reduce(masks, axis=(1,3))
    
    # unicode braille block starts at 0x2800
    unicodes = 0x2800 + codes
    return unicodes


# # # 
# UNICODE PARTIAL BLOCKS


PARTIAL_BLOCKS_ROW = [" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"]


def unicode_bar(
    proportion: float,
    total_width: int,
) -> list[str]:
    """
    Generates a Unicode progress bar as a list of characters.

    This function creates a fixed-width left-to-right bar using Unicode block
    elements to represent the proportion rounded down to nearest 1/8th of a
    block.

    Inputs:

    * proportion: float.
        The fraction of the bar to fill. Should be between 0.0 and 1.0
        inclusive.
    * total_width: int.
        The width of the full bar in characters. Should be positive.

    Returns:

    * bar: list[str].
        A list of unicode characters representing the bar. The length of the
        list is always equal to `total_width`.

    Examples:

    ```
    >>> ''.join(unicode_bar(0.5, 10))
    '█████     '
    >>> ''.join(unicode_bar(0.625, 10))
    '██████▎   '

    ```
    """
    # clip inputs to valid range
    proportion = max(0.0, min(1.0, proportion))
    total_width = max(1, total_width)

    # calculate number of filled 'eighths'
    full_eighths = int(proportion * total_width * 8)
    full_blocks, remainder = divmod(full_eighths, 8)

    # construct bar
    bar = ["█"] * full_blocks
    if remainder > 0:
        bar.append(PARTIAL_BLOCKS_ROW[remainder])
    bar.extend([" "] * (total_width - len(bar)))

    return bar


PARTIAL_BLOCKS_COL = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]


def unicode_col(
    proportion: float,
    total_height: int,
) -> list[str]:
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
    * total_height: int.
        The height of the full bar in characters. Should be positive.

    Returns:

    * bar: list[str]

        A list of unicode characters representing the bar. The length of the
        list is always equal to `total_height`.

    Examples:

    ```
    >>> unicode_col(0.5, 3)
    [' ','▄','█']
    
    ```
    """
    # clip inputs to valid range
    proportion = max(0.0, min(1.0, proportion))
    total_height = max(1, total_height)

    # calculate number of filled 'eighths'
    full_eighths = int(proportion * total_height * 8)
    full_blocks, remainder = divmod(full_eighths, 8)

    # construct col
    col = ["█"] * full_blocks
    if remainder > 0:
        col.append(PARTIAL_BLOCKS_COL[remainder])
    col.extend([" "] * (total_height - len(col)))
    col = col[::-1]

    return col


# # # 
# BOX DRAWING


class BoxStyle(str, enum.Enum):
    """
    A string enum defining preset styles for the `border` plot.

    Each style is a string of six characters representing the border
    elements in the following order: horizontal, vertical, top-left,
    top-right, bottom-left, and bottom-right.

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
    * `LIGHTX`: A heavy border with axis ticks.
    * `LOWERX`: A partial border with axis ticks.

    Demo:

    ```
    ┌──────┐ ┏━━━━━━┓ ╔══════╗ ┌╌╌╌╌╌╌┐ ⡤⠤⠤⠤⠤⠤⠤⢤ ╭──────╮
    │LIGHT │ ┃HEAVY ┃ ║DOUBLE║ ┊DASHED┊ ⡇DOTTED⢸ │ROUND │
    └──────┘ ┗━━━━━━┛ ╚══════╝ └╌╌╌╌╌╌┘ ⠓⠒⠒⠒⠒⠒⠒⠚ ╰──────╯
             ▛──────▜ ▛▀▀▀▀▀▀▜ █▀▀▀▀▀▀█ ▞▝▝▝▝▝▝▝ ▘▘▘▘▘▘▘▚
     BLANK   │BUMPER│ ▌BLOCK1▐ █BLOCK2█ ▖TIGER1▝ ▘TIGER2▗
             ▙──────▟ ▙▄▄▄▄▄▄▟ █▄▄▄▄▄▄█ ▖▖▖▖▖▖▖▞ ▚▗▗▗▗▗▗▗
    ┬──────┐ ┲━━━━━━┓ ╔══════╗ ╷        
    │LIGHTX│ ┃HEAVYX┃ ║DOUBLX║ │LOWERX  
    ┼──────┤ ╄━━━━━━┩ ╚══════╝ ┼──────╴ 
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
    def nw(self) -> str:
        """Northwest corner symbol."""
        return self[0]
    

    @property
    def n(self) -> str:
        """North edge symbol."""
        return self[1]
    

    @property
    def ne(self) -> str:
        """Norteast corner symbol."""
        return self[2]
    

    @property
    def e(self) -> str:
        """East edge symbol."""
        return self[3]
    

    @property
    def w(self) -> str:
        """West edge symbol."""
        return self[4]
    

    @property
    def sw(self) -> str:
        """Southwest corner symbol."""
        return self[5]
    

    @property
    def s(self) -> str:
        """South edge symbol."""
        return self[6]
    

    @property
    def se(self) -> str:
        """Southeast corner symbol."""
        return self[7]


# # # 
# 3D projection


def project3(
    xyz: np.ndarray,                                        # float[n, 3]
    camera_position: np.ndarray = np.array([0., 0., 2.]),   # float[3]
    camera_target: np.ndarray = np.zeros(3),                # float[3]
    scene_up: np.ndarray = np.array([0.,1.,0.]),            # float[3]
    fov_degrees: float = 90.0,
) -> tuple[
    np.ndarray,                                             # float[n, 2]
    np.ndarray,                                             # bool[n]
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

    * This implementation uses a left-handed coordinate system for the camera
      coordinate system, where X and Y point left and up respectively but then
      Z points towards the object ahead of the camera instead of away from the
      object behind the camera. I don't think there is any external effect of
      this left-handedness because the final projection logic takes it into
      account, but I think it is non-standard. Watch out.
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
