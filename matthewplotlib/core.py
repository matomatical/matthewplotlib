"""
A collection of classes and types used internally.

Constants:

* `BLANK : Char`

    A colourless space character used for padding.
"""

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
    

    def __bool__(self):
        """
        True if the character has visible content, false if it is blank.
        """
        return bool(self.c != " " or self.bg is not None)


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
    ) -> np.ndarray: # uint8[16,8,4]
        """
        Convert a Char to a small RGBA image patch, with the specified foreground
        color (or white) and background color (or a transparent background).
        """
        # decide colors
        if self.fg is not None:
            fg = np.array([*self.fg, 255], dtype=np.uint8)
        else:
            fg = np.array([255, 255, 255, 255], dtype=np.uint8)
        if self.bg is not None:
            bg = np.array([*self.bg, 255], dtype=np.uint8)
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
# BRAILLE HELPER FUNCTIONS


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

    * a: bool[4h, 2w]
        
        Array of booleans, height divisible by 4 and width divisible by 2.

    Returns:

    * bits: uint16[h, w]

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
