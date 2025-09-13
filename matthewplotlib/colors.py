"""
Configuring various plots involves specifying colours. In most cases, colours
can be specified in one of the following `ColorLike` formats:

1. **Named colours:** The following strings are recognised and translated to
   RGB triples: `"black"`, `"red"`, `"green"`, `"blue"`, `"cyan"`, `"magenta"`,
   `"yellow"`, `"white"`.

2. **Hexadecimal:** A hexadecimal string like ``"#ff0000"`` specifying the RGB
   values in the usual manner.

3. **Short hexadecimal:** A three-character hexadecimal string like `"#f00"`,
   where `"#RGB"` is equivalent to `"#RRGGBB"` in the usual hexadecimal format.

4. **Integer triple:** An array or tuple of three integers in the range 0 to
    255, converted directly to an RGB triple.

5. **Float triple:** An array or tuple of three floats in the range 0.0 to 1.0,
   converted to an RGB triple by multiplying by 255 and rounding down to the
   nearest integer.
   
   (Arrays or tuples with mixed integers and floats are promoted by NumPy to
   become float triples.)

The `Color` class is used internally to represent an RGB colour that has been
parsed from one of the above formats. In most cases, it is not used externally.

In some contexts, colours are specified through a colour map rather than
directly specified---see the `matthewplotlib.colormaps` module for details.
"""
from __future__ import annotations
from typing import Self, Iterator
import dataclasses

import numpy as np


type ColorLike = (
    str
    | np.ndarray # float[3] (0 to 1) or uint8[3] (0 to 255)
    | tuple[int, int, int]
    | tuple[float, float, float]
)

@dataclasses.dataclass(frozen=True)
class Color:
    """
    An RGB color triple.
    """
    r: int
    g: int
    b: int


    def __iter__(self) -> Iterator[int]:
        return iter((self.r, self.g, self.b))


    @staticmethod
    def parse(color: ColorLike | None) -> Color | None:
        """
        Accept and standardise RGB triples in various formats. See module-level
        documentation for a description of the possible formats.
        """
        if color is None:
            return None

        if isinstance(color, str):
            if color.startswith("#") and len(color) == 4:
                return Color(
                    r=17*int(color[1], base=16),
                    g=17*int(color[2], base=16),
                    b=17*int(color[3], base=16),
                )
            if color.startswith("#") and len(color) == 7:
                return Color(
                    r=int(color[1:3], base=16),
                    g=int(color[3:5], base=16),
                    b=int(color[5:7], base=16),
                )
            if color.lower() in NAMED_COLORS:
                return NAMED_COLORS[color.lower()]

        elif isinstance(color, (np.ndarray, tuple)):
            color_ = np.asarray(color)
            if color_.shape == (3,):
                if np.issubdtype(color_.dtype, np.floating):
                    rgb = (255*np.clip(color_, 0., 1.)).astype(int)
                    return Color(*rgb)
                if np.issubdtype(color_.dtype, np.integer):
                    rgb = np.clip(color_, 0, 255).astype(int)
                    return Color(*rgb)
        
        raise ValueError(f"invalid color {color!r}")


NAMED_COLORS = {
    "black":    Color(  0,   0,   0),
    "red":      Color(255,   0,   0),
    "green":    Color(  0, 255,   0),
    "blue":     Color(  0,   0, 255),
    "cyan":     Color(  0, 255, 255),
    "magenta":  Color(255,   0, 255),
    "yellow":   Color(255, 255,   0),
    "white":    Color(255, 255, 255),
}
