from __future__ import annotations

import numpy as np

from numpy.typing import NDArray


# # # 
# COLOR TYPES


type Color = NDArray # uint8[3]


type ColorLike = (
    str
    | NDArray # float[3] (0 to 1) or uint8[3] (0 to 255)
    | tuple[int, int, int]
    | tuple[float, float, float]
    | Color
)


# # # 
# COLOR HANDLING


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


NAMED_COLORS: dict[str, Color] = {
    "black":    np.array((  0,   0,   0), dtype=np.uint8),
    "red":      np.array((255,   0,   0), dtype=np.uint8),
    "green":    np.array((  0, 255,   0), dtype=np.uint8),
    "blue":     np.array((  0,   0, 255), dtype=np.uint8),
    "cyan":     np.array((  0, 255, 255), dtype=np.uint8),
    "magenta":  np.array((255,   0, 255), dtype=np.uint8),
    "yellow":   np.array((255, 255,   0), dtype=np.uint8),
    "white":    np.array((255, 255, 255), dtype=np.uint8),
}


