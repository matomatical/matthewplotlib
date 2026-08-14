"""
Specifying colors.

Wherever a plot takes a color, it accepts any of several convenient spellings,
and converts to a single internal representation. Wherever it takes the colors
of a whole series, it accepts either one of those or an array holding one per
point.

Types:

* `Color`: The internal representation, an RGB triple of bytes.
* `ColorLike`: Anything accepted in place of one---a named color, a hex string,
  or a triple of ints in 0 to 255 or floats in 0.0 to 1.0.
* `ColorSpec`: Anything accepted for the colors of a series---one `ColorLike`
  for all of it, or an array with a color per point.

Conversion:

* `parse_color`: Turn a `ColorLike` into a `Color`. See this function for the
  full list of accepted formats.
* `parse_colors`: Turn a `ColorSpec` into one `Color` for each of n points.
* `NAMED_COLORS`: The recognised color names.

For mapping data to colors, rather than naming one color, see
`matthewplotlib.colormaps`.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from numpy.typing import ArrayLike, NDArray


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


type ColorSpec = (
    None
    | ColorLike
    | ArrayLike # float[n, 3] (0 to 1) or uint8[n, 3] (0 to 255)
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

    elif isinstance(color, (np.ndarray, tuple, list)):
        color_ = np.asarray(color)
        if color_.shape == (3,):
            channels = _channel_bytes(color_)
            if channels is not None:
                return channels

    raise ValueError(f"invalid color {color!r}")


def parse_colors(
    spec: ColorSpec,
    n: int,
) -> NDArray: # uint8[n, 3]
    """
    One color for each of n points, from either a single `ColorLike` or an
    array holding a color per point.

    Channels are read the same way in both cases: floats as 0.0 to 1.0, ints as
    0 to 255. A spec of None means the default, white.
    """
    # an array of colors is the only spec with a second dimension; anything
    # else names one color, for all n of them
    if isinstance(spec, (np.ndarray, list, tuple)):
        colors = np.asarray(spec)
        if colors.ndim == 2:
            channels = _channel_bytes(colors)
            if channels is None:
                raise ValueError(f"invalid colors of type {colors.dtype}")
            if channels.shape != (n, 3):
                raise ValueError(
                    f"expected a color for each of {n} points, but got an "
                    f"array of shape {colors.shape}"
                )
            return channels

    color = parse_color(cast(ColorLike, spec))
    if color is None:
        return np.full((n, 3), 255, dtype=np.uint8)
    return np.full((n, 3), color, dtype=np.uint8)


def _channel_bytes(values: NDArray) -> NDArray | None:
    """
    Color channels as bytes, from floats in 0.0 to 1.0 or ints in 0 to 255.

    None if the values are neither, which is how both parsers above tell that
    what they are looking at is not a color at all.
    """
    if np.issubdtype(values.dtype, np.floating):
        return (255 * np.clip(values, 0., 1.)).astype(np.uint8)
    if np.issubdtype(values.dtype, np.integer):
        return np.clip(values, 0, 255).astype(np.uint8)
    return None


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


