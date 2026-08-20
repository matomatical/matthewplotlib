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
* `parse_colors`: Turn a `ColorSpec`, scalar array, RGB array, or colormapped
  array into RGB bytes with requested leading dimensions.
* `NAMED_COLORS`: The recognised color names.

Predefined functions for mapping data to colors live in
`matthewplotlib.colormaps` and can be passed to `parse_colors`.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from numpy.typing import ArrayLike, NDArray

from matthewplotlib.colormaps import ColorMap


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
"""
Anything accepted in place of a single color.

A named color or a hex string, an array or tuple of three integers from 0 to
255, or one of three floats from 0.0 to 1.0.
[`parse_color`][matthewplotlib.colors.parse_color] standardises each of them,
and documents the spellings it recognises.
"""


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
    n: int | None = None,
    *,
    shape: tuple[int | str | None, ...] | None = None,
    colormap: ColorMap | None = None,
) -> NDArray: # uint8[..., 3]
    """
    Parse colour specifications or array data into an array of RGB bytes.

    Exactly one output-shape specification is required:

    * `n` requests one colour for each of n points, preserving the original
      `parse_colors(spec, n)` interface.
    * `shape` describes any number of leading dimensions. An integer requires
      that exact size; `None` accepts any size and displays as `*` in errors;
      and a string such as `"h"` or `"w"` also accepts any size but retains
      that useful name in errors.

    Without a colormap, an array matching the requested leading shape is read
    as scalar data and repeated over the three RGB channels. An array with an
    additional final axis of length three is read directly as RGB. A single
    `ColorLike`, or `None` for white, can instead be broadcast over a fully
    specified shape. In the `n` form, a three-element array remains one RGB
    colour rather than becoming three greyscale point colours.

    If a colormap is supplied, the input array is passed to it without any
    shape interpretation. Its output must have the requested leading shape and
    a final RGB axis of length three. This permits a custom colormap to consume
    arbitrary array data, including feature vectors, provided it produces the
    requested colour array.

    In every form, float channels are read in the range 0.0 to 1.0 and integer
    channels in the range 0 to 255. Values outside those ranges are clipped.
    """
    if (n is None) == (shape is None):
        raise ValueError("provide exactly one of n or shape")
    if shape is not None and any(
        dimension is not None and not isinstance(dimension, (int, str))
        for dimension in shape
    ):
        raise ValueError("shape entries must be integers, strings, or None")
    expected = (
        (n,)
        if n is not None
        else cast(tuple[int | str | None, ...], shape)
    )
    rgb_shape = (*expected, 3)

    if colormap is not None:
        mapped = np.asarray(colormap(np.asarray(spec)))
        if not _shape_matches(mapped.shape, rgb_shape):
            raise ValueError(
                "expected colormap output of shape "
                f"{_format_shape(rgb_shape)}, not {mapped.shape}"
            )
        channels = _channel_bytes(mapped)
        if channels is None:
            raise ValueError(f"invalid colors of type {mapped.dtype}")
        return channels

    values = np.asarray(spec) if spec is not None else None

    # An explicit array with a final RGB axis is already a colour array.
    if values is not None and _shape_matches(values.shape, rgb_shape):
        channels = _channel_bytes(values)
        if channels is None:
            raise ValueError(f"invalid colors of type {values.dtype}")
        return channels

    # Preserve the point-colour shorthand: a single ColorLike (or None) is
    # broadcast over a concrete target shape. In particular, a three-element
    # vector remains one RGB colour rather than three greyscale point colours.
    if spec is None or isinstance(spec, str) or (
        values is not None and values.shape == (3,)
    ):
        color = parse_color(cast(ColorLike, spec))
        if any(not isinstance(size, int) for size in expected):
            raise ValueError(
                "cannot broadcast one color over an unspecified shape "
                f"{_format_shape(expected)}"
            )
        concrete = cast(tuple[int, ...], expected)
        if color is None:
            color = np.full(3, 255, dtype=np.uint8)
        return np.broadcast_to(color, (*concrete, 3)).copy()

    # A scalar array becomes greyscale by repeating its values over RGB.
    if values is not None and _shape_matches(values.shape, expected):
        channels = _channel_bytes(values)
        if channels is None:
            raise ValueError(f"invalid colors of type {values.dtype}")
        return np.repeat(channels[..., np.newaxis], 3, axis=-1)

    if n is not None:
        raise ValueError(
            f"expected a color for each of {n} points, but got an "
            f"array of shape {None if values is None else values.shape}"
        )
    raise ValueError(
        f"expected scalar colors of shape {_format_shape(expected)} or RGB "
        f"colors of shape {_format_shape(rgb_shape)}, not "
        f"{None if values is None else values.shape}"
    )


def _shape_matches(
    actual: tuple[int, ...],
    expected: tuple[int | str | None, ...],
) -> bool:
    """Whether an array shape matches a pattern containing wildcards."""
    return len(actual) == len(expected) and all(
        not isinstance(wanted, int) or got == wanted
        for got, wanted in zip(actual, expected)
    )


def _format_shape(shape: tuple[int | str | None, ...]) -> str:
    """A compact array-shape pattern for error messages."""
    dimensions = ("*" if size is None else str(size) for size in shape)
    return "[" + ",".join(dimensions) + "]"


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
