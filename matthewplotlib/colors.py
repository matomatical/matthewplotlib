from __future__ import annotations
from typing import Self, Iterator
import dataclasses

import numpy as np


type ColorLike = (
    str
    | np.ndarray # float[3] (0 to 1) or uint8[3] (0 to 255)
    | tuple[int, int, int]
    | tuple[float, float, float]
    | None
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
    def parse(color: ColorLike) -> Color | None:
        """
        Accept and standardise RGB triples in various formats.
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
