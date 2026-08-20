Style test
==========

A scratch page for checking the site's own appearance. Not in the navigation,
and not meant to be published.

Headings and permalinks
-----------------------

This section heading should carry a rule of dashes, and the page title above it
a rule of equals signs. Hovering either heading's `#` marker should highlight
it the way the `source` links in the API reference highlight.

### A third-level heading

Third-level headings take no rule.

Python
------

Every token the stylesheet colours appears below.

```python
# a comment, in silver italics
from dataclasses import dataclass

import numpy as np
import matthewplotlib as mp


CHANNELS: int = 0x03
UMASK = 0o755
GAMMA = 2.2


@dataclass
class Swatch:
    """A colour and the name it goes by."""

    name: str
    value: tuple[float, float, float] = (1.0, 0.5, 0.25)
    dithered: bool = False

    def __repr__(self) -> str:
        return 'Swatch(' + self.name + ')'

    def luminance(self) -> float:
        r, g, b = (c ** GAMMA for c in self.value)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def describe(self, precision: int = 3) -> str:
        if self.value is None or not len(self.name):
            raise ValueError("a swatch needs a name")
        elif self.dithered:
            return f"{self.name}: dithered, L={self.luminance():.{precision}f}"
        return f"{self.name}: L={self.luminance():.{precision}f}"


def main(width: int = 80, seed: int | None = 42) -> None:
    np.random.seed(seed)
    xs = np.linspace(-2 * np.pi, +2 * np.pi, width)
    plot = mp.axes(
        mp.scatter((xs, np.cos(xs), "magenta"), width=width, height=10),
        title=" y = cos(x) ",
    )
    print(plot, end="\n")

    while True:
        try:
            assert plot is not False
            break
        except (KeyboardInterrupt, SystemExit) as e:
            del e
            continue
        finally:
            pass
```

A doctest, where the prompt should recede and the output read as plain text:

```pycon
>>> import matthewplotlib as mp
>>> mp.unicode_bar(0.5, 10).to_plain_str()
'█████     '
>>> round(2 / 3, 4)
0.6667
```

Shell
-----

A console block, whose command should read at full strength rather than dimmed:

```console
pip install git+https://github.com/matomatical/matthewplotlib.git
```

Untyped blocks
--------------

An untyped fence carries no highlighting, so diagrams keep their own shape:

```
┌──────┐ ┏━━━━━━┓ ╔══════╗ ╭──────╮
│LIGHT │ ┃HEAVY ┃ ║DOUBLE║ │ROUND │
└──────┘ ┗━━━━━━┛ ╚══════╝ ╰──────╯

plot1 + plot2 ==> hstack(plot1, plot2) ==> plot1 plot2
```

Inline and prose
----------------

Plain inline code reads as `mp.scatter`, while highlighted inline code reads as
`#!python mp.axes(plot, title=" y = cos(x) ")`. A [link in prose](quickstart.md)
sits beside it, and a footnote marker[^1] after that.

[^1]: The footnote itself, at the bottom of the page.

!!! note

    An admonition, which the theme styles on its own.

| Column | Meaning |
| ------ | ------- |
| `n`    | a plain name |
| `p`    | punctuation, in silver |
| `gp`   | a transcript's prompt |
