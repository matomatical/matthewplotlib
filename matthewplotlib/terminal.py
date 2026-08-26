"""
What can be known about the screen a plot is printed to.

This library builds strings, and a string does not know where it will be shown.
The little that can be found out about the terminal on the other end is found
out here, so that the plots and the animations can share one answer and one
account of how much it is worth.

* `terminal_size`: How large the attached terminal is, if there is one.

Nothing in this library changes *what* it writes based on what this module
reports. A measurement can settle a size the caller left open, and it can
decide whether a warning is warranted; it can never override a size the caller
gave, and there is no fallback for its absence, because a plot that quietly
came out a different shape in a pipe than on a screen would be worse than one
that says it could not tell.
"""

from __future__ import annotations

import os
import sys


# # #
# MEASUREMENT


def terminal_size() -> os.terminal_size | None:
    """
    The size of the attached terminal, or None if stdout is not one.

    Returns:

    * size : os.terminal_size | None.
        The number of `columns` and `lines` the terminal has, or None if stdout
        is anything else: a pipe, a file, a captured stream.

    `shutil.get_terminal_size` is the wrong tool for this. Its job is to paper
    over the absence of a terminal by returning a fallback size, and a fallback
    is exactly what must not be mistaken for a measurement. Asking the file
    descriptor directly tells the two apart.

    The size is read fresh on each call, since a terminal can be resized
    between one call and the next.
    """
    try:
        return os.get_terminal_size(sys.stdout.fileno())
    except (AttributeError, OSError, ValueError):
        return None
