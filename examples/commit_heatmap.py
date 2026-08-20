"""
Commit heatmap: a year of work on the matthewplotlib repository.

Commits per day for the year ending 2026-08-20, counted with

    git log --date=short --pretty=%ad | sort | uniq -c

and frozen here, so that the picture stays put as the repository grows.

Quiet days are drawn too, in the palest colour of the scale, which is what
makes the year a solid block rather than a scattering of marks: a day with no
commits has a value like any other, and only a day the data says nothing about
is left blank.

By Claude Opus 5.
"""

import datetime

import numpy as np
import tyro

import matthewplotlib as mp


LAST = datetime.date(2026, 8, 20)   # the day the counts were taken
DAYS = 365                       # the year of days leading up to it


# Days with at least one commit. Every other day in the year had none.
COMMITS = {
    datetime.date(2025,  8, 29):  1, datetime.date(2025,  9,  3):  1,
    datetime.date(2025,  9, 12):  8, datetime.date(2025,  9, 13): 10,
    datetime.date(2025,  9, 14):  6, datetime.date(2025,  9, 15):  1,
    datetime.date(2025,  9, 16):  5, datetime.date(2025,  9, 18):  2,
    datetime.date(2025,  9, 20):  1, datetime.date(2025,  9, 27):  1,
    datetime.date(2025,  9, 28):  7, datetime.date(2025,  9, 29): 10,
    datetime.date(2025, 10,  2):  1, datetime.date(2025, 11, 26):  1,
    datetime.date(2025, 12, 17):  2, datetime.date(2025, 12, 18):  1,
    datetime.date(2026,  3,  6):  2, datetime.date(2026,  3, 10): 12,
    datetime.date(2026,  3, 11):  8, datetime.date(2026,  3, 13):  4,
    datetime.date(2026,  3, 14): 18, datetime.date(2026,  6, 23):  1,
    datetime.date(2026,  7, 14):  1, datetime.date(2026,  7, 25): 16,
    datetime.date(2026,  7, 26): 28, datetime.date(2026,  7, 27):  6,
    datetime.date(2026,  8, 14):  7, datetime.date(2026,  8, 16):  4,
    datetime.date(2026,  8, 17): 14, datetime.date(2026,  8, 18):  3,
    datetime.date(2026,  8, 19): 17, datetime.date(2026,  8, 20): 31,
}


# The five colours GitHub draws a contribution graph in: a grey for a day with
# nothing on it, then four greens. Interpolating between them gives a colormap
# that lands exactly on one colour per level, since the levels are evenly
# spaced over the range.
PALETTE = np.array([
    (235, 237, 240),
    (155, 233, 168),
    ( 64, 196,  99),
    ( 48, 161,  78),
    ( 33, 110,  57),
])


# The fewest commits reaching each level. Bucketing the counts this way keeps
# one 31-commit day from washing out a year of ordinary ones, which is what a
# linear scale over the raw counts would do.
LEVELS = (0, 1, 3, 7, 15)


def github(x):     # float[...] -> uint8[..., 3]
    """Colormap through GitHub's contribution colours."""
    stops = np.linspace(0.0, 1.0, len(PALETTE))
    channels = [np.interp(x, stops, PALETTE[:, c]) for c in range(3)]
    return np.stack(channels, axis=-1).astype(np.uint8)


def level(commits: int) -> int:
    """Which level a day's commit count reaches."""
    return sum(commits >= threshold for threshold in LEVELS) - 1


def main(width: int | None = None, save: str | None = None):
    """Commit heatmap of a year in the matthewplotlib repository.

    A year of square days is wider than a narrow terminal; pass a `width` to
    wrap it onto more than one band.
    """
    days = [LAST - datetime.timedelta(days=i) for i in reversed(range(DAYS))]
    levels = {day: level(COMMITS.get(day, 0)) for day in days}

    strip = mp.weeks(
        levels,
        vrange=(0, len(LEVELS) - 1),
        colormap=github,
        width=width,
    )
    total = sum(COMMITS.values())
    title = mp.text(f"matthewplotlib: {total} commits in the year to {LAST}")
    plot = title / strip

    print(plot)
    if save:
        plot.saveimg(save)


if __name__ == "__main__":
    tyro.cli(main)
