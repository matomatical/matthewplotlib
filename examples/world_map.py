"""
Flight routes on a world map, and why none of them look straight.

The shortest path between two places on a sphere is an arc of the great circle
through them, and a map projection has to distort that into something else. So
each route here is drawn twice: the great circle in colour, and in grey the
straight line between the same two cities *on the map*. The gap between them is
the projection talking. On an equirectangular map a flight from London to Tokyo
bows hundreds of miles north of the line a ruler would draw between them, and on
Mercator the bow is enormous; on the interrupted sinusoidal it goes the other
way. Pass `--projection` to see the same ten routes on each.

Where a projection needs the sphere is in the other direction from the globe in
`examples/globe.py`. That one inverts the projection at every pixel, because it
is filling an area. This one runs it forwards over a list of coordinates,
because it is drawing lines, and the whole map is 236 hand-listed vertices --
Natural Earth's coarsest coastlines, cut down with Douglas-Peucker until each
landmass was a shape you could type out, then rounded to whole degrees.

By Claude Opus 5.
"""

from typing import Callable, Literal, NamedTuple

import tyro
import numpy as np

import matthewplotlib as mp


EARTH_RADIUS_KM = 6371.0


# The coastlines, as whole degrees of longitude and latitude, each ring closed
# by repeating its first vertex. Simplified so hard that Java is a stroke and
# Cuba a triangle, which is the resolution a terminal-sized map can draw
# anyway. The Caspian Sea is in here because it is a hole in the land: at this
# scale a coastline is a coastline, whichever side the water is on.
COASTLINES: dict[str, list[tuple[int, int]]] = {
    "Afro-Eurasia": [
        (107, 77), (131, 71), (179, 69), (179, 63), (164, 60), (157, 51),
        (156, 57), (164, 63), (135, 55), (141, 52), (128, 40), (129, 35),
        (118, 39), (122, 37), (122, 28), (106, 20), (109, 13), (105, 9),
        (100, 13), (104, 1), (91, 23), (78, 8), (66, 25), (48, 30),
        (60, 22), (43, 13), (32, 30), (43, 12), (51, 11), (39, -5),
        (41, -15), (32, -29), (18, -34), (9, 4), (-9, 5), (-17, 12),
        (-17, 22), (-6, 36), (34, 31), (36, 37), (26, 39), (42, 42),
        (39, 47), (31, 47), (22, 36), (13, 46), (16, 38), (9, 44), (-9, 37),
        (-9, 43), (-1, 44), (-5, 49), (9, 57), (20, 54), (29, 60), (21, 61),
        (22, 66), (13, 55), (5, 62), (25, 71), (40, 68), (33, 67), (37, 64),
        (43, 69), (69, 68), (73, 73), (72, 66), (75, 73), (107, 77),
    ],
    "the Americas": [
        (-91, 69), (-81, 68), (-93, 62), (-92, 57), (-80, 51), (-78, 62),
        (-74, 62), (-56, 52), (-71, 47), (-60, 46), (-76, 39), (-80, 25),
        (-84, 30), (-97, 28), (-96, 19), (-87, 22), (-89, 16), (-81, 9),
        (-62, 11), (-35, -7), (-41, -22), (-58, -34), (-71, -54),
        (-76, -47), (-70, -20), (-81, -6), (-78, 8), (-104, 18), (-114, 32),
        (-112, 25), (-124, 40), (-123, 49), (-134, 58), (-151, 61),
        (-165, 54), (-157, 59), (-168, 66), (-157, 71), (-91, 69),
    ],
    "Antarctica": [
        (-59, -64), (-66, -68), (-61, -74), (-78, -79), (-58, -83),
        (-29, -80), (-35, -78), (-7, -71), (55, -66), (69, -68), (70, -72),
        (88, -66), (135, -65), (171, -72), (160, -81), (180, -85),
        (180, -90), (-180, -90), (-179, -84), (-143, -85), (-154, -84),
        (-157, -81), (-146, -80), (-158, -77), (-75, -74), (-59, -64),
    ],
    "Greenland": [
        (-27, 84), (-12, 81), (-20, 80), (-19, 74), (-26, 70), (-22, 70),
        (-43, 60), (-59, 76), (-73, 78), (-27, 84),
    ],
    "Australia": [
        (144, -14), (153, -26), (150, -37), (131, -31), (115, -34),
        (114, -22), (132, -11), (140, -18), (144, -14),
    ],
    "Ellesmere Island": [
        (-68, 83), (-62, 82), (-81, 76), (-89, 76), (-82, 80), (-92, 82),
        (-68, 83),
    ],
    "Baffin Island": [
        (-87, 73), (-62, 67), (-68, 66), (-66, 62), (-78, 64), (-73, 68),
        (-87, 73),
    ],
    "Victoria Island": [
        (-114, 73), (-101, 70), (-116, 69), (-112, 70), (-119, 72),
        (-114, 73),
    ],
    "Novaya Zemlya": [
        (58, 71), (52, 71), (56, 75), (69, 77), (58, 74), (58, 71),
    ],
    "New Guinea": [
        (134, -1), (151, -11), (138, -8), (134, -1),
    ],
    "Japan": [
        (141, 37), (130, 31), (140, 41), (141, 37),
    ],
    "Borneo": [
        (118, 2), (116, -4), (109, 0), (117, 7), (118, 2),
    ],
    "Sumatra": [
        (106, -6), (95, 5), (104, 0), (106, -6),
    ],
    "Java": [
        (109, -7), (116, -8), (109, -7),
    ],
    "Madagascar": [
        (50, -14), (47, -25), (44, -25), (44, -16), (50, -14),
    ],
    "Great Britain": [
        (-3, 59), (1, 51), (-5, 50), (-3, 54), (-6, 57), (-3, 59),
    ],
    "Iceland": [
        (-15, 66), (-14, 65), (-19, 63), (-24, 66), (-15, 66),
    ],
    "Cuba": [
        (-80, 23), (-74, 20), (-85, 22), (-80, 23),
    ],
    "New Zealand, South Island": [
        (173, -41), (173, -44), (167, -46), (173, -41),
    ],
    "New Zealand, North Island": [
        (175, -36), (179, -38), (175, -42), (173, -35), (175, -36),
    ],
    "the Caspian Sea": [
        (49, 41), (49, 38), (54, 37), (55, 41), (50, 45), (53, 47),
        (47, 45), (49, 41),
    ],
}

# Airports, to the nearest degree, which is as much precision as a map this
# coarse can spend.
CITIES: dict[str, tuple[int, int]] = {
    "LHR": (0, 51),         # London
    "JFK": (-74, 41),       # New York
    "LAX": (-118, 34),      # Los Angeles
    "MEX": (-99, 19),       # Mexico City
    "GRU": (-47, -24),      # Sao Paulo
    "EZE": (-58, -35),      # Buenos Aires
    "SCL": (-71, -33),      # Santiago
    "LOS": (3, 6),          # Lagos
    "JNB": (28, -26),       # Johannesburg
    "CAI": (31, 30),        # Cairo
    "SVO": (38, 56),        # Moscow
    "DXB": (55, 25),        # Dubai
    "DEL": (77, 29),        # Delhi
    "SIN": (104, 1),        # Singapore
    "PEK": (116, 40),       # Beijing
    "HND": (140, 36),       # Tokyo
    "SYD": (151, -34),      # Sydney
    "AKL": (175, -37),      # Auckland
}

# Long-haul pairs, chosen for the variety of ways a great circle can refuse to
# look like a straight line: over a pole, across the antimeridian, or deep into
# the southern ocean.
ROUTES: tuple[tuple[str, str], ...] = (
    ("LHR", "HND"),         # over the arctic
    ("JFK", "HND"),         # over the arctic
    ("LHR", "LAX"),
    ("LHR", "JFK"),
    ("MEX", "PEK"),         # across the antimeridian, over the arctic
    ("LAX", "SYD"),         # across the antimeridian
    ("SCL", "SYD"),         # deep south
    ("DXB", "SYD"),
    ("GRU", "JNB"),
    ("CAI", "SIN"),
)

GRATICULE = (48, 52, 62)
LAND = (108, 112, 116)
CHORD = (96, 132, 190)
CITY = (255, 240, 180)

# The arcs are coloured by length, over the bright end of `plasma` only: its
# dark end is a purple close enough to the chords' grey to be confused with
# them, which is the one distinction the picture cannot afford to lose.
SHORTEST_KM, LONGEST_KM = 3000.0, 18000.0
PALETTE_FROM, PALETTE_TO = 0.35, 0.95


class projection(NamedTuple):
    """A map from the sphere to the plane, and the rectangle it lands in.

    Every projection here takes radians and returns plane coordinates in its
    own units, which is why it has to carry its extents: they are what tell a
    plot how tall to be for a given width, and they differ by a factor of two
    between, say, mercator and equirectangular.
    """
    project: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
    xrange: tuple[float, float]
    yrange: tuple[float, float]
    describe: str

    def height_for(self, width: int) -> int:
        """The character rows that show this projection undistorted.

        A cell is twice as tall as it is wide, and a plot of `width` by `height`
        cells is a rectangle `width` by `2 * height` in units of cell width. So
        matching the projection's own proportions means
        `2 * height / width = y extent / x extent`.
        """
        x = self.xrange[1] - self.xrange[0]
        y = self.yrange[1] - self.yrange[0]
        return max(1, round(width * y / (2 * x)))


SQRT2 = np.sqrt(2)
MERCATOR_CUT = np.radians(85)   # the poles are infinitely far up and down


def equirectangular(lon, lat):
    return lon, lat


def mercator(lon, lat):
    return lon, np.log(np.tan(np.pi/4 + np.clip(lat, -MERCATOR_CUT, MERCATOR_CUT)/2))


def sinusoidal(lon, lat):
    return lon * np.cos(lat), lat


def mollweide(lon, lat):
    angle = _mollweide_angle(lat)
    return 2*SQRT2/np.pi * lon * np.cos(angle), SQRT2 * np.sin(angle)


def hammer(lon, lat):
    denominator = np.sqrt(1 + np.cos(lat)*np.cos(lon/2))
    return (
        2*SQRT2 * np.cos(lat) * np.sin(lon/2) / denominator,
        SQRT2 * np.sin(lat) / denominator,
    )


def _mollweide_angle(lat: np.ndarray) -> np.ndarray:
    """Solve `2t + sin 2t = pi sin(lat)` for the auxiliary angle `t`.

    Newton from `t = lat` is the usual advice and it is a trap: at the poles the
    derivative `2 + 2 cos 2t` is zero, so the iteration divides by zero there
    and crawls near it, and ten steps still leave the pole half a braille dot
    out of place. Seeding from the pole's own expansion fixes it. Writing
    `t = pi/2 - u`, the equation becomes `2u - sin 2u = pi (1 - sin lat)`, whose
    left side is `(4/3) u^3` for small `u`, so the cube root below starts within
    a fraction of a dot and three steps finish the job everywhere.
    """
    u = np.cbrt(3*np.pi*(1 - np.abs(np.sin(lat)))/4)
    t = np.sign(lat) * (np.pi/2 - u)
    for _ in range(3):
        residual = 2*t + np.sin(2*t) - np.pi*np.sin(lat)
        slope = 2 + 2*np.cos(2*t)
        t = t - np.where(slope < 1e-9, 0.0, residual/np.where(slope < 1e-9, 1.0, slope))
    return t


PROJECTIONS: dict[str, projection] = {
    "equirectangular": projection(
        project=equirectangular,
        xrange=(-np.pi, np.pi),
        yrange=(-np.pi/2, np.pi/2),
        describe="every degree the same width and height",
    ),
    "mercator": projection(
        project=mercator,
        xrange=(-np.pi, np.pi),
        yrange=(-mercator(0.0, MERCATOR_CUT)[1], mercator(0.0, MERCATOR_CUT)[1]),
        describe="angles preserved, cut at 85 degrees, Greenland enormous",
    ),
    "mollweide": projection(
        project=mollweide,
        xrange=(-2*SQRT2, 2*SQRT2),
        yrange=(-SQRT2, SQRT2),
        describe="equal area, an ellipse, meridians curved",
    ),
    "hammer": projection(
        project=hammer,
        xrange=(-2*SQRT2, 2*SQRT2),
        yrange=(-SQRT2, SQRT2),
        describe="equal area, an ellipse, parallels curved too",
    ),
    "sinusoidal": projection(
        project=sinusoidal,
        xrange=(-np.pi, np.pi),
        yrange=(-np.pi/2, np.pi/2),
        describe="equal area, pinched to a point at each pole",
    ),
}


def main(
    projection: Literal[
        "equirectangular", "mercator", "mollweide", "hammer", "sinusoidal",
    ] = "equirectangular",
    width: int = 76,
    routes: bool = True,
    chords: bool = True,
    save: str | None = None,
):
    """Great-circle flight routes drawn on a hand-listed world map.

    Pass `--projection mercator` and watch the northern routes bow off the top
    of the map, or `--no-chords` for the great circles without the straight
    lines to compare them against.
    """
    chosen = PROJECTIONS[projection]
    height = chosen.height_for(width)

    layers = [
        graticule(chosen, width, height),
        coastlines(chosen, width, height),
    ]
    if chords:
        layers.append(straight_lines(chosen, width, height))
    if routes:
        layers.append(great_circles(chosen, width, height))
    layers.append(airports(chosen, width, height))

    plot = (
        mp.border(mp.dstack2(*layers), title=f" {projection} ")
        / legend(chosen, width)
        / distance_table(width)
    )
    print(plot)

    if save:
        plot.saveimg(save, bgcolor="black")


def graticule(chosen: projection, width: int, height: int) -> mp.plot:
    """Meridians and parallels every thirty degrees.

    Straight lines on an equirectangular map and curves on every other one,
    which is the projection made visible: the coastlines alone cannot say
    whether it was the map or the coast that bent.
    """
    lines = []
    for lon in range(-180, 181, 30):
        latitudes = np.linspace(-90, 90, 64)
        lines.append(np.stack([np.full_like(latitudes, lon), latitudes], 1))
    for lat in range(-60, 61, 30):
        longitudes = np.linspace(-180, 180, 128)
        lines.append(np.stack([longitudes, np.full_like(longitudes, lat)], 1))
    return stroke(chosen, lines, width, height, color=GRATICULE)


def coastlines(chosen: projection, width: int, height: int) -> mp.plot:
    """Every ring, as one series broken by the gaps between them."""
    rings = [np.array(ring, dtype=float) for ring in COASTLINES.values()]
    return stroke(chosen, rings, width, height, color=LAND)


def great_circles(chosen: projection, width: int, height: int) -> mp.plot:
    """Each route as the arc a plane would actually fly, coloured by length."""
    arcs, colors = [], []
    lengths = np.array([distance_km(*pair) for pair in ROUTES])
    shades = mp.plasma(palette_position(lengths))
    for pair, shade in zip(ROUTES, shades):
        arcs.append(arc(*pair))
        colors.append(shade)
    return stroke(chosen, arcs, width, height, colors=colors)


def palette_position(km: np.ndarray) -> np.ndarray:
    """Where a distance falls along the part of the colormap in use."""
    along = np.clip((km - SHORTEST_KM) / (LONGEST_KM - SHORTEST_KM), 0, 1)
    return PALETTE_FROM + (PALETTE_TO - PALETTE_FROM)*along


def straight_lines(chosen: projection, width: int, height: int) -> mp.plot:
    """Each route as the map makes it look: the segment joining its ends."""
    chords = [
        np.array([CITIES[start], CITIES[end]], dtype=float)
        for start, end in ROUTES
    ]
    return stroke(chosen, chords, width, height, color=CHORD)


def airports(chosen: projection, width: int, height: int) -> mp.plot:
    """A dot at each city that a route touches."""
    used = sorted({city for pair in ROUTES for city in pair})
    points = np.array([CITIES[name] for name in used], dtype=float)
    x, y = chosen.project(*np.radians(points).T)
    return mp.scatter(
        (x, y, CITY),
        xrange=chosen.xrange,
        yrange=chosen.yrange,
        width=width,
        height=height,
    )


def stroke(
    chosen: projection,
    paths: list[np.ndarray],        # each float[n, 2], degrees
    width: int,
    height: int,
    color: tuple[int, int, int] | None = None,
    colors: list | None = None,
) -> mp.plot:
    """Project a batch of paths and draw them, one series each.

    A series per path, rather than one series with gaps between them, because
    each path may want its own colour -- and because the seam splitting below
    can turn one path into two.
    """
    series = []
    for index, path in enumerate(paths):
        x, y = chosen.project(*np.radians(path).T)
        x, y = split_at_seam(x, y, chosen)
        shade = color if colors is None else colors[index]
        series.append((x, y, shade))
    return mp.line(
        *series,
        xrange=chosen.xrange,
        yrange=chosen.yrange,
        width=width,
        height=height,
    )


def split_at_seam(
    x: np.ndarray,          # float[n]
    y: np.ndarray,          # float[n]
    chosen: projection,
) -> tuple[np.ndarray, np.ndarray]:
    """Break a path wherever it leaves one edge of the map and enters the other.

    A route from Los Angeles to Sydney crosses the antimeridian, and the
    projection sends the two halves to opposite edges. Joining them draws a line
    straight back across the whole map. Inserting a gap where consecutive points
    jump more than half the map's width leaves the two halves as they should be,
    each running off its own edge, since `line` treats a non-finite coordinate
    as a break in the stroke.
    """
    reach = (chosen.xrange[1] - chosen.xrange[0]) / 2
    jumped = np.nonzero(np.abs(np.diff(x)) > reach)[0]
    return (
        np.insert(x, jumped + 1, np.nan),
        np.insert(y, jumped + 1, np.nan),
    )


def arc(start: str, end: str, samples: int = 96) -> np.ndarray:  # float[n, 2]
    """The great-circle path between two cities, in degrees.

    Interpolating the two positions as vectors in space and normalising --
    slerp -- gives the arc directly: the shortest path on a sphere is the one
    that stays in the plane through both points and the centre.
    """
    first, second = unit_vector(CITIES[start]), unit_vector(CITIES[end])
    angle = np.arccos(np.clip(first @ second, -1, 1))
    step = np.linspace(0, 1, samples)[:, None]
    if angle < 1e-9:
        points = np.repeat(first[None], samples, axis=0)
    else:
        points = (
            np.sin((1 - step)*angle) * first + np.sin(step*angle) * second
        ) / np.sin(angle)
    return np.stack([
        np.degrees(np.arctan2(points[:, 1], points[:, 0])),
        np.degrees(np.arcsin(np.clip(points[:, 2], -1, 1))),
    ], axis=1)


def unit_vector(lonlat: tuple[int, int]) -> np.ndarray:     # float[3]
    """A position on the sphere as a point in the space around it."""
    lon, lat = np.radians(lonlat)
    return np.array([
        np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat),
    ])


def distance_km(start: str, end: str) -> float:
    """How far apart two cities are, along the surface."""
    first, second = unit_vector(CITIES[start]), unit_vector(CITIES[end])
    return EARTH_RADIUS_KM * float(np.arccos(np.clip(first @ second, -1, 1)))


def legend(chosen: projection, width: int) -> mp.plot:
    """The colour scale, what the two kinds of line mean, and the projection.

    Three short lines rather than one long one, so that a narrow map does not
    end up with a legend wider than itself: `center` pads a plot out to a width
    but will not crop one down to it.
    """
    scale = np.linspace(PALETTE_FROM, PALETTE_TO, 16)[None, :]
    lines = [
        mp.text("great circle  3000km ")
        + mp.image(np.repeat(scale, 2, axis=0), colormap=mp.plasma)
        + mp.text(" 18000km"),
        mp.image(np.full((2, 3), 0.5), colormap=flat(CHORD))
        + mp.text(" the same pair of cities joined straight"),
        mp.text(chosen.describe),
    ]
    return mp.vstack(*[mp.center(line, width=width + 2) for line in lines])


def distance_table(width: int) -> mp.plot:
    """The routes, longest first, with what the arc costs and what it saves."""
    ordered = sorted(ROUTES, key=lambda pair: -distance_km(*pair))[:5]
    return mp.center(
        mp.table({
            "route": [f"{start}-{end}" for start, end in ordered],
            "km": [round(distance_km(*pair)) for pair in ordered],
            "peak lat": [extreme_latitude(*pair) for pair in ordered],
        }),
        width=width + 2,
    )


def extreme_latitude(start: str, end: str) -> float:
    """The latitude furthest from the equator that the great circle reaches.

    The number the picture is about, and the answer to why a flight from London
    to Tokyo passes over the arctic: the shortest path leaves the latitudes of
    both its ends behind, by tens of degrees, and no straight line drawn on the
    map between them ever would.
    """
    path = arc(start, end)
    return round(float(path[np.argmax(np.abs(path[:, 1])), 1]), 1)


def flat(color: tuple[int, int, int]):
    """A colormap of one colour, for showing a swatch of it."""
    return lambda values: np.full((*values.shape, 3), color, dtype=np.uint8)


if __name__ == "__main__":
    tyro.cli(main)
