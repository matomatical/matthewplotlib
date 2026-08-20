"""
A spinning Earth, drawn by asking every pixel what it can see.

The naive way to draw a globe is to take points on a sphere and project them
onto the screen. That way lies grief: the projection sends both hemispheres into
the same disc, so the far side lands on top of the near side and has to be
sorted or culled before anything looks right.

Run the projection backwards instead and the problem disappears. Each *pixel*
of the plot is a position in the disc, and inverting the orthographic projection
there yields the one point of the *near* hemisphere seen at that position. The
far side is never computed, so it cannot bleed through, and the sphere needs no
depth test, no culling and no sorting -- just a per-pixel `arcsin`.

Everything else follows from what that inversion hands back:

* a latitude and longitude, which index the land mask and the climate ramp;
* the surface normal, which is `(x, y, sqrt(1 - x^2 - y^2))` and so is free,
  and which is all that day, night, the soft terminator and the lit limb need.

The Earth's colours are not a texture. They are a one-dimensional ramp from
rainforest to ice, read at a latitude that has been roughened by noise, which is
enough to put deserts near the tropics and taiga near the arctic without anyone
saying where the Sahara is. The clouds are the same noise, drifting east faster
than the planet turns.

By Claude Opus 5.
"""

from typing import NamedTuple

import tyro
import numpy as np

import matthewplotlib as mp


# The land, at five degrees to the character: 72 columns of longitude from
# 180W, 36 rows of latitude from the north pole. Coarsened from Natural Earth's
# 1:110m land polygons by asking which cell centres fall inside them, which is
# why the islands that survive are the ones that happen to cover a centre.
LAND = """\
........................................................................
...................#####...#####........................................
............#.....##..##########.......##...............................
...........####.#.###....#######..............#...#.##########..#.......
#..###############.#.##..####..........#####..##########################
...###############.##.#...#..........##################################.
....##...########....###...........#..#.########################...##...
..........###############.........##.############################..#....
............###########............##############################.......
...........###########............###.#.##..##.###############..#.......
...........##########.............##....######.#############.#..........
.............#######..............#####.#..#################............
.............####..#.............#############.#############............
...............#.................##########.####..#########.............
................##...............##############....##..##...............
.................................############......#....##..............
....................####.........#############..............#...........
....................######............#######...........#.##............
....................#######...........######............#.#....#........
....................#########..........#####.............##.....#.......
.....................#######...........#####..................#.#.......
......................######..........######.#..............#####.......
......................######...........####..#.............#######......
......................####.............####................########.....
......................####..............##.................##..###......
.....................####.......................................##......
.....................##..........................................#....#.
.....................##.................................................
.....................#..................................................
........................................................................
........................................................................
......................#.......................##....#############.......
.......................#.........#####################################..
....#################.........#######################################...
.....###################.############################################...
########################################################################
"""

# Where the colour comes from. A climate is read off a single ramp indexed by
# distance from the equator in degrees, so the whole palette is eight stops.
CLIMATE_LATITUDE = np.array([0, 12, 24, 34, 45, 58, 68, 78])
CLIMATE_COLOR = np.array([
    ( 38,  96,  52),        # equatorial forest
    (124, 136,  72),        # savanna
    (204, 179, 134),        # desert
    (139, 150,  86),        # dry grassland
    ( 71, 104,  61),        # temperate forest
    ( 49,  79,  59),        # boreal forest
    (146, 146, 129),        # tundra
    (244, 247, 252),        # permanent ice
])
DEEP_SEA = np.array([10, 28, 72])
SHALLOW_SEA = np.array([58, 122, 162])
CLOUD = np.array([250, 252, 255])
NIGHT_LIGHTS = np.array([255, 188, 108])
ATMOSPHERE = np.array([92, 142, 216])
SPACE = np.array([6, 7, 16])

# The sun sits still in front of the camera while the planet turns beneath it,
# so the terminator stays put on screen and the continents rotate through it.
# Its angle from the line of sight is what sets how much night is on show: this
# one is 68 degrees off, putting about a third of the disc in darkness, where a
# sun directly behind the camera would light very nearly all of it.
SUN = np.array([-0.86, 0.34, 0.38])
SUN /= np.linalg.norm(SUN)

# Everything random -- the noise fields and the stars -- is drawn from this, so
# the animation is the same one every time it runs.
SEED = 20260820


def main(
    num_frames: int = 72,
    fps: int = 20,
    width: int = 48,
    tilt_degrees: float = 20.0,
    spins: float = 1.0,
    clouds: bool = True,
    stars: bool = True,
    loop: bool = True,
    save: str | None = None,
):
    """The Earth, turning.

    The animation is periodic in `num_frames` for a whole number of `spins`, so
    it loops seamlessly. Pass `--no-clouds` for the surface alone, and
    `--tilt-degrees 90` to look down on the north pole.
    """
    frames = spin(
        num_frames=num_frames,
        width=width,
        tilt=np.radians(tilt_degrees),
        spins=spins,
        clouds=clouds,
        stars=stars,
    )
    animation = mp.animation(frames, fps=fps)
    animation.play(loop=loop)

    if save:
        # Each pixel of the animation is drawn as an eight by eight block of
        # the pixel font, so keeping every eighth pixel recovers the true
        # resolution of the frames.
        animation.savegif(save, downscale=8)


def spin(
    num_frames: int,
    width: int,
    tilt: float,
    spins: float,
    clouds: bool,
    stars: bool,
) -> np.ndarray:        # uint8[frames, width, width, rgb]
    """Every frame of the turning Earth.

    The globe is a disc, so the frames are square: `width` pixels across in
    `width` pixels down, which is `width // 2` rows of half-block characters.
    """
    rng = np.random.default_rng(SEED)
    relief = plane_waves(rng, count=28, wavenumber=9.0)
    weather = plane_waves(rng, count=32, wavenumber=14.0, falloff=0.5)
    # Cities are the one thing here that wants a short wavelength: the field
    # they are gated on has to break into specks the size of a conurbation,
    # where the climate's may cover a continent.
    cities = plane_waves(rng, count=36, wavenumber=30.0)
    sky = starfield(rng, width) if stars else None
    land = land_field()

    # The disc has radius one and the frame reaches a little past it, leaving
    # room for the atmosphere to glow outside the planet's edge.
    reach = 1.14
    axis = np.linspace(-reach, reach, width)
    x, y = np.meshgrid(axis, -axis)
    radius = np.hypot(x, y)
    on_globe = radius <= 1.0

    # The surface normal, in the camera's own frame: pointing out of the screen
    # at the middle of the disc and lying in it at the edge. This is the whole
    # of the lighting model's input, and it is the same in every frame.
    toward_viewer = np.sqrt(np.clip(1 - radius**2, 0, None))
    daylight = smoothstep(-0.09, 0.24, x*SUN[0] + y*SUN[1] + toward_viewer*SUN[2])

    frames = []
    for frame in range(num_frames):
        turn = 2*np.pi * spins * frame / num_frames
        lon, lat = unproject(x, y, radius, lon0=-turn, lat0=tilt)

        surface = terrain(lon, lat, land=land, relief=relief)
        if clouds:
            # The deck slides round the planet as well as with it, so weather
            # crosses the face at twice the speed the ground does. The drift
            # has to be a whole number of turns per loop or the animation
            # jumps when it repeats, and one turn is the slowest that is.
            cover = noise(weather, lon - turn, lat)
            alpha = 0.92 * smoothstep(0.3, 1.4, cover)
            surface = blend(surface, CLOUD, alpha[..., None])

        frames.append(illuminate(
            surface=surface,
            daylight=daylight,
            toward_viewer=toward_viewer,
            radius=radius,
            on_globe=on_globe,
            land=sample(land, lon, lat),
            lights=noise(cities, lon, lat),
            sky=sky,
        ))
    return np.stack(frames)


def illuminate(
    surface: np.ndarray,        # float[h, w, rgb], the unlit colours
    daylight: np.ndarray,       # float[h, w], one in full sun, zero at night
    toward_viewer: np.ndarray,  # float[h, w], the normal's z component
    radius: np.ndarray,         # float[h, w], distance from the disc's centre
    on_globe: np.ndarray,       # bool[h, w]
    land: np.ndarray,           # float[h, w]
    lights: np.ndarray,         # float[h, w], the field the cities follow
    sky: np.ndarray | None,     # float[h, w], star brightness, or none
) -> np.ndarray:                # uint8[h, w, rgb]
    """Light the surface, wrap it in atmosphere and set it against space."""
    # Day fades into a dim blue night rather than to black, so the continents
    # stay faintly readable on the far side of the terminator.
    lit = surface * (0.06 + 0.94*daylight[..., None])

    # Cities, on the land that is in darkness. The noise field standing in for
    # population is the one the terrain already used, which puts the lights in
    # the same places every time round.
    towns = smoothstep(0.8, 1.5, lights) * land * (1 - daylight)
    lit = lit + NIGHT_LIGHTS * (0.85*towns)[..., None]

    # Air scatters light towards the edge of the disc, where the line of sight
    # runs a long way through it, and only on the side the sun is on.
    limb = (1 - toward_viewer)**2.4
    lit = lit + ATMOSPHERE * (0.55 * limb * daylight)[..., None]

    # ...and past the edge, where there is no longer any ground to see, that
    # scattering is all there is: a halo, fading fast.
    halo = np.exp(-(np.clip(radius - 1, 0, None) / 0.045)) * 0.45
    space = SPACE + ATMOSPHERE * (halo * daylight)[..., None]
    if sky is not None:
        space = space + 255 * (sky * (1 - halo))[..., None]

    return np.clip(np.where(on_globe[..., None], lit, space), 0, 255).astype(np.uint8)


def unproject(
    x: np.ndarray,          # float[h, w]
    y: np.ndarray,          # float[h, w]
    radius: np.ndarray,     # float[h, w]
    lon0: float,
    lat0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The point of the near hemisphere seen at each position in the disc.

    The inverse of the orthographic projection about (`lon0`, `lat0`). Positions
    outside the disc have no preimage; they are computed anyway, since clamping
    the radius is cheaper than masking, and discarded by the caller.
    """
    rho = np.clip(radius, 1e-9, 1.0)
    c = np.arcsin(rho)                          # angular distance from centre
    sin_c, cos_c = np.sin(c), np.cos(c)
    lat = np.arcsin(np.clip(
        cos_c*np.sin(lat0) + y*sin_c*np.cos(lat0)/rho, -1, 1,
    ))
    lon = lon0 + np.arctan2(
        x * sin_c,
        rho*cos_c*np.cos(lat0) - y*sin_c*np.sin(lat0),
    )
    return lon, lat


def terrain(
    lon: np.ndarray,        # float[h, w]
    lat: np.ndarray,        # float[h, w]
    land: np.ndarray,       # float[rows, cols]
    relief: tuple,
) -> np.ndarray:            # float[h, w, rgb]
    """The colour of the ground and the water, before any light falls on it."""
    ground = sample(land, lon, lat)
    roughness = noise(relief, lon, lat)

    # A climate is a latitude, so roughening the latitude by a standard
    # deviation of nine degrees is enough to make the bands ragged and to
    # strand a desert or a forest well outside the one it belongs to.
    degrees = np.abs(np.degrees(lat)) + 9*roughness
    dry_land = ramp(CLIMATE_LATITUDE, CLIMATE_COLOR, degrees)

    # Sea ice, which is not on the land ramp because it is not on the land. The
    # pack takes much less licence from the noise than the climate bands do:
    # ragged ice is right, but ice reaching the temperate latitudes is not, and
    # a globe seen from above its pole shows a great deal of it at once.
    frozen = smoothstep(0.0, 1.0, (np.abs(np.degrees(lat)) - 73 + 4*roughness)/5)

    # The shelf: `land` sampled between cells already falls off across the
    # coast, and reading it through a curve widens that into shallow water.
    shelf = smoothstep(0.0, 0.55, ground)
    sea = blend(DEEP_SEA * np.ones_like(dry_land), SHALLOW_SEA, shelf[..., None])
    sea = blend(sea, CLIMATE_COLOR[-1], frozen[..., None])

    return blend(sea, dry_land, smoothstep(0.35, 0.65, ground)[..., None])


def land_field() -> np.ndarray:     # float[rows, cols]
    """The mask above, as ones and zeros, blurred by half a cell.

    The blur is what gives the coast somewhere to be. Sampling a hard mask
    between its cells produces a step; sampling a softened one produces a
    gradient a cell wide, which is the shore and the shallows both.
    """
    rows = [[1.0 if c == "#" else 0.0 for c in line] for line in LAND.split()]
    field = np.array(rows)
    # A cell averaged with its neighbours, wrapping in longitude, since the
    # column past the antimeridian is the column before it.
    return sum([
        0.5 * field,
        0.125 * np.roll(field, 1, axis=1),
        0.125 * np.roll(field, -1, axis=1),
        0.125 * np.roll(field, 1, axis=0),
        0.125 * np.roll(field, -1, axis=0),
    ])


def sample(
    field: np.ndarray,      # float[rows, cols]
    lon: np.ndarray,        # float[h, w]
    lat: np.ndarray,        # float[h, w]
) -> np.ndarray:            # float[h, w]
    """Read a lon/lat grid between its cells, bilinearly.

    Longitude wraps and latitude is held at the poles, so the sphere has no
    seam down the antimeridian and no hole at either end.
    """
    rows, cols = field.shape
    # cell coordinates of the sample, offset so that whole numbers are centres
    u = (np.degrees(lon) + 180) / 360 * cols - 0.5
    v = (90 - np.degrees(lat)) / 180 * rows - 0.5
    u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
    du, dv = u - u0, v - v0

    def at(dc: int, dr: int) -> np.ndarray:
        return field[np.clip(v0 + dr, 0, rows - 1), (u0 + dc) % cols]

    return (
        at(0, 0)*(1 - du)*(1 - dv) + at(1, 0)*du*(1 - dv)
        + at(0, 1)*(1 - du)*dv + at(1, 1)*du*dv
    )


class waves(NamedTuple):
    """A noise field on the sphere, as a sum of waves in the space it sits in.

    Value noise on a lon/lat grid has a seam at the antimeridian and a pinch at
    each pole, both of which a globe puts on show. Waves in three dimensions,
    sampled on the sphere's surface, have neither: the field is smooth
    everywhere because the space it is defined in is.
    """
    vector: np.ndarray          # float[count, 3], direction times wavenumber
    offset: np.ndarray          # float[count], each wave's phase
    amplitude: np.ndarray       # float[count]
    centre: float               # what to subtract for a zero mean
    spread: float               # what to divide by for unit variance


def plane_waves(
    rng: np.random.Generator,
    count: int,
    wavenumber: float,
    falloff: float = 1.0,
) -> waves:
    """Draw a wave field, and measure what it turns out to be.

    Predicting the field's spread from its amplitudes does not work: a wave
    whose wavelength is longer than the sphere does not complete a period
    anywhere on it, so it contributes far less variance than its amplitude
    suggests, and a large constant offset besides. Sampling the finished field
    is both exact and cheap, and it lets every threshold elsewhere in this file
    be read as a number of standard deviations.

    `falloff` is the slope of the spectrum: how fast amplitude drops as the
    waves get shorter. At one, the longest wave carries most of the field and
    the result is a few broad lobes, which is what a climate looks like. Lower
    it and the middle of the spectrum gets a say, which is what a cloud deck
    looks like.
    """
    direction = rng.normal(size=(count, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    # spread the octaves over the range, longest wave first
    scale = np.geomspace(1.4, wavenumber, num=count)
    field = waves(
        vector=direction * scale[:, None],
        offset=rng.uniform(0, 2*np.pi, size=count),
        amplitude=scale**-falloff,
        centre=0.0,
        spread=1.0,
    )
    lon, lat = np.meshgrid(
        np.linspace(-np.pi, np.pi, 128),
        np.arcsin(np.linspace(-1, 1, 64)),      # equal area, so the poles do
    )                                           # not count for more than they
    sample = noise(field, lon, lat)             # cover
    return field._replace(centre=sample.mean(), spread=sample.std())


def noise(
    field: waves,
    lon: np.ndarray,        # float[h, w]
    lat: np.ndarray,        # float[h, w]
) -> np.ndarray:            # float[h, w], zero mean and unit variance
    """Evaluate a wave field on the sphere.

    The field does not evolve. Advancing each wave's phase, so that a cloud
    deck grows and thins instead of drifting rigidly, was tried and dropped for
    two reasons: the loop stops being seamless, since the phases do not come
    back to where they started; and with a couple of dozen waves interfering,
    the field's own spread swings by a factor of two as they move past each
    other, which the sky shows as total overcast one moment and clear air the
    next. A rigid deck given a whole turn of drift is the better trade.
    """
    point = np.stack([
        np.cos(lat)*np.cos(lon),
        np.cos(lat)*np.sin(lon),
        np.sin(lat),
    ], axis=-1)
    angle = point @ field.vector.T + field.offset
    return (np.sin(angle) @ field.amplitude - field.centre) / field.spread


def starfield(rng: np.random.Generator, width: int) -> np.ndarray:
    """Stars, sparse and unequal, fixed behind the planet."""
    brightness = rng.random(size=(width, width))
    return np.where(brightness > 0.988, (brightness - 0.988)/0.012, 0.0)**1.5


def ramp(
    stops: np.ndarray,      # float[stops]
    colors: np.ndarray,     # float[stops, rgb]
    at: np.ndarray,         # float[...]
) -> np.ndarray:            # float[..., rgb]
    """Read a colour ramp, interpolating each channel between its stops."""
    return np.stack([
        np.interp(at, stops, colors[:, channel]) for channel in range(3)
    ], axis=-1)


def blend(under: np.ndarray, over: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """`over` laid on `under`, `alpha` of the way."""
    return under*(1 - alpha) + over*alpha


def smoothstep(low: float, high: float, at: np.ndarray) -> np.ndarray:
    """Zero below `low`, one above `high`, and an S between them."""
    t = np.clip((at - low) / (high - low), 0, 1)
    return t*t*(3 - 2*t)


if __name__ == "__main__":
    tyro.cli(main)
