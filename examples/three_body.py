"""
Three equal masses chasing one another around a shared figure-eight orbit.

This is not a parametric curve with three dots moved along it. The positions
are the result of integrating Newton's inverse-square law from the initial
conditions of the figure-eight choreography. Velocity Verlet keeps the
integration time-reversible and nearly closes the orbit after one period.

By GPT 5.6 Sol.
"""

import tyro
import numpy as np

import matthewplotlib as mp


# The equal-mass figure-eight initial conditions, in the centre-of-mass frame,
# and the orbit's period. The first two bodies have the same velocity; the
# third balances their momentum.
INITIAL_POSITIONS = np.array([
    [-0.97000436, 0.24308753],
    [0.97000436, -0.24308753],
    [0.0, 0.0],
])
INITIAL_VELOCITIES = np.array([
    [0.466203685, 0.432365730],
    [0.466203685, 0.432365730],
    [-0.932407370, -0.864731460],
])
PERIOD = 6.32591398

BODY_COLORS = np.array([
    [255, 91, 121],
    [89, 226, 255],
    [255, 218, 92],
], dtype=np.uint8)
BACKGROUND = np.array([5, 7, 18], dtype=np.uint8)

XRANGE = (-1.18, 1.18)
YRANGE = (-0.60, 0.60)
STEPS_PER_FRAME = 32


def main(
    num_frames: int = 72,
    fps: float = 20.0,
    width: int = 78,
    height: int = 20,
    trail_frames: int = 24,
    loop: bool = True,
    save: str | None = None,
):
    """The figure-eight three-body choreography under Newtonian gravity."""
    positions = integrate(num_frames, steps_per_frame=STEPS_PER_FRAME)
    frames = [
        draw_frame(positions, frame, trail_frames, width, height)
        for frame in range(num_frames)
    ]

    animation = mp.tstack(*frames, fps=fps)
    animation = animation.map(lambda frame: mp.border(
        frame,
        title=" three bodies, one orbit ",
    ))
    caption = mp.center(
        mp.text(
            "equal masses  ·  Newtonian gravity",
            fgcolor=(0.55, 0.55, 0.55),
        ),
        width=animation.width,
    )
    animation = animation.map(lambda frame: frame / caption)

    animation.play(loop=loop)

    if save:
        # A braille dot occupies four rendered pixels in either direction.
        # Keeping every fourth pixel writes one image pixel per plotted dot.
        animation.savegif(save, downscale=4, bgcolor="black")


def accelerations(positions: np.ndarray) -> np.ndarray:  # float[body, xy]
    """Acceleration of each unit mass due to every other unit mass."""
    displacement = positions[None, :, :] - positions[:, None, :]
    distance_squared = np.sum(displacement**2, axis=-1)
    np.fill_diagonal(distance_squared, np.inf)
    return np.sum(
        displacement / distance_squared[..., None] ** 1.5,
        axis=1,
    )


def integrate(
    num_frames: int,
    steps_per_frame: int,
) -> np.ndarray:  # float[frame, body, xy]
    """Integrate one period with velocity Verlet, sampling displayed frames."""
    positions = INITIAL_POSITIONS.copy()
    velocities = INITIAL_VELOCITIES.copy()
    acceleration = accelerations(positions)
    timestep = PERIOD / (num_frames * steps_per_frame)

    sampled = []
    for _ in range(num_frames):
        sampled.append(positions.copy())
        for _ in range(steps_per_frame):
            positions += (
                timestep * velocities
                + 0.5 * timestep**2 * acceleration
            )
            next_acceleration = accelerations(positions)
            velocities += 0.5 * timestep * (
                acceleration + next_acceleration
            )
            acceleration = next_acceleration
    return np.stack(sampled)


def draw_frame(
    positions: np.ndarray,
    frame: int,
    trail_frames: int,
    width: int,
    height: int,
) -> mp.plot:
    """Draw fading histories behind the three bodies at one instant."""
    trail_frames = min(max(trail_frames, 2), len(positions))
    history = (frame - np.arange(trail_frames - 1, -1, -1)) % len(positions)

    # Each body's colour emerges from the dark background along its trail.
    fade = np.linspace(0.05, 0.72, trail_frames)[:, None]
    trails = [
        (
            positions[history, body],
            (BACKGROUND + fade * (color - BACKGROUND)).astype(np.uint8),
        )
        for body, color in enumerate(BODY_COLORS)
    ]
    trails_plot = mp.line(
        *trails,
        xrange=XRANGE,
        yrange=YRANGE,
        width=width,
        height=height,
        thickness=1.35,
    )

    # A five-dot diamond gives each mass a visible centre without introducing
    # a second coordinate system or a special marker primitive.
    dot_x = (XRANGE[1] - XRANGE[0]) / (2 * width)
    dot_y = (YRANGE[1] - YRANGE[0]) / (4 * height)
    offsets = np.array([
        [0.0, 0.0],
        [-dot_x, 0.0],
        [dot_x, 0.0],
        [0.0, -dot_y],
        [0.0, dot_y],
    ])
    bodies = positions[frame, :, None, :] + offsets[None, :, :]
    bodies_plot = mp.scatter(
        (bodies.reshape(-1, 2), np.repeat(BODY_COLORS, len(offsets), axis=0)),
        xrange=XRANGE,
        yrange=YRANGE,
        width=width,
        height=height,
    )
    return mp.dstack2(trails_plot, bodies_plot)


if __name__ == "__main__":
    tyro.cli(main)
