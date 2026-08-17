"""
A moving incompressible flow, coloured by its velocity.

Each pixel contains a two-component vector rather than a scalar. The custom
colormap turns its direction into hue and its speed into brightness, reducing a
`[frames, rows, cols, 2]` velocity field to the RGB tensor `mp.animation`
renders. The flow itself comes from a periodic streamfunction, so it is
divergence-free and the animation loops exactly.

By GPT 5.6 Sol.
"""

import tyro
import numpy as np

import matthewplotlib as mp


# The almost-black colour of still water. Flow colours are mixed into this as
# speed rises, so stagnation points stay quiet while fast channels light up.
STILL = np.array([3, 5, 14], dtype=float)


def main(
    num_frames: int = 48,
    fps: float = 20.0,
    width: int = 78,
    height: int = 20,
    loop: bool = True,
    save: str | None = None,
):
    """An endlessly circulating chromatic flow."""
    phase = 2 * np.pi * np.arange(num_frames) / num_frames
    velocity = flow(phase, rows=2 * height, cols=width)

    # `velocity` ends in a two-vector, not RGB. A colormap is allowed to accept
    # any array data so long as its output has the colour shape animation asks
    # for; here it consumes the vector axis and returns three colour channels.
    animation = mp.animation(
        velocity,
        colormap=flow_colors,
        fps=fps,
    )
    animation = animation.map(lambda frame: mp.border(
        frame,
        title=" chromatic flow ",
    ))
    caption = mp.center(
        mp.text(
            "hue = direction  ·  brightness = speed",
            fgcolor=(0.55, 0.55, 0.55),
        ),
        width=animation.width,
    )
    animation = animation.map(lambda frame: frame / caption)

    animation.play(loop=loop)

    if save:
        # At half-block resolution each input pixel becomes an 8px square.
        # Keeping every fourth rendered pixel makes the field compact while
        # leaving the caption legible in the gallery.
        animation.savegif(save, downscale=4, bgcolor="black")


def flow(
    phase: np.ndarray,
    rows: int,
    cols: int,
) -> np.ndarray:  # float[frames, rows, cols, uv]
    """Velocity from a travelling, periodic streamfunction.

    For a streamfunction psi, `(u, v) = (dpsi/dy, -dpsi/dx)` has zero
    divergence by construction. Here psi is a sum of soft Gaussian vortices.
    Their centres circle around one another and return after one turn of
    `phase`, while their overlapping velocity fields bend each other's colour
    wheels into the channels between them.
    """
    aspect = cols / rows
    x = np.linspace(-aspect, aspect, cols)[None, None, :]
    y = np.linspace(-1, 1, rows)[None, :, None]
    t = phase[:, None, None]

    bases = np.array([
        [-0.82, -0.42],
        [0.82, -0.42],
        [-0.82,  0.42],
        [0.82,  0.42],
        [0.00,  0.00],
    ])
    strengths = np.array([1.0, -1.0, -1.0, 1.0, 0.65])
    offsets = np.linspace(0, 2 * np.pi, len(bases), endpoint=False)
    sigma = 0.48
    orbit = 0.16

    u = np.zeros((len(phase), rows, cols))
    v = np.zeros_like(u)
    for (x0, y0), strength, offset in zip(bases, strengths, offsets):
        cx = x0 + orbit * np.cos(t + offset)
        cy = y0 + orbit * np.sin(t + offset)
        dx, dy = x - cx, y - cy
        bell = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
        # psi = strength * bell
        u += -strength * dy / sigma**2 * bell
        v += strength * dx / sigma**2 * bell
    return np.stack([u, v], axis=-1)


def flow_colors(velocity: np.ndarray) -> np.ndarray:  # uint8[..., 3]
    """Map a velocity vector to hue by direction and light by speed."""
    u, v = np.moveaxis(velocity, -1, 0)
    direction = np.mod(np.arctan2(v, u) / (2 * np.pi), 1.0)
    speed = np.hypot(u, v)

    hue = mp.rainbow(direction).astype(float)
    amount = np.tanh(speed / 1.15)[..., None]
    return (STILL + amount * (hue - STILL)).astype(np.uint8)


if __name__ == "__main__":
    tyro.cli(main)
