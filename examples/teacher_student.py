"""
Train a teacher--student linear regression model with gradient descent.

By Matthew Farrugia-Roberts.
"""

import tyro
import matthewplotlib as mp
from jaxtyping import Float, Array # pip install jaxtyping

import jax # pip install jax
import jax.numpy as jnp


def main(
    num_steps: int = 400,
    learning_rate: float = 0.01,
    log_every: int = 100,
    save: str | None = None,
):
    # initialise teacher
    w_teacher = jnp.array([.5, -1.])

    # initialise student
    w_student = jnp.array([-1., 3.])

    # initialise input data
    x = jnp.linspace(-4, 4, 80)

    # training loop. `anim.print` puts a line above the plot rather than through
    # it, which is the only way to log from inside an animation.
    animation = mp.animate(
        fps=50,
        record=save is not None,
        stop_on_interrupt=True,
    )
    with animation as anim:
        anim.update(vis(w_student, w_teacher, x, 0))
        for t in range(num_steps):
            g_student = jax.grad(loss)(w_student, w_teacher, x)
            w_student = w_student - learning_rate * g_student
            anim.update(vis(w_student, w_teacher, x, t+1))
            if log_every and (t + 1) % log_every == 0:
                l = loss(w_student, w_teacher, x)
                anim.print(f"step {t+1:>4d}  loss {l:.5f}")

    if save:
        anim.frames.savegif(save, bgcolor="black")


def forward(
    w: Float[Array, "2"],
    x: Float[Array, "batch_size"],
) -> Float[Array, "batch_size"]:
    a, b = w
    return a * x + b


def loss(w_student, w_teacher, x):
    diff = forward(w_student, x) - forward(w_teacher, x)
    return jnp.mean(diff**2)


def vis(
    w_student: Float[Array, "2"],
    w_teacher: Float[Array, "2"],
    x: Float[Array, "batch_size"],
    step: int,
) -> mp.plot:
    return mp.axes(
        mp.scatter(
            mp.xaxis(-4, 4, 80),
            mp.yaxis(-4, 4, 80),
            (x, forward(w_student, x), 'magenta'),
            (x, forward(w_teacher, x), 'cyan'),
            height=20,
            width=40,
            xrange=(-4,4),
            yrange=(-4,4),
        ),
        title=f"step {step}",
        ylabel="y",
        xlabel="x",
    )

if __name__ == "__main__":
    tyro.cli(main)

