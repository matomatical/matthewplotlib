"""
Animated Lorenz Attractor in 3D.

Traces the chaotic paths of two slightly different Lorenz systems over time
while slowly orbiting the camera. This showcases both `mp.line3`, algebraic
plot composition (`+` for horizontal stacking), and the sensitive dependence
on initial conditions in chaotic systems.

By Gemini 3.1 Pro.
"""

import tyro
import numpy as np
import matthewplotlib as mp


# Lorenz parameters
SIGMA = 10.0
RHO = 28.0
BETA = 8.0 / 3.0


def lorenz_deriv(xyz):
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    return np.stack([
        SIGMA * (y - x),
        x * (RHO - z) - y,
        x * y - BETA * z
    ], axis=-1)


class LorenzSystem:
    def __init__(self, initial_state, dt=0.01):
        self.state = np.array(initial_state, dtype=float)
        self.dt = dt

    def step(self, num_steps=11):
        """Advance the system and return the new points."""
        points = np.zeros((num_steps, 3))
        for i in range(num_steps):
            k1 = lorenz_deriv(self.state)
            k2 = lorenz_deriv(self.state + k1 * self.dt / 2)
            k3 = lorenz_deriv(self.state + k2 * self.dt / 2)
            k4 = lorenz_deriv(self.state + k3 * self.dt)
            self.state = self.state + (k1 + 2*k2 + 2*k3 + k4) * self.dt / 6.0
            points[i] = self.state
        return points


def camera_pos(revolutions: float) -> np.ndarray:
    angle = 2 * np.pi * revolutions
    radius = 60.0
    return np.array([
        radius * np.sin(angle),
        25.0, # height
        radius * np.cos(angle),
    ])


def main(
    num_frames: int = 0,
    fps: int = 20,
    width: int = 40,
    height: int = 20,
    save: str | None = None,
    orbit_frames: int = 600,
    steps_per_frame: int = 11,
    window_size: int = 1500,
):
    """Animated 3D Lorenz attractor."""
    
    # Initialize two systems with a tiny difference in X
    sys1 = LorenzSystem([0.1, 1.0, 1.05])
    sys2 = LorenzSystem([0.10001, 1.0, 1.05])
    
    pts1 = np.zeros((0, 3))
    pts2 = np.zeros((0, 3))
    
    animation = mp.animate(
        fps=fps,
        record=save is not None,
        stop_on_interrupt=True,
    )
    
    with animation as anim:
        frame = 0
        total_points = 0
        while num_frames == 0 or frame < num_frames:
            # Generate new points and slide the window
            new1 = sys1.step(num_steps=steps_per_frame)
            new2 = sys2.step(num_steps=steps_per_frame)
            
            pts1 = np.concatenate([pts1, new1])[-window_size:]
            pts2 = np.concatenate([pts2, new2])[-window_size:]
            total_points += steps_per_frame
            
            # Map absolute indices to [0, 1] periodically with period = window_size.
            start_idx = max(0, total_points - window_size)
            indices = np.arange(start_idx, total_points)
            color_vals = (indices % window_size) / window_size
            colors = mp.rainbow(color_vals)
            
            p = camera_pos(frame / orbit_frames)
            target = np.array([0.0, 0.0, 25.0]) # center of mass of the attractor
            up = np.array([0., 0., 1.])
            
            plot1_tail = mp.line3(
                (pts1, colors),
                camera_position=p, camera_target=target, scene_up=up, 
                vertical_fov_degrees=60, width=width, height=height, thickness=1.0,
            )
            plot1_head = mp.line3(
                (np.vstack([pts1[-1], pts1[-1] + 1e-5]), "white"),
                camera_position=p, camera_target=target, scene_up=up, 
                vertical_fov_degrees=60, width=width, height=height, thickness=3.0,
            )
            plot1 = mp.border(plot1_tail @ plot1_head, title=" Initial X: 0.10000 ", style=mp.BoxStyle.HEAVY)
            
            plot2_tail = mp.line3(
                (pts2, colors),
                camera_position=p, camera_target=target, scene_up=up, 
                vertical_fov_degrees=60, width=width, height=height, thickness=1.0,
            )
            plot2_head = mp.line3(
                (np.vstack([pts2[-1], pts2[-1] + 1e-5]), "white"),
                camera_position=p, camera_target=target, scene_up=up, 
                vertical_fov_degrees=60, width=width, height=height, thickness=3.0,
            )
            plot2 = mp.border(plot2_tail @ plot2_head, title=" Initial X: 0.10001 ", style=mp.BoxStyle.HEAVY)
            
            # Compose them side by side
            anim.update(plot1 + plot2)
            frame += 1

    if save:
        anim.frames.savegif(save, bgcolor="black")


if __name__ == "__main__":
    tyro.cli(main)
