"""
Boids Flocking Simulation.

Simulates autonomous agents (boids) moving in 2D space based on three simple rules:
1. Separation: steer to avoid crowding local flockmates.
2. Alignment: steer towards the average heading of local flockmates.
3. Cohesion: steer to move toward the average position of local flockmates.

By Gemini 3.1 Pro.
"""

import tyro
import numpy as np
import matthewplotlib as mp


class Boids:
    def __init__(
        self,
        n: int,
        width: float,
        height: float,
        cohesion_radius: float,
        alignment_radius: float,
        separation_radius: float,
        cohesion_weight: float,
        alignment_weight: float,
        separation_weight: float,
        noise: float,
    ):
        np.random.seed(42)
        self.pos = np.random.rand(n, 2) * [width, height]
        angle = np.random.rand(n) * 2 * np.pi
        speed = 1.0
        self.vel = np.column_stack((np.cos(angle), np.sin(angle))) * speed
        self.width = width
        self.height = height
        
        self.coh_r = cohesion_radius
        self.ali_r = alignment_radius
        self.sep_r = separation_radius
        self.coh_w = cohesion_weight
        self.ali_w = alignment_weight
        self.sep_w = separation_weight
        self.noise = noise

    def step(self):
        n = len(self.pos)
        if n == 0:
            return
            
        # Vectorized differences
        diff = self.pos[:, np.newaxis, :] - self.pos[np.newaxis, :, :]
        
        # Torus topology wrapping for distance
        diff -= np.round(diff / [self.width, self.height]) * [self.width, self.height]
        dist = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dist, np.inf)
        
        coh_force = np.zeros_like(self.pos)
        align_force = np.zeros_like(self.vel)
        sep_force = np.zeros_like(self.pos)
        
        for i in range(n):
            # Limit field of view to 270 degrees (they can't see directly behind them)
            # This prevents them from forming one giant stable blob as easily.
            # Heading vector of boid i
            heading = self.vel[i] / (np.linalg.norm(self.vel[i]) + 1e-5)
            # Vector to other boids
            to_others = -diff[i] / (dist[i, :, np.newaxis] + 1e-5)
            # Dot product to find angle cosine
            cos_angles = np.dot(to_others, heading)
            
            # Mask out boids behind this one (e.g., cos < -0.707 for 270 deg FOV)
            # wait, cos < -0.5 means behind. Let's just use a simple mask:
            # cos_angles > -0.5 means they can see 240 degrees in front.
            in_fov = cos_angles > -0.5

            coh_mask = (dist[i] < self.coh_r) & in_fov
            if np.any(coh_mask):
                coh_force[i] = np.mean(-diff[i, coh_mask], axis=0)
                
            align_mask = (dist[i] < self.ali_r) & in_fov
            if np.any(align_mask):
                align_force[i] = np.mean(self.vel[align_mask], axis=0) - self.vel[i]
                
            sep_mask = dist[i] < self.sep_r # Separation doesn't strictly need FOV, they don't want to crash!
            if np.any(sep_mask):
                sep_force[i] = np.sum(diff[i, sep_mask] / (dist[i, sep_mask, np.newaxis]**2 + 1e-5), axis=0)

        # Apply forces
        self.vel += self.coh_w * coh_force + self.ali_w * align_force + self.sep_w * sep_force
        
        # Add a tiny bit of noise to prevent unnatural grid locks and make it look organic
        self.vel += (np.random.rand(n, 2) - 0.5) * self.noise
        
        # Constrain speed
        speed = np.linalg.norm(self.vel, axis=-1, keepdims=True)
        max_speed = 1.5
        min_speed = 0.5
        
        self.vel = np.where(speed > max_speed, self.vel / speed * max_speed, self.vel)
        self.vel = np.where(speed < min_speed, self.vel / speed * min_speed, self.vel)
        
        # Update and wrap positions
        self.pos += self.vel
        self.pos = np.mod(self.pos, [self.width, self.height])


def main(
    num_frames: int = 0,
    fps: int = 30,
    n_boids: int = 150,
    width: int = 80,
    height: int = 35,
    cohesion_radius: float = 12.0,
    alignment_radius: float = 10.0,
    separation_radius: float = 4.0,
    cohesion_weight: float = 0.003,
    alignment_weight: float = 0.03,
    separation_weight: float = 0.4,
    noise: float = 0.15,
    save: str | None = None,
    downscale: int = 1,
):
    """Animate a Boids flocking simulation."""
    boids = Boids(
        n_boids,
        width=100.0,
        height=100.0,
        cohesion_radius=cohesion_radius,
        alignment_radius=alignment_radius,
        separation_radius=separation_radius,
        cohesion_weight=cohesion_weight,
        alignment_weight=alignment_weight,
        separation_weight=separation_weight,
        noise=noise,
    )
    
    animation = mp.animate(
        fps=fps,
        record=save is not None,
        stop_on_interrupt=True,
    )
    
    with animation as anim:
        frame = 0
        while num_frames == 0 or frame < num_frames:
            boids.step()
            
            # Map headings to colors so flocks share identical colors
            angles = np.arctan2(boids.vel[:, 1], boids.vel[:, 0])
            colors = mp.rainbow((angles + np.pi) / (2 * np.pi))
            
            series_list = []
            for i in range(n_boids):
                head = boids.pos[i]
                # Tail points backwards along the velocity vector
                tail = head - boids.vel[i] * 2.5
                seg = np.stack([tail, head])
                # Provide segment and its assigned color
                series_list.append((seg, colors[i]))
            
            if series_list:
                plot = mp.line(
                    series_list[0],
                    *series_list[1:],
                    xrange=(0, 100),
                    yrange=(0, 100),
                    width=width,
                    height=height,
                    thickness=1.5,
                )
            else:
                plot = mp.blank(width=width, height=height)
                
            anim.update(plot)
            frame += 1

    if save:
        anim.frames.savegif(save, bgcolor="black", downscale=downscale)


if __name__ == "__main__":
    tyro.cli(main)
