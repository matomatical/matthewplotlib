import time
import collections

import tyro
import numpy as np
import psutil # pip install psutil

import matthewplotlib as mp


def main(
    num_frames: int = 0,
    fps: int = 10,
    history_seconds: int = 30,
    save: str | None = None,
):
    """Live system dashboard showing CPU and memory usage."""
    history_size = history_seconds * fps
    cpu_history = collections.deque(maxlen=history_size)
    time_points = -np.linspace(history_seconds, 0, history_size)
    plot = None
    frame = 0
    if save and num_frames > 0:
        all_frames = []
    while num_frames == 0 or frame < num_frames:
        # collect data
        cpu_percent = psutil.cpu_percent(percpu=False)
        cpu_percents_core = psutil.cpu_percent(percpu=True)
        mem = psutil.virtual_memory()
        cpu_history.append(cpu_percent)

        padded_history = np.pad(
            cpu_history, 
            (history_size - len(cpu_history), 0), 
            'constant', 
            constant_values=np.nan
        )

        # CPU utilisation
        cpu_plot = mp.border(mp.scatter(
            (time_points, padded_history, "cyan"),
            width=40,
            height=7,
            xrange=(-history_seconds, 0),
            yrange=(0, 100),
        ))
        cpu_title = mp.center(
            mp.text(f"CPU History ({history_seconds}s)"),
            width=cpu_plot.width,
        )

        # memory usage
        mem_plot = mp.border(mp.progress(
            mem.percent / 100, 
            width=cpu_plot.width - 2, 
            color="magenta"
        ))
        mem_title = mp.center(
            mp.text(f"Memory: {mem.percent:.1f}%"),
            width=cpu_plot.width,
        )
        left_panel = mp.vstack(
            cpu_title,
            cpu_plot,
            mem_title,
            mem_plot,
        )

        # per-core CPU usage
        core_plot = mp.border(mp.columns(
            cpu_percents_core,
            height=left_panel.height - 3,
            vrange=100,
            column_width=1,
            column_spacing=1,
            colors=mp.rainbow(np.linspace(0,1,len(cpu_percents_core))),
        ))
        core_title = mp.center(
            mp.text("CPU Core Usage"),
            width=core_plot.width,
        )
        right_panel = core_title / core_plot
        dashboard = left_panel + right_panel

        if plot:
            print(-plot, end="")
        plot = dashboard
        print(plot)
        if save and num_frames > 0:
            all_frames.append(plot)

        frame += 1
        time.sleep(1 / fps)

    if save and all_frames:
        mp.save_animation(
            plots=all_frames,
            filename=save,
            fps=fps,
            bgcolor='black',
        )


if __name__ == "__main__":
    try:
        tyro.cli(main)
    except KeyboardInterrupt:
        print()
