import time
import collections

import numpy as np
import psutil # pip install psutil

import matthewplotlib as mp

FPS = 10
HISTORY_SECONDS = 30
PLOT_WIDTH = 80

def main():
    history_size = HISTORY_SECONDS * FPS
    cpu_history = collections.deque(maxlen=history_size)
    time_points = -np.linspace(HISTORY_SECONDS, 0, history_size)
    plot = None
    while True:
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
            xrange=(-HISTORY_SECONDS, 0),
            yrange=(0, 100),
        ))
        cpu_title = mp.center(
            mp.text(f"CPU History ({HISTORY_SECONDS}s)"),
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
            column_width=2,
            color="yellow"
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

        time.sleep(1 / FPS)


if __name__ == "__main__":
    main()
