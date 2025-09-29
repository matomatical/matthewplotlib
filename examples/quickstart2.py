import time
import numpy as np
import matthewplotlib as mp

x = np.linspace(-2*np.pi, +2*np.pi, 150)

plot = None
# frames = []

try:
    # for i in range(49):
    while True:
        k = (time.time() % 3) * 2
        A = 0.85 + 0.15 * np.cos(k)
        y = A * np.cos(x - 2*np.pi*k/6)
        c = mp.rainbow(1-k/6)
        if plot is not None:
            print(-plot, end="")
        plot = mp.axes(
            mp.scatter(
                (x, y, c),
                width=75,
                height=10,
                yrange=(-1,1),
            ),
            title=f" y = {A:.2f} cos(x + 2π*{k:.2f}/6) ",
            xlabel="x",
            ylabel="y",
        )
        print(plot)
        # frames.append(plot)
        time.sleep(1/20)
except KeyboardInterrupt:
    print()
# mp.save_animation(
#     frames,
#     "images/quickstart.gif",
#     bgcolor="black",
#     fps=20,
# )
