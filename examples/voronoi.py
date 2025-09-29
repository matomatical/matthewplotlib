import numpy as np
import matthewplotlib as mp
from scipy.spatial import KDTree

NUM_SEEDS = 16
WIDTH = 70
HEIGHT = 20


def main():
    np.random.seed(42)
    seeds = np.random.rand(NUM_SEEDS, 2) * [WIDTH, 2 * HEIGHT]
    plot = (
        mp.function2(
            lambda xy: np.argmin(
                np.sum(
                    (seeds.reshape(-1, 1, 2) - (xy + 0.5))**2,
                    axis=-1,
                ),
                axis=0,
            ),
            xrange=(0, WIDTH),
            yrange=(0, 2 * HEIGHT),
            width=WIDTH,
            height=HEIGHT,
            colormap=lambda c: mp.rainbow(c)//2,
        ) @ mp.scatter(
            seeds,
            xrange=(0, WIDTH),
            yrange=(0, 2 * HEIGHT),
            width=WIDTH,
            height=HEIGHT,
        )
    )

    print(plot)
    plot.saveimg("images/voronoi.png")


if __name__ == "__main__":
    main()
