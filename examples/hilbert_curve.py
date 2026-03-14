import tyro
import numpy as np
import matthewplotlib as mp

def main(save: str | None = None):
    """Hilbert curve visualisation of binomial data."""
    N = 15565

    np.random.seed(42)
    data = np.random.binomial(1, p=np.linspace(0,1,N))
    data = data.astype(bool)
    plot = mp.hilbert(
        data=data,
        color=(1.,1.,1.),
    )

    print(plot)
    if save:
        plot.saveimg(save)

if __name__ == "__main__":
    tyro.cli(main)
