import numpy as np
import sklearn.decomposition # pip install scikit-learn
import mattplotlib as mp


def main(dimension: int = 10000, num_steps: int = 10000):
    # sample some brownian motion
    traj = np.random.normal(size=(num_steps, dimension)).cumsum(axis=0)
    print(mp.scatter(traj[:, :2], height=20, width=90, color=(.5,.5,.5)))

    # 3-dimensional PCA reduction
    proj = sklearn.decomposition.PCA(n_components=3).fit_transform(traj)
    print(
          mp.scatter(proj[:, (0,1)], color=(1,1,0))
        & mp.scatter(proj[:, (0,2)], color=(1,0,1))
        & mp.scatter(proj[:, (1,2)], color=(0,1,1))
    )


if __name__ == "__main__":
    import typer
    typer.run(main)

