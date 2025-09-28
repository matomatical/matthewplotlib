import numpy as np
import sklearn.decomposition # pip install scikit-learn

import matthewplotlib as mp


DIMENSION = 10_000
NUM_STEPS = 1_000

# sample some brownian motion
print("sample some high-dimensional brownian motion...")
traj = np.random.normal(size=(NUM_STEPS, DIMENSION)).cumsum(axis=0)

# 3-dimensional PCA reduction
print("conduct three dimensional PCA...")
proj = sklearn.decomposition.PCA(n_components=3).fit_transform(traj)
time = np.arange(NUM_STEPS)

# construct visualisation
plot = mp.vstack(
    mp.text("BROWNIAN MOTION"),
    mp.text("first two dimensions"),
    mp.scatter(
        xs=traj[:, 0],
        ys=traj[:, 1],
        color=(.5,.5,.5),
        height=20,
        width=75,
    ),
    mp.text("principal components 1, 2, 3 over time"),
    mp.scatter(xs=time, ys=proj[:,0], color="red", width=75)
    @ mp.scatter(xs=time, ys=proj[:,1], color="green", width=75)
    @ mp.scatter(xs=time, ys=proj[:,2], color="blue", width=75),
    mp.text("principal components 1, 2, 3 paired"),
    mp.hstack(
        mp.text("PC1 v PC2")
        / mp.scatter(
            xs=proj[:, 0],
            ys=proj[:, 1],
            color="yellow",
            width=25,
        ),
        mp.text("PC1 v PC3")
        / mp.scatter(
            xs=proj[:, 0],
            ys=proj[:, 2],
            color="magenta",
            width=25,
        ),
        mp.text("PC2 v PC3")
        / mp.scatter(
            xs=proj[:, 1],
            ys=proj[:, 2],
            color="cyan",
            width=25,
        )
    ),
)

print("printing plot...")
print(plot)
print("saving plot to images/lissajous.png...")
plot.saveimg('images/lissajous.png')
