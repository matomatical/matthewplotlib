import numpy as np
import sklearn.decomposition # pip install scikit-learn

import matthewplotlib as mp


DIMENSION = 10_000
NUM_STEPS = 1_000
SEED = 42


# sample some brownian motion
print("sample some high-dimensional brownian motion...")
np.random.seed(SEED)
traj = np.random.normal(size=(NUM_STEPS, DIMENSION)).cumsum(axis=0)

# 3-dimensional PCA reduction
print("conduct three dimensional PCA...")
proj = sklearn.decomposition.PCA(n_components=3).fit_transform(traj)
time = np.linspace(.1, 1., NUM_STEPS)

# construct visualisation
plot = mp.vstack(
    mp.text("BROWNIAN MOTION"),
    mp.text("first two dimensions"),
    mp.scatter(
        (traj[:, (0,1)], (.5,.5,.5)),
        height=20,
        width=75,
    ),
    mp.text("principal components 1, 2, 3 over time"),
    mp.scatter(
        (time, proj[:,0], "red"),
        (time, proj[:,1], "green"),
        (time, proj[:,2], "blue"),
        width=75,
    ),
    mp.text("principal components 1, 2, 3 paired, increasing lightness over time"),
    mp.hstack(
        mp.text("PC1 v PC2")
        / mp.scatter((proj[:, (0,1)], mp.yellows(time)), width=25),
        mp.text("PC1 v PC3")
        / mp.scatter((proj[:, (0,2)], mp.magentas(time)), width=25),
        mp.text("PC2 v PC3")
        / mp.scatter((proj[:, (1,2)], mp.cyans(time)), width=25),
    ),
)

print("printing plot...")
print(plot)
print("saving plot to images/lissajous.png...")
plot.saveimg('images/lissajous.png')
