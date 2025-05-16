import numpy as np
import sklearn.decomposition # pip install scikit-learn

import mattplotlib as mp


DIMENSION = 10_000
NUM_STEPS = 10_000

# sample some brownian motion
print("sample some high-dimensional brownian motion...")
traj = np.random.normal(size=(NUM_STEPS, DIMENSION)).cumsum(axis=0)
print("first two dimensions:")
print(mp.scatter(traj[:, :2], height=20, width=75, color=(.5,.5,.5)))

# 3-dimensional PCA reduction
print("three dimensional PCA...")
proj = sklearn.decomposition.PCA(n_components=3).fit_transform(traj)
print("plot:")
print(
      mp.text("PC1 v PC2") ^ mp.scatter(proj[:, (0,1)], color=(1,1,0), width=25)
    | mp.text("PC1 v PC3") ^ mp.scatter(proj[:, (0,2)], color=(1,0,1), width=25)
    | mp.text("PC2 v PC3") ^ mp.scatter(proj[:, (1,2)], color=(0,1,1), width=25)
)
