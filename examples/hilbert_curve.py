import numpy as np
import matthewplotlib as mp

N = 16
# N = 15565

for i in range(N):
    data = np.ones(i, dtype=bool)
    plot = mp.hilbert(
        data=data,
        dotcolor=(1.,1.,1.),
        bgcolor=(.2,.2,.2),
    )
    print(f"{i:2d}", plot)

# data = np.random.binomial(1, p=np.linspace(0,1,N))
# data = data.astype(bool)
plot = mp.hilbert(
    data=data,
    dotcolor=(1.,1.,1.),
    bgcolor=(0.,0.,0.),
    nullcolor=(.1,.1,.1),
)

print("printing plot...")
print(plot)
print("saving to 'out.png'...")
plot.saveimg('out.png')
