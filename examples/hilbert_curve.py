import numpy as np
import matthewplotlib as mp

N = 15565

data = np.random.binomial(1, p=np.linspace(0,1,N))
data = data.astype(bool)
plot = mp.hilbert(
    data=data,
    dotcolor=(1.,1.,1.),
    bgcolor=(0.,0.,0.),
    nullcolor=(.1,.1,.1),
)

print("printing plot...")
print(plot)
print("saving to 'examples/hilbert_curve.png'...")
plot.saveimg('examples/hilbert_curve.png')
