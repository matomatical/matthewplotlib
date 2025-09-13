import numpy as np
import matthewplotlib as mp
import hilbert


print("preparing test images...")
coords = hilbert.decode(
    hilberts=np.arange(256),
    num_dims=2,
    num_bits=4,
)
im_continuous = np.zeros((16,16), dtype=float)
im_continuous[coords[:,0], coords[:,1]] = np.linspace(0.0, 1.0, 256)
im_discrete = np.zeros((16, 16), dtype=int)
im_discrete[coords[:,0], coords[:,1]] = np.arange(256) // 16

print("generating plot...")
plot = (
    mp.text("test images:")
    ^ mp.hstack(
        mp.border(
            mp.text("linspace(0,1,256")
            ^ mp.image(im_continuous),
        ),
        mp.border(
            mp.text("arange(256)//16")
            ^ mp.image(im_discrete / 16),
        )
    )
    ^ mp.text("continuous colormaps:")
    ^ mp.wrap(*[
        mp.border(
            mp.text(c.__name__)
            ^ mp.image(im_continuous, colormap=c),
        )
        for c in [
            mp.reds, mp.greens, mp.blues,
            mp.yellows, mp.magentas, mp.cyans,
            mp.magma, mp.inferno, mp.plasma,
            mp.viridis, mp.cyber, mp.rainbow,
        ]
    ], cols=3)
    ^ mp.text("discrete colormaps:")
    ^ mp.wrap(*[
        mp.border(
            mp.text(c.__name__)
            ^ mp.image(im_discrete, colormap=c),
        )
        for c in [ mp.sweetie16, mp.pico8, ]
    ], cols=3)
)

print("rendering plot...")
print(plot)

print("saving plot to images/colormaps.png ...")
plot.saveimg("images/colormaps.png")

