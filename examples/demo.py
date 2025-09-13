import numpy as np
import matthewplotlib as mp

size = 14
u = np.random.rand(size**2).reshape(size,size)
i = np.eye(size)
g = np.clip(np.random.normal(size=(size, size)) + 3, 0, 6) / 6
plot = (
    mp.border(
        mp.center(mp.text("G'day matthewplotlib"), height=3, width=46),
        style=mp.border.Style.DOUBLE,
    ) ^ (
        mp.border(
            mp.text("uniform:") ^ mp.image(u, colormap=mp.reds),
            style=mp.border.Style.LIGHT,
        ) | mp.border(
            mp.text("identity:") ^ mp.image(i, colormap=mp.greens),
            style=mp.border.Style.HEAVY,
        ) | mp.border(
            mp.text("gaussian:") ^ mp.image(g, colormap=mp.blues),
            style=mp.border.Style.DOUBLE,
        )
    ) ^ (
        mp.border(
            mp.text("uniform:")  ^ mp.image(u, colormap=mp.yellows),
            style=mp.border.Style.ROUND,
        ) | mp.border(
            mp.text("identity:") ^ mp.image(i, colormap=mp.cyber),
            style=mp.border.Style.BLANK,
        ) | mp.border(
            mp.text("gaussian:") ^ mp.image(g, colormap=mp.cyans),
            style=mp.border.Style.BUMPER,
        )
    ) ^ mp.border(
        mp.scatter(
            data=np.random.normal(size=(300, 2)),
            height=18,
            width=46,
            xrange=(-5, +5),
            yrange=(-4, +4),
            color=(0.,1.,0.),
        ),
        style=mp.border.Style.ROUND,
    )
)
print(repr(plot))
print(plot)
print("saving to 'images/demo.png'...")
plot.saveimg('images/demo.png')
