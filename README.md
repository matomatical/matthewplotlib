matt's plotting library (mattplotlib)
=====================================

A Python plotting library that isn't painful.

Work in progress.

Contents:

* Image plots
* Scatter plots
* Arranging plots

TODO:

* axes and labels
* many more plot types
* animated plots
* many more things

Installation
------------

Dependencies:

```
pip install numpy einops
```

Installation:

* Just copy the single file `mattplotlib.py` into your project.
* TODO: pip installable github repository
* TODO (later): pip

Import:

```
import mattplotlib as mp
import numpy as np # required for some of the examples in this readme
```


Image plots
-----------

Example:

```
rgb_array = [
    [ (0,0,0), (1,1,1), (0,0,0), (0,0,0), (1,1,1), (0,0,0), ],
    [ (0,0,0), (1,1,1), (0,0,0), (0,0,0), (1,1,1), (0,0,0), ],
    [ (0,0,0), (1,1,1), (0,0,0), (0,0,0), (1,1,1), (0,0,0), ],
    [ (0,0,0), (0,0,0), (0,0,0), (0,0,0), (0,0,0), (0,0,0), ],
    [ (1,1,1), (0,0,0), (0,0,0), (0,0,0), (0,0,0), (1,1,1), ],
    [ (0,0,0), (1,1,1), (1,1,1), (1,1,1), (1,1,1), (0,0,0), ],
]
plot = mp.image(rgb_array)
print(plot)
```

Output:

```
 █  █ 
 ▀  ▀ 
▀▄▄▄▄▀
```

Other options:

* the array can be rgb (h, w, 3) or greyscale (h, w)
* colormap (applied to each pixel, returns rgb triple)
* see docstring


Scatter plots
-------------

WIP: mp.scatter

See docstring.


Arranging plots
---------------

Horizontal stacking with `&` (or `mp.hstack(*plots)`)

```
print( mp.image(left_image) & mp.image(right_image) )
```

Vertical stacking with `^` (or `mp.vstack(*plots)`)

```
print( mp.image(top_image) ^ mp.image(bottom_image) )
```

Both (recall `&` binds tighter than `^`)

```
print(
    mp.image(top_left_image)    & mp.image(top_right_image)
  ^ mp.image(bottom_left_image) & mp.image(bottom_right_image)
)
```

Automatically arranging a long list of plots into a wrapped grid with
`mp.wrap`:

```
print(mp.wrap(*plots, cols=4))
```

TODO:

* support indexing and slicing.
