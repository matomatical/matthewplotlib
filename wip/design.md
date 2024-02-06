matt's plotting library (mattplotlib)
=====================================

A Python plotting library that isn't painful.

This readme is a tutorial-by-example of the main interface elements. The
documentation for each function contains more about more configuration
options (todo).

Contents:

* Line plots
* Other plots
* Arranging plots
* Anotating plots
* 3D plots
* Animated plots
* Live-updating plots (dashboards)

Installation
------------

Dependencies

```
pip install numpy einops
```

Installation

```
pip install mattplotlib
```

Import (for the examples in this readme)

```
import numpy as np # used for some examples
import mattplotlib as mp
```


Line plots (todo)
----------

Functional

```
mp.show(mp.line(lambda x: x*x, xrange=(0, 2), samples=5))
```

Locus

```
mp.show(mp.line([(0, 0), (.5, .25), (1, 1), (1.5, 2.25), (2, 4)])))
```

Multiple lines

<!--
```
mp.show(mp.line(lambda x: x, lambda x: x*x, xrange=(0, 2)))
```

Or maybe:
-->

```
mp.show(
    mp.axis(xrange=(0,2), yrange=(0,4)),    # blank axis, stores settings
    + mp.line(lambda x: x, xrange=(0,2)),   # resize to fit prev axis
    + mp.line(lambda x: x*x, xrange=(0,2)), # resize to fit prev axis
    # can also combine with scatter, etc.
)
```

Other plots (todo)
-----------

Scatter

* TODO

Histogram

* TODO

Bar plot

* TODO

Heatmap

* TODO

Scattermap / 2d histogram

Image

* TODO

Vector field (ansi?)

* TODO


Arranging plots
---------------

Horizontal stacking with `&`

```
mp.show(
    mp.line(np.sin, xrange=(0,np.pi))
  & mp.line(np.cos, xrange=(0,np.pi))
)
```

Vertical stacking with `^`

```
mp.show(
    mp.line(np.sin, xrange=(0,np.pi))
  ^ mp.line(np.cos, xrange=(0,np.pi))
)
```

Both (recall `&` binds tighter than `^`)

```
mp.show(
    mp.line(np.sin, xrange=(0,np.pi)) & mp.line(np.sinh, xrange=(0,np.pi))
  ^ mp.line(np.cos, xrange=(0,np.pi)) & mp.line(np.cosh, xrange=(0,np.pi))
)
```

You can also use `mp.hstack(*plots)` and `mp.vstack(*plots)`.

TODO: support indexing and slicing.

TODO: design interface for easily dumping a bunch of similar plots into a grid

Annotating plots (todo)
----------------

Title

* TODO

Legend

* TODO

Colorbar

* TODO

Border

* TODO: spaces, thin (dashed) lines, bold (dashed) lines, double lines
* plus titles along the top or bottom (center or left or right etc)
* corners can be rounded/cut/double/plussed/crossed etc...
* see: https://symbl.cc/en/unicode/blocks/box-drawing/

3D plots (todo)
--------

Surface

```
mp.show(
    mp.surf3(lambda xy: xy[:,0]*xy[:,1], xrange=(-1,1), yrange=(-1,1)),
)
```

Line

```
mp.show(
    mp.axis3(xrange=(0,1), yrange=(0,1), zrange=(0,1)) # angle, etc.
    + mp.ray3([(0, 0, 0), (1, 0, 0)])
    + mp.ray3([(0, 0, 0), (0, 1, 0)])
    + mp.ray3([(0, 0, 0), (0, 0, 1)])
)
```
Animating plots (todo)
---------------

Building animations: Temporal stacking with `|`:

```
a = (
    mp.line(lambda x: np.sin(x+0.0*np.pi)), xrange=(0,np.pi))
  | mp.line(lambda x: np.sin(x+0.5*np.pi)), xrange=(0,np.pi))
  | mp.line(lambda x: np.sin(x+1.0*np.pi)), xrange=(0,np.pi))
  | mp.line(lambda x: np.sin(x+1.5*np.pi)), xrange=(0,np.pi))
)
```


Or with `tstack` (easier for long animations)

```
a = mp.tstack(*[
    mp.line(lambda x: np.sin(x-y)), xrange=(0,np.pi))
    for y in np.arange(0, 2*np.pi)
])
```

TODO: support indexing and slicing

Display (this takes control over the terminal, you need to call into the
library to advance the frames, and tell it when to relinquish control, and
use mp.print (todo: or print to a special IO object) to print during this)

```
import time

with mp.animate(a, loop=True) as anim:
    for t, frame in enumerate(anim):
        # each frame is repainted as it is generated
        time.sleep(0.04) # 25 fps
        # if you want to print you should use a special method
        anim.print("frame", t)
```

Live updating plots (todo)
-------------------

Build a dashboard like a normal plot, plus some specific stuff

Progress meters

* TODO: horizontal progress meters
* TODO: circular meters
* TODO: vertical meters
* TOOD: ticking number meters
* TODO: arbitrary animations as meters, which is also how I can implement the
  above?

Other

* TODO: scrolling text marquees?

Display it somehow, with some simple way to update each component and redraw
the dashboard periodically.

Output formats (todo)
--------------

ansi (print directly to the terminal)

* sure

png/gif

* shouldn't be a problem
* I have to figure out some fonts and stuff
* they will be pixel perfect.

Ti*k*Z src (pgfplots?)

* this will be harder but I am pretty keen for it

svg

* maybe one day

jupyter notebook integration

* over my dead body

Inspiration
-----------

Design:

* https://github.com/asciimoo/drawille
* https://github.com/JuliaPlots/UnicodePlots.jl
* https://github.com/fbbdev/plot
* https://github.com/piccolomo/plotext

Interface:

* ?

Avoid being like these things:

* Matplotlib
* ggplot
