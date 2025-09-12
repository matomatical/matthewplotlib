Matthew's plotting library (matthewplotlib)
===========================================

A Python plotting library that isn't painful.

Perpetual work in progress.

![](examples/lissajous.png)

![](examples/teapot.gif)

Features:

* Image plots.
* Scatter plots.
* Hilbert curves.
* Progress bars.
* Text boxes.
* Arranging plots.
* Various colour maps.
* Export to image.

TODO:

* Histograms and bar charts.
* Axes and labels.
* Use u8 for colours throughout.

Wishlist:

* Animated plots
* 3d visualisation
* Project coordinates onto world map with braille background, different
  projections, including 3d rotation

Installation
------------

Install:

```
pip install git+https://github.com/matomatical/matthewplotlib.git
```

Quickstart
----------

Import:

```
import matthewplotlib as mp
```

Demos:

TODO (for now, see the examples folder).

Related work
------------

Terminal plotting in Python

* Plotille: https://github.com/tammoippen/plotille
* Termgraph: https://github.com/sgeisler/termgraph
* Termplot: https://github.com/justnoise/termplot

Braille art

* Drawille (Python): https://github.com/asciimoo/drawille
* Rsille (Rust): https://github.com/nidhoggfgg/rsille
* Drawille (Lua): https://github.com/asciimoo/lua-drawille
* Drawille (NodeJS): https://github.com/madbence/node-drawille
* Python repo documents ports to various other languages

Terminal plotting in other languages

* Julia https://github.com/sunetos/TextPlots.jl
* GNU plot (dumb terminal mode) http://gnuplot.info/docs_6.0/loc19814.html

