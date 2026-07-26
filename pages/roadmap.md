Roadmap
=======

Roadmap to version 1
---------------------

Basic plot types:

* [x] Scatter plots.
* [ ] Line plots (connect the dots).
* [x] Image plots / matrix heatmaps.
* [x] Function heatmap plots.
* [x] Progress bars.
* [x] Basic bar charts and column charts.
* [x] Histograms.

Basic plot furnishings:

* [x] Basic text boxes.
* [x] Borders.
* [x] Axis ticks and tick labels for scatter plots.
* [x] Axis labels and titles.
* [ ] Labels and ticks for bar/column charts and histograms. See `bars` and
  `hist` in `notes/reference/myplot.py` for a reference.

Basic plot arrangement:

* [x] Horizontal and vertical stacking.
* [x] Naive layering plots on top of each other.
* [x] Automatically wrapping plots into a grid.
* [ ] Finalise operator assignment.

Styling plots with colors:

* [x] Basic colormaps.
* [x] BIDS colormaps.
* [x] Rainbow colormap.
* [x] Cyberpunk colormap.
* [x] Discrete colour palettes.

Specifying colors:

* [ ] Consistent API for color specification.
* [ ] Configurable colour scales and normalisation.
* [ ] Color bars, vertical or horizontal.

Rendering:

* [x] Render to string / terminal with ANSI control codes.
* [x] Export to image with pixel font.

Basic code improvements:

* [x] Split up monolithic file into a small number of modules.
* [ ] Split up plotting module with one file per plot type.
* [x] Comprehensive type annotations, static type checking with mypy.
* [ ] Robust input validation and error handling.

Testing:

* [x] Unit tests for core modules (colors, colormaps, data, core).
* [x] Integration tests (all examples run).
* [x] Adopt a more well-tested virtual terminal for testing ANSI control codes:
  drive a real terminal (tmux) and retire the hand-written emulator in the
  tests. See `notes/terminal-test-backend.md`.
* [ ] Audit the escape sequences we emit, preferring ones that do not vary
  between terminals over testing that they did not.
* [x] Snapshot testing for str output and image output regression detection:
  every example is replayed into a real terminal print by print and compared
  against a golden, cell by cell, along with the byte cost of each print and a
  digest of the image it saved. See `notes/closed/example-snapshot-tests.md`.

Documentation:

* [x] Minimal docstrings for everything user-facing.
* [x] Quick start guide.
* [x] Complete docstrings for modules, constants, etc.
* [x] Simple generated markdown documentation on GitHub.
* [x] Simple generated HTML/CSS documentation, hosted on web.

Repository:

* [x] Set up project, installable via git.
* [x] A simple example for the quick-start guide.
* [x] Changelog.
* [x] Version numbering and keep main branch working.
* [ ] List on PyPI.

Advanced features roadmap
-------------------------

More plot types:

* [x] Advanced scatter plots:
  * [x] Different colours for each point.
  * [x] Multiple point clouds on a single scatter plot.
  * [x] 3d scatter plots.
* [ ] Advanced line plots:
  * [ ] Error bars on line plots.
  * [ ] Fill plots.
* [ ] Advanced bar charts:
  * [x] Bar/column charts with configurable sizes and spacing.
  * [ ] Bar/column charts with other alignments.
  * [x] Bar/column charts with individual colours.
  * [ ] Negative values in bar/column charts.
* [ ] Hilbert curves:
  * [x] Basic Hilbert curves.
  * [ ] Non-square Hilbert curves.
  * [ ] 3d Hilbert curves.
* [ ] World maps:
  * [ ] Some 2d projections.
  * [ ] 3d globe projection.
* [ ] Advanced heatmaps:
  * [ ] RGB-channel 2d histograms (see `hist2d_rgb` in
    `notes/reference/myplot.py`).
  * [ ] Integer-factor down- and upsampling for `image`.
* [ ] Other:
  * [ ] Calendar heatmap plots (see calendar heatmap example for now).
  * [ ] Candlestick plots.
  * [ ] Box plots.
  * [ ] Vector field plots.

Advanced plot arrangement:

* [x] Animation context manager (`animate`), owning the printing state. See
  `notes/animations.md`.
  * [x] Opt-in frame timing, drift corrected.
  * [x] Opt-in frame collection, with the achieved frame rate.
  * [x] A way to print from inside a running animation without corrupting it.
  * [ ] Decide whether catching `KeyboardInterrupt` should be the default.
  * [ ] Routing third-party prints, by redirecting stdout for the block.
* [x] Temporal stacking (`tstack`): animations as first-class values.
  * [x] Indexing, slicing and mapping over frames.
  * [ ] Padding frames to a common size, so an animation cannot jitter.
  * [ ] An operator, folded into "finalise operator assignment" below.
* [ ] Mapping over the other composites, `hstack` and friends. See
  `notes/mapping-over-composites.md`.
* [ ] Indexing and slicing of plots.
* [ ] Crop plot composition primitive. See `notes/terminal-aware-printing.md`.
  * [ ] By default, clip plots to terminal width and (almost) height, to enable
    terminal-aware printing.

Advanced furnishings:

* [ ] Axis transformations (e.g. logarithmic scale).
* [ ] Legend construction (API needs thought).
* [x] Text embedded in borders.
* [ ] More border styles: dashed and bold lines, and corner treatments
  (rounded, cut, doubled, crossed).
* [ ] Dashboard meters: circular, vertical and ticking-number variants of
  `progress`, and scrolling text marquees.

Advanced rendering:

* [x] Export animations to gifs, at the requested or the achieved frame rate.
* [ ] Render plots to SVG (keep console aesthetic).
* [ ] Render plots to PDF (keep console aesthetic).
* [ ] Render plots to TikZ/pgfplots source.

Backend improvements:

* [x] Upgrade Char backend to use arrays of codepoints and colors.
* [x] Vectorised composition operations.
* [x] Vectorised bitmap rendering.
* [x] Intelligent ANSI rendering (only include necessary control codes and
  resets, e.g., if several characters in a row use the same colours).
* [x] Faster animated plot redraws (e.g., differential rendering with shortcut
  `-`).
* [ ] Clean up backend code e.g. using JAX PyTrees and vectorisation.
* [ ] Automatically optimise saved gifs (lossless compression). See
  `notes/gif-size.md`.

More elaborate documentation:

* [x] Links to source code from within documentation.
* [x] Links to mentioned functions/classes/methods/types within documentation
  (automatically linked to relevant release).
* [x] Documentation search.
* [ ] Tutorials and recipes.
* [ ] Freeze documentation with each version.
* [ ] Terminal support matrix: the escape sequences and behaviours the library
  relies on, against the terminals people actually use. Tells a reader whether
  their terminal will work, and doubles as the specification of what the
  library is allowed to emit. See `notes/terminal-test-backend.md`.

More examples:

* [x] Something to show bar/column plots and histograms.
* [x] Game of life as a demonstration of differential rendering.
* [ ] Webcam with ffmpeg

Future design directions.

* [ ] Reactive plots.

Related work
------------

Matthewplotlib aspires to achieve a similar levels of functionality as covered
by the following projects.

Terminal plotting in Python:

* Plotext: https://github.com/piccolomo/plotext
* Plotille: https://github.com/tammoippen/plotille
* Termgraph: https://github.com/sgeisler/termgraph
* Termplot: https://github.com/justnoise/termplot

Terminal plotting in other languages:

* Julia https://github.com/JuliaPlots/UnicodePlots.jl
  * See also https://github.com/sunetos/TextPlots.jl
* C++ https://github.com/fbbdev/plot
* R https://github.com/cheuerde/plotcli
  * See also https://github.com/bbnkmp/txtplot/ and
    https://github.com/geotheory/r-plot
* GNU plot (dumb terminal mode) http://gnuplot.info/docs_6.0/loc19814.html

Braille art:

* Drawille (Python): https://github.com/asciimoo/drawille
* Rsille (Rust): https://github.com/nidhoggfgg/rsille
* Drawille (Lua): https://github.com/asciimoo/lua-drawille
* Drawille (NodeJS): https://github.com/madbence/node-drawille
* Python repo documents ports to various other languages

TODO: Checklist of specific interesting target features that are and are not
implemented.

Other Python plotting libraries, most of which offer some level of
interactivity that there are no plans to replicate.

* Matplotlib https://github.com/matplotlib/matplotlib
* Seaborn https://github.com/mwaskom/seaborn
* Plotly.py https://github.com/plotly/plotly.py
* Pygal https://github.com/Kozea/pygal
* Bokeh https://github.com/bokeh/bokeh
* Altair https://github.com/vega/altair
  * Declarative API
* plotnine https://github.com/has2k1/plotnine
  * Compose subplots with `|` and `/`
