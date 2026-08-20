Roadmap
=======

Towards version 1
-----------------

Basic plot types:

* [x] Scatter plots.
* [x] Line plots (connect the dots).
* [x] Image plots / matrix heatmaps.
* [x] Function heatmap plots.
* [x] Progress bars.
* [x] Basic bar charts and column charts.
* [x] Histograms.
* [ ] Box plots, filled or outlined, lying either way, with `candles`
  becoming one setting of them. Designed in `notes/box-plots.md`.
* [x] Candlestick plots. See `notes/candlesticks.md`.

Basic plot furnishings:

* [x] Basic text boxes.
* [x] Borders.
* [x] Axis ticks and tick labels for scatter plots.
* [x] Axis labels and titles.
* [ ] Labels and ticks for bar/column charts, histograms, candlesticks, box
  plots. These lay marks out side by side, so the axis along them is a list of
  names rather than an interval of numbers, and there is no way to describe one
  yet. See `notes/categorical-axes.md`, and `bars` and `hist` in
  `notes/reference/myplot.py`.

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
* [ ] Color bars, in any of the four directions. See `notes/colorbars.md`.

Rendering:

* [x] Render to string / terminal with ANSI control codes.
* [x] Export to image with pixel font.

Basic code improvements:

* [x] Split up monolithic file into a small number of modules.
* [ ] Split up plotting module with one file per plot type.
* [x] A window value type for the mapping from data coordinates onto the grid,
  used by axes. See `notes/plot-windows.md`.
* [x] Comprehensive type annotations, static type checking with mypy.
* [ ] Robust input validation and error handling.

Testing:

* [x] Unit tests for core modules (colors, colormaps, data, core).
* [x] Integration tests (all examples run).
  
Documentation:

* [x] Minimal docstrings for everything user-facing.
* [x] Quick start guide.
* [ ] Complete docstrings for modules, constants, etc.
  * [x] Everything exported from the package.
  * [ ] Private parsers and constants too.
* [x] Simple generated markdown documentation on GitHub.
* [x] Simple generated HTML/CSS documentation, hosted on web.

Repository:

* [x] Set up project, installable via git.
* [x] A simple example for the quick-start guide.
* [x] Changelog.
* [x] Version numbering and keep main branch working.
* [ ] List on PyPI (when we agree we have reached version 1).

Advanced features
-----------------

More exotic plot types:

* [x] Calendar heatmap plots:
  * [x] Month-chunked calendar (`calendar`).
  * [x] Github contributions-style continuous calendar (`weeks`).
* [x] Tables (see `notes/tables.md`):
  * [x] List of dictionaries.
  * [x] Dictionary of lists.
  * [x] 2d list with/without header.
  * [x] Configurable format strings.
* [ ] Vector field plots:
  * [x] color (`vfunction2`), including domain colouring of complex functions
    (`cfunction2`).
  * [ ] line (see boids example). Parked on legibility rather than
    machinery; see `notes/quiver-plots.md`.
* [ ] Hilbert curves:
  * [x] Basic Hilbert curves.
  * [ ] Non-square Hilbert curves.
  * [ ] 3d Hilbert curves.
* [ ] World maps:
  * [ ] Some 2d projections.
  * [ ] 3d globe projection.
* [ ] Dashboard meters (similar to `progress`):
  * [ ] circular
  * [ ] vertical
  * [ ] ticking-number
  * [ ] Scrolling text marquees

Advancements of basic plot types:

* [x] Advanced scatter plots:
  * [x] Different colours for each point.
  * [x] Multiple point clouds on a single scatter plot.
  * [x] 3d scatter plots.
* [x] Advanced line plots (See `notes/lines.md`).
  * [x] Configurable stroke thickness, with round caps and filled joins.
  * [x] Gaps, where a coordinate is non-finite.
  * [x] 3d line plots / wireframes.
* [ ] Advanced bar charts:
  * [x] Bar/column charts with configurable sizes and spacing.
  * [ ] Bar/column charts with other alignments.
  * [x] Bar/column charts with individual colours.
  * [ ] Negative values in bar/column charts.
* [ ] Advanced heatmaps:
  * [ ] RGB-channel 2d histograms (see `hist2d_rgb` in
    `notes/reference/myplot.py`).
* [ ] Advanced image options:
  * [ ] Integer-factor down- and upsampling.
  * [ ] Normalisation colormaps.
* [ ] Advanced text formatting parser (`rich=True` or something):
  * [ ] Bold, italic, underline, based on markdown?
  * [ ] Some way of specifying colors, html-style tags?

Advanced plot arrangement:

* [x] Animation context manager (`animate`), owning the printing state. See
  `notes/animations.md`.
  * [x] Opt-in frame timing, drift corrected, with the frame's own render and
    write time inside its budget.
  * [x] Opt-in frame collection, reporting the achieved frame rate.
  * [x] A way to print from inside a running animation without corrupting it,
    as a method (`anim.print`) and as a stream (`anim.out`).
* [x] Temporal stacking (`tstack`): animations as first-class values.
  * [x] Indexing, slicing and mapping over frames.
  * [x] Padding frames to a common size, so an animation cannot jitter.
  * [x] Building one straight from an array with a time axis (`animation`).
  * [ ] Operator for temporal stacking.
  * [ ] Per-frame durations.
* [ ] Mapping over the other composites, `hstack` and friends. See
  `notes/mapping-over-composites.md`.
* [ ] Indexing and slicing of plots.
* [ ] Crop plot composition primitive. See `notes/terminal-aware-printing.md`.
  * [ ] Clip plots to terminal width and (almost) height (see
    `docs/src/compatibility.md`).
  * [ ] `animate` context manager should handle this, and window resizes.

Advanced furnishings:

* [ ] Axis transformations (e.g. logarithmic scale).
* [ ] Axis segments (see `notes/axis-series.md`).
* [x] Axes on any subset of the four sides, each blank, ruled, or ruled with
  ticks and labels, so that a colorbar can be labelled without a full border.
  See `notes/axes-sides.md`.
* [x] Text embedded in borders.

Advanced rendering:

* [x] Export animations to gifs, at the requested or the achieved frame rate.
  * [x] Control over the palette sharing and size. See `notes/gif-size.md`.
  * [x] Automatically optimise saved gifs (lossless compression). See
    `notes/gif-size.md`.

Backend improvements:

* [x] Upgrade Char backend to use arrays of codepoints and colors.
  * [x] Vectorised composition operations.
  * [x] Vectorised bitmap rendering.
* [x] Intelligent ANSI rendering (only include necessary control codes and
  resets, e.g., if several characters in a row use the same colours).
* [x] Differential rendering with shortcut `plot_new - plot_old`.
* [ ] Vectorised animations (3-D `CharArray`, `codes[T,H,W]`). See
  `notes/animations.md`.

More elaborate documentation:

* [x] Links to source code from within documentation.
* [ ] Links to mentioned functions/classes/methods/types within documentation
  (automatically linked to relevant release).
* [x] Documentation search.
* [x] Versioned documentation.
* [ ] Logo and favicon.
* [ ] Link previews (Open Graph tags).
* [x] Terminal support matrix / compatibility docs.

Advanced testing:

* [x] Tests that drive a virtual terminal for testing ANSI control codes (see
  `notes/terminal-test-backend.md`).
  * [x] hand-written emulator (retired)
  * [x] tmux
* [ ] More virtual terminal test backends
  * [ ] zmx (backed by ghostty-vt)
  * [ ] pyte and friends
  * [ ] more?
* [x] Test the set of escape sequences we emit.
* [x] Regression testing for str output and image output of examples (See
  `notes/closed/example-snapshot-tests.md`).

Support non-24-bit-colour modes:

* [ ] Reduced-colour modes (See `docs/src/compatibility.md`.)
* [ ] Colormaps that are legible in a reduced-colour mode.

Advanced examples:

* [x] Sort by feature or maybe style.
* [ ] Systematically ensure there is at least one example per major plot type
  or feature.
  * [ ] Need more dashboard and progress bar demos.
* [ ] Advanced example categories:
  * [ ] Real-time signal processing (music visualiser, webcam with filters?)
  * [ ] Games / real-time input collection with animation (snake? hexagon?)
* [ ] Specific example requests:
  * [ ] 2d/3d water physics simulation.
  * [ ] Curl field, floating specs, wind simulation.
  * [ ] Playable super hexagon.
  * [ ] Lightbike game, AI bots vs 0,1,2 players.
  * [ ] Enhance vaporwave with procgen city skyline, motorway, light cycle;
    plus special edition red/white/black color scheme?
  * [ ] Blowup kinda like https://far.in.net/blowing-up
  * [ ] 3b1b fourier analysis video plots.
  * [ ] Markdown renderer.

Longer term
-----------

Further plot enhancements:

* [ ] More advanced line plots:
  * [ ] Error bars on line plots.
  * [ ] Fill plots.
  * [ ] Dashed/dotted strokes (allows animated lines if we advance the start
    index)
* [ ] Advanced wireframes:
  * [ ] An explicit edge list, so a mesh with shared vertices is projected once
    per vertex rather than once per wire through it.
  * [ ] Hidden line removal, or some depth ordering. Colouring by depth is the
    stand-in for now.
* [ ] Legend construction (API needs thought).
* [ ] Better and more flexible border parameterisation.

More rendering formats:

* [ ] Render plots to SVG (keep console aesthetic).
* [ ] Render plots to PDF (keep console aesthetic).
* [ ] Render plots to TikZ/pgfplots source.

Advanced glyphs and fonts:

* [ ] Fallback for terminals whose font lacks braille.
* [ ] Opt-in octants (Symbols for Legacy Computing Supplement, U+1CC00–U+1CEBF,
  new in Unicode 16.0) as an alternative to braille.
  * [ ] Instructions for installing or patching a font that has them.
* [ ] Other font extensions, creating new density and shape possibilities?

More elaborate documentation:

* [ ] Tutorials and recipes.

Future design directions.

* [ ] Better tools for letting AI agents see the results?
* [ ] Better performance and/or cleaner code by switching backend to JAX?
* [ ] Reactive plots.
* [ ] Manim import/export? Could that make sense?

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
* GNU plot (dumb terminal mode) http://gnuplot.info/docs_6.0/loc19814.md

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
