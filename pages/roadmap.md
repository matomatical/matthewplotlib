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
* [ ] Finalise operator assignment, including an operator for temporal
  stacking (`tstack`), which the pre-library sketch spelled `|`.

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
* [ ] An opt-in reduced-colour mode, chosen by the caller rather than sniffed
  from the terminal, so that the fallback on a terminal without 24-bit colour
  is ours to control rather than the terminal's to guess. Also saves bytes.
  See `pages/compatibility.md`.
  * [ ] Colormaps that are legible in a reduced-colour mode, since a continuous
    map quantised to 256 colours bands badly.

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
* [x] Audit the escape sequences we emit, preferring ones that do not vary
  between terminals over testing that they did not. Retired `CHA`, `CNL` and
  `ECH`, leaving a VT100-only vocabulary apart from the SGR colours, and pinned
  it with `TestEmittedVocabulary`. See `notes/closed/escape-vocabulary.md`.
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
  * [x] Opt-in frame timing, drift corrected, with the frame's own render and
    write time inside its budget.
  * [x] Opt-in frame collection, reporting the achieved frame rate.
  * [x] A way to print from inside a running animation without corrupting it,
    as a method (`anim.print`) and as a stream (`anim.out`).
* [x] Temporal stacking (`tstack`): animations as first-class values.
  * [x] Indexing, slicing and mapping over frames.
  * [x] Padding frames to a common size, so an animation cannot jitter.
  * [x] Building one straight from an array with a time axis (`animation`).
  * [ ] An operator for temporal stacking. `|` is vstack's; `>>` reads as
    "then". Part of "finalise operator assignment" above.
  * [ ] Per-frame durations as the primitive instead of a single frame rate, so
    that concatenating animations of different rates keeps each part at its own
    speed, and a gif can be written with non-uniform frame delays throughout
    rather than only from a recording.
* [ ] Mapping over the other composites, `hstack` and friends. See
  `notes/mapping-over-composites.md`.
* [ ] Indexing and slicing of plots.
* [ ] Crop plot composition primitive. See `notes/terminal-aware-printing.md`.
  * [ ] By default, clip plots to terminal width and (almost) height, to enable
    terminal-aware printing. A plot wider than the screen is currently
    undefined behaviour, and silently so: see `pages/compatibility.md`.
  * [ ] Have `animate` crop for the caller, and keep cropping correctly when
    the terminal is resized mid-animation — the session already knows the
    terminal's height, and a resize invalidates every cursor calculation the
    next diff is written against.

Advanced furnishings:

* [ ] Axis transformations (e.g. logarithmic scale).
* [ ] Legend construction (API needs thought).
* [x] Text embedded in borders.
* [ ] More border styles: dashed and bold lines, and corner treatments
  (rounded, cut, doubled, crossed).
* [ ] Dashboard meters: circular, vertical and ticking-number variants of
  `progress`, and scrolling text marquees.

Glyphs and fonts:

* [ ] A fallback for terminals whose font lacks braille, which is the densest
  block the library draws with and the one most likely to be missing. See
  `pages/compatibility.md`.
* [ ] Opt-in octants (Symbols for Legacy Computing Supplement, U+1CC00–U+1CEBF,
  new in Unicode 16.0) as an alternative to braille. Octants give the same 2 by
  4 resolution per cell as braille dots but as solid blocks, so they tile
  without the dot-matrix look. Newer than braille and so less widely present in
  fonts, hence opt-in, and worth shipping with instructions for installing or
  patching a font that has them.

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
* [ ] A 3-D `CharArray` (`codes[T,H,W]`), so animations vectorise the way
  plots do rather than being a tuple of them. See `notes/animations.md`.
* [ ] Automatically optimise saved gifs (lossless compression). See
  `notes/gif-size.md`.

More elaborate documentation:

* [x] Links to source code from within documentation.
* [x] Links to mentioned functions/classes/methods/types within documentation
  (automatically linked to relevant release).
* [x] Documentation search.
* [ ] Tutorials and recipes.
* [ ] Freeze documentation with each version.
* [x] Terminal support matrix: the escape sequences and behaviours the library
  relies on, against the terminals people actually use. Tells a reader whether
  their terminal will work, and doubles as the specification of what the
  library is allowed to emit. Published as `pages/compatibility.md`; the
  terminal column is honest that only tmux is under test.
  * [ ] Automate a second terminal, so that tmux's behaviour stops being the de
    facto specification by default. `zmx` is the cheapest second opinion and is
    already installed: it is a session-persistence layer over ghostty's VT
    engine, an implementation wholly independent of tmux's, and
    `zmx history --vt` reads a session's screen back with colour intact, which
    is the capability the harness needs. Spot-checked by hand already; see the
    terminals table in `pages/compatibility.md`. `Xvfb` plus the installed
    `xterm` remains the reference VT if a real terminal is wanted too.

More examples:

* [x] Something to show bar/column plots and histograms.
* [x] Game of life as a demonstration of differential rendering.
* [x] The Amiga Boing Ball, as a demonstration of animations as values: the
  frames are computed as one array and played, rather than printed in a loop,
  and it spins by palette cycling as the original did.
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
