Changelog
=========

In development
--------------

New examples:

* `boids.py`: 2D flocking simulation using `mp.line` to draw short directional
  segments and `mp.rainbow` for coordinated coloring.
* `doomfire.py`: Vectorized implementation of the classic 1997 PSX Doom fire
  effect mapped through a 37-color palette using `mp.image`.
* `lorenz.py`: Animated 3D Lorenz attractors racing side-by-side using
  `mp.scatter3`, `mp.line3`, and `mp.wrap`.
* `sorting.py`: Various sorting algorithms racing in parallel, visualised with
  `mp.columns` and dynamic layout wrapping.

Version 0.6.2
-------------

Changed:

* `parse_colors` can now standardise scalar and RGB arrays of a requested shape
  and apply a colormap. `image` and `animation` use it as their shared
  colour-input path; custom colormaps may accept arrays of any shape provided
  they return `[h,w,3]` or `[t,h,w,3]`.

Fixed:

* Text, titles, and labels reject terminal control characters instead of
  emitting them verbatim. Raw ANSI styling was never structurally supported:
  its invisible bytes were counted as character cells, breaking composition,
  differential redraws, and image rendering.
* `function2` values outside `zrange` and `histogram2` counts above `max_count`
  saturate at the colour-scale endpoints instead of wrapping around with some
  colormaps. All-zero histograms no longer divide by zero, and an explicit
  `max_count` must be positive.
* `image` rejects arrays with non-RGB channel counts instead of producing
  malformed terminal colour sequences or failing later during image rendering.

Examples:

* Attribute examples to their designers.
  * Do you have a cool, standalone matthewplotlib example? Consider sharing!
* Streamline descriptions on example page.
* `quickstart2.py` and `quickstart3.py` amplitude fix.

New examples:

* `chromatic_flow.py`: a periodic incompressible velocity field whose custom
  colormap turns vector direction into hue and speed into brightness.
* `three_body.py`: three equal masses integrated under Newtonian gravity in a
  shared figure-eight orbit, with fading trails.

Version 0.6.1
-------------

New:

* `savegif` takes `palette` and `colors`, choosing how an animation's colours
  are fitted into the palettes a gif stores instead of a colour per pixel.
  * `palette='unified'` (the default) builds one palette for the whole
    animation, so a colour is the same colour in every frame.
  * `palette='per-frame'` gives each frame its own, spending the whole budget
    on each frame separately.
  * `colors` (2 to 256, default 256) is how many colours a palette may hold.
    Fewer means a smaller file.

Changed:

* Gifs are saved in the colours the plots were drawn in, for an animation of
  256 colours or fewer. Previously each frame was reduced on its own, and to
  far fewer colours than a gif allows, which banded smooth colourmaps and let
  still content change colour from frame to frame. Animations with more colours
  than the budget are now reduced once for the whole animation instead.
  * Colourful gifs are larger as a result. `colors=32` asks for small files
    back explicitly. See `notes/gif-size.md` for the measurements.

Fixed:

* `savegif` no longer ghosts an animation with a transparent background.
  Frames after the first kept the pixels of the frames before them, so moving
  content smeared over everywhere it had been.

Examples:

* `teapot.py` orbits once over `num_frames` rather than once every two seconds,
  so a saved gif loops at any length. The showcase gifs are regenerated.

Version 0.6.0
-------------

New:

* `mp.line(series, ...)`: line plots, connecting the points of each series in
  order, drawn with braille dots like `mp.scatter`.
  * Variable `thickness` measured in dots, round caps.
  * NaN breaks lines.
  * Colors interpolate along each segment.
* `mp.line3(series, ...)`: the same for points in space, seen from a camera
  configured as for `mp.scatter3`.

Changed:

* `mp.scatter` and `mp.line` map data onto dots the same way, so a scatter's
  points move by up to one dot: the limits of the data now land on the centres
  of the outermost dots rather than the outer edges of the outermost bins.
* `parse_range` ignores non-finite values, and gives a range to data that
  reaches no distance: a constant series now reports the range it is drawn in
  rather than one of zero width.
* 3d projection moves out of `data` into a new module, `camera`, which gains
  `perspective` alongside `view_matrix` and the two `project3*` functions.
* Color specs move out of `data` into `colors`, as `parse_colors`.

Fixed:

* `mp.scatter3` is no longer a subclass of `mp.scatter`.
  * As a consequence, `mp.axes` no longer takes it (the axis labels would have
    been based on the projection anyway).
* Floats in color arays now have range [0.0, 1.0], as was already the case for
  single float colors.
  * Previously, floats in color arrays were interpreted in [0.0, 255.0].
* Improved color-parsing error handling and error messages.

New examples:

* `vaporwave.py` example: a wireframe landscape scrolling under a banded sun,
  terrain showcasing `mp.line3`.
* `lines.py` example: showcasing `mp.line`, two loss curves, one of them
  measured sparsely and with a stretch missing, and the same spiral drawn with
  four widths of pen.
* `starburst.py` example: a turning rose of rays whose stroke swells from one
  dot to six and back.

Version 0.5.0
-------------

New:

* `mp.tstack(*plots)`: animations as values. The third stacking operation,
  arranging plots in time rather than across the screen. Supports `len`,
  indexing and slicing (`a[0]`, `a[10:20]`, `a[::-1]`), `map` for applying a
  combinator to every frame, `play` for showing it in the terminal, and
  `savegif`. Frames are padded to a common size, so an animation cannot change
  shape while it plays.
* `mp.animation(array)`: a `tstack` straight from an array with a time axis,
  taking what `mp.image` takes with a leading frame index.
* `mp.animate()`: a context manager that runs an animation loop.
  `anim.update(plot)` writes one frame and returns the string it wrote.
  Optionally caps the frame rate (`fps=`), keeps the frames (`record=True`,
  readable as `anim.frames`), and ends quietly on Ctrl-C
  (`stop_on_interrupt=True`). Reports the rate actually achieved as
  `anim.achieved_fps`.
* `anim.print(...)`: print a line above a running animation instead of through
  it. Also available as a stream, `anim.out`, for `print(file=...)`,
  `logging.StreamHandler`, or redirecting `sys.stdout` for the block.
* `mp.tstack(...).savegif(f, fps="achieved")` encodes a recorded animation at
  the frame rate it really ran at, rather than the one that was requested.
* `life.py` example: Conway's Game of Life, demonstrating differential redraw.
* `quickstart3.py` example: `quickstart2.py` with the animation loop handed to
  `mp.animate`.
* `boing.py` example: the Amiga Boing Ball, built with `mp.animation` from a
  computed array of frames, spinning by palette cycling as the original did,
  with the cycling palette shown underneath it.
* A compatibility page (`pages/compatibility.md`): every escape sequence the
  library emits, the terminal behaviours it relies on, the glyph blocks it
  draws with, and which terminals are actually tested.
* `terminal_test.py` example: does your terminal render matthewplotlib
  correctly? Four stages exercising every escape sequence the library can emit,
  each saying what it should look like. Measures the terminal's width and draws
  to it, so that the last stage puts a plot against the right margin.

Change:

* `mp.save_animation(plots, filename, ...)` is retired in favour of
  `mp.tstack(*plots).savegif(filename, ...)`.
* `teacher_student.py` takes a `--log-every` argument, and logs its loss with
  `anim.print`.
* Animated redraws now speak only VT100, apart from the SGR colours. `CHA`
  (absolute column) became a carriage return plus a cursor forward, `CNL` (next
  line) a carriage return plus a cursor down, and `ECH` (erase character)
  written spaces. The screens are identical -- across every example snapshot
  only the byte counts moved -- and the point is `CHA`, whose old use depended
  on it cancelling a deferred wrap, which is the thing terminals disagree about
  most. A full repaint now costs less rather than more; a sparse diff costs one
  byte more. See `notes/closed/escape-vocabulary.md`.

Fix:

* `dashboard.py` no longer raises `NameError` when asked to save an unbounded
  run.
* `axes` no longer paints the `ylabel` down the plot's right-hand border when
  the y tick labels are narrower than `ypad + 1`. The tick gutter now widens to
  make room. An absent `ylabel` no longer erases that border either.
* Correct the Python version classifiers, which still advertised 3.10 and 3.11
  after 0.4.0 raised the requirement to 3.12.

Dev:

* `tests/test_exports.py` checks that everything `plots`, `colormaps` and
  `animations` define is reachable as `mp.something`, deriving the expectation
  from the modules rather than from a list, so adding a feature does not mean
  editing a third file.
* Add module docstrings for `core`, `colors` and `data`, so every module now
  introduces itself in the API reference.
* Escape sequences are now tested against a real terminal (a tmux pane, see
  `tests/test_terminal.py` and `tests/tmux.py`) rather than a hand-written
  emulator, which retires the emulator and makes tmux a development dependency.
  See `notes/terminal-test-backend.md`.
* The example smoke tests are replaced by snapshot tests. Every example is
  replayed into a real terminal print by print and compared against a golden in
  `tests/goldens/`, cell by cell in both glyph and colour, along with the byte
  cost of each print and a digest of the image it saved. Regenerate with `make
  goldens`. See `notes/closed/example-snapshot-tests.md`.
* `life.py` seeds `np.random.seed` rather than `np.random.default_rng`, whose
  stream NumPy does not guarantee across releases. Its initial board, and
  `images/life.gif`, change accordingly.
* `TestEmittedVocabulary` pins the set of escape sequences the renderer is
  allowed to emit, over every path that emits any. It is the executable form of
  the compatibility page, so the page cannot go quietly out of date.

Version 0.4.0
-------------

New:

* Add differential redraw for animated plots: `print(plot - prev)` repaints only
  the cells that changed. Subtract `None` for the first frame.

Change:

* Every string the library returns is now shaped for a plain `print`. In
  particular `clearstr` (`-plot`) must no longer be printed with `end=""`.

Fix:

* `clearstr` (`-plot`) erases only the plot's own rows, rather than everything
  below it on screen, and handles a plot with no rows.
* Stop passing the deprecated `mode` argument to Pillow, which will be an error
  in Pillow 13.
* Declare the Python version actually required (3.12, for `type` aliases); the
  package never supported the 3.10 it claimed.

Version 0.3.8
-------------

Fix:

* Make automatic `wrap` layout work without an attached terminal.

Version 0.3.7
-------------

Fix:

* Fix BIDS colormaps (magma, inferno, plasma, viridis) returning wrong dtype.

Dev:

* Add unit test suite and `make test` target.
* Add integration tests for all examples.
* Add `pytest` to dev dependencies.
* Add `tyro` CLI argument parsing to long-running examples.

Version 0.3.6
-------------

Fix:

* Fix operator precedence bug in `isblank`/`isnonblank` (affected `dstack` overlays).
* Fix `axes.__repr__` returning `"border(...)"` instead of `"axes(...)"`.
* Accept `list` as a valid color input in `parse_color`.
* Fix `save_animation` playing GIFs twice instead of once when `repeat=False`.
* Fix mypy errors.

Version 0.3.5
-------------

New:

* Diverging colormaps divreds, divgreens, divblues.

Fix:

* Fix bug in bar chart layout.

Version 0.3.4
-------------

New:

* Per-column and per-bar colours in column and bar plots.

Fix:

* Fix bug in column/bar spacing implementation.

Version 0.3.3
-------------

New:

* `dstack2` for stacking data, extend axes to more datatypes.
* `teacher_student.py` example

Fix:

* Missing title parameter from border.

Notes:

* Plausibly `dstack2` should be the default and `dstack` should be removed.

Version 0.3.2
-------------

New:

* Transpose parameter for wrap.

Version 0.3.1
-------------

New:

* Axes subplot type. Takes a scatter plot or function2 plot as input, and adds
  axes with labels and ticks. Basic API.
* Animated version of quickstart example.

Version 0.3.0
-------------

Breaking changes:

* scatter and scatter3 take xs, ys, (zs), and color as series tuples in
  positional arguments.
* removed function plot type (since scatter is now much easier to use).

New:

* scatter and scatter3 accept cs, an array of colors (one for each point), and
  plot using them, using weighted averaging to combine plots.
* scatter and scatter3 now accept multiple series at once.
* special series for X/Y/Z axes.
* some new examples (deigned by Gemini 2.5 pro): voronoi, dashboard,
  mandelbrot.

Version 0.2.1
-------------

Fix:

* Regenerate documentation.
* Update version number properly.

Version 0.2.0
-------------

Breaking changes:

* Various argument name changes, especially for colors.
* Inverted `cyber` colormap.
* Move `plots.border.Style` to `core.BoxStyle`.

New:

* Configurable background colour for image rendering.
* 3d scatterplot.
* Discrete colourmaps are now cyclic.
* New discrete colourmaps `tableau`, `nouveau`.
* New border styles.
* Export animations as GIFs.
* New configuration options for bar/column sizes.

Internal:

* Refactor backend to use numpy arrays rather than nested lists.

Version 0.1.2
-------------

Breaking changes:

* Change operators used for shortcuts.
* Rename `fimage` to `function2`.

New:

* New plot types: `bars`, `columns`, `histogram`, `vistogram`, `histogram2`,
  `function`.
* More documentation.
* Generated markdown documentation.
* Additional examples.

Dependencies:

* Make example dependency on `scikit-learn` explicit.

Version 0.1.1
-------------

New:

* Add type annotations.

Dependencies:

* Add `mypy` as a dev dependency.
* Remove dependency on `unscii` (bundle the specific version of the font we
  want).

Internal:

* Refactor from long single-file script to multi-file library.

Version 0.1.0
-------------

Much unstructured development.
