Changelog
=========

In development
--------------

New:

* `life.py` example: Conway's Game of Life, demonstrating differential redraw.
* A compatibility page (`pages/compatibility.md`): every escape sequence the
  library emits, the terminal behaviours it relies on, the glyph blocks it
  draws with, and which terminals are actually tested.

Change:

* Animated redraws now speak only VT100, apart from the SGR colours. `CHA`
  (absolute column) became a carriage return plus a cursor forward, `CNL` (next
  line) a carriage return plus a cursor down, and `ECH` (erase character)
  written spaces. The screens are identical -- across all nineteen example
  snapshots only the byte counts moved -- and the point is `CHA`, whose old use
  depended on it cancelling a deferred wrap, which is the thing terminals
  disagree about most. A full repaint now costs less rather than more (-121
  bytes per frame in `mandelbrot.py`); a sparse diff costs one byte more. See
  `notes/closed/escape-vocabulary.md`.

Fix:

* `axes` no longer paints the `ylabel` down the plot's right-hand border when
  the y tick labels are narrower than `ypad + 1`. The tick gutter now widens to
  make room. An absent `ylabel` no longer erases that border either.
* Correct the Python version classifiers, which still advertised 3.10 and 3.11
  after 0.4.0 raised the requirement to 3.12.

Dev:

* `TestEmittedVocabulary` pins the set of escape sequences the renderer is
  allowed to emit, over every path that emits any. It is the executable form of
  the compatibility page, so the page cannot go quietly out of date.
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
