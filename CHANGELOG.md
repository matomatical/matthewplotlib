Changelog
=========

In development
--------------

New examples:

* `planisphere.py`: the whole sky over Oxford as a turning star chart, with
  the sun computed alongside the stars so twilight washes them out and dusk
  hands them back.

Version 0.7.1
-------------

New:

* `crop`: limit a plot to a maximum height and width, keeping the top left
  rectangle. The sizes default to what the attached terminal can show.

Internal:

* New `terminal` module, measuring the attached terminal on behalf of the plots
  and the animations that had been duplicating it between them.

Version 0.7.0
-------------

New:

* `heatmap`: a replacement for many uses of `image` for displaying a grid of
  values through a colormap. It normalises the values onto the colormap rather
  than asking the caller to scale them first. A pre-computed grid of colours,
  or palette indices, or of values already scaled onto the range 0.0 to 1.0 can
  still be an `image`.
* Finally, colour bars! Any way you want them!
  * `colorbar`: a gradient heatmap showing a range of colours.
  * `Direction`: which way along the screen a colorbar runs. See
    the `colorbars` note.
* Vector colormaps and the plots that use them:
  * `chroma`: a colormap over the plane, turning a vector's direction into hue
    and its magnitude into brightness. Vectors may be spelled as complex
    numbers or as pairs.
  * `domain`: a colormap for domain colouring, turning a complex number's
    phase into hue and its modulus into lightness. A zero is black and a pole
    is white, and a dark contour ring falls at every doubling of the modulus,
    so the rings count the order of a zero or a pole. Its scale is absolute
    rather than normalised, since the modulus is part of what it is showing.
  * `vfunction2`: a colour field over a rectangle, sampling a vector-valued
    function of the plane. `vrange` scales the magnitudes into the unit disc,
    from zero to the largest by default, leaving the directions alone.
  * `cfunction2`: a domain colouring over a rectangle of the complex plane,
    sampling a complex-valued function of one complex variable.
* Calendar plots and support:
  * `calendar`: a heatmap of values observed on dates, drawing a block per
    month with a cell per day and wrapping the months into a grid.
  * `weeks`: the same days as one unbroken strip, a column per week and a row
    per weekday, captioned with the months and years along the top. A `width`
    wraps a strip too long for the terminal onto further bands, each captioned
    again.
  * `DateSeries`: the forms dated data arrives in, accepted by `calendar`. A
    mapping from dates to values, separate sequences of dates and values, or one
    date standing in for the consecutive days from there. Dates may be spelled as
    `datetime` dates or datetimes, NumPy `datetime64`s, or ISO 8601 strings.
* `table`: a grid of values, formatted into aligned columns and ruled. Takes
  a list of dicts, a dict of lists, or a 2d array, and sizes each column to
  what is in it.
  * A float shows four significant figures and a column of numbers aligns
    right so its digits line up, until a format or an alignment says otherwise,
    given for the whole table, per column, or per column name.
  * Each of its eight rules---`toprule`, `midrule`, `rowrule` and `bottomrule`
    across, `leftrule`, `indexrule`, `colrule` and `rightrule` down---takes a
    `Rule` of its own, defaulting to a line above, a double line under the
    header and a line below.
  * Cells take colours one at a time, so a table shaded by its own values reads
    as a heatmap whose numbers can still be read off exactly. See
    the `tables` note.
  * `TableData`: the forms a grid of values arrives in, accepted by `table`. A
    list of dicts, a dict of lists, or a sequence of rows or 2d array, the
    first two naming their columns and the last not.
  * `Rule`: what a table draws between its cells---`"skip"` for nothing at
    all, `"blank"` for a row or column of space, `"single"` for a light line
    or `"double"` for a double one.
  * `Align`: where a value sits in its cell, `"left"`, `"center"` or
    `"right"`.
* Box plots and candle plots:
  * `boxes`: a box plot, one box per group of samples. The box spans the first
    and third quartiles and is divided at the median, whiskers reach the
    furthest sample within `whisker_iqrs` times the interquartile range of the
    quartiles, and every sample beyond them is drawn as a point. Takes raw
    samples, ragged groups allowed, and works the quartiles out itself.
    * Horizontal and vertical modes. Which is which? We label by the direction
      of the box.
    * Outlined (unicode box-drawing characters) and filled (unicode eigths).
      Filled has slightly higher resolution but requires a solid background.
  * `candles`: a candlestick chart, one candle per period, its body spanning
    the opening and closing values and its wick reaching out of the body to the
    high and the low. Bodies are coloured by whether the period closed above or
    below where it opened, and land on an eighth of a character cell.
  * `Orientation`: which way a mark lies, and so which way its value axis
    runs. Both take one, as `box_direction` and `candle_direction`.
* `window`: the interval of data a plot covers on each axis, and the rectangle
  of character cells it covers them with. Every 2d plot carries one, and it
  provides the conversions from data coordinates to the grids of dots and
  pixels the plots are drawn in.
* A range given descending inverts its axis: `xrange=(1, 0)` mirrors a plot
  left to right and a descending `yrange` turns it over. The plots that place
  points and sample functions did this already, by arithmetic rather than by
  design; `histogram2` now does it too, binning with its edges ascending and
  turning the counts around afterwards, so that the same data always lands in
  the same bin.

New examples:

* `axes_gallery.py`: demonstrating different ways of attaching axes to some
  data.
* `candlesticks.py`: a simulated price series as candles, one column to a
  period.
* `colorbars.py`: colour scales, and the bars that stand for them.
* `commit_heatmap.py`: a year of commits to this repository as a strip of
  weeks.
* `domain_coloring.py`: six complex functions, painted onto their own input
  plane.
* `phase_portrait.py`: six planar vector fields, as colour fields.
* `tables.py`: a hyperparameter sweep reported in tables.
* `globe.py`: a spinning Earth. See the `world-maps` note.
* `world_map.py`: great-circle flight routes in different projections.

Changed:

* `axes` draws each of its four sides independently. Every side takes a `Side`:
  `"crop"` for nothing at all, `"pad"` for a blank cell, `"rule"` for a line,
  or `"label"` for a line with ticks at its ends and its coordinate's limits
  outside it. Left unspecified, each coordinate the plot carries is labelled
  once, below it and to its left, and the remaining sides are ruled when the
  plot carries both coordinates and dropped when it carries only one. So a
  gradient with one coordinate is labelled along one side and left alone on the
  others.
* `axes` takes a `LineStyle`---`LIGHT`, `HEAVY`, `ROUND` or `DOUBLE`---rather
  than a `BoxStyle`. The characters where its rules meet, and the ticks
  reaching out towards its labels, are derived from which sides are drawn
  rather than written down, which an eight-character `BoxStyle` cannot
  express. `border` still takes a `BoxStyle`, with all of its styles.
* `axes` no longer garbles the limits of a plot too narrow to hold them, or
  raises when such a plot is also given an axis name. The limits use the whole
  width available, including the gutter under the y labels, and are replaced
  by hashes if even that will not fit them, as a spreadsheet does. Axis names
  are truncated as before.
* `axes` writes a title into the north side when that side is blank or ruled,
  and gives it a row of its own otherwise.
* `axes` refuses a plot carrying no coordinates at all, rather than silently
  framing it. `border` is for that.
* `BoxStyle` no longer offers `LIGHTX`, `HEAVYX` or `LOWERX`. They existed so
  that a border could carry the ticks an `axes` needed, and an `axes` derives
  its own now.
* `function2` takes its colour scale as `vrange` rather than `zrange`, the name
  the rest of the library already gave the interval of values a plot covers.
* `vrange` unification and changes:
  * A `vrange` is a pair of numbers or nothing at all. `bars`, `columns`,
    `calendar` and `weeks` used to accept a single number too, meaning zero up
    to it; write `(0, hi)` instead. `histogram`, `vistogram` and `histogram2`
    keep their own `max_count`, which is that shorthand under a name that says
    what it is counting.
  * A plot keeps the interval it settled on as `plot.vrange`, one pair, rather
    than as `plot.vmin` and `plot.vmax`.
  * `image` keeps no interval, since its data is already colours or already
    scaled, so a colorbar for one has to be told the interval outright.
  * A `vrange` the caller wrote that covers no interval is an error, rather
    than quietly colouring everything at the bottom of the scale or dividing by
    something close enough to nothing. Every plot that measures its values
    against an interval says so the same way, whether that interval becomes a
    colour scale or a length along the screen: `heatmap` and the plots built on
    one, `calendar`, `weeks`, `vfunction2`, `bars`, `columns`, `boxes` and
    `candles`.
  * An interval *inferred* from values that are all the same still puts them
    all at the bottom of a colour scale, there being nothing else it could
    mean. The plots that draw a position along the interval---`boxes` and
    `candles`---have nowhere to put them instead, and say so.
  * A value that is not a number is left out of an inferred interval, and comes
    out at the bottom of the scale wherever it appears: at the bottom of the
    colormap for a `heatmap`, and as a bar or column of zero width for `bars`
    and `columns`.
* A sample that is not finite is left out of a `boxes` summary, as a
  measurement that was not made, rather than shifting the quartiles or
  counting as a point beyond the whiskers. A group with no finite samples at
  all is an error. `candles` instead refuses a value that is not a number
  outright, a period with an unknown high having no candle to draw.
* `text` takes a `width`, a `height` and an `align`. The width and the height
  are the least it may take, so a text plot can be held to a size larger than
  its text, and the alignment says where each line sits in the width---which
  is what a `table` cell is, and what it now uses.
* The examples that had values on a colour scale draw them with `heatmap`,
  rather than scaling them into the unit interval by hand first. The ones
  drawing colours, or palette indices for a discrete colormap, still use
  `image`, which is what it is for.
* `time_series_histogram.py` gives its 2d histogram the colorbar it had a
  standing TODO for, along the foot of the panel so that all three panels stay
  the same width.

Fixed:

* `text` documented its foreground colour argument under the wrong name.
* `text("")` is a plot of no rows rather than a `ValueError` from taking the
  longest of no lines. `text("\n")`, which has one empty line, is still a plot
  of one row.
* `hilbert`'s repr closes its parenthesis.
* One value that is not a number no longer draws every bar of a `bars` or
  `columns` chart full. It poisoned the largest value, and so the interval, and
  so every bar, which then saturated at the top of a scale that was itself not
  a number.
* `colorbar` says what is wrong with a plot whose values are all the same,
  rather than reporting a `vrange` covering no interval against an argument the
  caller never wrote. Such a plot has one colour and no axis to label it along,
  so there is no bar to draw for it.
* `plot.xrange` and `plot.yrange` have moved onto the window, as
  `plot.window.xrange` and `plot.window.yrange`.
* `axes` and `dstack2` accept any plot carrying a window, rather than a listed
  union of plot types.
* `image` accepts an `xrange` and a `yrange`, so that an image can be given
  axes or overlaid on another plot. Without them it carries no coordinates, as
  before. An image with an odd number of pixel rows cannot be given them, since
  it half-fills its last character row.
* Plots report their window in their reprs, rather than separately spelling out
  their dimensions and their ranges.
* `dstack2` requires its plots to share one window---the same intervals in the
  same number of character cells---and refuses mismatches, plots without
  coordinates, and an empty stack with a `ValueError`. It used to compare only
  the intervals, and to do so with assertions, which vanish under `python -O`.
* `function2` shows the value of the function at the centre of each grid
  square, rather than at its lower left corner, so that the picture is no
  longer biased half a square towards the low end of each range. With
  `endpoints=True` it still samples the ends of both ranges exactly.

Version 0.6.3
-------------

Documentation:

* Rebuilt library documentation using MkDocs and a modification of the
  [terminal](https://ntno.github.io/mkdocs-terminal/) theme.
* The website is now versioned, one directory per release, published to the
  `gh-pages` branch rather than committed to the repository. The site root
  redirects to the newest release, and page addresses have changed to suit:
  the API reference is now at `/latest/api/`, the quickstart at
  `/latest/quickstart/`, and so on.
* The website now carries every released version, each at its own address,
  with a menu in the top bar for moving between them. The root redirects to
  the newest.
* Reorganise example page by topic.

New examples:

* `boids.py`: 2D flocking simulation using `mp.line` to draw short directional
  segments and `mp.rainbow` for coordinated coloring.
* `doomfire.py`: Vectorized implementation of the classic 1997 PSX Doom fire
  effect mapped through a 37-color palette using `mp.image`.
* `lorenz.py`: Animated 3D Lorenz attractors racing side-by-side using
  `mp.scatter3`, `mp.line3`, and `mp.wrap`.
* `sorting.py`: Various sorting algorithms racing in parallel, visualised with
  `mp.columns` and dynamic layout wrapping.

Documented:

* The type aliases the modules introduce and then left out of the reference:
  `Series` and `Series3`, `ColorLike`, and the colormap types.
* The `xaxis`, `yaxis` and `zaxis` series, and `unscii16.bitmaps`.

Fixed:

* `doomfire.py` seeds its randomness, so the example draws the same fire on
  every run, as the other stochastic examples do.

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
    back explicitly. See the `gif-size` note for the measurements.

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
  byte more. See the `escape-vocabulary` note.

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
  See the `terminal-test-backend` note.
* The example smoke tests are replaced by snapshot tests. Every example is
  replayed into a real terminal print by print and compared against a golden in
  `tests/goldens/`, cell by cell in both glyph and colour, along with the byte
  cost of each print and a digest of the image it saved. Regenerate with `make
  goldens`. See the `example-snapshot-tests` note.
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
