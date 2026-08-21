# Visual regression tests for the examples

Investigated and built 2026-07-26 (Matthew + Claude), after the tmux test
backend landed. This is the reasoning behind `tests/examples.py`,
`tests/test_examples.py` and `tests/goldens/`, which replaced the example smoke
tests. See "What was built", below, for the result.

## The problem

The old `tests/test_examples.py` ran each example as a subprocess and asserted
three things: exit code zero, stdout non-empty, and (where the example saves)
the file exists and is non-empty. Its own docstring carried the two TODOs this
note answers. It cost about 12.8 s of the suite's 17.3 s and could not
distinguish a correct plot from a plot of the wrong thing, in the wrong place,
in the wrong colour. Every regression the differential rendering work
introduced and fixed would have passed it.

The examples are the only place where the whole library runs end to end on
realistic input: eighteen scripts covering every plot type, every composition
operator, most colormaps, and -- crucially -- long sequences of real
differential redraws, which the property sweeps in `tests/test_terminal.py`
only reach with small random cases.

## What there is to snapshot

An example's output passes through two disjoint renderings of the same
`CharArray`:

    example -> plot.chars ->  to_ansi_str / to_ansi_diff_str  -> terminal screen
                          ->  to_rgba_array                   -> PNG / GIF

So there are four things a test could pin down, and they are not
interchangeable:

* **text** -- the glyph at each cell of the terminal screen;
* **colour** -- the fg/bg pair at each cell of that screen;
* **bytes** -- how many bytes each `print` cost to say it;
* **image** -- the pixels `saveimg`/`save_animation` produce.

The image is a pure function of the same `CharArray` the terminal shows, so it
is not an independent check on *content*; its unique coverage is the pixel font
lookup and RGBA composition in `CharArray.to_rgba_array`. The byte count is the
only one of the four that sees the difference between saying something
efficiently and saying it at all.

## What was measured

**The examples are deterministic.** Every example run twice as a subprocess,
comparing stdout byte for byte: 17 of 18 identical. `dashboard.py` differs, and
always would -- it plots live `psutil` readings, and its width depends on the
host's core count. Nothing else consults the clock for content; the
`time.sleep` calls only pace the animations, and `life.py`'s cost series is
computed from `len(update)`, not from elapsed time.

**Replaying stdout through tmux is nearly free.** Feeding an example's captured
stdout into a pane and capturing the screen costs about 10 ms per example --
0.18 s for all 17 deterministic ones, against the 12.8 s the suite already
spent running them. Feeding is not a bottleneck at scale either: a single
800 kB write to the pane tty completed without a partial write, and the 178 kB
of `mandelbrot.py` fed in 20 ms.

**Snapshotting per frame is affordable, and necessary.** Running the examples
in process, with `time.sleep` stubbed and stdout redirected to a recorder,
takes 5.3 s for all 17 (against 12.8 s as subprocesses) and yields each `print`
payload separately. Replaying all of them frame by frame costs about 1 s.

Necessary, because the last screen of an animation is not representative. The
zoom in `mandelbrot.py` is `geomspace(3, 1e-12, num_frames)`, so at the
`--num-frames 5` the tests use, the final frame is fully inside the set: a
solid black rectangle. A last-screen-only golden for mandelbrot would assert
almost nothing. The intermediate frames are where the picture is.

**The image path is exactly reproducible.** Each example saved twice: the
PNG/GIF file bytes are identical, and so are the decoded pixels. All 17 come to
233 kB at test parameters and 4.4 s to generate. (Compare decoded pixels rather
than file bytes anyway: PNG is lossless so the two agree, but a GIF's palette
depends on Pillow's quantiser and is the kind of thing that shifts under a
version bump without any pixel changing.)

**Golden size, for all frames of all examples**, in four candidate formats,
measured before the pane sizes were settled:

    plain text only                      75 kB
    text + palette-indexed colour grid  318 kB
    text + colour grid, gzipped          92 kB
    raw ANSI                            627 kB

## The mutation study

Four single-line mutations to the library, scored against all four layers, over
seven examples (`demo`, `colormaps`, `life`, `quickstart2`, `teapot`,
`mandelbrot`, `voronoi`), with the pane 210 columns wide. Cells count how many
of the seven caught it:

| mutation | text | colour | bytes | image |
|---|---|---|---|---|
| `reset_colour` made a no-op (the BCE bug) | -- | 3 | 4 | -- |
| last-column cursor arithmetic restored (the original bug) | -- | -- | 1 | -- |
| `viridis` index shifted by one | -- | 1 | 1 | 1 |
| one corner glyph changed in `BoxStyle.ROUND` | 3 | -- | -- | 3 |

Three conclusions, each of which changed the design:

**A text-only golden is blind to the bug this test backend exists for.**
Deleting the `reset_colour` call before an erase -- the bug
the `terminal-test-backend` note records as caught by 6 new tests and 0 old
ones -- changes not one glyph in any example. Only the colour layer and the
byte counts see it. So the colour layer is not optional.

**Byte counts are a sharper instrument than expected.** They caught the BCE bug
in more examples than the colour layer did, and they were the *only* layer to
catch the last-column cursor bug at this pane width. That is not incidental: an
encoder change that leaves the screen identical is invisible to every other
layer, and "the differential renderer stopped differentiating" is a real
regression that no screen snapshot can see.

**A generous pane hides the entire right-margin bug class.** Re-run with each
pane sized to the example rather than to 210 columns, the cursor mutation
corrupts `teapot`'s visible screen and is caught by text *and* colour. The
deferred wrap at the last column is only exercised when a plot actually reaches
the last column, which is the case a user in a snug terminal hits and a roomy
test pane never does. Pane size is a coverage parameter, not a convenience.

The image layer caught nothing that text and colour missed, which is what the
pipeline above predicts.

## What was built

`tests/examples.py` holds the machinery and the table of examples;
`tests/test_examples.py` is the pytest surface; `tests/goldens/*.txt` are the
snapshots. Per example:

1. **Run it in a subprocess** via `runpy.run_path(..., run_name="__main__")`,
   so it goes through its own `tyro.cli` call and the command line stays under
   test, with stdout redirected to a `Recorder`, `time.sleep` stubbed to a
   no-op, and `psutil` replaced (see below). `print(x)` writes the payload and
   then the newline as two calls, and no example passes `end=` or touches
   `sys.stdout`, so a write of exactly `"\n"` is what separates one print from
   the next.
2. **Replay each print into a tmux pane sized to that example**, through
   `tests/tmux.py`, capturing the `Screen` after each.
3. **Compare against the golden**: every cell's glyph, every cell's fg/bg, the
   byte count of the print, the cursor, and the lines scrolled.
4. **Compare the saved image** by its decoded pixels, as a digest.

414 tests in 16.9 s, from 394 in 17.3 s: the suite is slightly *faster* than it
was with the smoke tests, because the examples were already being run and only
the looking is new. Goldens come to 449 kB, of which `mandelbrot.txt` is 137 kB
and `colormaps.txt` 66 kB.

**Verification.** The same four mutations, applied to the library and run
against the suite as built. Every one is caught, by many more examples than in
the prototype, because every example is now covered, at its own size, on all
four layers:

| mutation | examples that fail |
|---|---|
| `reset_colour` no-op | dashboard, life, mandelbrot, quickstart2, teacher_student, teapot |
| last-column cursor arithmetic | mandelbrot, teapot |
| `viridis` index shifted | colormaps, image, jointplot, scatter |
| `BoxStyle.ROUND` corner glyph | colormaps, dashboard, demo, functions, jointplot, life, time_series_histogram |

The failure message for the cursor mutation reads

    frame 4 cell 9,23: glyph ' ' -> '⠐'
    frame 4 cell 9,26: glyph '⠐' -> '⣄'
    ...

-- a whole row shifted one column, which is that bug's signature. Getting that
rather than "the colours changed somewhere" is the whole reason the colour
layer is stored in full instead of digested.

### Decisions taken

* **The colour layer is stored in full**, as a palette legend plus a grid of
  symbols aligned cell for cell with the text grid. A digest would catch every
  mutation the grid catches, at 80 kB of goldens instead of 449 kB, but it
  could not name the cell that moved. Nobody reads a 2000-line grid of palette
  letters in a `git diff` either way, which is why `--diff` and `--show` exist.
* **Byte counts are asserted exactly.** Every encoder change then shows up as a
  small, readable diff in the goldens, which is informative: the cost of a
  frame is a number worth watching. The text goldens do not move at the same
  time, so the signal stays clean.
* **One subprocess per example**, not one process for all of them. It costs
  about 4 s, and it keeps the `tyro` CLI and the `__main__` block under test,
  stops one example's module state reaching the next, and contains a crash.
* **The image layer is a digest**, not 233 kB of committed binaries. Its
  failure mode is legible: if the rasteriser changes, all eighteen digests move
  and no screen golden does.
* **`psutil` is stubbed** for `dashboard.py`, with a seeded `RandomState` and a
  fixed four cores. That makes the eighteenth example testable and stops its
  width depending on the host.
* **`life.py` now calls `np.random.seed`** rather than `default_rng`. NumPy
  guarantees the legacy stream across releases and explicitly does not
  guarantee `Generator`'s, and this board is now pinned by a golden.
  `images/life.gif` was regenerated to match, at the same 64 frames.
* **Frame counts were left at 5.** Mandelbrot's black final frame was the
  symptom of a last-screen-only test, not of a bad parameter: frames 1-5 span
  the whole zoom, and all five are now snapshotted.

### Sizing the terminals

The first attempt sized each pane to the *ink* -- the bounding box of non-blank
cells -- and three examples scrolled. A plot with a blank top or bottom row is
taller than its ink: `teapot.py`'s plot is 20 rows and draws in 18 of them, so
in a 19-row pane every frame scrolled. `calendar_heatmap.py` was worse, since
its plot is 64 columns wide with two blank ones at the right, so a 62-column
pane wrapped every row and 25 lines went off the top.

`python -m tests.examples --sizes` measures it instead: replay into a pane far
taller than needed and read the row the cursor reaches. Width cannot be
measured the same way -- a diff only mentions the cells it repaints, and
stripping its cursor moves runs a whole frame's glyphs into one line -- so it
reports the widest line of a full redraw as a suggestion, identifying a full
redraw as a payload whose every escape is an SGR. That still over-reports for
`demo.py`, which prints a 1233-character `repr` that is meant to wrap, so the
number is a prompt rather than an answer and the table stays hand-maintained.

## What this does not cover

* **Frames that a later frame overwrites.** Differential rendering makes errors
  sticky -- a wrong cell survives until something writes over it -- so most
  mid-animation errors do reach a later screen. Snapshotting every print
  narrows this to "wrong, and corrected by the very next frame".
* **The rasterised font, in detail.** The image digest notices that
  `to_rgba_array` changed but cannot say how. A glyph the library emits that is
  missing from `unscii16` would render in the terminal and blank in the image;
  the digest sees it, and a targeted unit test would explain it.
* **`demo.py`'s right margin.** Its plot is 48 columns in an 80-column pane,
  because the alternative is wrapping its `repr` over 26 rows. Margin coverage
  comes from the examples that fill their width: teapot, mandelbrot,
  quickstart1 and 2, time_series_histogram.

## Risks

**Cross-machine reproducibility is the real exposure.** Every measurement here
is from one machine, and the goldens are exact. Two sources of drift remain now
that `life.py` is on the legacy stream:

* `teacher_student.py` runs JAX. Deterministic run to run here; XLA across
  versions and backends is a different question.
* Quantisation to braille and half-block cells absorbs small float differences
  almost everywhere, but not for a value sitting on a cell boundary.

Neither is a reason not to do it -- a golden that moves on a version bump is
telling you something -- but the goldens should be understood as generated in
one reference environment, and a mismatch on a new machine investigated rather
than accepted.

**Over-sensitivity is the usual way suites like this die.** The layers are
chosen against that: the text layer moves only when the picture moves, the
colour layer only when the colours move, and the byte counts -- the one layer
that moves whenever the encoder does -- are small enough to review as numbers.
Raw stdout was measured (627 kB, and it changes on every encoder tweak) and
rejected for exactly this reason.
