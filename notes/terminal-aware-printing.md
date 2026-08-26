# Terminal-aware printing / clipping — framing notes

Raised 2026-07-25 (Matthew's wishlist). What follows down to "Questions to
answer first" is the framing written before anything was designed: what the
differential rendering work had established about the constraints, so that the
design would not start from scratch. `crop` was built against it on 2026-08-26,
and the sections after that record what it decided and what it left alone.

## The ask

A way to clip output to the terminal's width and height, so a plot too big for
the screen degrades instead of tearing.

## What we already know about the edges

From the differential rendering work (see `CharArray.to_ansi_diff_str` and the
screen-edge tests in `tests/test_core.py`):

* **Height.** Animating needs a spare row below the plot for the newline that
  `print` appends, so the usable height is `R-1`, not `R`. A plot of exactly
  the screen height cannot animate by any path: printing H rows plus a newline
  into H rows must scroll, and the plot's top row is then lost off the screen.
* **Clear-and-redraw needs one more.** `clearstr` steps a row above the plot,
  so `print(-plot)` then `print(plot)` wants `R-2`. On the top row of the
  screen the step clamps and the redraw settles one row lower, once.
* **Width.** A plot exactly as wide as the screen is fine, but only because the
  renderer stops trusting its own column arithmetic after writing the final
  column (terminals defer the wrap there). Anything *wider* than the screen
  wraps and all the cursor arithmetic is void -- that is out of contract today,
  silently.

So "too big" has three different thresholds depending on what you are doing
with the plot, which is probably the first thing the design has to pin down.

## The hard part is position, not size

`shutil.get_terminal_size()` gives the size (already used by `wrap`, and made
robust without an attached terminal in 0.3.8). What no string-returning
function can know is **where on the screen it is being printed**. That killed
two ideas during the differential rendering work:

* growing a plot cannot push the content below it down, because pre-scrolling
  to make room scrolls somewhere between 0 and k times depending on position,
  and the compensating move back is ambiguous by exactly that amount;
* `clearstr` cannot tell whether it has a row above it to step onto.

The only fix is querying the terminal (`\x1b[6n` and *reading the reply*),
which makes the API interactive rather than pure. Worth deciding deliberately
whether clipping is allowed to cross that line. An animation context manager
could legitimately do it, since it already owns the terminal for the duration
of a `with` block; `str(plot)` could not.

## Questions to answer first

* What does clipping *mean* for a composed plot -- truncate the character grid
  at the edge, or push the constraint down into the leaf plots so axes and
  borders stay coherent? The former is easy and ugly; the latter is a layout
  pass.
* Is it opt-in (`plot.clip()`, or a parameter) or automatic? Automatic changes
  what existing programs print.
* Does it warn? Silent truncation of a plot is the kind of thing that wastes an
  afternoon.
* Interaction with image export: `saveimg`/`tstack.savegif` have no terminal,
  so clipping must not apply there.

## What `crop` settled

Built 2026-08-26. The draft and the decisions are MFR's, the write-up and the
review that prompted them Claude's (Opus 5). `crop` answers the size half of
the ask and leaves the position half alone.

**It truncates the character grid.** The easy and ugly option, not the layout
pass: a cropped `axes` loses its east and south edges rather than redrawing
itself smaller. Pushing the constraint into the leaf plots is still available
later, and would not change this API.

**It is opt-in.** An explicit `crop(plot)`, so no existing program prints
anything different. Nothing became automatic.

**It marks instead of warning.** The last row or column of a cut direction is
given over to a marker character, `#` by default. A warning can be filtered,
missed, or arrive after the plot has already scrolled past; a marker is in the
output, next to the thing it is about. `marker=None` opts out and takes the
full rectangle of content instead. The marker costs a row: cropping ten rows
to eight shows seven of them.

**A defaulted size errors without a terminal, rather than falling back.** This
is the one place the design had a choice with teeth. `wrap` already takes
`shutil.get_terminal_size(fallback=(80, 24))` for its column count, so there
was precedent for a fallback -- but `wrap` only picks a layout with it, and a
wrong guess costs a line break. Here a fallback would delete content: piped to
a file, `crop(plot)` would silently truncate to 80x24. The invariant recorded
at `_terminal_rows` -- nothing in this library changes *what* it writes based
on whether a terminal is attached -- is worth more than the convenience, so a
measurement is required and `os.get_terminal_size` is what takes it, since a
`shutil` fallback cannot be told apart from a real answer. `_terminal_size` in
`plots.py` is now the single place that reads it, and `_terminal_rows` defers
to it.

**Of the three thresholds, the default is R-1.** The differential animation
case, the one `animate` drives. R is offered by passing it, and R-2 for
clear-and-redraw likewise; neither is the default, because a plot that only
just fits is the case that goes wrong silently.

**Image export is unaffected.** `crop` produces a plot like any other, so
`saveimg` and `savegif` see whatever it produced and never consult a terminal
themselves. A `crop` with defaulted sizes cannot run headless at all, which is
the error above rather than a special case.

## Still open

* Position on screen, unchanged. `crop` needs to know only how big the screen
  is, never where on it the plot lands, which is why it could be built at all
  while the questions above about `clearstr` and pre-scrolling stay unanswered.
* Cropping from any of the nine directions. The top left rectangle is kept
  today.
* `animate` doing this itself, and following a resize while it runs. This is
  where the remaining questions live: an animation context manager owns the
  terminal for the duration of its block, so it is the one caller that could
  legitimately query it.
