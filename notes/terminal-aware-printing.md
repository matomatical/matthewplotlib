# Terminal-aware printing / clipping — framing notes

Raised 2026-07-25 (Matthew's wishlist). Not designed; this note only records
what the differential rendering work established about the constraints, so the
design does not start from scratch.

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
