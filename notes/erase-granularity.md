# Erasing whole lines, or only the plot's own columns

Raised 2026-07-26 (Matthew + Claude), noticed during the escape-sequence audit
(`notes/closed/escape-vocabulary.md`). Not decided; nothing changed. This note
exists because the answer depends on work that has not been done yet, and the
measurement is cheap to record now.

## The inconsistency

The library blanks cells in two places, and they disagree about what a plot owns.

**Only its own columns.** `CharArray.to_ansi_diff_str`, when a plot narrows,
blanks exactly the columns the old frame covered and the new one does not
(`trailing = PW - W`, blanked with spaces from column `W`). It is careful not to
reach past them, and `test_narrowing_erases_only_the_lost_columns` marks the
column to the right of both plots and asserts it survives.

**The whole terminal line.** Two other paths use `EL` (`CSI 2 K`), which erases
the line from margin to margin regardless of how wide the plot is:

* `plot.clearstr` (`plots.py`), which erases each of the plot's rows;
* `to_ansi_diff_str`'s lost-rows path (`core.py`), which erases each row a
  shrinking plot gave up.

So a differential redraw preserves whatever sits beside a plot, and a
clear-and-redraw or a vertical shrink destroys it. Nothing to the *left* is at
risk -- plots are always rendered from column 0 -- so this is only ever about
the columns to the right.

`clearstr`'s docstring says it "erases the plot's own rows and nothing else",
which is true about rows and silent about columns. That is the sentence to fix
whichever way this goes.

## Why it is not obviously a bug

Composition happens inside the library. `hstack` merges plots into one
`CharArray`, so anything the library puts beside a plot is part of that plot and
is redrawn anyway. To be bitten you have to put something beside a plot that the
library does not know about -- your own status line, another program's output --
and then shrink or clear the plot. That is rare.

It is not hypothetical, though, and it contradicts a claim now made in public.
`pages/compatibility.md` says the library "only ever moves relative to the
cursor and never addresses the screen absolutely, [so] output composes with
whatever else is on screen. It does not own the display." Whole-line erasing is
the one place that is false.

## What the options cost

Measured on real plots, comparing the current `clearstr` against the two
alternatives (`redraw` is `renderstr`, the full repaint that a clear is normally
followed by):

    plot             H x W   clearstr   spaces    ECH    redraw
    image            42x80        301     3535    343    115345
    image            24x80        175     2023    199     65906
    image            18x73        133     1393    151     45112
    bordered text      4x7         33       49     33        31

Reading it:

* **Spaces cost about one byte per cell of the plot.** Against a fully coloured
  redraw, which runs about 35 bytes per cell, that is a steady **+2.8%** at
  every size. Against a *monochrome* redraw, which runs about one byte per cell,
  it is close to double -- the bordered-text row, +16 bytes on a 31-byte redraw.
  The percentages are the honest way to read this; the absolute numbers are
  small either way, a few kB at the very top end.
* **`ECH` costs about five bytes per row**, near enough free at any size.
* Doing nothing costs nothing.

There is a second cost to spaces that the table does not show. `EL` does not
move the cursor, so `clearstr` is currently a closed-form string with no cursor
bookkeeping in it at all:

    f"\x1b[{H}A" + "\x1b[2K" + "\x1b[B\x1b[2K" * (H - 1) + f"\x1b[{H}A"

Writing spaces moves the cursor to column `W`, which may be the screen's last
column, which defers a wrap -- so each row needs a carriage return after it, and
the closing jump can no longer assume the column is preserved. That turns the
simplest function in the library into one that has to think. Not hard, but the
simplicity is worth something.

## The awkward part

The obvious tool for erasing exactly `W` columns is `ECH`, and the audit just
retired it. `ECH` would make this fix almost free -- five bytes a row against
one byte a cell -- but it would put a non-VT100 sequence back in the vocabulary
to serve a case that, on today's evidence, nobody has hit.

That is a real tension and it should be decided on its merits rather than by
consistency with a decision made an hour earlier. The argument for spaces over
`ECH` here is the same one as in the audit: the vocabulary is the thing being
protected, and 2.8% of a redraw is a low price for keeping it at eleven
sequences. The argument for `ECH` is that this is precisely the case it exists
for, and 2.8% is not nothing on a slow link.

## Recommendation

Leave it for now, and fix it with spaces when terminal-aware printing lands.

The roadmap has "crop plot composition primitive" and "by default, clip plots to
terminal width and (almost) height" under advanced arrangement, with framing in
`notes/terminal-aware-printing.md`. That work is about a plot coexisting with a
terminal it does not fill, which is exactly the situation where owning columns
you did not draw starts to matter. Today the inconsistency is a wart; after that
work it would be a bug, and the invariant "a plot writes only inside its own
rectangle" would be load-bearing rather than decorative.

Doing it then also means doing it once, with the cursor bookkeeping designed
alongside the clipping rather than bolted onto `clearstr` first.

What would change the recommendation:

* Anyone actually reporting eaten output beside a plot -- fix it immediately,
  with `ECH` if the byte cost is what is biting.
* A plotting style dominated by monochrome output, where the spaces overhead is
  nearer 100% than 3%. The examples are overwhelmingly coloured, so this is
  theoretical today.

Until then, the docstring on `clearstr` should say what it does: erases the
plot's rows, full width, and does not preserve anything to the right of the
plot.
