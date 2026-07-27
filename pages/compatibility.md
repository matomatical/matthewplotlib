Terminal compatibility
======================

Matthewplotlib draws by writing bytes to a terminal, so what you see depends on
what your terminal does with them. This page is the complete list of what the
library sends, and what each part of it needs.

It is written to answer two questions. If you are using the library: will it
work in your terminal, and what will it look like if something is missing. If
you are changing the library: this is the specification of what it is allowed
to emit, and `TestEmittedVocabulary` in `tests/test_core.py` enforces it.

The library never asks what terminal it is talking to. It does not read `TERM`
or `COLORTERM`, and it has no capability database and no fallback modes. Every
terminal gets the same bytes. (The one exception is `wrap`, which reads the
terminal *width* from `shutil.get_terminal_size` to decide a grid layout.) So
this page is a complete description, not a negotiation.


The short version
-----------------

Three things can vary between terminals, and they fail very differently:

| What | If your terminal falls short | Severity |
|------|------------------------------|----------|
| **Glyphs** — braille, blocks, box-drawing | Missing characters show as boxes or gaps | Cosmetic, and obvious |
| **Colour** — 24-bit RGB | Colours snap to the nearest the terminal has | Cosmetic, sometimes ugly |
| **Escape sequences** — cursor movement and erasing | Static plots are unaffected; animation could misplace or smear | The only one that corrupts |

Only the third can produce a scrambled screen rather than a plain-looking one,
which is why most of this page is about it — and why the set of sequences the
library uses was deliberately cut down to a size that fits on one screen.

If you print a plot and never animate it, you are only using colour, glyphs, and
newlines. That works essentially everywhere.


Escape sequences
----------------

The complete set. Everything here is in the VT100's repertoire except the SGR
colour codes, which are noted as such.

| Sequence | Name | Effect | Used for |
|----------|------|--------|----------|
| `ESC [ n A` | CUU | Cursor up n rows | Animation: reaching the frame above |
| `ESC [ n B` | CUD | Cursor down n rows | Animation: reaching a later row |
| `ESC [ n C` | CUF | Cursor forward n columns | Animation: skipping unchanged cells |
| `ESC [ n D` | CUB | Cursor back n columns | Animation: stepping back within a row |
| `ESC [ 2 K` | EL | Erase the whole line | Clearing a plot, and rows a shrinking plot gave up |
| `ESC [ 0 m` | SGR | Reset all attributes | Ending a coloured run |
| `ESC [ 39 m` / `ESC [ 49 m` | SGR | Default foreground / background | Returning one channel to default. *Not VT100; ECMA-48.* |
| `ESC [ 38;2;r;g;b m` | SGR | 24-bit foreground | Colour. *Not VT100; ISO 8613-6.* |
| `ESC [ 48;2;r;g;b m` | SGR | 24-bit background | Colour. *Not VT100; ISO 8613-6.* |
| `CR` (`\r`) | — | Column 0 | Reaching a known column from an unknown one |
| `LF` (`\n`) | — | Next row, scrolling at the bottom | Line breaks, and growing a plot taller |

That is the whole vocabulary. A terminal that handles those eleven things
handles every control sequence this library can produce; the rest of what it
sends is printable characters, which is the Glyphs section below.

Three sequences were removed in the escape-sequence audit, and the library is
tested not to reintroduce them. If you are comparing against an older version,
or reading code that predates it:

* **CHA** (`ESC [ n G`, absolute column) is now `CR` plus `CUF`. CHA was only
  ever reached in the one situation where the renderer had lost track of the
  cursor column — after filling a plot's last column, where terminals defer the
  line wrap — so it depended on CHA cancelling a pending wrap. Wrap handling is
  the single least consistent corner of real terminals (see
  [wraptest](https://github.com/mattiase/wraptest)); carriage return is not.
* **CNL** (`ESC [ n E`, next line) is now `CR` plus `CUD`. One byte more.
* **ECH** (`ESC [ n X`, erase n characters) is now written spaces. The renderer
  already resets the colour to default before blanking anything, so a space
  paints exactly what the erase painted.

See `notes/closed/escape-vocabulary.md` for the measurements behind those three.


Behaviours, not just sequences
------------------------------

Recognising a sequence is not the same as agreeing what it does. These are the
behaviours the library counts on. Each is asserted against a real terminal in
`tests/test_terminal.py`, most of them in `TestTheHarness`.

* **An omitted count means one.** The library writes `ESC [ B` where the
  distance is always one row. This is the standard default — terminfo records
  xterm's own cursor-up capability as `cuu1=\E[A` — and every curses program
  relies on it. It never writes a count of zero, which would mean one and not
  none.
* **Carriage return cancels a deferred wrap.** After a glyph lands in the last
  column, terminals leave the cursor on the margin with a wrap pending rather
  than moving past the edge. The library stops trusting its column arithmetic
  at that point and issues a carriage return before counting again.
* **Cursor movement clamps; it does not scroll.** Cursor-down at the bottom row
  stays on the bottom row. This is why a plot that grows taller is extended with
  newlines rather than cursor movement — a newline at the bottom scrolls, which
  is what is wanted there.
* **Blanking paints the current background.** On terminals with
  background-colour erase, an erase fills with the active background rather than
  the default; a written space does so everywhere. Rather than depend on which,
  the library resets the colour to default before it blanks anything, so both
  behaviours produce the same screen.
* **`print` adds the newline.** Every string the library returns is shaped for a
  plain `print` and ends one row short, expecting the newline `print` appends.
  On a tty the line discipline turns that into carriage-return plus line-feed,
  which is what returns the cursor to column 0.

And two things the library requires of the *screen*, rather than of the
terminal's escape handling:

* **A plot must not be wider than the screen.** A plot exactly as wide as the
  screen is fine. Anything wider wraps, and every column calculation after that
  is void. This is out of contract today, and silently so.
* **Animation needs a spare row.** Printing a plot of height H plus a newline
  needs H+1 rows, so an animation in an R-row terminal wants a plot of at most
  R-1 rows; clear-and-redraw wants R-2, since it steps a row above the plot.


What the library never sends
----------------------------

Worth stating, because these are where terminals genuinely diverge and where a
plotting library could easily have reached:

* **No cursor save/restore.** There are two incompatible conventions for it
  (`ESC 7`/`ESC 8` and `ESC [ s`/`ESC [ u`) and terminals differ over which they
  implement.
* **No absolute cursor positioning** (`CUP`), no scroll regions, no alternate
  screen buffer, no tab stops, no private modes, no OSC. The library moves
  relative to where the cursor already is, and stays inside the rows it printed.
* **No queries.** It never asks the terminal a question and waits for a reply,
  so it cannot hang on a terminal that does not answer.

A consequence worth knowing: because the library only ever moves relative to the
cursor and never addresses the screen absolutely, output composes with whatever
else is on screen. It does not own the display.


Colour
------

Colour is the only part of the vocabulary that is not VT100, and the only one
where terminals differ in a way you will actually notice.

The library always emits 24-bit colour. On a terminal with fewer colours the
usual behaviour is to approximate to the nearest available, which is a rendering
difference and not a correctness problem: the layout, the cursor arithmetic and
the animation all still work, and the plot is simply a coarser colour. Some very
old terminals ignore the sequence instead and render in the default colour,
which is legible but flat.

Continuous colormaps are what suffer: a 256-colour terminal renders `viridis`
as visible bands. Discrete palettes with a handful of well-separated colours
survive almost anything.

There is no reduced-colour mode yet. It is on the roadmap under "configurable
colour scales and normalisation".


Glyphs
------

The plots are drawn with characters, so the font matters as much as the
terminal. The library uses four blocks:

| Range | Block | Used by |
|-------|-------|---------|
| U+2800–U+28FF | Braille Patterns | `scatter`, `hilbert`, dotted borders — 2×4 dots per cell |
| U+2580–U+259F | Block Elements | `image` (half blocks), `bars` and `columns` (eighth blocks), blocky borders |
| U+2500–U+257F | Box Drawing | `border`, `axes` |
| ASCII | — | `text`, tick labels, titles |

Braille is the one to check. It is the densest and the least likely to be in an
older bitmap font, and a font that lacks it turns a scatter plot into a field of
replacement boxes. Most modern terminal fonts have it.

Two further font properties matter, and neither is about coverage:

* **The glyphs must be single-width.** The library assumes one character per
  cell throughout its cursor arithmetic. Braille, blocks and box-drawing are all
  narrow characters, but some fonts render braille double-width, which will
  smear an animated plot.
* **Cells should abut.** Half-blocks and eighth-blocks are meant to tile into
  continuous areas. Terminals that add letter-spacing, or fonts whose block
  glyphs do not fill the cell, leave visible seams in an `image` plot. This is
  cosmetic, but it is the most common complaint about block-drawing plots.

If glyphs are the problem rather than the terminal, `saveimg` renders the same
plot to a PNG through an embedded pixel font, with no dependency on your font at
all.


Which terminals are tested
--------------------------

Honestly: one.

| Terminal | Status | How |
|----------|--------|-----|
| tmux 3.5a | **Verified continuously** | The test suite drives a real tmux pane. Every behaviour listed above is asserted there, and every example is snapshotted frame by frame — glyph, colour, cursor and scroll position of every cell |
| Everything else | **Expected, not verified** | Reasoned from the vocabulary being VT100 |

This is a smaller claim than a table of ticks would be, and it is the true one.
What makes it a reasonable position rather than an evasion is the shape of the
vocabulary: the eleven sequences above are the ones that forty years of software
has made unskippable, so a terminal that got one of them wrong would break far
more than this library. The audit that produced this list existed precisely so
that the compatibility question could be answered by argument instead of by a
matrix nobody can keep current.

The places to be sceptical, in order:

1. **24-bit colour**, which genuinely varies, and degrades visibly.
2. **Deferred wrap at the last column**, if you animate a plot exactly as wide
   as the screen. Terminals really do disagree here; the library is written not
   to depend on the disagreement, but this is the assumption most worth testing.
3. **Background-colour erase**, if a plot shrinks mid-animation and leaves a
   coloured smear behind. The library resets the colour first specifically to
   avoid this, so a smear is a bug worth reporting.


Reporting a terminal
--------------------

If you have a terminal not listed above, two things are useful.

The strong check, if you can install tmux, is the suite itself — though note
that it tests tmux, not your terminal:

```console
pytest tests/test_terminal.py
```

The check that actually exercises *your* terminal is an animation that resizes,
which is the case that uses every sequence on this page:

```python
import time
import numpy as np
import matthewplotlib as mp

prev = None
for i in range(60):
    w = 30 + int(25 * np.sin(i / 6))          # a plot that grows and shrinks
    rng = np.random.default_rng(i)
    plot = mp.border(mp.image(rng.random((12, w))))
    print(plot - prev)                        # differential redraw
    prev = plot
    time.sleep(1 / 20)
print()
```

(`mp.animate` is the comfortable way to write that loop; it is spelled out here
because the raw form is exactly what goes down the wire, and it needs nothing
but the library.)

It should animate in place, with no drifting, no leftover columns to the right
as it narrows, and no coloured smear. If it misbehaves, please open an issue
with your terminal, its version, and `echo $TERM` — and if the plot drifts or
smears rather than merely looking wrong, say so, because that is the class of
problem this page exists to prevent.
