Terminal compatibility
======================

Matthewplotlib draws plots by writing bytes to a terminal. What you see depends
on what your terminal does with them. This page is the complete list of the
kinds of bytes the library produces and what different terminals might do with
them.

The library produces three kinds of byte sequences:

* **Glyphs**: including plain ASCII but also lots of unicode for braille,
  blocks, and box-drawing. If your terminal font is missing some characters,
  you might see boxes or gaps.
* **Colour control codes**: 24-bit RGB ANSI escape sequences. If your terminal
  doesn't support true-colour mode, it should probably still be able to read
  these and fall back to the nearest colours it does have.
* **Movement/erasure control codes**: cursor movement and erasing escape
  sequences. If your terminal is missing certain codes or interprets them
  differently than the library expects, rendering can become garbled.

Movement/erasure control codes are only used for plot redrawing/animation. If
you print a plot and never animate it, you are only using colour, glyphs, and
newlines, which should work essentially everywhere. Otherwise, we try to keep
the set of escape sequences minimal to maximise compatibility.

The library produces strings for the terminal via pure functions that don't
check the state of the terminal. It looks at the terminal in only two places,
and neither one changes a byte of what is written: `wrap` reads the terminal
width to choose a grid layout, and `animate` reads the terminal height to
decide whether your plot is too tall to animate and you should be warned.
Therefore:

* The control codes used can't depend on the terminal.
* The library assumes the terminal is large enough to fit whatever you want to
  plot. Printing something larger than the screen is undefined behaviour.


Glyphs
------

The plots are drawn with characters, so the font matters as much as the
terminal. The library uses four blocks:

| Range         | Block            | Used by |
|---------------|------------------|---------|
| ASCII         | —                | tick labels, titles |
| U+2500–U+257F | Box Drawing      | `border`, `axes` |
| U+2580–U+259F | Block Elements   | `image` (half blocks), `bars` and `columns` (eighth blocks), blocky borders |
| U+2800–U+28FF | Braille Patterns | `scatter`, `hilbert`, dotted borders |

(The library also allows you to render arbitrary glyphs through `text` plots,
those glyphs are your responsibility.)

Braille is the one to check. It is the densest and the least likely to be in an
older bitmap font, and a font that lacks it turns a scatter plot into a field
of replacement boxes. Most modern terminal fonts have it.

The library generally operates on the following assumption about how these
characters are rendered:

* **The glyphs must be single-width.** The library assumes one character per
  cell throughout its cursor arithmetic. Braille, blocks and box-drawing are all
  narrow characters, but some fonts render braille double-width, which will
  smear an animated plot.
* **Cells should abut.** Half-blocks and eighth-blocks are meant to tile into
  continuous areas. Terminals that add letter-spacing, or fonts whose block
  glyphs do not fill the cell, leave visible seams in an `image` plot. This is
  cosmetic, but it is the most common complaint about block-drawing plots.

If glyphs are the problem rather than the terminal, `saveimg` renders the same
plot to a PNG through an embedded pixel font, with no dependency on your font
at all.

On the roadmap:

* Fallbacks for missing braille glyphs.
* Opt-in octant glyphs from the new Symbols for Legacy Computing Supplement
  instead of braille + instructions to install/patch fonts with these glyphs.


Colour
------

The library emits the following control codes for controlling the
forground/background colour of terminal glyphs.

| Sequence | Name | Effect | Used for | Standard |
|----------|------|--------|----------|----------|
| `ESC [ 39 m` / `ESC [ 49 m` | SGR | Default foreground / background | Returning one channel to default. | ECMA-48 |
| `ESC [ 38;2;r;g;b m` | SGR | 24-bit foreground | Colour. | ISO 8613-6 |
| `ESC [ 48;2;r;g;b m` | SGR | 24-bit background | Colour. | ISO 8613-6 |

In particular, the library always emits 24-bit colour control codes. On a
terminal with fewer colours the usual behaviour is to approximate to the
nearest available, which might still be readable. Some very old terminals might
ignore the sequence instead and render in the default colour, which is legible
but flat.

On the roadmap:

* Manually configured library-level reduced-colour mode to guarantee the
  fallback and save bytes on older terminals.
* Reduced-colour colourmaps.

Movement/Erasure
----------------

For standard plotting, the library relies on only newlines.

| Sequence | Name | Effect | Used for | Standard |
|----------|------|--------|----------|----------|
| `LF` (`\n`) | — | Next row, scrolling at the bottom | Line breaks | ASCII |

During plot erasure/redrawing, such as for animation, the library emits the
following escape sequences in different contexts. A terminal that handles these
eleven things the way we expect should support animation.

| Sequence | Name | Effect | Used for | Standard |
|----------|------|--------|----------|----------|
| `CR` (`\r`) | — | Column 0 | Reaching a known column from an unknown one | ASCII |
| `LF` (`\n`) | — | Next row, scrolling at the bottom | Growing a plot taller | ASCII |
| `ESC [ n A` | CUU | Cursor up n rows | Animation: reaching the frame above | VT100 |
| `ESC [ n B` | CUD | Cursor down n rows | Animation: reaching a later row | VT100 |
| `ESC [ n C` | CUF | Cursor forward n columns | Animation: skipping unchanged cells | VT100 |
| `ESC [ n D` | CUB | Cursor back n columns | Animation: stepping back within a row | VT100 |
| `ESC [ 2 K` | EL | Erase the whole line | Clearing a plot, and rows a shrinking plot gave up | VT100 |
| `ESC [ 0 m` | SGR | Reset all attributes | Ending a coloured run | VT100 |
| `ESC [ 39 m` / `ESC [ 49 m` | SGR | Default foreground / background | Returning one channel to default. | ECMA-48 |
| `ESC [ 38;2;r;g;b m` | SGR | 24-bit foreground | Colour. | ISO 8613-6 |
| `ESC [ 48;2;r;g;b m` | SGR | 24-bit background | Colour. | ISO 8613-6 |

Note that recognising a sequence is not the same as agreeing what it does.
These are the behaviours the library counts on.

* **An omitted count means one.** The library writes `ESC [ B` where the
  distance is always one row. This is the standard default — terminfo records
  xterm's own cursor-up capability as `cuu1=\E[A` — and every curses program
  relies on it. It never writes a count of zero, which would mean one and not
  none.
* **Carriage return cancels a deferred wrap.** After a glyph lands in the last
  column, terminals do not move the cursor past the edge. They leave it on the
  margin and set a flag — the Last Column Flag — so that the wrap happens when
  the *next* printable character arrives. The library stops trusting its column
  arithmetic at that point and issues a carriage return before counting again.

  This is the one behaviour on this page where the disagreements are documented
  and substantial. [wraptest](https://github.com/mattiase/wraptest) is a test
  suite for exactly this, and its finding is blunt: *no emulator tested so far
  matches the specification* (STD-070). The axes it probes are

  * whether the terminal defers the wrap at all, or wraps immediately;
  * which operations clear the flag. STD-070 says cursor movement (`CUU`,
    `CUD`, `CUF`, `CUB`), cursor positioning (`CUP`, `HVP`), the control
    characters `BS`, `HT`, `CR` and `LF`, and the erase, delete and insert
    operations (`ECH`, `DCH`, `ICH`) all should. In practice which ones
    actually do varies;
  * what column the terminal reports while the flag is set. tmux reports the
    width rather than width − 1 — pinned by
    `test_wrap_is_deferred_at_the_right_margin` — and a model that reports
    width − 1 computes every subsequent relative move one column off.

  Even the hardware disagrees: wraptest reports that the VT100 diverges from
  STD-070 considerably, the VT220 follows it to the letter, and the VT510
  differs again; among emulators only recent xterm is described as faithful.

  So the library's position is not that carriage return is specified to clear
  the flag where other operations are not — the specification lists a dozen
  operations that clear it, and no emulator implements the list faithfully.
  It is that a specification nobody matches is not something to reason from,
  and `CR` at the right margin is the single most exercised path in any
  terminal, because it is what every progress bar and every `\r`-terminated
  line in forty years of software does. An operation that common is where
  implementations are least likely to be wrong. That is a bet on ubiquity
  rather than on standards, and it is deliberately the more conservative bet.
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

Three things we don't use, since terminals genuinely diverge in how they are
handled:

* **No cursor save/restore.** There are two incompatible conventions for it
  (`ESC 7`/`ESC 8` and `ESC [ s`/`ESC [ u`) and terminals differ over which they
  implement.
* **No absolute cursor positioning** (`CUP`), no scroll regions, no alternate
  screen buffer, no tab stops, no private modes, no OSC. The library moves
  relative to where the cursor already is, and stays inside the rows it printed.
* **No queries.** It never asks the terminal a question and waits for a reply,
  so it cannot hang on a terminal that does not answer.

Because the library only ever moves relative to the cursor and never addresses
the screen absolutely, output composes with whatever else is on screen. It does
not own the display.

Legacy:

* An earlier version of the library previously used three additional sequences,
  which were replaced with combinations of the above.
  * **CHA** (`ESC [ n G`, absolute column) is now `CR` plus `CUF`. CHA was only
    ever reached in the one situation where the renderer had lost track of the
    cursor column—after filling a plot's last column, where terminals defer the
    line wrap—so it depended on CHA cancelling a pending wrap. Both are cursor
    movements, and by the specification both clear the flag; the difference is
    how heavily each is exercised in terminals that do not follow the
    specification. See the deferred-wrap behaviour above.
  * **CNL** (`ESC [ n E`, next line) is now `CR` plus `CUD`. One byte more.
  * **ECH** (`ESC [ n X`, erase n characters) is now written spaces. The renderer
    already resets the colour to default before blanking anything, so a space
    paints exactly what the erase painted.
  See `notes/closed/escape-vocabulary.md` for the measurements behind this.

Screen size
-----------

The library produces strings independently of the terminal, as above. These
strings generally assume the rendered plot will fit neatly onto the screen.
Otherwise, line wrap and scrollback may upset the effect of the control
sequences. Specifically:

* **A plot must not be wider than the screen.** A plot exactly as wide as the
  screen is fine. Anything wider wraps, and every column calculation after that
  is void. This is out of contract today, and silently so.

* **Animation needs a spare row.** Printing a plot of height H plus a newline
  needs H+1 rows, so an animation in an R-row terminal wants a plot of at most
  R-1 rows; clear-and-redraw wants R-2, since it steps a row above the plot.

On the roadmap:

* Plot primitives that allow automatically cropping a plot so that it will fit
  on the screen.
* Managed animations automatically cropping including when the screen resizes?

Supported terminals
-------------------

At the moment, we only officially guarantee support for tmux.

| Terminal | Status | How |
|----------|--------|-----|
| tmux 3.5a | **Verified continuously** | The test suite drives a real tmux pane. Every behaviour listed above is asserted there, and every example is snapshotted frame by frame—glyph, colour, cursor and scroll position of every cell. |

The author also uses Alacritty and zmx, but without automated testing to detect
regressions.

If you have a terminal that is not listed above and you see visual distortion,
please report it. There is an example that exercises every escape sequence on
this page, in the situations where they are hardest, and says what each stage
should look like so that you can judge it:

```console
python examples/terminal_test.py
```

It measures your terminal's width and draws to it, so that stage 4 puts a plot
against the right margin — the deferred-wrap case above. That is the one stage
testing something your terminal does not already do constantly, so it is the
one to watch. (If it cannot measure the width, because output is redirected
somewhere that is not a terminal, it says so and stage 4 tests nothing.)

If a stage misbehaves, please open an issue saying which one, along with your
terminal, its version, and `echo $TERM`.
