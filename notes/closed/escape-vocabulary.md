# Auditing the escape sequences we emit

Audited and implemented 2026-07-26 (Matthew + Claude). This is step 4 of
`notes/terminal-test-backend.md`, which argued that "testing against N terminals
is worth less than not depending on the sequences that vary" and left the audit
itself for later. Step 5, the support matrix, became `pages/compatibility.md`.

## What we actually emitted

The list in `terminal-test-backend.md` was already stale. It named `ED`
(`CSI 0 J`, erase to end of screen), which commit 334393d ("Clear only the
plot's own rows") had removed a session earlier; there is no `J` anywhere in the
package. The real list, from grepping `matthewplotlib/` for `\x1b`:

    CUU CUD CUF CUB   cursor moves        core.py, plots.py
    CNL               down and to col 0   core.py, parking at end of a frame
    CHA               absolute column     core.py, in paint() and before ECH
    ECH               erase n characters  core.py, trailing columns when narrowing
    EL (2K)           erase whole line    core.py lost rows, plots.py clearstr
    SGR 0/39/49/38;2/48;2                 core.py, both renderers
    CR LF                                 core.py

Tiered by how much they actually vary:

* **VT100 core, no realistic risk.** `CUU`, `CUD`, `CUF`, `CUB`, `EL`, `CR`,
  `LF`, `SGR 0`.
* **ECMA-48, near-universal but not guaranteed.** `CHA`, `CNL`, `ECH`,
  `SGR 39/49`.
* **Genuinely varies.** 24-bit `SGR 38;2` / `48;2`.

The audit was to see how much of the middle tier could be moved to the top for
free. Three of the four could.

## The three swaps

**`CHA` → `CR` + `CUF`.** The one that is about correctness and not tidiness.
`CHA` was only ever reached from one place: `paint` sets `cur_col = None` after
filling a plot's last column, because terminals defer the wrap there and the
cursor is not where counting glyphs says. Recovering from that state with `CHA`
assumes `CHA` cancels a pending wrap. tmux does (pinned by
`test_reading_the_screen_leaves_the_wrap_pending`), and xterm and its
descendants are believed to, but wrap semantics are the single least consistent
corner of real terminals -- `https://github.com/mattiase/wraptest` exists for
this reason alone. Carriage return cancelling a pending wrap is not in doubt
anywhere. The recovery is now `goto_col`, which carriage-returns first and then
counts forward.

**`CNL` → `CR` + `CUD`.** `CSI n E` is down-n-and-to-column-0. Two universal
sequences do the same for one more byte, once per frame.

**`ECH` → written spaces.** `reset_colour` already puts the colour back to
default before any blanking, so a space paints exactly what the erase painted.
`ECH` was the least-exercised sequence we emitted -- it fires only when a plot
narrows -- and the only one a reader would have to look up.

`SGR 39/49` stayed. The alternative is `SGR 0` plus re-emitting the other
channel, which is more bytes and no more portable; the codes are ECMA-48 and
universally implemented.

## What it cost

The example goldens record text, colour, cursor and scrolled lines per frame as
well as byte count, so they answer the equivalence question directly: across all
nineteen examples, **only the byte counts moved**. Same screens, different
bytes. That is the whole argument that the swaps are semantically identical, and
it is worth more than any reasoning about what the sequences mean.

Per frame, measured:

    mandelbrot        -121   full repaint, plot exactly the pane width
    dashboard           +1
    life                +1
    quickstart2         +1
    teacher_student     +1
    teapot           +1/+2   grows 73 to 80 wide across its frames
    13 others            0   no animation, so no diff path at all

The two directions have different causes, both confirmed by rendering the same
frame pair through both versions and counting sequences (the throwaway script is
not kept; it bound `main`'s `to_ansi_diff_str` alongside the new one):

    full width 42x80, every cell changing   63180 -> 63057  (-123)
        CHA 41 -> 0, CR 1 -> 42
    20x60, every cell changing              22635 -> 22578   (-57)
        CHA 19 -> 0, CR 1 -> 20
    full width, 30 scattered cells            793 ->   791    (-2)
    narrowing 80 -> 60                      22937 -> 23181  (+244)
        ECH 20 -> 0, CHA 20 -> 0, CUB 19 -> 0, CUF 0 -> 20, CR 1 -> 40
    narrowing 80 -> 79                      30003 -> 29887  (-116)

So:

* A **full repaint** hits the plot's last column on every row, so every row
  starts with a recovery. `CSI 1 G` (4 bytes) became `CR` (1 byte): -3 per row.
  This is why mandelbrot got cheaper, and it is the common case for any
  animation that changes most of its cells.
* A **sparse diff** rarely paints a last column, so it never recovers and never
  paid for `CHA`. It pays +1 for the parking swap and nothing else.
* **Narrowing** is the only regression. `CSI n X` is 3 + digits bytes against
  `n` spaces, so spaces are cheaper up to n = 4, equal at 5, and dearer above.
  At n = 20 over 20 rows that is +244 bytes on the single frame that narrows,
  about 1% of it. No example narrows, so this path shows up only in the property
  sweeps in `tests/test_terminal.py`.

## The vocabulary as a test

`TestEmittedVocabulary` in `tests/test_core.py` renders thirteen scenarios --
every path that emits anything, including each resize direction and both
`clearstr` cases -- and asserts that every sequence in them is one of six final
bytes. It also insists that every escape in the string is a plain CSI, so an
`OSC` or a two-byte escape cannot slip past a regex written for the sequences we
know about.

This is what keeps `pages/compatibility.md` honest: the page is a claim about
what we emit, and the test is the same claim, executable. A fourth sequence
cannot appear without someone widening `ALLOWED_FINALS` and noticing the page.

Writing it caught one thing: a bare `CSI B` in both row-erase loops
(`to_ansi_diff_str`'s lost-rows path and `clearstr`). An omitted count means one
-- terminfo records xterm's own as `cuu1=\E[A`, so every curses program on earth
depends on it -- so it stays. The test now forbids what it was actually reaching
for, an explicit `0`, which means one rather than none and is the classic
off-by-one here.

## Left open

* **Colour depth.** 24-bit `SGR` is the only genuinely varying thing left, and
  the answer is a reduced-colour mode, which is a feature rather than an audit
  item. It degrades to a nearby colour, so it is a rendering difference and not
  a correctness bug. Roadmap: "configurable colour scales and normalisation".
* **Erase granularity.** `EL 2` erases a whole terminal line, while the trailing
  path erases only the plot's own columns, so the two disagree about whether the
  library owns the columns beside a plot. See `notes/erase-granularity.md`.
* **Cursor hiding.** The animation context manager
  (`notes/animation-context-manager.md`) will add `CSI ? 25 l` / `h`, a DEC
  private mode rather than ECMA-48. It is the most widely implemented private
  mode there is and failure is benign -- a visible cursor -- but it will be the
  first non-VT100 sequence added since this audit, and it needs a row on the
  compatibility page and an entry in `ALLOWED_FINALS`.
