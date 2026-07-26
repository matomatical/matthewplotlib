# Testing ANSI output against a real terminal

Investigated 2026-07-25 (Matthew + Claude), after the differential rendering
work. Steps 1-3 of the plan built 2026-07-26 (see "What was built", below).
Steps 4 and 5 were done later the same day: the audit is
`notes/closed/escape-vocabulary.md`, and the matrix is `pages/compatibility.md`.

## The problem

`tests/test_core.py` contains `_Term`, a hand-written ANSI screen emulator used
to assert what our escape sequences do to a screen. It grew three times during
one session, each time because a test could not express something:

* no deferred wrap at the right margin -- which is what *hid* the last-column
  cursor bug (see the commit "Fix cursor bookkeeping and animation idiom");
* no scrolling -- so it could not answer whether a plot of the terminal's own
  height can animate;
* no cursor clamping, and no `EL`/`ECH`.

Every one of those gaps was found by luck, when a test happened to probe that
area. The gaps that remain are unknown. A test double whose fidelity is
unverified fails silently: the suite passes and the terminal disagrees.

## What was measured

**A real terminal distinguishes the bug we fixed.** A 6-wide plot in a 6-column
tmux pane, updating cell (0,5) then cell (1,3):

    fixed  (absolute \x1b[4G)  -> ['.....X', '...Y..']   Y at column 3
    buggy  (relative \x1b[3D)  -> ['.....X', '..Y...']   Y at column 2

`_Term` agrees with tmux on *both*. So the bug was real, and our emulator is
faithful on the axis that caught it.

**pyte would not have caught it.** Its `draw` lets `cursor.x` reach `columns`
and wraps only when the next glyph arrives, so a cursor movement issued from
that state computes from `columns` rather than `columns - 1`. It does not
implement the Last Column Flag. That is inference from reading `screens.py`,
not from running it -- but if right, pyte would have scored the buggy sequence
as correct. Adopting it would have been a fidelity *downgrade* on the one axis
that matters most here.

**tmux can replace `_Term` outright.** The objection was colour: our assertions
check per-cell fg/bg. But `capture-pane -p -e` round-trips 24-bit colour
exactly:

    \x1b[38;2;12;34;56m\x1b[48;2;200;100;50mA\x1b[39m\x1b[49mB...

**Timings** (tmux 3.5a): spawn + capture + kill is 23.6 ms per case;
`capture-pane` alone is 3.4 ms. (An in-session feed measured at 296 ms, but
that harness was broken -- the pane was in raw mode while the reader expected
newline-terminated input -- so it is not evidence of anything.)

## The plan

1. **Build a tmux-backed harness and delete `_Term`.** Keeping both would mean
   maintaining two models, which is the worst of the options. The decisive
   argument is that tmux takes over exactly the error-prone part: all three
   emulator bugs were cursor semantics. What remains to write is an SGR parser
   over an already-rendered static screen, with no cursor model in it at all --
   far lower risk than what we have now.

   Scroll assertions, which `_Term` served with a counter, become
   `capture-pane -S` against the pane history.

2. **Require tmux as a development dependency.** Documented in CONTRIBUTING
   alongside uv, make and pandoc. Reasonable to assume a developer can install
   tmux. (Written up as "tests skip with a message"; corrected on
   implementation to a hard dependency -- see below.)

3. **Watch the cost.** The property sweeps are the valuable tests -- 120 random
   resizes, 40 random diffs, 40 random right-edge cases -- and at ~24 ms plus a
   settle poll they take the suite from ~13s to perhaps 20-25s. Mitigations if
   that hurts: reuse one session and reset between cases, or push a whole sweep
   into one pane program that self-checks. If it still hurts, cut the number of
   random cases rather than reintroduce a homemade emulator.

4. **Audit the emitted vocabulary.** Testing against N terminals is worth less
   than not depending on the sequences that vary. We emit: `CUU`/`CUD`/`CUF`/
   `CUB`, `CHA` (`G`), `CNL` (`E`), `ED` (`0J`), `EL` (`2K`), `ECH` (`X`),
   truecolor `SGR`, `CR`, `LF`.

   (Done, and this list was already wrong when written: `ED` had been removed by
   334393d a session earlier. Predictions below scored in
   `notes/closed/escape-vocabulary.md` -- `ECH` and `CNL` went as guessed, and
   `CHA` turned out to matter more than either, for the wrap reason rather than
   the byte count.)

   * `ECH` is the one to scrutinise -- least commonly exercised of the set, and
     only used when a plot narrows. Writing spaces is universally supported and
     costs bytes in that one case only.
   * `CNL` could be `CR` plus `CUD`.
   * truecolor varies but degrades to the nearest colour: a rendering
     difference, not a correctness bug.
   * deferred wrap we already do not depend on -- the bookkeeping was made
     conservative instead. That is the pattern to generalise.

5. **Build a terminal support matrix** (Matthew's idea, and the eventual public
   artifact; now `pages/compatibility.md`, which answers the terminal question
   by argument from a small vocabulary rather than by a table of ticks, and says
   plainly that only tmux is tested): the sequences and behaviours we rely on,
   against the terminals people actually use -- what supports what, and where behaviour differs.
   Publish it in the documentation. It is more useful to a user than any test
   result, since it tells them whether their terminal will work, and it doubles
   as the specification of what the library is allowed to emit.

   Only tmux is testable on the nook today (xterm is installed but there is no
   display and no Xvfb; no screen, kitty, alacritty, foot or st). Filling the
   matrix means either installing more (`screen` is the cheapest second
   opinion; `Xvfb` + the existing `xterm` is the reference VT; `pyte` is the
   most informative *because* it is wrong in a known way, so it acts as a
   canary for terminals that simplify wrap the same way) or collecting reports
   from real users on real terminals.

## What was built

`tests/tmux.py`: one tmux server on a socket private to the test process, one
session whose single pane runs `sleep` to hold a pty open, created on first use
and killed (socket unlinked) at exit. `Terminal(height, width)` claims that
pane; `Screen` is a snapshot of it. `tests/test_terminal.py` holds every test
that needs one; `_Term` is gone. 394 tests in 17.3s, from 381 in 13.8s.

tmux is a **hard** dependency, deliberately (Matthew, on review): importing the
harness without it raises, so the suite fails with an install prompt rather than
skipping. A suite that quietly skips the only tests of what we emit reports
success while checking nothing. The terminal cost is worth paying every run; if
it ever stops being worth it, optimise then.

Four mechanism choices, each measured rather than assumed:

* **Feeding** writes bytes straight to the pane's slave tty (`#{pane_tty}`), so
  there is no `send-keys` escaping, no terminal echo, and no line-discipline
  fight. It is also more faithful: ONLCR turns `"\n"` into CR LF exactly as it
  does for a program calling `print`. (The in-session feed that measured 296 ms
  was `send-keys` into a raw-mode reader; that whole approach is unnecessary.)
* **Synchronising.** A tty write returns before tmux has read it, so a read
  first appends `\x1b]2;...\x07` (OSC 2, set title) and polls `#{pane_title}`.
  tmux applies pty bytes in order, so an arrived title proves everything before
  it has been applied. The first poll always succeeded in practice. OSC 2 was
  chosen for being inert; `TestTheHarness` pins down that it does not disturb
  the last-column flag, which would otherwise hide the bug this work began with.
* **Resetting** between cases is `send-keys -R ; clear-history ; resize-window`
  in one invocation -- screen cleared, cursor homed, attributes dropped, scroll
  count back to zero, any size from 1x1 up (with `window-size manual`, which a
  detached session needs). Note `\x1b[2J` is *not* a usable reset: in tmux it
  pushes the screen into the history, which is what the scroll count reads.
* **Reading** is `capture-pane -p -e -N` plus one `display-message` for
  `#{cursor_y}`, `#{cursor_x}`, `#{history_size}` and `#{pane_dead}`, batched
  into a single tmux invocation. Two subtleties: the capture is one continuous
  SGR stream, so colour set on one row carries to the next and the parser must
  not reset per line; and rows are emitted only as far as tmux tracks them used,
  so short rows pad with default cells (`-N` is what keeps a *coloured* blank at
  the right margin from being trimmed away, which plain `-e` does trim).

Cost is 4 ms per tmux invocation and ~7 ms per observation (claim + read), so a
test case that compares a screen against a reference screen costs ~15 ms. The
per-case spawn the original timings assumed (23.6 ms) is avoided by reusing one
pane. If it ever needs to be faster, a control-mode client (`tmux -C`) removes
the fork+exec from every invocation; the sweeps did not need it.

**What this catches that the emulator did not.** tmux paints erases in the
active background colour (BCE) -- `EL`, `ECH` and `ED` all do. `_Term` always
erased to default, so it could not see the invariant `to_ansi_diff_str` observes
in `reset_colour`: colour must be back to default before any erase. Deleting
that call is caught by 6 of the new tests and by none of the old ones (the whole
old suite passed with the bug present). Reintroducing the original cursor bug
fails the two right-edge tests, and swapping the appended-row newlines for
cursor-down fails the bottom-of-screen scroll test.

`TestTheHarness` is where tmux's behaviour is written down as the specification:
deferred wrap at the margin (and that the reported column is then the width, not
width - 1), clamping on every cursor move, line feed at the bottom scrolling and
being counted, BCE, 24-bit colour round-tripping. It is the natural place to add
rows as step 5's support matrix grows.

## Caveat

tmux is a specific emulator with specific choices, so testing solely against it
makes tmux's behaviour the de facto specification. That is a far better
specification than a model we wrote ourselves, but it is still a choice, and
real terminals genuinely disagree about wrap semantics -- see
`https://github.com/mattiase/wraptest`. The support matrix in (5) is what
eventually answers this properly.
