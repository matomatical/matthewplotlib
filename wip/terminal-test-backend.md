# Testing ANSI output against a real terminal

Investigated 2026-07-25 (Matthew + Claude), after the differential rendering
work. Plan agreed, not implemented.

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
   alongside uv, make and pandoc; tests skip with a message telling you to
   install it. Reasonable to assume a developer can install tmux.

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

   * `ECH` is the one to scrutinise -- least commonly exercised of the set, and
     only used when a plot narrows. Writing spaces is universally supported and
     costs bytes in that one case only.
   * `CNL` could be `CR` plus `CUD`.
   * truecolor varies but degrades to the nearest colour: a rendering
     difference, not a correctness bug.
   * deferred wrap we already do not depend on -- the bookkeeping was made
     conservative instead. That is the pattern to generalise.

5. **Build a terminal support matrix** (Matthew's idea, and the eventual public
   artifact): the sequences and behaviours we rely on, against the terminals
   people actually use -- what supports what, and where behaviour differs.
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

## Caveat

tmux is a specific emulator with specific choices, so testing solely against it
makes tmux's behaviour the de facto specification. That is a far better
specification than a model we wrote ourselves, but it is still a choice, and
real terminals genuinely disagree about wrap semantics -- see
`https://github.com/mattiase/wraptest`. The support matrix in (5) is what
eventually answers this properly.
