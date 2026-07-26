# Animations — design notes

Written 2026-07-26 (Claude), alongside building the roadmap's animation
entries, once differential rendering had been confirmed against a real terminal
(`tests/test_terminal.py`). Covers `matthewplotlib.animations`: `tstack`,
`animation` and `animate`.

## Two halves of one loop

Animating in a terminal splits cleanly into a value and a session, and the
library offers both.

* **`tstack`** is an animation as a value: a sequence of plots with a frame
  rate. Pure — no terminal, no clock, no I/O. It slices, maps, plays and saves
  to a gif, in the same way a `plot` composes and saves to a PNG.
* **`animate`** is an animation as a session: a context manager that owns the
  terminal for the duration of a `with` block, so it can do the things no
  string-returning function can.

They are not rival APIs for the same job, which is the trap this design spent a
while in. They are a value and the thing that plays it, and each produces the
other:

* `animate(record=True).frames` hands back a `tstack`.
* `tstack.play()` opens an `animate` session and pushes its own frames through
  it.

So "pull" — iterating a finished animation — is literally implemented as the
push loop, and there is one code path for both. That is the whole argument for
building the two together rather than either alone.

Push has to be the primitive, not pull, because pull needs every frame up
front. Game of Life, a training curve, a webcam and a system dashboard — four of
the six animated examples — do not know frame `t+1` until frame `t` has been
shown, and cannot be expressed as a finished value at all.

## What the session is for

Before this, the six animated examples in `examples/` each hand-rolled the same
four incidental concerns, and agreed with each other on none of them:

* the `prev = None` sentinel and the assignment maintaining it: all six;
* frame timing: a flat `time.sleep(1/fps)` in four, drift-corrected sleeping in
  only `mandelbrot` and `life`;
* tidying the terminal on the way out: a bare `print()` from a
  `KeyboardInterrupt` handler in four, nothing in the other two;
* collecting frames for export: six spellings of `frames = [] if save else
  None` plus a guarded `append`.

None of that belongs in a plotting call and all of it belongs to animating in a
terminal, which is the library's business.

`update()` writes one frame in one `print`: a full render for the seed frame, a
diff thereafter, and the newline bookkeeping never surfaces. It returns the
string it wrote, so a caller that wants to know what a frame cost — which is the
entire point of `examples/life.py` — can still measure it. That also makes the
session testable without a terminal.

## Everything the session adds is opt-in

Matthew's framing, and the reason the plain `print(plot - prev)` loop stays
first-class and fully supported: the library is a library, not a framework. A
context manager takes over the client's control flow, which should always be
avoidable, so within the session each extra service is opt-in too and the caller
chooses how much framework they want.

* `mp.animate()` — printing only: the `prev` bookkeeping and one write per
  frame.
* `mp.animate(fps=20)` — also owns the clock. Worth having because it makes
  drift-corrected sleeping the default everywhere instead of the naive
  `sleep(1/fps)` that four of the six examples got wrong. An upper bound on the
  frame rate, never a guarantee.
* `mp.animate(record=True)` — also accumulates frames, readable afterwards as
  `anim.frames`. Opt-in because holding every frame is a memory cost, in the way
  `fps` is opt-in because owning the clock is a control-flow cost. Reading
  `.frames` without it raises rather than returning empty: silently losing a
  ten-minute run to a forgotten keyword is the worst failure available here.
* `mp.animate(stop_on_interrupt=True)` — Ctrl-C ends the animation instead of
  propagating, so the statements after the block still run and a recording
  survives to be saved. Off by default, because swallowing `KeyboardInterrupt`
  is exactly the control-flow capture the rule above is about; the examples pass
  it explicitly, which is also how they shed their `try/except` blocks.

## `anim.print`

The one service that genuinely cannot be had outside a session. Once something
owns the cursor, a bare `print()` from user code lands in the middle of the plot
and corrupts it, so debugging an animated program today means choosing between
not printing and not animating.

The mechanism needs no new machinery at all — three prints of strings the
library already emits:

    print(-prev)      # clearstr: erase the plot, step to the row above it, so
                      # print's newline lands the cursor at the plot's top row
    print(message)    # the message takes that row
    print(prev)       # full redraw, one row lower than before

The plot ends up one row further down with the message above it, and near the
bottom of the screen the terminal scrolls, which is the wanted behaviour: a log
that grows upward with the plot pinned below it, as `tqdm` does.

The subtle part is that `prev` survives unchanged. `CharArray.to_ansi_diff_str`
states its cursor contract relative to the cursor, not absolutely on the screen,
so moving the whole plot down a row does not invalidate the next diff. After the
redraw the screen shows `prev` and the cursor sits below it, which is exactly the
invariant `update()` assumes.

The cost is one full repaint per printed line. Printing is human-paced, so that
is free in practice. It inherits `clearstr`'s requirement of a spare row above
the plot: printed from the very top row of the screen, the plot drops one extra
row, once.

Rejected: scroll regions (`\x1b[r`) and insert-line (`\x1b[L`). Both are
narrower in terminal support than the handful of sequences the library already
relies on, and neither buys anything over a repaint that costs nothing.

`anim.out` is the same thing as a file, for the callers that want somewhere to
write rather than something to call — `print(..., file=anim.out)`,
`logging.StreamHandler(anim.out)`. Line buffered, because the plot can only be
moved out of the way a whole line at a time, and flushed on the way out of the
block so a `print(..., end="")` is not silently swallowed.

Having it also settles the question of routing *third-party* prints, without the
library making that decision for anyone. A caller who wants a print from deep
inside someone else's training loop to land above the plot writes:

    with mp.animate(fps=20) as anim, contextlib.redirect_stdout(anim.out):

which is explicit, scoped, and theirs to opt into. The one thing that has to be
true for it to work is that the session must not write frames through
`sys.stdout` at the time it writes them, or its own output would recurse back
through `anim.out` forever. So the session captures the real stdout in
`__enter__` and writes there for the rest of the block. That is a better rule
independently: an animation should be drawn where it started being drawn, whatever
the program does to `sys.stdout` in the meantime.

## Frame rate, requested and achieved

`fps` is an upper bound, and the whole of a frame's cost is spent *inside* its
budget rather than on top of it. There are three costs, and getting all three
inside took two goes:

* the caller's compute, which is inside because the sleep happens at the top of
  `update`, before the write, rather than at the bottom;
* the diff and the write — working out `updatestr` and pushing the bytes down the
  wire — which are inside because the next frame is scheduled from the moment
  this frame's slot *opened*, not from the moment its write finished;
* the sleep's own overshoot, which does not accumulate because the schedule
  advances by exactly one period from the last deadline.

So: sleep until `deadline`; take `due = now`; write; then
`deadline += 1/fps`, and only if that lands before `due` — meaning the frame ran
more than a whole period late — resynchronise to `due + 1/fps`.

Both of the later two were bugs the tests caught, and both are invisible without
a clock that charges for things:

* The first draft scheduled from `max(now, previous_deadline + 1/fps)` with `now`
  read *after* the write. Wrong brackets: a 250ms frame at 20fps was followed
  immediately by the next write with no delay at all, because the overrun was
  paid back out of the following frame's slot, so one slow frame became a
  stutter rather than a recovery.
* Reading the clock after the write also put the write's own cost outside the
  budget, so the achieved period was `1/fps` *plus* the render and write time —
  the same flaw as the flat `sleep(1/fps)` the examples used, one term smaller.
  A fake clock only catches this if writing to it costs something, which is why
  `tests/test_animations.py` has a stdout that charges the clock per write.

Four of the six examples used a flat `sleep(1/fps)` after the write, which runs
slow by the sum of all three.

The session times every frame whether or not it is recording, since that costs
two floats, and exposes `anim.achieved_fps` — the rate the animation actually
managed, or `None` before two frames have been written. Asking for 20 and
getting 6 is the diagnostic you want when an animation feels sluggish, and no
example could report it before.

When recording, the per-frame intervals travel with the frames, so a gif can be
written at the rate that was asked for (the default, and what every example did
implicitly), at the rate actually achieved (`fps="achieved"`, faithful right
down to reproducing a stall — honest, and rarely what a showcase gif wants), or
at any other rate.

Measured 2026-07-26, Pillow 12.3.0, against `tstack.savegif`:

* The standing TODO in the old `save_animation` claiming gif durations were
  broken by an internal RGBA→P conversion is **stale**. Both a scalar
  `duration` and a per-frame list round-trip exactly.
* GIF stores delays in centiseconds, so a requested duration quantises to 10 ms.
  `fps=12` asks for 83 ms, gets 80 ms, and plays at 12.5. `fps=60` asks for
  17 ms, gets 10 ms, and plays at 100. Above roughly 50 fps the number in the
  file is fiction, and many viewers additionally clamp delays under 20 ms.

## Composition

`tstack.map(f)` applies a plot-to-plot function to every frame, which is how an
animation composes: `anim.map(lambda p: mp.border(p, title=" life "))` puts a
static border around a moving interior. One method, and it covers every
combinator in `plots.py` including ones not written yet.

Two roads not taken, both worth naming so they are not re-proposed:

* **Teaching the existing combinators to dispatch on animations**, so
  `mp.border(anim)` works directly. That is a change to every class in
  `plots.py`, and `map` gets the same result for the price of a lambda.
* **Making `tstack` a subclass of `plot`** whose `chars` is the first frame.
  Composition would work for free, and `hstack(anim, anim)` would silently
  collapse to two static first frames. It is the obvious shortcut and it is a
  lie about the type.

n-ary composition of animations stays a comprehension for now —
`mp.tstack(*[mp.border(f) + panel for f, panel in zip(a, b)])` — because the
common case is lifting one unary combinator, which `map` covers. The symmetric
question of whether the *static* composites should have `map` too is a separate
design with its own note: `notes/mapping-over-composites.md`.

## Deliberately not done

* **No cursor hiding.** The visible block cursor sitting inside an animated plot
  has never actually bothered anyone using this library, and hiding it means
  owning restoration through SIGTERM and hard crashes, not just the
  `KeyboardInterrupt` that `__exit__` sees. A library that leaves an invisible
  cursor behind fails worse than one that leaves a flickering one.
* **No iteration protocol on the session.** `for t in anim:` was drafted — it
  would have absorbed the `frame = 0` / `while num_frames == 0 or frame <
  num_frames` / `frame += 1` trio that all six examples also duplicate. It reads
  badly at the call site, and the loop bound is the caller's business.
* **No tty detection.** That is, nothing changes *what is written* depending on
  whether a terminal is attached. Nothing in the tree needs it now that the
  example tests replay into a real terminal, and adding it would quietly damage
  them: examples run under `contextlib.redirect_stdout` (`tests/examples.py`), so
  `sys.stdout.isatty()` is false inside every example the goldens cover, and a
  library that branched on it would have its whole example suite exercising the
  degraded path. If it is ever wanted, the test recorder needs `isatty()` to
  answer true and the degraded path needs tests of its own.

  The height check below is the one place that asks, and it asks a different
  question: not "should the output differ" but "is there a terminal here to
  measure". `_terminal_rows` reads the descriptor rather than calling
  `shutil.get_terminal_size`, because the latter's whole job is to substitute a
  fallback size, and a fallback must not be mistaken for a measurement -- it
  would have the session warning about a 40-row plot in a redirected run on the
  strength of an invented 24-row terminal.
* **No alternate screen buffer.** `\x1b[?1049h` gives full-screen-application
  mode and restores the terminal perfectly on exit — including erasing the plot,
  which is the wrong model here. Every string this library emits assumes inline
  output that stays on the scrollback where it was printed.
* **No clipping.** A plot taller than `R-1` cannot animate by any path (see
  `notes/terminal-aware-printing.md`); the session warns once, naming both
  sizes, and draws it anyway rather than raising, because killing a long run
  over a small window is worse than a torn frame. Clipping proper is a layout
  pass, not a session feature.

## Settled

* **`stop_on_interrupt` stays opt-in**, defaulting off, even though every
  converted example turns it on. Catching `KeyboardInterrupt` is the one service
  here that changes what the *program* does rather than what the terminal shows,
  and a library should be asked before it does that. The examples passing it
  explicitly is the API working, not evidence against the default.
* **Frames are padded to a common size** on construction, top-left aligned.
  Differential rendering is what makes this free: the padding is blank in every
  frame, so no cell of it is ever sent twice, and the alternative was every
  caller having to know to pin whatever made the plot change size — as
  `examples/life.py` pins its axis ranges.

## Open

Roadmap entries exist for each of these.

* **An operator for `tstack`.** The old sketch wanted `|`, which vstack has.
  `>>` reads as "then" and would give time-concatenation a spelling. Part of
  finalising operator assignment, since the budget is nearly spent.
* **A 3-D `CharArray`** (`codes[T,H,W]`) instead of a tuple of plots. It would
  vectorise composition and fits the "clean up the backend with pytrees and
  vectorisation" entry, at the cost of losing the per-frame plot subclass. Worth
  doing when building a long `tstack` shows up in a profile.
* **Per-frame durations as the primitive, instead of one frame rate.** A `tstack`
  carries a scalar `fps` plus, if it was recorded, a duration per frame; only
  `savegif(fps="achieved")` reads the latter. If durations were the primitive,
  concatenating a 30fps animation onto a 5fps one would keep both parts playing
  at their own speed instead of flattening to the first one's rate, and
  `savegif` would need no special case. Gifs already support it — a per-frame
  delay list round-trips exactly (measured above) — and so could `play`.
