# Animations — design notes

Written 2026-07-26 (Matthew + Claude), alongside building the roadmap's
animation entries, once differential rendering had been confirmed against a real
terminal (`tests/test_terminal.py`). Covers `matthewplotlib.animations`:
`tstack` and `animate`.

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

Deferred: redirecting `sys.stdout` for the duration of the block. It is the only
option that fixes *third-party* code that prints — a training loop deep in
someone else's library — which is a real pull. But it makes the session's
behaviour depend on global state, and it cannot tell a stray print apart from
the session's own writes. If it is built it should be an explicit opt-in, not
the default. A file-like `anim.out` for `print(file=...)` is the cheaper half of
the same idea and would compose with `logging`.

## Frame rate, requested and achieved

`fps` is an upper bound. The session schedules each write for
`max(now, previous_deadline + 1/fps)`: a frame that overruns its budget does not
accumulate drift, and does not then burst to catch up. Four of the six examples
used a flat `sleep(1/fps)`, which pays the sleep *on top of* the compute and so
runs slower than asked by exactly the compute time.

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
* **No tty detection.** Nothing in the tree needs it now that the example tests
  replay into a real terminal, and adding it would quietly damage them: examples
  run under `contextlib.redirect_stdout` (`tests/examples.py`), so
  `sys.stdout.isatty()` is false inside every example the goldens cover, and a
  library that branched on it would have its whole example suite exercising the
  degraded path. If it is ever wanted, the test recorder needs `isatty()` to
  answer true and the degraded path needs tests of its own.
* **No alternate screen buffer.** `\x1b[?1049h` gives full-screen-application
  mode and restores the terminal perfectly on exit — including erasing the plot,
  which is the wrong model here. Every string this library emits assumes inline
  output that stays on the scrollback where it was printed.
* **No clipping.** A plot taller than `R-1` cannot animate by any path (see
  `notes/terminal-aware-printing.md`); the session warns once, naming both
  sizes, and draws it anyway rather than raising, because killing a long run
  over a small window is worse than a torn frame. Clipping proper is a layout
  pass, not a session feature.

## Open

* Whether `stop_on_interrupt` is opt-in by the right default. It is off here on
  principle, and every example turns it on, which is mild evidence that the
  default is backwards.
* An operator for `tstack`. The old sketch wanted `|`, which vstack has. `>>`
  reads as "then" and would give time-concatenation a spelling, but the operator
  budget is nearly spent — folded into the roadmap's "finalise operator
  assignment".
* A 3-D `CharArray` (`codes[T,H,W]`) instead of a tuple of plots. It would
  vectorise composition and fits the "clean up the backend with pytrees and
  vectorisation" entry, at the cost of forcing uniform frame sizes and losing
  the per-frame plot subclass. Worth revisiting only if building a long `tstack`
  shows up in a profile.
* Frames of differing sizes. `savegif` pads to the largest, top-left aligned, as
  `save_animation` did; the terminal path renders them correctly frame to frame
  (that is what `plot - prev` falls back to) but the plot visibly jitters, and
  `examples/life.py` pins its axis ranges specifically to avoid it. A `tstack`
  that padded its frames to a common size on construction would fix the jitter
  for everyone, and is probably the right default.
