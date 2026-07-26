# Animation context manager — design notes

Designed 2026-07-25 (Matthew + Claude), deferred to the roadmap after the
differential rendering work landed. Not implemented.

## Why

With `print(plot - prev)` the *rendering* is settled, but the five animated
examples still each hand-roll the same four incidental concerns, and they do
not agree with each other on any of them:

* the `prev = None` sentinel and the assignment that maintains it
* frame timing (`time.sleep(1/fps)` in four of them, drift-corrected
  `sleep(max(0, start + 1/fps - now))` only in `mandelbrot`)
* tidying the terminal on exit (three end with a bare `print()`, from a
  `KeyboardInterrupt` handler)
* the cursor stays visible, so a block cursor flickers inside every plot

None of that belongs in a plotting call, but all of it belongs to *animating in
a terminal*, which is the library's business.

## Shape

    with mp.animation(fps=20) as anim:
        while ...:
            anim.update(mp.axes(...))

`update()` writes one frame in one write: a full render for the seed frame, a
diff thereafter, and the newline bookkeeping never surfaces. The context
manager hides the cursor on entry and, on exit -- including via
`KeyboardInterrupt` or an exception -- restores it and leaves the terminal on a
fresh line below the plot.

Naming follows the library's lowercase-noun convention (`plot`, `axes`,
`border`), and `update` matches the existing `updatestr`.

## Everything opt-in

Matthew's framing: the library is a library, not a framework, so the plain
`print(plot - prev)` loop must stay first-class and fully supported -- a
context manager takes over the client's control flow, which should always be
avoidable. Within the context manager, each extra service is also opt-in, so
the user chooses how much framework they want:

* `mp.animation()` -- printing and cursor handling only.
* `mp.animation(fps=20)` -- also owns frame timing. Worth having because it
  makes drift-corrected sleeping the default everywhere instead of the naive
  `sleep(1/fps)` that four of the five examples get wrong. An upper bound on
  the frame rate, not a guarantee.
* `mp.animation(track=True)` -- also accumulates frames for export. The object
  outlives the `with` block, so they are just read off afterwards:

      with mp.animation(fps=20, track=True) as anim:
          for ...:
              anim.update(plot)
      mp.save_animation(anim.frames, "out.gif", fps=20)

  Opt-in because holding every frame is a memory cost, in the same way `fps` is
  opt-in because owning the clock is a control-flow cost.

## Open questions

* **Non-tty.** When stdout is redirected, cursor codes are noise. Options:
  detect and fall back to plain full frames separated by newlines (consistent
  with the 0.3.8 fix that made `wrap` work without an attached terminal);
  suppress output; or do not special-case it. Leaning on the first.
* **Height check.** Animation needs the plot to fit in the terminal with a
  spare row for the trailing newline (H <= R-1; see the screen-edge notes in
  `CharArray.to_ansi_diff_str`). The context manager is the natural place to
  check that and say something useful, since it can call
  `shutil.get_terminal_size()`. The pure string API cannot.
* **Which examples convert?** Not necessarily all of them: at least one should
  keep the bare `print(plot - prev)` loop, since that is the idiom the README
  teaches and it needs to stay visibly first-class.
* Whether `update()` should return anything (the string it wrote? the frame?).

## An older, different shape

Added 2026-07-26, harvested from `notes/closed/design.md` — the pre-library
design sketch, which had already reached for an animation context manager and
arrived somewhere else. Worth reading before building the above, because it
disagrees on the fundamentals rather than the details.

There, an animation is a **value** first: frames are stacked into one object
with `tstack` (or the `|` operator), and the context manager *plays* it.

    a = mp.tstack(*[mp.line(...) for y in np.arange(0, 2*np.pi)])

    with mp.animate(a, loop=True) as anim:
        for t, frame in enumerate(anim):
            time.sleep(0.04)
            anim.print("frame", t)

So the loop **pulls** frames out of a finished animation, where the design
above has the user **push** frames in with `anim.update(plot)`. The difference
is not cosmetic:

* **Pull needs every frame up front.** That is exactly what the `track=True`
  option above collects, but as an output rather than an input. Pull cannot
  animate anything reactive — Game of Life, a training curve, a webcam — since
  those do not know frame `t+1` until frame `t` has been shown. Push handles
  both, and can be given a `tstack` to iterate over trivially.
* **Pull makes an animation a first-class value,** which push does not. It can
  be sliced, looped, composed, exported, or `hstack`ed with another animation
  before anything is printed. `save_animation` currently takes a list of frames
  and would be the natural consumer.

Push is the right default, and the design above should stay as it is. But
`tstack` as a composition primitive is worth having on its own terms — it is on
the roadmap now — and if it exists, `mp.animate(tstack_value)` is a thin
convenience over the push loop rather than a competing API.

### `anim.print`

The sharper idea, and one the design above has no answer for: **printing while
an animation is running.** Once the context manager owns the cursor, a bare
`print()` from user code lands in the middle of the plot and corrupts it. The
sketch's answer is to route it — `anim.print(...)` — so the context manager can
scroll the plot, emit the line above it, and repaint.

This is a real gap. Debugging an animated program is a normal thing to want to
do, and today the only options are to not print or to not animate. It also
interacts with the height budget in `notes/terminal-aware-printing.md`: a plot
occupying `R-1` rows has nowhere to put a printed line without scrolling the
plot off, so `anim.print` and the height check are the same question asked
twice.

Open: whether it is a method, a file-like object the user can pass to
`print(file=...)` (the sketch's own parenthetical alternative), or a
redirect of `sys.stdout` for the duration of the block. The last is the most
convenient and the most magical.

## Prerequisite

Confirm the differential rendering foundation behaves in a real terminal
first. Everything about it so far was verified against the ANSI emulator in
`tests/test_core.py`, which is a model of a terminal, not a terminal.
