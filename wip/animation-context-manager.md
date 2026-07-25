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

## Prerequisite

Confirm the differential rendering foundation behaves in a real terminal
first. Everything about it so far was verified against the ANSI emulator in
`tests/test_core.py`, which is a model of a terminal, not a terminal.
