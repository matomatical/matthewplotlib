Notes from MFR using the library
================================

OK, Claude and I have added a lot of new features. Mostly I'm happy with the
look of the API and the AI-generated examples, but the API needs refining
through human use and I'm a little worried about some of the internals getting
messy. I'll try dogfooding it for the schedule/spirit apps and see where the
issues are and note them here to follow up later.

Issues encountered building scheduling app:

* Tables API is good, looks great!
* I would like to be able to highlight specific rows though. At the moment the
  only way to highlight cells is to supply a full grid of colours. It would be
  nice to be able to do the following:
  * Provide a list of colours for only rows, or only columns, and have it
    broadcast over the other.
  * Provide the list/grid of colourlikes, including named colours, as the
    docstring suggests should be possible?
    * I glanced at the parse_colors code, it looks frightening.
    * This is due for rethinking, already on the roadmap so seems fine.
  * Provide a *partial* colour specification. At the moment I don't think it's
    possible to get the default fg and bg color (off-white, transparent in my
    case) if you want to colour any cells.
    * How best to achieve this for tables is a different question. One basic
      method would be allowing leaving None in some parts of the grid/list (for
      default fg/bg).
    * Better yet, a *sparse* partial colour specification, with targeted
      cell/row/cols to specify the colour of (e.g. a dict[tuple[int,int],
      ColorLike incl. str]).
* I wonder if it ever makes sense to integrate the animation context manager
  with an infinite loop. Something like:
  ```
  anim = mp.animate(fps=1, stop_on_interrupt=True)
  with anim:
    while True:
      pass
  ```
  Becomes:
  ```
  anim = mp.animate(fps=1, stop_on_interrupt=True)
  for _ in anim:
    pass
  ```
  Probably not worth it...
* Bars API needs to be able to set background color as well as fg color.
* Plots file should have types lifted to the top. I'm still not convinced they
  belong in here. Likewise helpers. But it would probably be easier to swallow
  if we split the file.
* Blank should maybe get a bgcolor param so we can make coloured patches, atm I
  am using text with blank spaces.
* Why didn't mp.text("", width=5, bgcolor="cyan") work? Zero lines, zero
  height.

Other notes for later:

* We should revisit the policy of using `_names` to indicate 'not part of the
  public `mp.*` api' for export testing purposes, but allowing these to be
  exported/imported within mp internals.
* Think more carefully about global policy for erroring or silently ignoring
  values outside of scale bounds for colours and data axes. Currently
  inconsistent, need to make sure it's principled and predictable, or change
  the policy.

Notes on axes:

* The realisation that these scales are actually the things window should store
  as an axis makes sense, I guess it would be even more discoverable if we
  renamed the xrange, yrange, vrange etc. args to xaxis, yaxis, vaxis, caxis
  etc.? Maybe this is something I want to consider before v1. I know it would
  be a breaking change, but it's one I'm certainly comfortable making before
  v1.

Plans for the next release:

* We've added a bunch of features mainly around data and colour scales.
  Also made a few minor changes like how axes are interpreted. I think this
  might be worth bumping to v0.8.0, but then I want to wait to bundle a few
  more axis-related things:
  * Make value-axes on bar charts and etc.
  * Histogram category axis is a value axis or a categorical axis? Maybe
    neither?
  * Design plumbing for coordinate axes, and, harder, rendering. Rendering is
    kinda easy for vertical categorical axes since we have one line per
    category. Harder for horizontal, but we can think of something, and OK if
    it's a compromise?
  * While we're there, backgrounds, alignment, and negative values for
    bar/column charts please. We learned some relevant lessons for this when
    doing candlestick plots.
* Check: Did we lift types to the top of each file during the plot split? Check
  docs for each module read well.
* Miscellaneous blank bgcolor and empty-string rethink may as well.
* Deeper colour specification refactor and the table colour specification
  enhancements I enumerated above need careful design, unsure if I want them in
  this release, see how we feel after we finish the axes and bar charts
  sidequest.
