# Axes, one side at a time

Designed 2026-08-20 by MFR and Claude (Opus 5), written up by Claude, while
adding colorbars. Nothing here is built yet. The decisions are MFR's; the
measurements and the arm table are Claude's, and are reproducible from the
snippets below.

Colorbars are what prompted it. A colorbar is a gradient `image` one or two
cells thick, and the point of it is the scale beside it, so it needs labels on
one side and nothing on the other three. `axes` cannot do that.

## What `axes` does now, and where it breaks

`axes` draws a box with `unicode_box` and paints four numbers around it: the y
limits in a left gutter, the x limits in a row below. The box comes from a
`BoxStyle`, a string of eight characters, one per edge and corner. Ticks are
part of those characters: `LIGHTX = "┬─┐││┼─┤"` has a left arm on its
north-west corner pointing at the ymax label, and a left and a down arm on its
south-west pointing at ymin and xmin.

Three things are wrong with it.

**It is all or nothing.** Every side is drawn, and the two labelled sides are
always west and south. A one-axis plot gets a box it does not want, and a plot
with no x coordinate gets an x label row with nothing to put in it.

**Narrow plots garble.** The x labels are painted at fixed opposite ends of the
row, so when they do not both fit they overwrite each other:

    axes(scatter(xy, width=2, height=3))

    1.0┬──┐
       │ ⠌│
       │⢀⠁│
       │⡐ │
    0.0┼──┤
       01.0        <- "0.0" and "1.0" overlapping

**With an axis name it crashes.** `xroom = plot.width + 2 - len(xmin_label) -
len(xmax_label)` goes negative, `xlabel[:xroom]` then truncates from the end
rather than to nothing, `.center(xroom)` passes the result through unchanged,
and the assignment fails:

    axes(scatter(xy, width=2, height=3), xlabel="time")
    ValueError: could not broadcast input array from shape (2,) into shape (0,)

## The model

Each of the four cardinal sides is independently one of four modes, a ladder:

    "crop"    no cells at all
    "pad"     one blank cell, holding the space
    "rule"    one cell, the line
    "label"   the line, ticks at its ends, and a row or column of labels
              outside it holding the two limits and the axis name

So `"label"` implies `"rule"`, and the axis name rides with the ticks. A side
may only be labelled if the matching coordinate exists: north and south need an
`xrange`, east and west a `yrange`.

Spelling: four keyword arguments taking `None` or one of those four strings.

    axes(p, north=None, east=None, south=None, west=None)

## Corners come from arms, not from a style string

Eight characters cannot express this, because a corner's glyph depends on which
of its neighbours are drawn and which carry ticks. Instead each border cell is
a set of arms --- up, down, left, right --- and the glyph is a lookup on the
four-bit mask. A ruled side contributes the arms running along itself; a
labelled side contributes an arm pointing outward at each of its ends.

    U,D,L,R = 1,2,4,8
    LIGHT = {0:" ", U:"╵", D:"╷", L:"╴", R:"╶", U|D:"│", L|R:"─",
             D|R:"┌", D|L:"┐", U|R:"└", U|L:"┘",
             U|D|R:"├", U|D|L:"┤", L|R|D:"┬", L|R|U:"┴", U|D|L|R:"┼"}

All sixteen combinations exist for light lines, and for heavy. Round shares the
light stubs and junctions and differs only in its corners. Double is missing
the four single-arm stubs. The block and tiger styles have no notion of an arm
at all.

The table reproduces today's appearance exactly. All four sides ruled, ticks on
west and south, derives `┬┐ / ┼┤`, which is `LIGHTX` character for character.
`LOWERX = "╷   │┼─╴"` comes out only approximately: its south-east is `╴`, a
bare left stub, so it ticks its origin corner and not the far one. Per-end tick
control would reproduce it; per-side is what is proposed, and nothing yet wants
the finer grain.

The consequence is that `axes` stops taking a `BoxStyle` and takes a line
weight, deriving its glyphs. `border` keeps `BoxStyle` untouched. The two
have been sharing a mechanism that only one of them wants: `border` is a
decorative box and should keep its blocks and tigers; `axes` is line art with
ticks in it.

## What `None` infers

1. Each present axis gets one labelled side: south for x, west for y.
2. The other sides are ruled only when both axes are present. A frame is for a
   2d canvas; a one-axis window gets its one labelled side and nothing else.
3. Labelling a side explicitly demotes the opposite side from `"label"` to
   `"rule"`, so an axis is never labelled twice unless asked.

Rule 2 is what gives colorbars a minimal look without `colorbar` knowing
anything about decoration:

     2d plot, unchanged           axes(colorbar(...))    axes(colorbar(..., "right"))

    1.0┬────────┐                  1.0┐▓                  ▒▒▓▓▓█████
       │   ⡀⠔⠊  │                     │▓                  ┌────────┐
       │⡠⠊      │                     │▓                  0.0    1.0
    0.0┼────────┤                  0.0┘▓
       0.0    1.0

Those corners are derived, not chosen. The west rule's top cell has a downward
arm, because the rule continues, and a leftward arm, because the end is ticked:
`┐`. The south-only bar's ends have a rule arm and a downward tick: `┌` and `┐`.

So `colorbar` stays a bare gradient with a window, and `axes(colorbar(...))` is
the idiom. One decoration mechanism, not two.

## Titles

A title embeds into the north side when that side is `"pad"` or `"rule"` --- a
padded north is a blank row, which a title fills. When north is `"crop"` or
`"label"` there is no room to embed it, and the title takes its own centred
line above everything.

## Labels that do not fit

Three failure modes were rejected before the fourth was chosen.

* **Raising** kills a running animation over a cosmetic overflow. Ranges move
  every frame, so label lengths move with them.
* **Truncating a number** prints a wrong number, which is the one thing a
  plotting library must not do. `0.` is worse than nothing.
* **Growing the plot** to fit its labels makes a plot's width depend on its
  label lengths, so a live plot jitters horizontally as its range moves, and
  drags everything laid out beside it.

What is chosen instead is the spreadsheet's answer: a number that does not fit
is replaced by `#` filling the space it had. Loud, never wrong, never resized,
never raised. Names are different: a name is text, and a shortened word does
not lie, so names are truncated as they are today.

The row is `gutter + width + 2` cells wide, and the labels may use all of it,
including the gutter columns under a labelled west side, which are blank. Each
end label is allotted half the row; one that fits is drawn against its outer
end, one that does not becomes hashes. The name takes what is left between
them, truncated. Working through it:

    w=30 g=3   |0.0             time            1.0|   ordinary
    w=30 g=3   |0.0    a very long axis name    1.0|   name truncated to the gap
    w=2  g=3   |0.0 1.0|                             garbles today; fits now
    w=2  g=3   |#######|                             genuinely too narrow
    w=4  g=0   |######|                              a short horizontal colorbar
    w=8  g=3   |-1.5   1000.0|                       uneven, both still fit

Using the whole row is what fixes the width-2 case; hashing only happens when
even that is impossible. East and west never need it: their labels are one to a
row, and the gutter is sized to the longest of them.

## Not in this design

* **Interior ticks.** Ticks live at the ends of a side, which is where the
  labels are. Anything else needs a position along the axis, which is what
  `window.dots` would provide, and a rule for choosing tick values.
* **Per-end tick control**, which is what `LOWERX` would need to be reproduced
  exactly.
* **Axis transformations.** A log axis changes where a tick falls, not which
  sides are drawn, so it is independent of all of this.

## Consequences elsewhere

`axes` carries no window. Its rectangle is the child's plus the gutters, rules
and label rows, so it is not the child's window and claiming otherwise would
misplace every coordinate in it. Practically: overlay with `dstack2` first,
then add axes. The same goes for `border`.
