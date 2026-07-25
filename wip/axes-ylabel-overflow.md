# `axes` writes the ylabel onto the right border when y ticks are narrow

Found 2026-07-25 (Matthew + Claude) while building the Game of Life example.

`mp.axes` pads a gutter on the left as wide as its widest y tick label, and
paints the (vertical) ylabel one column in from the right of that gutter:

    L = max(len(ymin_label), len(ymax_label))
    chars_padded = chars_boxed.pad(left=L, below=1, fgcolor=color)
    ...
    chars_padded.codes[1:-2, L-1-ypad] = ords(ylabel)

When the tick labels are one character wide -- a y range of `(0, 7)` with
`yfmt="{y:.0f}"`, say -- `L` is 1 and the column index is `1-1-1 = -1`. numpy
reads that as the *last* column, so the ylabel is painted down the right-hand
border of the plot, silently replacing it:

    7┬────────────────────┐
     │                     
     │  ⠁                 k     <- 'k' where the border should be
     │                    B     <- 'B'
     │⡀                    
    0┼────────────────────┤
     0        xlab       10

`ypad` does not help: it only shifts the column further left, so `ypad=2`
lands on the second-to-last column and so on. A four-character ylabel with a
one-character gutter walks four characters down the right-hand edge.

There is no error and no visual clue other than the missing border, so this is
easy to ship without noticing.

## Fix

The index needs to be in range, but clamping alone would let the ylabel
collide with the tick labels in the same column. Better to guarantee the room
when there is a label to place:

    L = max(len(ymin_label), len(ymax_label))
    if ylabel:
        L = max(L, ypad + 1)

Tick labels are already right-aligned into the gutter
(`codes[0, L-len(ymax_label):L]`), so widening it does not disturb them, and
the ylabel column `L-1-ypad` is then always at least 0. Note this changes the
width of any `axes` whose y ticks are narrower than `ypad + 1`, so a couple of
the example images may shift by a column.

Worth a test that asserts the border is intact for a one-character y range,
since the failure is silent.

## Workaround in the meantime

Pad the tick format so the gutter is wide enough: `yfmt="{y:2.0f}"` gives a
two-character gutter regardless of the values, which is what `examples/life.py`
does.
