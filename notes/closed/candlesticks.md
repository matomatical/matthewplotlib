# Candlestick bodies, and the eighth that Unicode is missing

Written 2026-08-20 by Claude (Opus 5) at MFR's direction, while adding
`candles`. The measurements are reproducible from the snippets below; the
decision to require a background colour is MFR's.

A candle body is an *interval*: it spans the opening and closing values, and
unlike a bar it is anchored to neither end of the column. That one difference
is what makes it awkward to draw in character cells, and it is the whole
subject of this note.

## Three substrates, measured

The plot needs many candles side by side, bodies that read as solid, and wicks
that read as hairlines. Three vocabularies were rendered on the same 40-period
random walk at `height=14` and compared.

**Braille** (2 by 4 dots per cell, so 4x the vertical resolution) lost on
looks. A filled body is a dot matrix, and its ends come out ragged; body and
wick are both made of dots and stop being distinguishable. The same experiment
rejected braille for box plots, where a small outlined rectangle reads as noise
(`⡯⠭⢽`).

**Half-blocks** (` ▄▀█`, 2x vertical) with wicks from the vertical line stubs
(` ╷╵│`) look right immediately: fat body, thin wick. But the resolution costs
something real. At `height=14` a half-block grid has 28 levels, and over 40
candles:

    distinct body heights:  8      bodies collapsing to zero height:  7/40

**Eighth-blocks** (`▁▂▃▄▅▆▇█`, 8x vertical) fix that:

    distinct body heights: 20      bodies collapsing to zero height:  2/40

and then fail for a different reason. Unicode's eighth-blocks are
*bottom-anchored only*. The upper-anchored set is three characters: `▔` (one
eighth), `▀` (a half), `█`. An interval needs both a top and a bottom edge, so
a body whose bottom edge lands high in a cell has nowhere to go but `▔`, a
hairline that reads as a *wick*:

    half-blocks, symmetric        eighths, asymmetric
    ╷ ▄██      ╷ ▄▄██╷╷          ╷ ▃▀█      ╷ ▁▁██╷╷
    ▄ █╵█╷╷    │▄█████▀█         ▂ █╵█╷╷    │▃▀████▀▅
    █▄█ ▀▀█▄▄  █▀│████ █▄╷       █▂█ ▔▔█▂▂  █▔│██▀▀ ▀▃
     ▀▀ ╵ ││█ │█  ▀▀╵╵ ╵█▄╷       ▔▔ ╵ ││█ │█  ▔▔╵╵ ╵▀▃

Worse than coarse: a rising and a falling candle of identical geometry render
differently, because one anchors where the glyphs exist and the other does not.
Rendered to PNG, bodies visibly break into floating fragments.

## What the background buys

There is a third option, and it is the one built. A cell has a foreground and a
background colour, so "the body fills the upper five eighths" is the same
picture as "the block filling the lower three eighths, painted in the
background colour, over a cell whose background is the body colour". Every
eighth becomes reachable, symmetrically, by drawing half of them as negatives.

The catch is that this needs the background *named*. ANSI can select a default
foreground (SGR 39) and a default background (SGR 49), but there is no code for
"use the default background as a foreground", so a plot that leaves its
background to the terminal cannot invert against it. So `candles` takes a
`background` and paints its whole rectangle, which no other plot in the library
does.

Two things were considered and dropped in favour of just requiring it. Making
the background optional and silently falling back to half-blocks gives the user
a quarter of the resolution for a reason they cannot see. Reaching only the
eighths that need no inversion is the asymmetric render above.

Rendered on a dark background this is also, by some distance, the best-looking
of the three. That was not the argument for it, but it is not an accident
either: precise bodies are continuous bodies.

## What is still approximate, and by how much

`unicode_candles` places a body to the nearest eighth of a cell and a wick to
the nearest half, and two approximations survive that.

**A body confined to a single cell.** The expressible fills within one cell are
those anchored to its top edge and those anchored to its bottom edge; a fill
floating between the two is not among them. Such a body keeps its length and
shifts to whichever edge is nearer. Length is preserved rather than position
because length is the open-to-close move, which is what a candle is for, and
because the alternative — growing the body out to the nearer edge — exaggerates
that move. A body of zero length is drawn as one eighth, so a candle that
opened and closed at the same value reads as a hairline rather than vanishing.

The shift is bounded at three eighths of a cell: it is `min(above, below)`
where `above + below = 8 - length` and both are at least 1, which is largest at
`length = 1`. Measured over 3000 random candles at `height=10`, checked against
the rendered pixels rather than the chosen glyphs:

    body length exact:                     3000/3000
    position exact (body spans two cells):  1613/1613
    bodies confined to one cell:            1387/3000
    shift, eighths:                         mean 1.18, max 3

**The wick, where it meets the body.** The body is drawn over the wick, so in
the one cell holding a body's edge, whatever the body does not fill shows
background rather than wick. A gap of up to seven eighths of a cell can open
between a body and the wick leaving it. Letting the wick win instead would cost
the body's edge, which is the more important of the two. Both ends of a wick
round inward, under-claiming the high and the low rather than over-claiming
them, which is the convention `unicode_bar` already follows.

Neither approximation is a candidate for fixing with more colours; both are
about what one glyph can show. The octants of Unicode 16 (U+1CC00–U+1CEBF,
already on the roadmap as an opt-in) would settle both, since a 2 by 4 grid of
independently settable blocks can float a fill inside a cell and can put a
narrow wick beside it.

## Reproducing the numbers

The resolution comparison, given `opens`, `highs`, `lows`, `closes` over `n`
periods and a `height`:

```python
levels = height * per_cell        # per_cell = 2 for half-blocks, 8 for eighths
def level(v):
    return np.clip(np.floor((vmax - v) / (vmax - vmin) * levels), 0, levels - 1)
sizes = level(np.minimum(opens, closes)) - level(np.maximum(opens, closes))
len(set(sizes.tolist())), (sizes == 0).sum()
```

The placement check reads the pixels back, which is the only way to see a
negative and a direct fill as the same mark. `CharArray.to_rgba_array` renders
each cell sixteen pixel rows tall through the bundled font, so an eighth of a
cell is exactly two rows; find the rows carrying the body colour and compare
their extent against `_sub_cell_rows`. `TestUnicodeCandlesBodies` in
`tests/test_core.py` does this one case at a time.
