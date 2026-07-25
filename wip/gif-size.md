# Shrinking animation gifs

Measured 2026-07-25 (Matthew + Claude) getting `images/life.gif` down to a size
fit for the README. Applies to any gif `save_animation` produces, since they
all share the same character: a handful of flat, maximally distinct colours and
no gradients.

## What does not work

**Lossy compression does nothing.** `gifsicle --lossy` nudges pixels toward
*nearby* palette entries so that LZW runs get longer. Terminal output has no
nearby colours to nudge toward -- pure black background, bright white, saturated
green -- so there is nothing to trade away:

    gifsicle -O3            1247 kB   (82% of original)
    gifsicle -O3 --lossy=20 1249 kB   (82%)
    gifsicle -O3 --lossy=50 1249 kB   (82%)
    gifsicle -O3 --lossy=200 1230 kB  (81%)

Anyone reaching for `--lossy` on a plot gif is wasting their time, and risking
artefacts on braille dots for nothing.

**Tightening the palette does nothing either.** The whole Game of Life
animation, panels and scatter blends included, uses **28-37 distinct colours**.
`gifsicle -O3 --colors 32` came back byte-identical in size to `-O3` alone.

**There is no expensive prefix to trim.** Cost accumulates at a fairly steady
~10 kB per frame at 624x480; the chaotic opening frames cost only modestly more
than the settled ones. Starting the recording later saves nothing much.

## What does work

Only two things: **how many frames** and **how many pixels in each**.

| board | frames | dimensions | saved | + `gifsicle -O3` |
|-------|--------|------------|-------|------------------|
| 76x20, panel 5 | 128 | 624x480 | 1517 kB | 1247 kB |
| 76x20, panel 5 |  96 | 624x480 | 1236 kB | 1021 kB |
| 76x20, panel 5 |  64 | 624x480 |  856 kB |  704 kB |
| 60x14, panel 4 |  96 | 496x368 |  643 kB |  543 kB |
| 60x14, panel 4 |  64 | 496x368 |  466 kB |  402 kB |

`gifsicle -O3` is a flat ~15% and is **lossless** (verified: maximum channel
difference 0 across every frame).

A Pillow pass -- convert every frame to one shared palette, then save with
`optimize=True` -- is worth a further ~13%, and the two stack (96 frames at
full size: 1236 -> 1138 with Pillow, -> 980 with gifsicle after it). But doing
it with `Image.quantize()` is *not* bit-exact: it introduced a maximum channel
difference of 3, from the quantiser's nearest-colour matching rather than from
any shortage of palette entries.

## Do not decimate frames

Halving the file by keeping every second frame is tempting and wrong for Life
specifically: blinkers have period 2, so sampling every other generation
freezes every oscillator into an apparent still life, and the animation loses
the most visibly alive thing on the board. Shorten the run instead.

The general form of the trap: check the sampling rate against the periods
actually present in the animation before decimating.

## Library follow-up

`save_animation` calls `Image.save(...)` without `optimize=True`, so every gif
the library produces is ~13% larger than it needs to be, for users who do not
have gifsicle. Worth passing it, together with converting frames to a single
shared palette first, since Pillow only does inter-frame delta encoding when
consecutive frames share one. Build the palette with an exact index mapping
rather than `quantize()`, and it is lossless as well as smaller.

Note that the delta encoding is only worth so much here: GIF's inter-frame
optimisation crops to the changed *bounding box*, and in an animation whose
changes are sparse but scattered -- which is exactly the animation that
differential terminal rendering is best at -- the bounding box is most of the
frame anyway.
