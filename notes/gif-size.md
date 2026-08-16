# Shrinking animation gifs

Measured 2026-07-25 (Matthew + Claude) getting `images/life.gif` down to a size
fit for the README. Applies to any gif `tstack.savegif` produces, since they
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

`tstack.savegif` calls `Image.save(...)` without `optimize=True`, so every gif
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

## What happened when it was done

Measured 2026-08-16 (Claude, reviewed by Matthew) carrying out the follow-up
above. The ~13% did not appear, and the reason is the paragraph directly above
it: on a 64-frame Life-like animation, the shared palette plus `optimize=True`
came to 431 kB either way, **0.0%**. The changes are scattered, so the bounding
box is the whole frame, and there is no delta to win. The two halves of that
section were written as a recommendation and its own refutation; only the first
half survived into the roadmap.

What the shared palette is actually worth is **colour**, which this note did
not think to measure. Saving frame by frame lets Pillow pick a palette per
frame, and it picks a small one: on frames holding 228 distinct colours it
stored **29**, altering 98.7% of pixels, maximum channel error 17 -- visible
banding across any smooth ramp. Worse, it re-picks per frame, so content that
never moves changes colour underneath it: in a 12-frame animation with a static
ramp across the top, 10 frames disagreed with frame 0 about a region that is
byte-identical in the source.

One palette for the whole animation fixes both, and is exact when the animation
has 256 colours or fewer -- which is most terminal plots, though not the
colourmapped ones. The cost is size, and it is the opposite of what was
predicted: 154 kB to 245 kB on that animation, since flat bands compress better
than the gradients they were flattening.

So `savegif` grew `palette` and `colors` rather than a fixed choice. Small
files are still available, by asking for fewer colours (`colors=32`), which is
an honest way to trade colour for bytes; the old behaviour was making that
trade without saying so.

Palette *order* is not a lever: sorting the 253 colours of a viridis animation
by luminance instead of by packed RGB gave a byte-identical file. Size here is
colour entropy, not layout.

## The ghosting bug, found the same day

Sharing a palette makes Pillow store frames as differences, and that exposed a
bug that predates it. A difference frame marks an unchanged pixel with the
transparent index -- the same index that means "draw a hole". An animation with
a transparent background therefore smeared: on 0.6.0, scrolling text left every
position it had passed through still painted, its last frame drawing 823 pixels
where 143 belonged. Every frame after the first was wrong.

It went unnoticed because every example either fills each cell (`image` plots)
or sets a `bgcolor`, and nothing read a saved gif back.

The fix is disposal method 2, "restore to background before the next frame",
which is the only setting that keeps the two meanings apart. It costs the
difference encoding for transparent animations -- they go back to whole frames,
a few percent larger. `tests/test_animations.py::TestSaveGifTransparency`
pins it; both tests fail if the disposal is dropped.
