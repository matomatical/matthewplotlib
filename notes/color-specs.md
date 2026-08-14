# Color specs, and how they are told apart

Written 2026-08-14 by Claude (Opus 5) at MFR's direction, after `line3` on a
mesh turned out to spend most of its time formatting exception messages.

A `ColorSpec` is either one color for a whole series or an array with one per
point, and `parse_colors` has to tell which. It used to do that by trying
`parse_color` and catching the `ValueError`, which cost more than it looked:

* the message names the offending value, so every successful per-point parse
  formatted its whole array into a string nobody reads. About a millisecond a
  series -- 18.7 ms of a 20 ms `line3` call over a 37-wire mesh;
* a misspelled name is also a parse failure, so `"puce"` was classified as an
  array of colors and reported as `invalid literal for int() with base 10`;
* and the two branches had drifted: `parse_color` reads a float triple as 0.0 to
  1.0, while the array branch cast straight to bytes, so an array of 0.5 grey
  came out black.

One cause, three symptoms. Deciding by shape instead fixes all three, and the
float convention now lives in one place (`_channel_bytes`) that both branches
use, so it cannot drift again.

Rejected on the way:

* Making the message cheap and keeping the try/except. Fixes the cost only, and
  needs a helper in `colors` whose job is formatting error text. NumPy already
  summarises arrays over a thousand elements, so the expensive case was small
  arrays formatted often, not the huge dump that helper was defending against.
* A predicate, `is_color`. Names the classification honestly, but is either a
  second list of accepted forms to keep in step with `parse_color`, or the same
  try/except wearing a hat.
* A non-raising `as_color`. `parse_color(None)` already returns None for "use
  the default", so "not a color" would need a sentinel.
* Dispatching on shape in `data`, where the parser used to live. Two modules
  would then both know what color input looks like. Moving the type and the
  parser into `colors` was the point.
