# Mapping over composites — framing notes

Raised 2026-07-26 by Matthew, on seeing `tstack.map` land with the animation
work (`notes/animations.md`); written by Claude. Not designed. This note pins
down what the question actually is, because the obvious phrasing hides two
different operations.

## The ask

`tstack.map(f)` applies a plot-to-plot function to every frame:

    anim.map(lambda p: mp.border(p, title=" life "))

`tstack` is a composite that keeps its children. So do `hstack`, `vstack`,
`dstack` and `wrap` — all five store `self.plots`. If mapping is the right way to
lift a combinator over the time axis, it looks like it should be the right way to
lift one over the other axes too.

## Two operations, not one

The phrase "map for hstack and vstack" resolves two ways, and they have different
signatures, different call sites, and different amounts of risk.

**(a) An instance method: relift an existing composite.** `f : plot -> plot`,
same shape as `tstack.map`.

    row = plotA + plotB + plotC
    row.map(mp.border)              # -> hstack of three bordered plots

Available today for free — `hstack.__init__` keeps `self.plots`, so the method is
`type(self)(*[f(p) for p in self.plots])`. Genuinely useful for the
"give every panel the same treatment" case that `examples/dashboard.py` does by
hand three times over.

The catch is that the constructors are not all uniform. `wrap` takes `cols` and
`transpose` keywords, and `dstack2` validates that its children share `xrange`
and `yrange`. A generic `type(self)(*mapped)` loses `cols`, silently reflowing a
grid. So this wants either a per-class `map`, or composites that record enough to
rebuild themselves. Neither is hard; both are more than a one-liner, which is why
it is not in the animation branch.

**(b) A constructor helper: build a composite from a sequence.** `f : A -> plot`,
a different signature — this is sugar for a comprehension.

    hmap(f, xs)     ==  hstack(*[f(x) for x in xs])
    vmap(f, xs)     ==  vstack(*[f(x) for x in xs])
    tmap(f, xs)     ==  tstack(*[f(x) for x in xs])

This is the one the examples would actually use most: `colormaps.py`,
`life.py` and `dashboard.py` all build `stack(*[... for ... in ...])`. It is also
the one with a naming hazard — `vmap` means something specific and different to
anyone arriving from JAX, and this library's `vmap` would map over the *vertical
layout axis*, not a batch axis. `vstack_map`, or accepting an iterable directly
in the existing constructors, both dodge it.

Note that (b) subsumes nothing: `row.map(f)` cannot be written as `hmap` without
already having the children in hand, and `hmap(f, xs)` cannot be written as
`.map` without building a throwaway composite first.

## The third option

Let the existing constructors take an iterable as well as varargs:

    hstack(mp.border(x) for x in xs)

No new names, no `vmap` collision, and it reads like `sum` or `max`. Costs a
type check at the head of five constructors, and makes `hstack(gen)` and
`hstack(*gen)` both legal, which is either forgiving or sloppy depending on
taste. Probably the cheapest thing that addresses (b).

## Questions to answer first

* Is (a) or (b) the one that is actually wanted? They are independent and could
  land separately.
* If (a): per-class `map`, or a general one plus the machinery for composites to
  remember their own keywords? The latter is the start of a plot-as-pytree
  refactor, which the roadmap already wants for other reasons.
* If (b): free functions, classmethods (`hstack.of(f, xs)`), or iterable-accepting
  constructors? And what to call the vertical one, given JAX.
* Does `dstack2`'s range validation survive a map that changes the ranges? It
  should refuse rather than silently keep the old `xrange` attribute.
* Whether any of this earns its keep against just writing the comprehension,
  which is one line and already works.
