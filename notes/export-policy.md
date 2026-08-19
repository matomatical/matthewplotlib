# What counts as public — framing notes

Raised 2026-07-26 by Matthew, after `mp.animation` shipped without being added to
`__init__.py` and nothing failed; written by Claude. Not designed — this note
records the shape of the question and the counts a decision would rest on, so the
next session does not start by re-measuring.

## The problem

`__init__.py` re-exports names one at a time, so every new feature has two edit
sites, and until now only a checklist item in `CONTRIBUTING.md` connected them.
Every test imports from the defining module (`from matthewplotlib.plots import
scatter`), so a missing re-export breaks no test and ships. `mp.animation` did.

`tests/test_exports.py` now closes the hole for three modules by deriving the
expectation from the code: whatever `plots`, `colormaps` and `animations` define,
`mp.*` must re-export. Nothing to maintain per feature, and it catches an
unexported plot type, not just the one that prompted it.

It stops there because a blanket rule is false for the other three. Measured
2026-07-26, counting module-level names without a leading underscore whose
`__module__` is the module itself:

| module | defines | not exported |
|---|---|---|
| `plots` | 22 | — |
| `colormaps` | 22 | — |
| `animations` | 3 | — |
| `data` | 15 | `number`, `ColorSpec`, `parse_range`, `parse_color_spec`, `parse_series`, `parse_multiple_series`, `parse_series3`, `parse_multiple_series3`, `axis`, `project3` |
| `colors` | 3 | `Color`, `parse_color` |
| `core` | 8 | `CharArray`, `ords`, `unicode_braille_array`, `unicode_bar`, `unicode_col`, `unicode_box`, `unicode_image` |

So 19 names are public by naming convention and private by intent. The rule
cannot be "no underscore means exported" until that is untrue.

## Matthew's proposal: underscore the internals

Rename those 19 with a leading underscore. Then "public" has one definition
everywhere, `test_exports.py` covers all six modules, the module list in it goes
away, and the policy is a sentence.

It also shrinks the API reference, which is the larger prize and easy to
overlook: pdoc documents every name without a leading underscore, so
`matthewplotlib.core`'s page currently advertises `unicode_bar` and `ords` to
readers who have no use for them. Whatever fixes the export check fixes that too.

## Why it is not just a rename

The 19 are not one kind of thing. `parse_range` and `project3` are genuinely
internal: nothing outside `data` and its own tests names them, and an underscore
is simply true. `CharArray` is different, and it is the case the design has to
answer:

* `plot.chars` is a documented public attribute, and it holds a `CharArray`. A
  public attribute of a private type is a contradiction — `_CharArray` while
  `plot.chars` hands one out says "you may have this, but not know what it is".
* Its methods are the extension surface. Someone writing a new plot type builds a
  `CharArray` and passes it to `plot.__init__`, exactly as every class in
  `plots.py` does. That is a supported thing to do, or the base class would not
  take one.
* Three notes in `notes/` name it and `to_ansi_diff_str` when stating what the
  rendering guarantees, which is the sort of claim `notes/README.md` asks to be
  recheckable against the code.
* `tests/test_core.py` and `tests/test_terminal.py` import it and
  `unicode_image` directly. That is only a rename, but it is a signal: these are
  the names the test suite reasons in.

Nothing outside the library reaches for `.chars` — checked `examples/`, `docs/`
and the README, no hits — so the contradiction is currently theoretical. It stops
being theoretical the moment anyone writes a plot type outside this repository.

## The real shape: three tiers, not two

An underscore encodes one bit, and there seem to be three states:

1. **Top-level API.** `mp.scatter`, `mp.animate`. Reachable as `mp.x`, documented,
   stable.
2. **Extension surface.** `CharArray`, `unicode_image`, `BoxStyle`. Documented and
   importable from their module, deliberately *not* in the top-level namespace
   because they are not what plotting looks like — but a library is allowed to
   have these, and pretending otherwise is how a plotting library ends up with no
   way to add a plot type.
3. **Internal.** `parse_range`, `project3`, `parse_color`, `ords`. No promises.

Tier 2 is what makes the two-tier proposal awkward. `BoxStyle` is already in tier
2 and *is* exported, which shows the current boundary is drawn by taste rather
than by rule.

## Options

* **(a) Underscore everything internal.** Cheapest policy, one rule, smallest
  reference. Forces tier 2 to collapse into tier 1 or tier 3: either export
  `CharArray` at the top level, or call it `_CharArray` and accept that
  `plot.chars` is undocumentable. Roughly 19 renames plus their references in
  tests and notes.
* **(b) `__all__` per module.** Expresses all three tiers: `__all__` names tier 1
  and 2, the underscore-free-but-absent names are tier 2 or 3 as the module
  chooses, and pdoc honours it. `test_exports.py` becomes
  `set(mp.__dict__) >= set(module.__all__)` over every module with no exclusions.
  Costs a list per module, which is the bookkeeping this was trying to remove —
  but one list per module is not one edit per feature, and it is the convention
  Python readers already know.
* **(c) Both.** `__all__` for the tier 1/2 boundary, underscores for the genuinely
  internal helpers in `data` and `colors`. Probably where this lands, and it can
  be done module by module.
* **(d) Leave it.** The partial test covers the three modules where features
  actually get added, which is where the mistake happened and where it would
  happen again. The other three modules have not gained a public name in months.

## Questions to answer first

* Is there a tier 2, or is `plot.chars` a mistake to be hidden behind methods?
  That is the load-bearing question; everything else follows from it.
* Should the API reference document tier 2 at all, or only tier 1? Right now it
  documents tiers 1, 2 and 3 indiscriminately.
* If `__all__`: does it go in every module, or only the ones with something to
  hide? Uniformity is easier to check; sparseness is less to read.
* Does `BoxStyle` belong in the top-level namespace, given `CharArray` does not?
  Whichever way, the answer names the tier 2 rule.
