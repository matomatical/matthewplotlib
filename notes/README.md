# Design notes

Reasoning behind design decisions: what was measured, what was chosen, what was
rejected. `docs/src/roadmap.md` is the list of what to build; these are the notes
behind individual entries, which link here.

Write one when an investigation outruns its commit message — measurements worth
keeping, alternatives that took real thought, a design that will not be built
this week. Bugs are not design notes; those go to the issue tracker.

    notes/            Open questions.
    notes/closed/     Settled, kept for the reasoning.
    notes/reference/  Raw material: codepoint tables, superseded code kept for
                      the parts that have no equivalent yet.

Refer to a note by its name and not its path: "the `box-plots` note", never
"`notes/box-plots.md`". A note moves from one directory to the other when the
question it asks is answered, and a path spelled out in a dozen places goes
stale the moment it does.

Open with when it was written, by whom, and what prompted it. Give the numbers.
Say what was tried and did not work. Name the functions a claim depends on, so
it can be rechecked.
