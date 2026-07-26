"""
Integration tests: every example, against a snapshot of what it drew.

Each example is run as a subprocess, its prints are replayed one at a time into
a real terminal, and the resulting screens are compared against the golden in
`tests/goldens/`, cell by cell, together with the byte cost of each print and a
digest of the image the example saved. The machinery, and why each of those
layers is worth its keep, is in `tests/examples.py`.

When one of these fails, the assertion names the cells that moved. To look at
the two screens instead:

    python -m tests.examples --diff <example>

and to accept the new output once you have:

    make goldens
"""

import pytest

from tests import examples
from tests.examples import EXAMPLES, Example


@pytest.mark.parametrize("example", EXAMPLES, ids=[e.name for e in EXAMPLES])
def test_example_matches_its_golden(example: Example):
    """Every example draws exactly what it drew when the golden was taken.

    One test per example rather than one per layer, so that an example is run
    once: the layers are checked together and every difference is reported, not
    just the first.
    """
    assert example.golden.exists(), (
        f"no golden for {example.script}; run `make goldens`"
    )
    snapshot = examples.take(example)
    lines = examples.differences(snapshot, example.golden.read_text())
    assert not lines, (
        f"{example.script} no longer matches tests/goldens/{example.name}.txt:\n"
        + "\n".join("  " + line for line in lines)
        + f"\n\nLook at both with `python -m tests.examples --diff {example.name}`"
        + ", accept with `make goldens`."
    )


def test_all_examples_covered():
    """Every .py file in examples/ should have an entry in EXAMPLES."""
    on_disk = sorted(p.name for p in examples.EXAMPLES_DIR.glob("*.py"))
    in_table = sorted(e.script for e in EXAMPLES)
    assert on_disk == in_table


def test_every_example_has_a_golden():
    """A missing golden should fail here, not as eighteen confusing failures."""
    missing = [e.script for e in EXAMPLES if not e.golden.exists()]
    assert not missing, f"no goldens for {missing}; run `make goldens`"


def test_goldens_are_not_stale():
    """Every golden belongs to an example that still exists."""
    known = {f"{e.name}.txt" for e in EXAMPLES}
    stray = sorted(
        p.name for p in examples.GOLDENS_DIR.glob("*.txt") if p.name not in known
    )
    assert not stray, f"goldens with no example: {stray}"


@pytest.mark.parametrize("example", EXAMPLES, ids=[e.name for e in EXAMPLES])
def test_golden_round_trips(example: Example):
    """The golden format parses back to what was written.

    Cheap, and it means a failure in the tests above is a real difference in
    the output rather than a hole in the parser.
    """
    text = example.golden.read_text()
    header, frames = examples.loads(text)
    assert header["example"] == example.script
    assert header["terminal"] == f"{example.height}x{example.width}"
    again = examples.dumps(examples.Snapshot(
        example=example,
        frames=frames,
        image=_image_of(header),
    ))
    assert again == text


def _image_of(header: dict[str, str]) -> examples.ImageDigest | None:
    """Rebuild an ImageDigest from the header line that described it."""
    line = header.get("image")
    if line is None:
        return None
    frames, size, digest = line.split(", ")
    height, width = size.split("x")
    return examples.ImageDigest(
        frames=int(frames.split()[0]),
        height=int(height),
        width=int(width),
        digest=digest.removeprefix("sha256:"),
    )
