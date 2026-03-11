import subprocess
import sys
from pathlib import Path

import pytest


EXAMPLES_DIR = Path(__file__).parent.parent / "examples"

# (script, args, output_file or None)
EXAMPLES = [
    ("braille_test.py", [], None),
    ("calendar_heatmap.py", [], "images/calendar_heatmap.png"),
    ("colormaps.py", [], "images/colormaps.png"),
    ("dashboard.py", ["--num-frames", "5"], None),
    ("demo.py", [], "images/demo.png"),
    ("functions.py", [], None),
    ("hilbert_curve.py", [], "images/hilbert_curve.png"),
    ("image.py", [], None),
    ("lissajous.py", [], "images/lissajous.png"),
    ("mandelbrot.py", ["--num-frames", "5"], "images/mandelbrot.gif"),
    ("quickstart1.py", [], "images/quickstart.png"),
    ("quickstart2.py", ["--num-frames", "5"], None),
    ("scatter.py", [], "images/scatter.png"),
    ("teacher_student.py", ["--num-steps", "5"], "images/teacher_student.gif"),
    ("teapot.py", ["--num-frames", "5"], None),
    ("time_series_histogram.py", [], None),
    ("voronoi.py", [], "images/voronoi.png"),
]


def _run_example(script, args, tmp_path):
    """Run an example script in a tmp dir with images/ created."""
    (tmp_path / "images").mkdir()
    result = subprocess.run(
        [sys.executable, str(EXAMPLES_DIR / script)] + args,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=tmp_path,
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    return result


@pytest.mark.parametrize(
    "script, args",
    [(s, a) for s, a, _ in EXAMPLES],
    ids=[s for s, _, _ in EXAMPLES],
)
def test_example_str(script, args, tmp_path):
    """Each example should run successfully and produce terminal output.

    TODO: snapshot stdout content to detect rendering regressions.
    """
    result = _run_example(script, args, tmp_path)
    assert len(result.stdout) > 0


EXAMPLES_WITH_IMAGE = [(s, a, f) for s, a, f in EXAMPLES if f is not None]


@pytest.mark.parametrize(
    "script, args, output_file",
    EXAMPLES_WITH_IMAGE,
    ids=[s for s, _, _ in EXAMPLES_WITH_IMAGE],
)
def test_example_image(script, args, output_file, tmp_path):
    """Each example that saves a file should produce the expected output.

    TODO: compare output images against reference snapshots to detect
    rendering regressions.
    """
    _run_example(script, args, tmp_path)
    output_path = tmp_path / output_file
    assert output_path.exists(), f"expected output file {output_file}"
    assert output_path.stat().st_size > 0


def test_all_examples_covered():
    """Every .py file in examples/ should have a test entry."""
    example_files = sorted(p.name for p in EXAMPLES_DIR.glob("*.py"))
    tested_files = sorted(s for s, _, _ in EXAMPLES)
    assert example_files == tested_files
