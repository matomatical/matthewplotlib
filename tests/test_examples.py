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
    ("jointplot.py", [], "images/jointplot.png"),
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


@pytest.mark.parametrize(
    "script, args, output_file",
    EXAMPLES,
    ids=[s for s, _, _ in EXAMPLES],
)
def test_example(script, args, output_file, tmp_path):
    """Each example should run successfully and produce terminal output.

    For examples that save a file, also check the output file exists.

    TODO: snapshot stdout content to detect rendering regressions.
    TODO: compare output images against reference snapshots to detect
    rendering regressions.
    """
    (tmp_path / "images").mkdir()
    result = subprocess.run(
        [sys.executable, str(EXAMPLES_DIR / script)] + args,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=tmp_path,
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert len(result.stdout) > 0
    if output_file is not None:
        output_path = tmp_path / output_file
        assert output_path.exists(), f"expected output file {output_file}"
        assert output_path.stat().st_size > 0


def test_all_examples_covered():
    """Every .py file in examples/ should have a test entry."""
    example_files = sorted(p.name for p in EXAMPLES_DIR.glob("*.py"))
    tested_files = sorted(s for s, _, _ in EXAMPLES)
    assert example_files == tested_files
