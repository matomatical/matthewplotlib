# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

matthewplotlib is a terminal-based plotting library for Python that renders charts using Unicode characters (braille, box drawing) with ANSI color codes. Plots are composable expressions combined via operators (`+` for hstack, `/` for vstack). Outputs to terminal via `print()` or to PNG/GIF via an embedded pixel font.

## Development Commands

```bash
# Install (use uv for package management)
uv pip install -e ".[dev]"

# Run tests
pytest tests/ -v            # or: make test
pytest tests/test_colors.py::TestParseColorNamed::test_named_colors -v  # single test

# Type checking (primary quality gate)
mypy matthewplotlib/

# Regenerate API docs
make DOCS.md
```

## Architecture

Six core modules in `matthewplotlib/`:

- **core.py** — `CharArray` dataclass: the fundamental rendering primitive (grid of unicode chars with optional fg/bg RGB). Handles ANSI escape codes, composition (stacking/layering/padding), PIL image rendering, and GIF export.
- **plots.py** — All plot types inherit from base `plot` class (provides height/width, render methods, operator overloads). Categories: data plots (scatter, image, bars, histogram, etc.), furnishings (text, border, axes), and arrangement (hstack, vstack, wrap, center, blank).
- **data.py** — `Series`/`Series3` types for flexible plot input parsing. Accepts arrays, tuples, axis objects. Includes `project3()` for 3D-to-2D camera projection.
- **colormaps.py** — Continuous colormaps (viridis, magma, etc.) and discrete palettes (sweetie16, pico8, tableau). Types: `ContinuousColorMap`, `DiscreteColorMap`.
- **colors.py** — `Color`/`ColorLike` types, `parse_color()` supporting named colors, hex, int/float RGB tuples.
- **unscii16.py** — Embedded UNSCII-16 bitmap font data for image export (do not edit manually).

Public API is aggregated in `__init__.py`. New features must be exported there.

## Pre-merge Checklist

From CONTRIBUTING.md — before merging any branch:
1. `mypy matthewplotlib/` passes
2. `pytest tests/ -v` passes (includes integration tests for all examples)
3. `make DOCS.md` is up to date
4. Look through README and make sure it's up to date, including roadmap
   section.
5. Update CHANGELOG.md
6. Export new features in `__init__.py`

See CONTRIBUTING.md for instructions on how to prepare a new version for
release.

## Build Configuration

- Build backend: Hatchling
- Python: >= 3.10
- Core deps: numpy, einops, numpy-hilbert-curve, Pillow
- Version is tracked in both `pyproject.toml` and `matthewplotlib/__init__.py` (must be bumped together)
