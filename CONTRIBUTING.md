Contributing to matthewplotlib
==============================

This is a personal project developed by MFR. It's best to discuss with me
before attempting to contribute, for example by raising an issue. Before that,
please read the following.

Development dependencies
------------------------

System dependencies:

* uv for managing virtual environment
* make for building docs, running tests, making releases
* pandoc for building some parts of docs
* tmux for the tests that drive a real terminal

Then, install the package and its (Python) development dependencies into your uv venv
with:

```
uv pip install -e ".[dev]"
```


Workflow
--------

To work on a new feature:

1. Make the change on a new branch
2. Make sure the branch passes the checklist:
   * mypy checks pass (`mypy matthewplotlib/`).
   * Tests pass (`pytest tests/ -v`), including integration tests for all
     examples.
   * Docs up to date (`make docs`).
   * Roadmap (`pages/roadmap.md`) is up to date.
   * Changelog (`CHANGELOG.md`) is up to date.
   * All new features are exported in `__init__.py`.
3. Then merge into main

Notes:

* CHANGELOG entries should be concise and describe API-level changes, not
  implementation details.
* Longer design notes and investigations live in `notes/`, and the roadmap
  entries they belong to link to them. See `notes/README.md`.

Code style
----------

* Modules are divided into sections by `# # #` / `# Title` comment blocks,
  which double as fold markers. Preserve the pattern when adding a section.
* Every module has a module-level docstring introducing what is in it, since
  these are published as the API reference.

Testing
-------

Unit tests live in `tests/test_*.py` for the core modules (colors, data,
colormaps, core). Integration tests in `tests/test_examples.py` run every
example script as a subprocess and check for successful execution and output.

Tests in `tests/test_terminal.py` check what the library's escape sequences do
to a real terminal, driving a tmux pane through the harness in `tests/tmux.py`.
Claims about the sequences as strings go in `test_core.py` instead.

When adding a new example to `examples/`, add a corresponding entry to the
`EXAMPLES` list in `test_examples.py`. The `test_all_examples_covered` test
will fail if any example is missing from the list.

Long-running or animated examples should accept a `--num-frames` (or similar)
argument via `tyro` so that integration tests can run them with a small value.
Use `num_frames=0` to mean "loop forever" (the default for interactive use).

Images in `images/` are used in the README as showcase material. They should
be generated manually with full-length runs, not overwritten by test runs.

Releasing a new version
-----------------------

To release a new version:

0. Make sure the checklist above is complete.
1. Decide on a new version number (V).
2. Move changelog items from 'In development' to a new 'Version V' section.
3. Bump `__version__` in `__init__.py` to V.
4. Bump `version` in `pyproject.toml` to V.
5. Rebuild the docs website: `make docs`.
6. Commit: `git commit -m "Version V"`.
7. Tag: `git tag vV`.
8. Push: `git push origin main --tags`.
9. On GitHub, create a new release from the tag.

Steps 3-7 can be automated by `make release V=<new version number>`
