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
   * Tests pass (`pytest tests/ -v`), including the example snapshots. If a
     snapshot changed, look at the change before accepting it with
     `make goldens` -- see Testing, below.
   * Documentation site still builds (`make docs`).
   * Roadmap (`docs/roadmap.md`) is up to date.
   * Changelog (`CHANGELOG.md`) is up to date.
   * All new features are exported in `__init__.py`. For `plots`, `colormaps`
     and `animations` this is checked by `tests/test_exports.py`, which derives
     what to expect from what those modules define. `data`, `colors` and `core`
     keep some things back deliberately and are not covered; widening the rule
     to them is open, see `notes/export-policy.md`.
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
colormaps, core).

Tests in `tests/test_terminal.py` check what the library's escape sequences do
to a real terminal, driving a tmux pane through the harness in `tests/tmux.py`.
Claims about the sequences as strings go in `test_core.py` instead.

`tests/test_examples.py` runs every example and compares what it drew against a
snapshot in `tests/goldens/`. Each example is run as a subprocess through its
own `tyro` command line, its prints are replayed one at a time into a tmux pane
sized to that example, and the resulting screens are compared cell by cell --
glyph and colour -- along with the byte cost of each print, the cursor, and a
digest of the image the example saved. The machinery is in `tests/examples.py`,
and the reasoning, including which layer catches what, is in
`notes/closed/example-snapshot-tests.md`.

When one of these fails, the assertion names the cells that moved. To look at
the two screens in colour, and to accept the new output once you have:

```
python -m tests.examples --diff <example>   # golden against a fresh run
python -m tests.examples --show <example>   # the golden, as the terminal drew it
make goldens                                # accept, reporting what changed
```

When adding a new example to `examples/`, add an entry to the `EXAMPLES` table
in `tests/examples.py` (`test_all_examples_covered` fails if you forget) and
run `make goldens` to record it. The entry carries the terminal size to
snapshot in, which is part of the test rather than a convenience: a pane wider
than the plot never exercises the deferred wrap at the final column, which is
where the cursor arithmetic is hardest. Get the size from

```
python -m tests.examples --sizes [example ...]
```

which measures the height and suggests a width. Size the pane to the plot, one
row taller than its output.

Long-running or animated examples should accept a `--num-frames` (or similar)
argument via `tyro` so that integration tests can run them with a small value.
Use `num_frames=0` to mean "loop forever" (the default for interactive use).

Images in `images/` are used in the README as showcase material. They should
be generated manually with full-length runs, not overwritten by test runs.

Documentation
-------------

The website is built from `docs/` with mkdocs, using mkdocstrings to render
the API reference from the docstrings. `docs/index.md`, `docs/examples.md` and
`docs/changelog.md` include `README.md`, `examples/README.md` and
`CHANGELOG.md`, so each of those files serves both GitHub and the site.
Overrides for the theme's partials and for mkdocstrings' templates live in
`templates/`, and the palette is in `docs/css/`.

```
make serve                  # preview on localhost, reloading as you edit
make docs                   # build once into site/, failing on any bad link
```

Each version gets its own directory on the `gh-pages` branch, published by
mike. The `latest` alias follows the newest release, and the site root
redirects there. `make deploy V=<version>` builds the current tree as that
version and moves the alias.

The API reference links each object to its source on GitHub. Those links are
derived from the `origin` remote, so a checkout without one silently builds a
site with no source links.

Releasing a new version
-----------------------

To release a new version:

0. Make sure the checklist above is complete.
1. Decide on a new version number (V).
2. Move changelog items from 'In development' to a new 'Version V' section.
3. Bump `__version__` in `__init__.py` to V.
4. Bump `version` in `pyproject.toml` to V.
5. Commit: `git commit -m "Version V"`.
6. Tag: `git tag vV`.
7. Push: `git push origin main --tags`.
8. Publish the version's documentation: `make deploy V=V`, then
   `git push origin gh-pages`.
9. On GitHub, create a new release from the tag.

Steps 3-6 can be automated by `make release V=<new version number>`
