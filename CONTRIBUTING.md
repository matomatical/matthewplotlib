Contributing to matthewplotlib
==============================

Best to discuss before attempting to contribute. This file contains some notes
to self.

Workflow
--------

To work on a new feature:

1. Make the change on a new branch
2. Make sure the branch passes the checklist:
   * mypy checks pass (`mypy matthewplotlib/`).
   * Docs up to date `make DOCS.md`.
   * All examples run without errors (`make examples`).
   * Road map (in README.md) is up to date.
   * Change log (CHANGELOG.md) is up to date.
   * All new features are exported in `__init__.py`.
3. Then merge into main

To release a new version:

1. Decide on a new version number (V).
2. Bump `__version__` in `__init__.py` to V.
3. Bump `version` in `pyproject.toml` to V.
4. Move changelog items from 'In development' to a new 'Version V' section.
6. Commit: `git commit -m "Version V"`.
7. Tag: `git tag vV`.
8. Push: `git push origin main --tags`.
9. On GitHub, create a new release from the tag.

