Note to self
------------

Things to check before updating a new release:

* mypy checks pass.
* All examples run without errors.
* Road map is up to date.
* Change log is up to date.

Steps to update the version:

* Decide on a new version number.
* Bump `__version__` in `__init__.py`.
* Bump version in `pyproject.toml`.
* `make DOCS.md`.
* Push/merge new code into main branch.
* On github, make a new release with a new tag.

In development:
---------------

Breaking changes:

* scatter and scatter3 take xs, ys, (zs), and color as series tuples in
  positional arguments.
* removed function plot type (since scatter is now much easier to use).

New:

* scatter and scatter3 accept cs, an array of colors (one for each point), and
  plot using them, using weighted averaging to combine plots.
* scatter and scatter3 now accept multiple series at once.
* special series for X/Y/Z axes.

TODO:

* Improve specification of colours; separate from series? More like image with
  a colormap?
* Labelled axes.

Version 0.2.1:
--------------

Fix:

* Regenerate documentation.
* Update version number properly.

Version 0.2.0:
--------------

Breaking changes:

* Various argument name changes, especially for colors.
* Inverted `cyber` colormap.
* Move `plots.border.Style` to `core.BoxStyle`.

New:

* Configurable background colour for image rendering.
* 3d scatterplot.
* Discrete colourmaps are now cyclic.
* New discrete colourmaps `tableau`, `nouveau`.
* New border styles.
* Export animations as GIFs.
* New configuration options for bar/column sizes.

Internal:

* Refactor backend to use numpy arrays rather than nested lists.

Version 0.1.2:
--------------

Breaking changes:

* Change operators used for shortcuts.
* Rename `fimage` to `function2`.

New:

* New plot types: `bars`, `columns`, `histogram`, `vistogram`, `histogram2`,
  `function`.
* More documentation.
* Generated markdown documentation.
* Additional examples.

Dependencies:

* Make example dependency on `scikit-learn` explicit.

Version 0.1.1:
--------------

New:

* Add type annotations.

Dependencies:

* Add `mypy` as a dev dependency.
* Remove dependency on `unscii` (bundle the specific version of the font we
  want).

Internal:

* Refactor from long single-file script to multi-file library.

Version 0.1.0
-------------

Much unstructured development.
