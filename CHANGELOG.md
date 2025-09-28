Note to self
------------

Things to check before updating a new release:

* All examples run without errors.
* Road map is up to date.
* Change log is up to date.

Steps to update the version:

* Decide on a new version number.
* Bump `__version__` in `__init__.py`.
* `make DOCS.md`.
* Push/merge new code into main branch.
* On github, make a new release with a new tag.

In development:
---------------

Plans:

* [ ] Different colours for each point.
* [ ] Multiple point clouds on a single scatter plot.
* [ ] Line plots (connect the dots).
* [ ] Axis ticks and tick labels for scatter plots.
* [ ] Labels and ticks for bar/column charts and histograms.

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
