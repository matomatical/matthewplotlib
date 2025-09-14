DOCS.md: generate_docs.py $(wildcard matthewplotlib/*.py)
	python generate_docs.py \
		matthewplotlib/__init__.py \
		matthewplotlib/plots.py \
		matthewplotlib/colors.py \
		matthewplotlib/colormaps.py \
		matthewplotlib/core.py \
		matthewplotlib/unscii16.py \
	> $@


copy:
	tail -n +1 pyproject.toml README.md matthewplotlib/*.py examples/*.py | pbcopy


.PHONY: copy
