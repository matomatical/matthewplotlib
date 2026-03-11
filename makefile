DOCS.md: generate_docs.py README.md $(wildcard matthewplotlib/*.py)
	python generate_docs.py \
		README.md \
		matthewplotlib/__init__.py \
		matthewplotlib/plots.py \
		matthewplotlib/colormaps.py \
		matthewplotlib/core.py \
		matthewplotlib/unscii16.py \
	> $@

examples:
	python examples/braille_test.py
	python examples/calendar_heatmap.py
	python examples/colormaps.py
	python examples/dashboard.py --num-frames 5
	python examples/demo.py
	python examples/functions.py
	python examples/hilbert_curve.py
	python examples/image.py
	python examples/jointplot.py
	python examples/lissajous.py
	python examples/mandelbrot.py --num-frames 5
	python examples/quickstart1.py
	python examples/quickstart2.py --num-frames 5
	python examples/scatter.py
	python examples/teacher_student.py --num-steps 5
	python examples/teapot.py --num-frames 5
	python examples/time_series_histogram.py
	python examples/voronoi.py

copy:
	tail -n +1 pyproject.toml README.md matthewplotlib/*.py examples/*.py | pbcopy

# release:
# 	@test -n "$(V)" || (echo "Usage: make release V=0.3.7" && exit 1)
# 	sed -i 's/__version__ = ".*"/__version__ = "$(V)"/' matthewplotlib/__init__.py
# 	sed -i 's/^version = ".*"/version = "$(V)"/' pyproject.toml
# 	$(MAKE) DOCS.md
# 	git add matthewplotlib/__init__.py pyproject.toml DOCS.md CHANGELOG.md
# 	git commit -m "Version $(V)"
# 	git tag v$(V)

test:
	pytest tests/ -v

.PHONY: copy examples test # release
