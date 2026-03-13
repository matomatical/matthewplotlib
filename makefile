DOCS.md: generate_docs.py README.md $(wildcard matthewplotlib/*.py)
	python generate_docs.py \
		README.md \
		matthewplotlib/__init__.py \
		matthewplotlib/plots.py \
		matthewplotlib/colormaps.py \
		matthewplotlib/core.py \
		matthewplotlib/unscii16.py \
	> $@


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

docs: templates/custom.css $(wildcard matthewplotlib/*.py)
	pdoc matthewplotlib/ \
		--no-show-source \
		-e matthewplotlib=https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/ \
		-t templates/ \
		-o docs/

test:
	pytest tests/ -v

.PHONY: copy test docs # release
