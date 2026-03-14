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

PDOC_CSS := $(shell python -c "import pdoc; from pathlib import Path; print(Path(pdoc.__file__).parent / 'templates')")

docs: docs/api docs/index.html docs/changelog.html docs/quickstart.html docs/examples.html docs/roadmap.html docs/images docs/pdoc.css

docs/api: templates/custom.css templates/module.html.jinja2 $(wildcard matthewplotlib/*.py)
	pdoc matthewplotlib/ \
		--no-show-source \
		-e matthewplotlib=https://github.com/matomatical/matthewplotlib/blob/main/matthewplotlib/ \
		-t templates/ \
		-o docs/
	@touch $@

docs/pdoc.css: templates/custom.css
	cat $(PDOC_CSS)/resources/bootstrap-reboot.min.css \
		$(PDOC_CSS)/syntax-highlighting.css \
		$(PDOC_CSS)/theme.css \
		$(PDOC_CSS)/layout.css \
		$(PDOC_CSS)/content.css \
		templates/custom.css > $@

GITHUB := https://github.com/matomatical/matthewplotlib/blob/main

docs/index.html: README.md templates/page.html docs/pdoc.css docs/api
	pandoc README.md -o $@ --template=templates/page.html \
		-V title="Home" \
		-V source="$(GITHUB)/README.md"

docs/changelog.html: CHANGELOG.md templates/page.html docs/pdoc.css
	pandoc CHANGELOG.md -o $@ --template=templates/page.html \
		-V title="Changelog" \
		-V source="$(GITHUB)/CHANGELOG.md"

docs/quickstart.html: pages/quickstart.md templates/page.html docs/pdoc.css
	pandoc $< -o $@ --template=templates/page.html \
		-V title="Quickstart" \
		-V source="$(GITHUB)/pages/quickstart.md"

docs/examples.html: pages/examples.md templates/page.html docs/pdoc.css
	pandoc $< -o $@ --template=templates/page.html \
		-V title="Examples" \
		-V source="$(GITHUB)/pages/examples.md"

docs/roadmap.html: pages/roadmap.md templates/page.html docs/pdoc.css
	pandoc $< -o $@ --template=templates/page.html \
		-V title="Roadmap" \
		-V source="$(GITHUB)/pages/roadmap.md"

docs/images: images
	rm -rf $@
	cp -r $< $@

test:
	pytest tests/ -v

.PHONY: copy test docs # release
