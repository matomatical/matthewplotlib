# # # 
# Documentation website

PDOC_CSS := $(shell python -c "import pdoc; from pathlib import Path; print(Path(pdoc.__file__).parent / 'templates')")
IMAGES := $(wildcard images/*)
DOCS_IMAGES := $(IMAGES:images/%=docs/images/%)

docs: docs/api docs/index.html docs/changelog.html docs/quickstart.html docs/examples.html docs/roadmap.html docs/images docs/pdoc.css

# Copies the images, then drops any left over from a deleted source image.
# Phony, because the timestamp of a directory says nothing about whether its
# contents are stale -- which is the trap the per-image targets exist to avoid.
docs/images: $(DOCS_IMAGES)
	@rm -f $(filter-out $(DOCS_IMAGES),$(wildcard docs/images/*))

docs/images/%: images/%
	@mkdir -p $(@D)
	cp $< $@

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
	pandoc README.md -o $@ --template=templates/page.html --wrap none \
		--metadata title="Home" \
		-V source="$(GITHUB)/README.md"

docs/changelog.html: CHANGELOG.md templates/page.html docs/pdoc.css
	pandoc CHANGELOG.md -o $@ --template=templates/page.html --wrap none \
		--metadata title="Changelog" \
		-V source="$(GITHUB)/CHANGELOG.md"

docs/quickstart.html: pages/quickstart.md templates/page.html docs/pdoc.css
	pandoc $< -o $@ --template=templates/page.html --wrap none \
		--metadata title="Quickstart" \
		-V source="$(GITHUB)/pages/quickstart.md"

docs/examples.html: pages/examples.md templates/page.html docs/pdoc.css
	pandoc $< -o $@ --template=templates/page.html --wrap none \
		--metadata title="Examples" \
		-V source="$(GITHUB)/pages/examples.md"

docs/roadmap.html: pages/roadmap.md templates/page.html docs/pdoc.css
	pandoc $< -o $@ --template=templates/page.html --wrap none \
		--metadata title="Roadmap" \
		-V source="$(GITHUB)/pages/roadmap.md"


# # # 
# Tests

test:
	pytest tests/ -v


# # #
# Release (TODO)

# release:
# 	@test -n "$(V)" || (echo "Usage: make release V=0.3.7" && exit 1)
# 	sed -i 's/__version__ = ".*"/__version__ = "$(V)"/' matthewplotlib/__init__.py
# 	sed -i 's/^version = ".*"/version = "$(V)"/' pyproject.toml
# 	$(MAKE) DOCS.md
# 	git add matthewplotlib/__init__.py pyproject.toml DOCS.md CHANGELOG.md
# 	git commit -m "Version $(V)"
# 	git tag v$(V)

.PHONY: test docs docs/images # release
