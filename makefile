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

mypy:
	mypy

test:
	pytest tests/ -v

# Rewrite the example snapshots in tests/goldens/, reporting what changed in
# each before it writes. See tests/examples.py for the other subcommands
# (--diff, --show, --sizes).
goldens:
	python -m tests.examples --update


# # #
# Release
# Usage: make release V=<new version number, e.g. "0.3.7">

release:
	# guards
	@test -n "$(V)" || (echo "Usage: make release V=0.3.7" && exit 1)
	@grep -q '^Version $(V)$$' CHANGELOG.md \
		|| (echo "CHANGELOG.md has no 'Version $(V)' section yet" && exit 1)
	@test "$$(git rev-parse --abbrev-ref HEAD)" = main \
		|| (echo "Must release from main (you are on $$(git rev-parse --abbrev-ref HEAD))" && exit 1)
	$(MAKE) mypy
	$(MAKE) test
	# version bump
	sed -i 's/__version__ = ".*"/__version__ = "$(V)"/' matthewplotlib/__init__.py
	sed -i 's/^version = ".*"/version = "$(V)"/' pyproject.toml
	# rebuild docs
	$(MAKE) docs
	# commit
	git add matthewplotlib/__init__.py pyproject.toml docs CHANGELOG.md
	git commit -m "Version $(V)"
	git tag v$(V)
	# prepare to push
	@echo "ready to release:"
	@echo "git push origin main --tags"
	@echo "(then make the release on github)"

.PHONY: docs docs/images mypy test goldens release
