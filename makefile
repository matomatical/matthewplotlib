# # # 
# Documentation website
#
# Everything the site is built from lives in docs/: the sources under
# docs/src, the templates it overrides, and the mkdocs configuration tying
# them together. Published to the gh-pages branch by mike, one directory per
# version. See CONTRIBUTING.md.

CONFIG := docs/mkdocs.yml

# The alias the site root redirects to, moved to each new release.
DOCS_ALIAS := latest

# Source links in the API reference are derived from the github remote, and
# are silently omitted when it cannot be found.
docs:
	@git remote get-url origin 2>/dev/null | grep -q github.com \
		|| echo "warning: no github remote, API source links will be missing"
	mkdocs build --config-file $(CONFIG) --strict

serve:
	mkdocs serve --config-file $(CONFIG)

# `make docs` leaves the site it built behind, for looking over.
clean:
	rm -rf site

# Usage: make deploy V=<version number, e.g. "0.6.3">
# The alias is a copy rather than mike's default symlink, which github pages
# serves as a 404 instead of following.
deploy:
	@test -n "$(V)" || (echo "Usage: make deploy V=0.6.3" && exit 1)
	mike deploy --config-file $(CONFIG) --update-aliases --alias-type=copy $(V) $(DOCS_ALIAS)
	mike set-default --config-file $(CONFIG) $(DOCS_ALIAS)
	@echo "deployed. to publish:"
	@echo "git push origin gh-pages"

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
	$(MAKE) docs
	# version bump
	sed -i 's/__version__ = ".*"/__version__ = "$(V)"/' matthewplotlib/__init__.py
	sed -i 's/^version = ".*"/version = "$(V)"/' pyproject.toml
	# commit
	git add matthewplotlib/__init__.py pyproject.toml CHANGELOG.md
	git commit -m "Version $(V)"
	git tag v$(V)
	# prepare to push
	@echo "ready to release:"
	@echo "git push origin main --tags"
	@echo "make deploy V=$(V)"
	@echo "(then make the release on github)"

.PHONY: docs serve deploy clean mypy test goldens release
