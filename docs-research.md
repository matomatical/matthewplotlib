# HTML Documentation Research

## Context

matthewplotlib currently generates API docs via a bespoke AST-walking script
(`generate_docs.py`) that outputs a single `DOCS.md`. The goal is to replace
this with HTML documentation hosted on GitHub Pages (in `docs/`).

**Constraints:**
- Docstrings stay in markdown (readable from source code)
- Minimalist aesthetic matching the project's character
- Simple workflow (ideally one makefile target)
- Future-proof for: cross-linking between symbols, search, versioned docs

## Tool evaluation

Evaluated pdoc, pydoctor, griffe+mkdocstrings, handsdown, lazydocs, and a
bespoke script extension. pdoc was the clear winner: clean minimalist HTML,
native markdown docstrings, near-zero config, 4 deps, built-in search and
cross-linking, Jinja2 template customisation. See git history for full
comparison.

## Roadmap

### Styling

- [x] Dark/light theme with sweetie16 palette, toggle button, system preference
- [x] Monospace font for headings and code attributes
- [x] Flat sidebar (no shadow)
- [x] "matthewplotlib" title in sidebar linking to landing page
- [x] Back button in main content header on submodule pages
- [x] "Edit on GitHub" links via `-e` flag
- [ ] Logo in sidebar (needs design)
- [ ] Fine-tune colours if needed after more content is added

### Content

- [ ] Richer landing page (include README content: images, quickstart, etc.)
- [ ] Exclude private variables from docs (pdoc hides `_`-prefixed by default;
      verify this covers `_UNSCII_16_DATA` and `CharArray` internals)
- [ ] Example gallery page (images + links to source)
- [ ] Changelog page (rendered from CHANGELOG.md)

### Deployment

- [ ] GitHub Pages from `docs/` on `main` (no Actions needed)
- [ ] Versioned docs (`docs/v0.3.7/` with tag-pinned source links)
