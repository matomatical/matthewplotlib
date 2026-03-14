# HTML Documentation Research

## Architecture

- **API docs:** pdoc generates HTML from Python docstrings into `docs/`
- **Standalone pages:** pandoc converts markdown (README.md, CHANGELOG.md) to
  HTML using `templates/page.html`, sharing pdoc's CSS via `docs/pdoc.css`
- **CSS:** pdoc's built-in CSS is concatenated with `templates/custom.css`
  into `docs/pdoc.css` for pandoc pages; pdoc inlines CSS directly
- **Templates:**
  - `templates/module.html.jinja2` — pdoc template override (sidebar nav tree,
    theme toggle, back button, "Getting Started" links)
  - `templates/page.html` — pandoc template (sidebar, search, theme support)
  - `templates/custom.css` — sweetie16 dark/light theme, layout tweaks
- **Build:** `make docs` runs pdoc + pandoc + copies images
- **Serving locally:** `python -m http.server -b 0.0.0.0 -d docs/ 8081`

### Key implementation details

- pdoc's `all_modules` is a `Mapping[str, pdoc.doc.Module]` — full module
  objects available in templates, used for the persistent nav tree
- Pandoc pages hardcode only "matthewplotlib" link under API Reference;
  pdoc dynamically lists submodules. No module list duplication.
- Search on pandoc pages loads pdoc's `search.js` index directly
- Nav tree indentation uses padding on `<a>` elements (not `<ul>` padding)
  so hover backgrounds span full sidebar width
- Theme toggle uses `data-theme` attribute on `<html>` + localStorage
- pdoc excludes `_`-prefixed names by default; `--no-show-source` keeps
  file sizes small; `@private` in docstrings available if needed
- pdoc's `.. include::` directive can embed markdown files in docstrings
  (not currently used, but available if needed)

## Roadmap

### Styling

- [x] Dark/light theme with sweetie16 palette, toggle button, system preference
- [x] Monospace font for headings and code attributes
- [x] Flat sidebar (no shadow)
- [x] "matthewplotlib" title in sidebar linking to landing page
- [x] Back button in main content header on submodule pages
- [x] "Edit on GitHub" links via `-e` flag
- [x] Sidebar navigation tree with expandable submodules, consistent across
      all pages (pdoc and pandoc)
- [x] Search on all pages (including pandoc-generated pages)
- [ ] Logo in sidebar (needs design)
- [ ] Fine-tune colours if needed after more content is added

### Content

- [x] Landing page (README rendered via pandoc with images)
- [x] Changelog page (CHANGELOG.md rendered via pandoc)
- [x] Exclude private variables from docs
- [x] Example gallery page (images + links to source)

### Deployment

- [ ] GitHub Pages from `docs/` on `main` (no Actions needed)
  - Configure GitHub Pages to serve from `docs/` directory on main branch
  - Run `make docs` before committing
- [ ] Versioned docs (`docs/v0.3.7/` with tag-pinned source links)
  - Use `-e matthewplotlib=.../blob/v0.3.7/matthewplotlib/` for source links
  - Script in `make release` to generate versioned docs folder
