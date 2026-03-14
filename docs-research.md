# HTML Documentation Research

## Architecture

- **API docs:** pdoc generates HTML from Python docstrings into `docs/`
- **Standalone pages:** pandoc converts markdown (README.md, CHANGELOG.md) to
  HTML using `templates/page.html`, sharing pdoc's CSS via `docs/pdoc.css`
- **CSS:** pdoc's built-in CSS is concatenated with `templates/custom.css`
  into `docs/pdoc.css` for pandoc pages; pdoc inlines CSS directly
- **Templates:**
  - `templates/module.html.jinja2` — pdoc template override (sidebar nav tree,
    theme toggle, back button, SVG icon buttons, "Overview" links)
  - `templates/page.html` — pandoc template (sidebar, search, theme toggle,
    edit on GitHub, SVG icons)
  - `templates/custom.css` — sweetie16 dark/light theme, layout tweaks,
    nav heading styles, page image constraints
- **Build:** `make docs` runs pdoc + pandoc + copies images
- **Serving locally:** `python -m http.server -b 0.0.0.0 -d docs/ 8081`

### Key implementation details

- pdoc's `all_modules` is a `Mapping[str, pdoc.doc.Module]` — full module
  objects available in templates, used for the persistent nav tree
- Pandoc pages hardcode only "matthewplotlib" link under Reference;
  pdoc dynamically lists submodules. No module list duplication.
- Pandoc pages source in `pages/` dir (quickstart, examples, roadmap)
  plus README.md and CHANGELOG.md
- SVG icons (Feather-style sun/moon, GitHub octocat) inlined in templates,
  use `stroke="currentColor"`/`fill="currentColor"` for theme compatibility
- Page images get `max-width: 100%` and dark background for transparency
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
- [x] SVG icon buttons (Feather sun/moon, GitHub octocat) on all pages
- [x] Nav sections renamed: "Overview" and "Reference"
- [x] Nav heading size matched to logo text
- [x] Page images constrained to content width with dark background
- [ ] Logo in sidebar (needs design)
- [ ] Fine-tune colours if needed after more content is added

### Content

- [x] Landing page (README rendered via pandoc with images)
- [x] Changelog page (CHANGELOG.md rendered via pandoc)
- [x] Exclude private variables from docs
- [x] Example gallery page (images + links to source)
- [x] Quickstart page (extracted from README)
- [x] Roadmap page with related work (extracted from README)
- [ ] Generate images for all examples (some currently missing)

### Deployment

- [x] GitHub Pages from `docs/` on `main` at matthewplotlib.far.in.net
  - Run `make docs` before committing
- [ ] Versioned docs (`docs/v0.3.7/` with tag-pinned source links)
  - Use `-e matthewplotlib=.../blob/v0.3.7/matthewplotlib/` for source links
  - Script in `make release` to generate versioned docs folder
