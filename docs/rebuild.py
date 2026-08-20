"""
Rebuild the documentation website for every released version.

Every tag predates this site, so there is nothing to build from in any of
them. For each one, this checks the tag out into a worktree, generates a
documentation tree for it -- a configuration, a page for each source the
version has, and one for each module it defines -- and hands that to mike to
publish alongside the others.

The generated pages carry the version's own files, rather than including them,
so that image paths written for a file at the root of the repository can be
adjusted for the page that now shows it. Everything else about the site, the
theme and the templates and the palette, comes from the working tree: these
are the old versions as the site renders them today, not a reproduction of
what they looked like at the time.

Usage: python docs/rebuild.py [--alias latest] [tag ...]
"""

import argparse
import os
import pathlib
import re
import shutil
import subprocess
import tempfile

import yaml


REPO = pathlib.Path(__file__).resolve().parent.parent

# The order the navigation introduces the modules in, matching the order the
# package docstring introduces them. Anything unrecognised follows, sorted.
MODULE_ORDER = [
    "plots",
    "animations",
    "data",
    "colors",
    "colormaps",
    "core",
    "camera",
    "unscii16",
]

# Each page of the site, the file the version keeps its content in, and where
# the API reference falls among them.
PAGES = [
    ("Home", "index.md", ["README.md"]),
    ("Quickstart", "quickstart.md", ["pages/quickstart.md"]),
    ("Examples", "examples.md", ["pages/examples.md"]),
    ("API Reference", None, None),
    ("Compatibility", "compatibility.md", ["pages/compatibility.md"]),
    ("Changelog", "changelog.md", ["CHANGELOG.md"]),
    ("Roadmap", "roadmap.md", ["pages/roadmap.md", "ROADMAP.md"]),
]


def git(*args, cwd=REPO):
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def released_tags():
    """Every version tag, oldest first. Tags without the v are not releases."""
    def key(tag):
        release = tag[1:].split("-")[0]
        numbers = tuple(int(part) for part in release.split("."))
        # an alpha of a version precedes the version itself
        return numbers + (0 if "-" in tag else 1,)
    tags = [t for t in git("tag").split() if re.fullmatch(r"v\d[\d.]*(-\w+)?", t)]
    return sorted(tags, key=key)


def modules_of(tree):
    """The modules a checkout defines, in the order the navigation wants."""
    found = {
        path.stem
        for path in (tree / "matthewplotlib").glob("*.py")
        if path.stem != "__init__"
    }
    ordered = [m for m in MODULE_ORDER if m in found]
    return ordered + sorted(found - set(ordered))


def write_page(path, source, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\nsource: {source}\n---\n\n{body}")


def generate(tree, tag):
    """Write a documentation tree for the checkout, and return its config."""
    docs = tree / "docs"
    src = docs / "src"
    src.mkdir(parents=True)

    # the palette, and the templates the theme and mkdocstrings render through
    shutil.copytree(REPO / "docs" / "src" / "css", src / "css")
    shutil.copytree(REPO / "docs" / "templates", docs / "templates")

    # images are referenced from the root of the repository, which is where
    # every page's content was written to sit
    os.symlink("../../images", src / "images")

    nav = []
    for title, page, candidates in PAGES:
        if page is None:
            modules = modules_of(tree)
            write_page(src / "api.md", "matthewplotlib/__init__.py",
                       "::: matthewplotlib\n")
            for module in modules:
                write_page(src / "api" / f"{module}.md",
                           f"matthewplotlib/{module}.py",
                           f"::: matthewplotlib.{module}\n")
            nav.append({title: [{"matthewplotlib": "api.md"}] +
                               [{m: f"api/{m}.md"} for m in modules]})
            continue
        source = next((c for c in candidates if (tree / c).exists()), None)
        if source is None:
            continue
        body = (tree / source).read_text()
        # raw html is not rewritten for the page it lands on, and every page
        # but the home page is served one directory deep
        if page != "index.md":
            body = body.replace('src="images/', 'src="../images/')
        write_page(src / page, source, body)
        nav.append({title: page})

    # an image a docstring refers to has to resolve beside the module's page
    wanted = set()
    for module in (tree / "matthewplotlib").glob("*.py"):
        wanted |= set(re.findall(r"]\(images/([^)]+)\)", module.read_text()))
    if wanted:
        (src / "api" / "images").mkdir(parents=True, exist_ok=True)
        for name in sorted(wanted):
            if (tree / "images" / name).exists():
                os.symlink(f"../../../../images/{name}",
                           src / "api" / "images" / name)

    config = yaml.safe_load((REPO / "docs" / "mkdocs.yml").read_text())
    config["docs_dir"] = "src"
    config["site_dir"] = "../site"
    config["nav"] = nav
    # the pages are generated, so git holds no history for them to date, and
    # the links back to them belong to this version rather than to main
    config["plugins"] = [p for p in config["plugins"] if p != "git-revision-date"]
    config.setdefault("extra", {})["source_ref"] = tag
    config.pop("draft_docs", None)
    config.pop("watch", None)
    path = docs / "mkdocs.yml"
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    return path


def rebuild(tags, alias):
    for index, tag in enumerate(tags):
        version = tag[1:]
        last = index == len(tags) - 1
        with tempfile.TemporaryDirectory() as area:
            tree = pathlib.Path(area) / "tree"
            git("worktree", "add", "--quiet", "--detach", str(tree), tag)
            try:
                config = generate(tree, tag)
                command = [
                    "mike", "deploy",
                    "--config-file", str(config),
                    "--update-aliases",
                    "--alias-type=copy",
                    version,
                ]
                if last and alias:
                    command.append(alias)
                subprocess.run(command, cwd=REPO, check=True)
                print(f"  {version}: {len(modules_of(tree))} modules"
                      + (f", aliased {alias}" if last and alias else ""))
            finally:
                git("worktree", "remove", "--force", str(tree))
    if alias:
        subprocess.run(["mike", "set-default", "--config-file",
                        str(REPO / "docs" / "mkdocs.yml"), alias],
                       cwd=REPO, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tags", nargs="*", help="tags to rebuild (default: all)")
    parser.add_argument("--alias", default="latest",
                        help="alias for the newest tag rebuilt (default: latest)")
    args = parser.parse_args()
    tags = args.tags or released_tags()
    print(f"rebuilding {len(tags)} versions: {' '.join(tags)}")
    rebuild(tags, args.alias)


if __name__ == "__main__":
    main()
