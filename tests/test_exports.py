"""Every public name is reachable as `mp.something`.

`__init__.py` is a second place to edit whenever a feature is added, and no other
test reaches for the library through the top-level namespace -- they all import
from the defining module -- so an export left out breaks nothing and ships. That
happened to `animation`. CONTRIBUTING.md has always asked for the export; this
asks on its behalf.

The expectation is derived from the code rather than listed here, so adding a
plot type or a colormap does not mean editing a third file: whatever a covered
module defines, the top-level namespace has to re-export.
"""

import importlib
import types

import pytest

import matthewplotlib as mp


# Modules whose entire public surface belongs in the top-level namespace.
#
# The others cannot be covered this way, and it is not an oversight: `data`
# keeps its parsers back (`parse_range`, `parse_multiple_series`, ...), `colors`
# keeps `Color` and `parse_color`, `camera` keeps its projections, and `core` is
# the character-array backend, where only `BoxStyle` is meant to be reached
# for. Covering those needs an
# explicit `__all__` per module, or a leading underscore on each internal name.
# Both would also stop pdoc documenting the internals. The trade-offs, and the
# counts to decide with, are in the `export-policy` note.
FULLY_PUBLIC = ("plots", "colormaps", "animations")


def defined_in(module_name: str) -> list[str]:
    """The public names a module defines itself, ignoring what it imports."""
    module = importlib.import_module(f"matthewplotlib.{module_name}")
    return sorted(
        name
        for name, value in vars(module).items()
        if not name.startswith("_")
        and not isinstance(value, types.ModuleType)
        and getattr(value, "__module__", None) == module.__name__
    )


@pytest.mark.parametrize("module_name", FULLY_PUBLIC)
def test_every_public_name_is_exported(module_name):
    missing = [name for name in defined_in(module_name) if not hasattr(mp, name)]
    assert not missing, (
        f"matthewplotlib.{module_name} defines {missing}, which "
        f"`import matthewplotlib as mp` cannot reach. Add them to the "
        f"`from matthewplotlib.{module_name} import (...)` list in "
        f"matthewplotlib/__init__.py, or make them private with a leading "
        f"underscore if they are not meant to be public."
    )


@pytest.mark.parametrize("module_name", FULLY_PUBLIC)
def test_the_module_defines_something(module_name):
    """So that a renamed or emptied module fails here rather than passing vacuously."""
    assert defined_in(module_name)


def test_exports_resolve_to_the_module_they_came_from():
    """`mp.scatter` should be `plots.scatter`, not a shadowed name.

    Two modules defining the same name would otherwise let one silently win,
    and the test above would still pass.
    """
    clashes = []
    seen: dict[str, str] = {}
    for module_name in FULLY_PUBLIC:
        for name in defined_in(module_name):
            if name in seen:
                clashes.append(f"{name} in both {seen[name]} and {module_name}")
            seen[name] = module_name
            expected = importlib.import_module(f"matthewplotlib.{module_name}")
            if getattr(mp, name, None) is not getattr(expected, name):
                clashes.append(f"mp.{name} is not matthewplotlib.{module_name}.{name}")
    assert not clashes, clashes
