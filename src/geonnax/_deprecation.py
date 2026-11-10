"""Deprecated re-exports of the maths that moved to functax.

The bases, sphere geometry, field derivatives and transforms now live in
functax (jejjohnson/geonnax#153). Their old geonnax paths keep working for one
release: a module lists its moved names in a `Moved` table and installs the
``__getattr__`` built here, which returns the functax object and emits a
`DeprecationWarning` naming the new path. The names are not in ``__all__``, so
``import geonnax`` and ``from geonnax.basis import *`` stay silent.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any, NamedTuple


#: the geonnax release that deletes every alias registered here
REMOVED_IN = "0.0.13"


class Moved(NamedTuple):
    """One moved name: where it lives now, and the object the old path returns.

    Attributes:
        target: The new import path, quoted in the warning
            (``"functax.basis.local.rbf_basis"``).
        obj: The functax object itself, or a geonnax shim that keeps the old
            signature when functax changed it (see `geonnax._compat`).
        note: Extra advice for the warning, such as the keyword that changed.
    """

    target: str
    obj: Any
    note: str = ""


def deprecated_getattr(module: str, moved: dict[str, Moved]) -> Callable[[str], Any]:
    """Build a module ``__getattr__`` that serves ``moved`` with a warning.

    Args:
        module: The module's dotted name, quoted in messages.
        moved: Moved names and where each one went.

    Returns:
        A function to bind as the module's ``__getattr__``. It raises
        `AttributeError` for names that are not in ``moved``.

    Examples:
        >>> import warnings
        >>> getattr_ = deprecated_getattr(
        ...     "pkg.mod", {"f": Moved("functax.f", len)}
        ... )
        >>> with warnings.catch_warnings(record=True) as caught:
        ...     warnings.simplefilter("always")
        ...     getattr_("f") is len
        True
        >>> caught[0].category.__name__
        'DeprecationWarning'
    """

    def __getattr__(name: str) -> Any:
        try:
            entry = moved[name]
        except KeyError:
            msg = f"module {module!r} has no attribute {name!r}"
            raise AttributeError(msg) from None
        message = (
            f"{module}.{name} moved to {entry.target}; the geonnax path is "
            f"deprecated and will be removed in geonnax {REMOVED_IN}."
        )
        if entry.note:
            message = f"{message} {entry.note}"
        warnings.warn(message, DeprecationWarning, stacklevel=2)
        return entry.obj

    return __getattr__


def deprecated_dir(
    names: list[str], moved: dict[str, Moved]
) -> Callable[[], list[str]]:
    """Build a module ``__dir__`` that lists the moved names next to ``names``."""

    def __dir__() -> list[str]:
        return sorted({*names, *moved})

    return __dir__
