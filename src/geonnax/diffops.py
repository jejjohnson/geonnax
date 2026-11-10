"""The old home of the field derivatives, which moved to `functax.diffops`.

`geonnax.diffops` moved to `functax.diffops` (jejjohnson/geonnax#153). Each
name still resolves here for one release, with a `DeprecationWarning` naming
the functax path. Two names changed so that they no longer shadow JAX:
``grad`` is ``functax.diffops.gradient`` and ``jacobian`` is
``functax.diffops.jacobian_field``. ``surface_grad_enu`` keeps geonnax's
degrees default here; functax takes ``unit=`` with no default.
"""

from __future__ import annotations

import functax.diffops as _diffops

from geonnax import _compat
from geonnax._deprecation import Moved, deprecated_dir, deprecated_getattr


#: names that moved to functax, served with a DeprecationWarning until removal
_MOVED: dict[str, Moved] = {
    "curl2d": Moved("functax.diffops.curl2d", _diffops.curl2d),
    "divergence": Moved("functax.diffops.divergence", _diffops.divergence),
    "grad": Moved("functax.diffops.gradient", _diffops.gradient),
    "jacobian": Moved("functax.diffops.jacobian_field", _diffops.jacobian_field),
    "laplace_beltrami": Moved(
        "functax.diffops.laplace_beltrami", _diffops.laplace_beltrami
    ),
    "laplacian": Moved("functax.diffops.laplacian", _diffops.laplacian),
    "surface_grad": Moved("functax.diffops.surface_grad", _diffops.surface_grad),
    "surface_grad_enu": Moved(
        "functax.diffops.surface_grad_enu",
        _compat.surface_grad_enu,
        "Note that it takes unit= in place of input_unit=, with no default.",
    ),
}

__all__: list[str] = []

__getattr__ = deprecated_getattr(__name__, _MOVED)
__dir__ = deprecated_dir(__all__, _MOVED)
