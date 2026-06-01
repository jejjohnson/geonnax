"""Slepian positional encoders for localized spherical features.

Tier-A deterministic cores. The Bayesian variant (which puts NumPyro
sites on the cap radius and centre) lives as a bespoke wrapper in the
consuming probabilistic library.
"""

from __future__ import annotations

import math
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from geonnax._basis import SlepianCapBasis, slepian_cap_basis
from geonnax.encoders import SphericalHarmonicEncoder
from geonnax.geo import lonlat_to_cartesian3d


def _unit_xyz(
    x: Float[Array, " 3"] | Float[Array, " 2"],
    input_mode: Literal["cartesian", "lonlat"],
) -> Float[Array, " 3"]:
    if input_mode == "cartesian":
        if x.ndim != 1 or x.shape[-1] != 3:
            raise ValueError(
                f"x must be (3,) when input_mode='cartesian'; got shape {x.shape}."
            )
        return x
    if x.ndim != 1 or x.shape[-1] != 2:
        raise ValueError(f"x must be (2,) when input_mode='lonlat'; got {x.shape}.")
    return lonlat_to_cartesian3d(x[None, :], input_unit="radians")[0]


class SlepianEncoder(eqx.Module):
    """Deterministic Slepian-cap positional encoder on :math:`S^2`.

    Slepian functions are the band-limited modes maximally concentrated
    inside a spherical cap; each is a weighted sum of spherical harmonics
    up to ``l_max`` with concentration eigenvalue ``λ ∈ [0, 1]``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax.slepian import SlepianEncoder
        >>> enc = SlepianEncoder.from_cap(
        ...     l_max=3,
        ...     cap_radius_deg=20.0,
        ...     cap_centre_lonlat_deg=(0.0, 0.0),
        ...     eig_threshold=0.0,
        ...     n_modes=4,
        ... )
        >>> enc(jnp.array([0.0, 0.0])).shape == (enc.num_features,)
        True
    """

    basis: SlepianCapBasis
    input_mode: Literal["cartesian", "lonlat"] = eqx.field(
        static=True, default="lonlat"
    )
    weight_by_eigenvalue: bool = eqx.field(static=True, default=True)

    def __post_init__(self) -> None:
        if self.input_mode not in {"cartesian", "lonlat"}:
            raise ValueError(
                f"input_mode must be 'cartesian' or 'lonlat'; got {self.input_mode!r}."
            )

    @classmethod
    def from_cap(
        cls,
        *,
        l_max: int,
        cap_radius_deg: float,
        cap_centre_lonlat_deg: tuple[float, float],
        eig_threshold: float = 0.05,
        n_modes: int | None = None,
        input_mode: Literal["cartesian", "lonlat"] = "lonlat",
        weight_by_eigenvalue: bool = True,
    ) -> SlepianEncoder:
        """Construct an encoder from cap geometry specified in degrees.

        Examples:
            >>> from geonnax.slepian import SlepianEncoder
            >>> enc = SlepianEncoder.from_cap(
            ...     l_max=3,
            ...     cap_radius_deg=20.0,
            ...     cap_centre_lonlat_deg=(0.0, 0.0),
            ...     eig_threshold=0.0,
            ...     n_modes=4,
            ... )
            >>> enc.num_features
            4
        """
        # Degrees in the public API → radians for the basis solver.
        basis = slepian_cap_basis(
            l_max=l_max,
            cap_radius=math.radians(cap_radius_deg),
            n_modes=n_modes,
            eig_threshold=eig_threshold,
            lonlat_centre=jnp.radians(jnp.asarray(cap_centre_lonlat_deg)),
        )
        return cls(
            basis=basis,
            input_mode=input_mode,
            weight_by_eigenvalue=weight_by_eigenvalue,
        )

    @property
    def num_features(self) -> int:
        """Number of output features (retained Slepian modes).

        Examples:
            >>> from geonnax.slepian import SlepianEncoder
            >>> enc = SlepianEncoder.from_cap(
            ...     l_max=3,
            ...     cap_radius_deg=20.0,
            ...     cap_centre_lonlat_deg=(0.0, 0.0),
            ...     eig_threshold=0.0,
            ...     n_modes=4,
            ... )
            >>> enc.num_features
            4
        """
        return self.basis.num_modes

    def __call__(
        self,
        x: Float[Array, " 3"] | Float[Array, " 2"],
    ) -> Float[Array, " K"]:
        """Evaluate the ``K`` retained Slepian modes at a single point.

        Examples:
            >>> import jax.numpy as jnp
            >>> from geonnax.slepian import SlepianEncoder
            >>> enc = SlepianEncoder.from_cap(
            ...     l_max=3,
            ...     cap_radius_deg=20.0,
            ...     cap_centre_lonlat_deg=(0.0, 0.0),
            ...     eig_threshold=0.0,
            ...     n_modes=4,
            ... )
            >>> enc(jnp.array([0.0, 0.0])).shape
            (4,)
        """
        unit_xyz = _unit_xyz(x, self.input_mode)  # (2,)/(3,) -> (3,)
        # (3,) -> (1, 3) -> evaluate K modes -> (1, K) -> drop batch -> (K,)
        features = self.basis.evaluate(unit_xyz[None, :])[0]
        if self.weight_by_eigenvalue:
            # Scale mode k by √λ_k so well-concentrated modes dominate.
            features = features * jnp.sqrt(self.basis.eigenvalues)
        return features


class HybridSphericalSlepianEncoder(eqx.Module):
    """Concatenate low-bandwidth global SHs with local Slepian cap modes.

    Output is ``[Y_l^m features | Slepian cap features]`` — global
    structure from spherical harmonics plus localized detail from the cap.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax.slepian import HybridSphericalSlepianEncoder
        >>> enc = HybridSphericalSlepianEncoder.from_cap(
        ...     sh_l_max=1,
        ...     slepian_l_max=3,
        ...     cap_radius_deg=20.0,
        ...     cap_centre_lonlat_deg=(0.0, 0.0),
        ...     eig_threshold=0.0,
        ...     n_modes=2,
        ... )
        >>> enc(jnp.array([0.0, 0.0])).shape == (enc.num_features,)
        True
    """

    sh_encoder: SphericalHarmonicEncoder
    slepian_encoder: SlepianEncoder

    @classmethod
    def from_cap(
        cls,
        *,
        sh_l_max: int,
        slepian_l_max: int,
        cap_radius_deg: float,
        cap_centre_lonlat_deg: tuple[float, float],
        eig_threshold: float = 0.05,
        n_modes: int | None = None,
        input_mode: Literal["cartesian", "lonlat"] = "lonlat",
    ) -> HybridSphericalSlepianEncoder:
        """Construct the hybrid encoder used by Slepian positional encodings.

        Examples:
            >>> from geonnax.slepian import HybridSphericalSlepianEncoder
            >>> enc = HybridSphericalSlepianEncoder.from_cap(
            ...     sh_l_max=1,
            ...     slepian_l_max=3,
            ...     cap_radius_deg=20.0,
            ...     cap_centre_lonlat_deg=(0.0, 0.0),
            ...     eig_threshold=0.0,
            ...     n_modes=2,
            ... )
            >>> # sh_l_max=1 → (1 + 1)^2 = 4 SH features, plus 2 Slepian modes.
            >>> enc.num_features
            6
        """
        return cls(
            sh_encoder=SphericalHarmonicEncoder(l_max=sh_l_max, input_mode=input_mode),
            slepian_encoder=SlepianEncoder.from_cap(
                l_max=slepian_l_max,
                cap_radius_deg=cap_radius_deg,
                cap_centre_lonlat_deg=cap_centre_lonlat_deg,
                eig_threshold=eig_threshold,
                n_modes=n_modes,
                input_mode=input_mode,
            ),
        )

    @property
    def num_features(self) -> int:
        """Number of concatenated output features (SH modes + Slepian modes).

        Examples:
            >>> from geonnax.slepian import HybridSphericalSlepianEncoder
            >>> enc = HybridSphericalSlepianEncoder.from_cap(
            ...     sh_l_max=1,
            ...     slepian_l_max=3,
            ...     cap_radius_deg=20.0,
            ...     cap_centre_lonlat_deg=(0.0, 0.0),
            ...     eig_threshold=0.0,
            ...     n_modes=2,
            ... )
            >>> enc.num_features
            6
        """
        return self.sh_encoder.num_features + self.slepian_encoder.num_features

    def __call__(
        self,
        x: Float[Array, " 3"] | Float[Array, " 2"],
    ) -> Float[Array, " F"]:
        """Concatenate SH and Slepian features for a single point.

        Examples:
            >>> import jax.numpy as jnp
            >>> from geonnax.slepian import HybridSphericalSlepianEncoder
            >>> enc = HybridSphericalSlepianEncoder.from_cap(
            ...     sh_l_max=1,
            ...     slepian_l_max=3,
            ...     cap_radius_deg=20.0,
            ...     cap_centre_lonlat_deg=(0.0, 0.0),
            ...     eig_threshold=0.0,
            ...     n_modes=2,
            ... )
            >>> enc(jnp.array([0.0, 0.0])).shape
            (6,)
        """
        # (num_sh,) ++ (num_slepian,) -> (num_sh + num_slepian,)
        return jnp.concatenate([self.sh_encoder(x), self.slepian_encoder(x)], axis=-1)


__all__ = [
    "HybridSphericalSlepianEncoder",
    "SlepianEncoder",
]
