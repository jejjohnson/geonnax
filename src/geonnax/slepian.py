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
    """Deterministic Slepian-cap positional encoder on :math:`S^2`."""

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
        """Construct an encoder from cap geometry specified in degrees."""
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
        """Number of output features."""
        return self.basis.num_modes

    def __call__(
        self,
        x: Float[Array, " 3"] | Float[Array, " 2"],
    ) -> Float[Array, " K"]:
        unit_xyz = _unit_xyz(x, self.input_mode)
        features = self.basis.evaluate(unit_xyz[None, :])[0]
        if self.weight_by_eigenvalue:
            features = features * jnp.sqrt(self.basis.eigenvalues)
        return features


class HybridSphericalSlepianEncoder(eqx.Module):
    """Concatenate low-bandwidth global SHs with local Slepian cap modes."""

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
        """Construct the hybrid encoder used by Slepian positional encodings."""
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
        """Number of concatenated output features."""
        return self.sh_encoder.num_features + self.slepian_encoder.num_features

    def __call__(
        self,
        x: Float[Array, " 3"] | Float[Array, " 2"],
    ) -> Float[Array, " F"]:
        return jnp.concatenate([self.sh_encoder(x), self.slepian_encoder(x)], axis=-1)


__all__ = [
    "HybridSphericalSlepianEncoder",
    "SlepianEncoder",
]
