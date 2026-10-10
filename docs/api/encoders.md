# Encoders & Geometry

Feature maps that turn raw coordinates — degrees, lon/lat, points on the sphere — into
representations a downstream network can use, plus the geometric helpers behind them.

## Coordinate encoders

`equinox.Module` encoders: degree↔radian conversion, lon/lat rescaling, cartesian
projection, cyclic (sin/cos) features, and spherical-harmonic embeddings.

`GeoContextEncoder` composes these into a single **context vector** — longitude,
latitude, day of year, and arbitrary physical covariates (wind, retrieval
uncertainty, viewing geometry) concatenated into one array. That vector is what you
pass as the `condition` of a conditional normalizing flow, an anomaly scorer, or any
of the [conditioning layers](representation.md).

Feature layout is fixed: `[lon/lat block, time block, extra block]`, with covariates
in sorted key order. Inputs are in degrees, one observation per call — use
`jax.vmap` to encode a batch.

```python
import jax
import jax.numpy as jnp
from geonnax import GeoContextEncoder

encoder = GeoContextEncoder(latlon_encoding="spherical", time_encoding="cyclic")

context = jax.vmap(
    lambda lat, lon, doy, extra: encoder(
        lat=lat, lon=lon, day_of_year=doy, extra=extra
    )
)(lat, lon, day_of_year, {"wind_u": wind_u, "sigma": sigma})

context.shape[-1] == encoder.output_dim(extra_dim=2)
```

::: geonnax.encoders

## Slepian encoders

Slepian functions are optimally concentrated on a spherical cap — a localised
alternative to global spherical harmonics.

::: geonnax.slepian

## Geometry helpers

Pure functions for lon/lat math: degree conversion, rescaling, lon/lat → cartesian,
cyclic encoding, and spherical-harmonic evaluation; plus coordinate frames —
WGS84 ↔ ECEF (`lonlat_to_ecef` / `ecef_to_lonlat`), the local east/north/up
basis (`enu_basis`), east/north ↔ 3-D tangent vectors
(`tangent_to_cartesian` / `cartesian_to_tangent`), tangent-plane projections
(`local_tangent_plane`) and the rotation taking a centre to the pole
(`rotation_to_pole`).

::: geonnax.geo
