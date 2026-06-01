# Encoders & Geometry

Feature maps that turn raw coordinates — degrees, lon/lat, points on the sphere — into
representations a downstream network can use, plus the geometric helpers behind them.

## Coordinate encoders

`equinox.Module` encoders: degree↔radian conversion, lon/lat rescaling, cartesian
projection, cyclic (sin/cos) features, and spherical-harmonic embeddings.

::: geonnax.encoders

## Slepian encoders

Slepian functions are optimally concentrated on a spherical cap — a localised
alternative to global spherical harmonics.

::: geonnax.slepian

## Geometry helpers

Pure functions for lon/lat math: degree conversion, rescaling, lon/lat → cartesian,
cyclic encoding, and spherical-harmonic evaluation.

::: geonnax.geo
