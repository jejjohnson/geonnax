"""Earth constants for geonnax's lon/lat helpers, re-exported from `geonnax.geo`.

functax keeps no Earth constants (it takes ``semi_major`` and ``flattening``),
so the values geonnax's ellipsoid names stand for live here. geonnax sits
below geophysix and must not import it, so they repeat geophysix's
``WGS84.semi_major``, ``WGS84.flattening`` and ``SPHERE.semi_major``;
``tests/test_geophysix_constants.py`` keeps the copies equal.
"""

from __future__ import annotations


WGS84_A = 6378137.0
"""WGS84 semi-major axis, in metres."""
WGS84_F = 1.0 / 298.257223563
"""WGS84 flattening."""
EARTH_MEAN_RADIUS_M = 6371008.771415059
"""IUGG mean Earth radius R₁ = (2a + b) / 3 of WGS84, in metres."""
