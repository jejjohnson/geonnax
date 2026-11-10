"""Earth constants for geonnax's lon/lat helpers, re-exported from `geonnax.geo`.

functax keeps no Earth constants (it takes ``semi_major`` and ``flattening``),
so the values geonnax's ellipsoid names stand for live here.
"""

from __future__ import annotations


WGS84_A = 6378137.0
"""WGS84 semi-major axis, in metres."""
WGS84_F = 1.0 / 298.257223563
"""WGS84 flattening."""
EARTH_MEAN_RADIUS_M = 6371008.8
"""IUGG mean Earth radius, in metres."""
