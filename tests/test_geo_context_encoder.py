"""Tests for `geonnax.encoders.GeoContextEncoder`."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from geonnax import GeoContextEncoder


PERIOD = 365.25


def _call(encoder, lat, lon, doy, extra=None):
    return encoder(lat=lat, lon=lon, day_of_year=doy, extra=extra)


# -- shape and sizing contracts -------------------------------------------------


def test_default_output_shape_matches_output_dim():
    encoder = GeoContextEncoder()
    out = encoder(lat=45.0, lon=10.0, day_of_year=200.0)
    assert out.shape == (5,)
    assert out.shape[-1] == encoder.output_dim()


@pytest.mark.parametrize("latlon_encoding", ["spherical", "sincos", "raw"])
@pytest.mark.parametrize("time_encoding", ["cyclic", "raw"])
@pytest.mark.parametrize("extra_dim", [0, 1, 3])
def test_output_dim_matches_encoded_width(latlon_encoding, time_encoding, extra_dim):
    encoder = GeoContextEncoder(
        latlon_encoding=latlon_encoding, time_encoding=time_encoding
    )
    extra = {"a": jnp.arange(extra_dim, dtype=jnp.float32)} if extra_dim else None
    out = _call(encoder, 12.0, -34.0, 56.0, extra)
    assert out.shape == (encoder.output_dim(extra_dim=extra_dim),)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, 5),
        ({"latlon_encoding": "raw"}, 4),
        ({"time_encoding": "raw"}, 4),
        ({"use_time": False}, 3),
        ({"use_latlon": False}, 2),
        ({"use_latlon": False, "time_encoding": "raw"}, 1),
    ],
)
def test_output_dim_table(kwargs, expected):
    assert GeoContextEncoder(**kwargs).output_dim() == expected


# -- numerics and feature layout ------------------------------------------------


@pytest.fixture
def latlon_only():
    return GeoContextEncoder(use_time=False, include_extra=False)


@pytest.mark.parametrize(
    ("lat", "lon", "axis"),
    [
        (0.0, 0.0, [1.0, 0.0, 0.0]),
        (0.0, 90.0, [0.0, 1.0, 0.0]),
        (90.0, 0.0, [0.0, 0.0, 1.0]),
    ],
)
def test_spherical_axes(latlon_only, lat, lon, axis):
    out = latlon_only(lat=lat, lon=lon)
    assert jnp.allclose(out, jnp.asarray(axis), atol=1e-6)


def test_spherical_features_are_unit_norm(latlon_only):
    out = latlon_only(lat=37.4, lon=-122.1)
    assert jnp.allclose(jnp.linalg.norm(out), 1.0, atol=1e-6)


def test_sincos_is_cos_first_with_normalized_latitude():
    encoder = GeoContextEncoder(
        latlon_encoding="sincos", use_time=False, include_extra=False
    )
    out = encoder(lat=45.0, lon=90.0)
    # lon = 90° → cos = 0, sin = 1; lat feature is ϕ/90.
    assert jnp.allclose(out, jnp.asarray([0.0, 1.0, 0.5]), atol=1e-6)


def test_raw_latlon_is_lon_then_lat_scaled_to_unit_range():
    encoder = GeoContextEncoder(
        latlon_encoding="raw", use_time=False, include_extra=False
    )
    assert jnp.allclose(encoder(lat=90.0, lon=180.0), 1.0, atol=1e-6)
    assert jnp.allclose(encoder(lat=-90.0, lon=-180.0), -1.0, atol=1e-6)
    # Column order is (lon, lat): a mid-latitude at the dateline separates them.
    out = encoder(lat=0.0, lon=180.0)
    assert jnp.allclose(out, jnp.asarray([1.0, 0.0]), atol=1e-6)


@pytest.mark.parametrize(
    ("doy", "expected"),
    [(0.0, [1.0, 0.0]), (PERIOD / 4.0, [0.0, 1.0]), (PERIOD / 2.0, [-1.0, 0.0])],
)
def test_cyclic_time_is_cos_then_sin(doy, expected):
    encoder = GeoContextEncoder(use_latlon=False, include_extra=False)
    assert jnp.allclose(encoder(day_of_year=doy), jnp.asarray(expected), atol=1e-6)


def test_cyclic_time_wraps_across_the_period():
    encoder = GeoContextEncoder(use_latlon=False, include_extra=False)
    assert jnp.allclose(
        encoder(day_of_year=0.0), encoder(day_of_year=PERIOD), atol=1e-6
    )


def test_custom_period_rescales_the_cycle():
    encoder = GeoContextEncoder(use_latlon=False, include_extra=False, period=100.0)
    assert jnp.allclose(encoder(day_of_year=25.0), jnp.asarray([0.0, 1.0]), atol=1e-6)


def test_raw_time_is_day_over_period():
    encoder = GeoContextEncoder(
        use_latlon=False, include_extra=False, time_encoding="raw"
    )
    assert jnp.allclose(encoder(day_of_year=100.0), 100.0 / PERIOD, atol=1e-6)


def test_block_ordering_is_latlon_then_time_then_extra():
    encoder = GeoContextEncoder()
    out = _call(encoder, 0.0, 0.0, 0.0, {"z": jnp.asarray([7.0, 8.0])})
    # [x, y, z] = [1, 0, 0]; [cos, sin] = [1, 0]; then the covariates.
    assert jnp.allclose(
        out, jnp.asarray([1.0, 0.0, 0.0, 1.0, 0.0, 7.0, 8.0]), atol=1e-6
    )


# -- extra covariates -----------------------------------------------------------


def test_extra_is_concatenated_in_sorted_key_order():
    encoder = GeoContextEncoder(use_latlon=False, use_time=False)
    out = encoder(extra={"wind": jnp.asarray([3.0, 4.0]), "sigma": 12.0})
    # "sigma" sorts before "wind" regardless of insertion order.
    assert jnp.allclose(out, jnp.asarray([12.0, 3.0, 4.0]))


def test_extra_ordering_is_insertion_independent():
    encoder = GeoContextEncoder(use_latlon=False, use_time=False)
    first = encoder(extra={"a": 1.0, "b": 2.0})
    second = encoder(extra={"b": 2.0, "a": 1.0})
    assert jnp.allclose(first, second)


def test_extra_none_contributes_no_features():
    encoder = GeoContextEncoder()
    assert _call(encoder, 1.0, 2.0, 3.0, None).shape == (encoder.output_dim(),)


def test_extra_only_mode():
    encoder = GeoContextEncoder(use_latlon=False, use_time=False)
    out = encoder(extra={"a": jnp.asarray([1.0, 2.0, 3.0])})
    assert out.shape == (encoder.output_dim(extra_dim=3),)


# -- transform compatibility ----------------------------------------------------


def test_vmap_matches_per_example_calls():
    encoder = GeoContextEncoder()
    lat = jnp.linspace(-60.0, 60.0, 4)
    lon = jnp.linspace(-120.0, 120.0, 4)
    doy = jnp.asarray([1.0, 90.0, 180.0, 270.0])
    extra = {"wind": jnp.arange(8.0).reshape(4, 2), "sigma": jnp.ones(4)}

    batched = jax.vmap(lambda *a: _call(encoder, *a))(lat, lon, doy, extra)
    assert batched.shape == (4, encoder.output_dim(extra_dim=3))

    for i in range(4):
        single = _call(
            encoder,
            lat[i],
            lon[i],
            doy[i],
            {"wind": extra["wind"][i], "sigma": extra["sigma"][i]},
        )
        assert jnp.allclose(batched[i], single, atol=1e-6)


def test_filter_jit_matches_eager():
    encoder = GeoContextEncoder()
    args = (45.0, 10.0, 200.0, {"a": 1.5})
    assert jnp.allclose(
        eqx.filter_jit(_call)(encoder, *args), _call(encoder, *args), atol=1e-6
    )


def test_integer_inputs_are_promoted_to_floating():
    encoder = GeoContextEncoder()
    out = _call(encoder, 45, 10, 200, {"a": 3})
    assert jnp.issubdtype(out.dtype, jnp.floating)
    assert jnp.allclose(out, _call(encoder, 45.0, 10.0, 200.0, {"a": 3.0}), atol=1e-6)


def test_encoded_context_is_finite_across_the_globe():
    encoder = GeoContextEncoder()
    lat = jnp.linspace(-90.0, 90.0, 19)
    lon = jnp.linspace(-180.0, 180.0, 19)
    doy = jnp.linspace(0.0, PERIOD, 19)
    out = jax.vmap(lambda a, b, c: encoder(lat=a, lon=b, day_of_year=c))(lat, lon, doy)
    assert bool(jnp.all(jnp.isfinite(out)))


# -- construction-time validation -----------------------------------------------


def test_invalid_latlon_encoding():
    with pytest.raises(ValueError, match="latlon_encoding must be one of"):
        GeoContextEncoder(latlon_encoding="polar")


def test_invalid_time_encoding():
    with pytest.raises(ValueError, match="time_encoding must be one of"):
        GeoContextEncoder(time_encoding="seasonal")


def test_fourier_time_encoding_reports_it_is_unimplemented():
    with pytest.raises(ValueError, match="not yet implemented"):
        GeoContextEncoder(time_encoding="fourier")


@pytest.mark.parametrize("period", [0.0, -1.0])
def test_non_positive_period(period):
    with pytest.raises(ValueError, match="period must be positive"):
        GeoContextEncoder(period=period)


def test_all_feature_groups_disabled():
    with pytest.raises(ValueError, match="at least one of"):
        GeoContextEncoder(use_latlon=False, use_time=False, include_extra=False)


# -- call-time validation -------------------------------------------------------


@pytest.mark.parametrize(("lat", "lon"), [(None, 10.0), (45.0, None), (None, None)])
def test_missing_latlon(lat, lon):
    with pytest.raises(ValueError, match="lat and lon must both be provided"):
        GeoContextEncoder()(lat=lat, lon=lon, day_of_year=1.0)


def test_latlon_supplied_while_disabled():
    encoder = GeoContextEncoder(use_latlon=False)
    with pytest.raises(ValueError, match="must be omitted when use_latlon=False"):
        encoder(lat=45.0, day_of_year=1.0)


def test_missing_day_of_year():
    with pytest.raises(ValueError, match="day_of_year must be provided"):
        GeoContextEncoder()(lat=45.0, lon=10.0)


def test_day_of_year_supplied_while_disabled():
    encoder = GeoContextEncoder(use_time=False)
    with pytest.raises(ValueError, match="must be omitted when use_time=False"):
        encoder(lat=45.0, lon=10.0, day_of_year=1.0)


def test_extra_supplied_while_disabled():
    encoder = GeoContextEncoder(include_extra=False)
    with pytest.raises(ValueError, match="must be omitted when include_extra=False"):
        _call(encoder, 45.0, 10.0, 1.0, {"a": 1.0})


@pytest.mark.parametrize("name", ["lat", "lon", "day_of_year"])
def test_non_scalar_inputs_are_rejected(name):
    encoder = GeoContextEncoder()
    kwargs = {"lat": 45.0, "lon": 10.0, "day_of_year": 1.0}
    kwargs[name] = jnp.zeros((3,))
    with pytest.raises(ValueError, match=f"{name} must be a scalar"):
        encoder(**kwargs)


def test_non_scalar_error_points_at_vmap():
    with pytest.raises(ValueError, match=r"jax\.vmap"):
        GeoContextEncoder()(lat=jnp.zeros((2,)), lon=10.0, day_of_year=1.0)


def test_multi_dimensional_extra_value_is_rejected():
    encoder = GeoContextEncoder()
    with pytest.raises(ValueError, match=r"extra\['grid'\] must be a scalar"):
        _call(encoder, 45.0, 10.0, 1.0, {"grid": jnp.zeros((2, 2))})


def test_output_dim_rejects_negative_extra_dim():
    with pytest.raises(ValueError, match="extra_dim must be non-negative"):
        GeoContextEncoder().output_dim(extra_dim=-1)


def test_output_dim_rejects_extra_dim_when_extra_disabled():
    with pytest.raises(ValueError, match="extra_dim must be 0 when"):
        GeoContextEncoder(include_extra=False).output_dim(extra_dim=2)
