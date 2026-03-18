import numpy as np
import pytest

import healpix_geo
from healpix_geo import auto


@pytest.mark.parametrize(
    ["scheme", "expected"],
    (
        ("nested", healpix_geo.nested),
        ("ring", healpix_geo.ring),
        ("zuniq", healpix_geo.zuniq),
    ),
    ids=["nested", "ring", "zuniq"],
)
def test_dispatch_module(scheme, expected):
    actual = auto._dispatch_module(scheme)

    assert actual is expected


@pytest.mark.parametrize(
    ["level", "indexing_scheme", "ellipsoid", "expected"],
    (
        (5, "nested", "sphere", np.array([12, 84, 104, 72], dtype="uint64")),
        (3, "ring", "WGS84", np.array([340, 245, 244, 277], dtype="uint64")),
        (
            6,
            "zuniq",
            "WGS84",
            np.array(
                [
                    6825768185233408,
                    47358164831567872,
                    58617163899994112,
                    40602765390512128,
                ],
                dtype="uint64",
            ),
        ),
    ),
)
def test_lonlat_to_healpix(level, indexing_scheme, ellipsoid, expected):
    lon = np.array([45.0, 64.6875, 47.8125, 53.4375], dtype="float64")
    lat = np.array([5.9791568, 18.20995686, 18.20995686, 13.24801491], dtype="float64")

    grid = auto.Grid(level=level, indexing_scheme=indexing_scheme, ellipsoid=ellipsoid)
    print(grid)

    actual = auto.lonlat_to_healpix(lon, lat, grid)

    np.testing.assert_equal(actual, expected)
