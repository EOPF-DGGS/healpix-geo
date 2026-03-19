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
    ["grid", "expected"],
    (
        (
            auto.Grid(level=5, indexing_scheme="nested", ellipsoid="sphere"),
            np.array([12, 84, 104, 72], dtype="uint64"),
        ),
        (
            auto.Grid(level=3, indexing_scheme="ring", ellipsoid="WGS84"),
            np.array([340, 245, 244, 277], dtype="uint64"),
        ),
        (
            auto.Grid(level=6, indexing_scheme="zuniq", ellipsoid="WGS84"),
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
def test_lonlat_to_healpix(grid, expected):
    lon = np.array([45.0, 64.6875, 47.8125, 53.4375], dtype="float64")
    lat = np.array([5.9791568, 18.20995686, 18.20995686, 13.24801491], dtype="float64")

    actual = auto.lonlat_to_healpix(lon, lat, grid)

    np.testing.assert_equal(actual, expected)


@pytest.mark.parametrize(
    ["grid", "cell_ids", "expected_lon", "expected_lat"],
    (
        (
            auto.Grid(level=5, indexing_scheme="nested", ellipsoid="sphere"),
            np.array([12, 104], dtype="uint64"),
            np.array([45.0, 47.8125], dtype="float64"),
            np.array([5.9791568, 18.20995686], dtype="float64"),
        ),
        (
            auto.Grid(level=3, indexing_scheme="ring", ellipsoid="sphere"),
            np.array([340, 245, 244], dtype="uint64"),
            np.array([45.0, 61.875, 50.625]),
            np.array([4.78019185, 19.47122063, 19.47122063]),
        ),
        (
            auto.Grid(level=6, indexing_scheme="zuniq", ellipsoid="WGS84"),
            np.array([6825768185233408], dtype="uint64"),
            np.array([45.0]),
            np.array([5.40338952]),
        ),
    ),
)
def test_healpix_to_lonlat(grid, cell_ids, expected_lon, expected_lat):
    actual_lon, actual_lat = auto.healpix_to_lonlat(cell_ids, grid)

    np.testing.assert_allclose(actual_lon, expected_lon)
    np.testing.assert_allclose(actual_lat, expected_lat)


@pytest.mark.parametrize(
    ["grid", "cell_ids", "expected_lon", "expected_lat"],
    (
        (
            auto.Grid(level=2, indexing_scheme="nested", ellipsoid="WGS84"),
            np.array([3, 54], dtype="uint64"),
            np.array([[45.0, 56.25, 45.0, 33.75], [326.25, 337.5, 330.0, 315.0]]),
            np.array(
                [
                    [19.55202227, 30.11125172, 41.93785391, 30.11125172],
                    [30.11125172, 41.93785391, 54.46234938, 41.93785391],
                ]
            ),
        ),
        (
            auto.Grid(level=3, indexing_scheme="ring", ellipsoid="WGS84"),
            np.array([19, 67, 94], dtype="uint64"),
            np.array(
                [
                    [225.0, 240.0, 225.0, 210.0],
                    [115.71428571, 120.0, 108.0, 105.0],
                    [135.0, 141.42857143, 135.0, 128.57142857],
                ],
                dtype="float64",
            ),
            np.array(
                [
                    [66.53737405, 72.46140572, 78.33504545, 72.46140572],
                    [48.26869833, 54.46234938, 60.54441647, 54.46234938],
                    [41.93785391, 48.26869833, 54.46234938, 48.26869833],
                ],
                dtype="float64",
            ),
        ),
        (
            auto.Grid(level=None, indexing_scheme="zuniq", ellipsoid="sphere"),
            np.array(
                [1963569437533536256, 824158731808800768, 5116089176692883456],
                dtype="uint64",
            ),
            np.array(
                [
                    [326.25, 337.5, 330.0, 315.0],
                    [146.25, 154.28571429, 150.0, 141.42857143],
                    [45.0, 67.5, 45.0, 22.5],
                ],
                dtype="float64",
            ),
            np.array(
                [
                    [30.0, 41.8103149, 54.3409123, 41.8103149],
                    [41.8103149, 48.14120779, 54.3409123, 48.14120779],
                    [-41.8103149, -19.47122063, 0.0, -19.47122063],
                ],
                dtype="float64",
            ),
        ),
    ),
    ids=["nested", "ring", "zuniq"],
)
def test_vertices(grid, cell_ids, expected_lon, expected_lat):
    actual_lon, actual_lat = auto.vertices(cell_ids, grid)

    np.testing.assert_allclose(actual_lon, expected_lon)
    np.testing.assert_allclose(actual_lat, expected_lat)
