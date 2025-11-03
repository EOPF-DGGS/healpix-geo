import numpy as np
import pytest

import healpix_geo.nested


@pytest.mark.parametrize(
    ["lon", "lat", "depth", "ellipsoid", "expected"],
    (
        (
            *np.mgrid[-120 : 120 + 1 : 120, -45 : 45 + 1 : 45],
            2,
            "sphere",
            np.array([], dtype="uint64"),
        ),
    ),
)
def test_interpolation_quadrilaterals(lon, lat, depth, ellipsoid, expected):
    actual = healpix_geo.nested.interpolation_quadrilateral(lon, lat, depth, ellipsoid)

    np.testing.assert_equal(actual, expected)
