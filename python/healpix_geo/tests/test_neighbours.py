import cdshealpix.nested
import cdshealpix.ring
import numpy as np
import pytest

import healpix_geo


@pytest.mark.parametrize("depth", [2, 8])
@pytest.mark.parametrize("ring", [0, 1])
@pytest.mark.parametrize(
    "indexing_scheme",
    [
        "nested",
        "ring",
    ],
)
def test_kth_neighbourhood(depth, ring, indexing_scheme):
    if indexing_scheme == "nested":
        kth_neighbourhood = healpix_geo.nested.kth_neighbourhood
        neighbours = cdshealpix.nested.neighbours
    elif indexing_scheme == "ring":
        kth_neighbourhood = healpix_geo.ring.kth_neighbourhood

        def neighbours(ipix, depth):
            return cdshealpix.to_ring(
                cdshealpix.nested.neighbours(cdshealpix.from_ring(ipix, depth), depth),
                depth=depth,
            )

    ipixels = np.array([50, 100], dtype="int64")

    actual = kth_neighbourhood(depth=depth, ipix=ipixels, ring=ring)
    if ring == 0:
        expected = np.reshape(ipixels, (-1, 1))
    else:
        expected = neighbours(ipix=ipixels, depth=depth)

    np.testing.assert_equal(np.sort(actual, axis=-1), np.sort(expected, axis=-1))
    np.testing.assert_equal(actual[:, 0], ipixels)
