import cdshealpix
import numpy as np
import pytest

from healpix_geo import nested


@pytest.mark.parametrize("depth", [2, 8])
@pytest.mark.parametrize("ring", [0, 1])
@pytest.mark.parametrize(
    "indexing_scheme",
    [
        "nested",
        pytest.param("ring", marks=pytest.mark.skip(reason="not implemented yet")),
    ],
)
def test_neighbours_in_kth_ring(depth, ring, indexing_scheme):
    if indexing_scheme == "nested":
        neighbours_in_kth_ring = nested.neighbours_in_kth_ring
        neighbours = cdshealpix.nested.neighbours

    ipixels = np.array([50, 100], dtype="int64")

    actual = neighbours_in_kth_ring(depth=depth, ipix=ipixels, ring=ring)
    if ring == 0:
        expected = np.reshape(ipixels, (-1, 1))
    else:
        expected = neighbours(ipix=ipixels, depth=depth)

    np.testing.assert_equal(actual, expected)
