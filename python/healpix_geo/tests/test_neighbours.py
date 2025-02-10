import cdshealpix
import numpy as np
import pytest

from healpix_geo import neighbours_in_kth_ring


@pytest.mark.parametrize("depth", [2, 8])
@pytest.mark.parametrize("ring", [0, 1])
def test_neighbours_in_kth_ring(depth, ring):
    ipixels = np.array([50, 100], dtype="int64")

    actual = neighbours_in_kth_ring(depth=depth, ipix=ipixels, ring=ring)
    if ring == 0:
        expected = np.reshape(ipixels, (-1, 1))
    else:
        expected = cdshealpix.nested.neighbours(ipix=ipixels, depth=depth)

    np.testing.assert_equal(actual, expected)
