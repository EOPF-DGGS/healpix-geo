import numpy as np
import pytest

import healpix_geo


@pytest.mark.parametrize(
    ["depth", "new_depth", "indexing_scheme"],
    (
        pytest.param(1, 1, "nested", id="identity"),
        pytest.param(1, 0, "nested", id="parents-one step-base cells"),
        pytest.param(2, 0, "nested", id="parents-two step-base cells"),
        pytest.param(2, 1, "nested", id="parents-one step-normal"),
        pytest.param(1, 2, "nested", id="children-one step"),
        pytest.param(0, 2, "nested", id="children-two step-base cells"),
    ),
)
def test_zoom_to(depth, new_depth, indexing_scheme):
    cell_ids = np.arange(12 * 4**depth)
    if depth == new_depth:
        expected = cell_ids
    elif depth > new_depth:
        relative_depth = depth - new_depth
        expected = np.repeat(
            np.arange(12 * 4**new_depth),
            4**relative_depth,
        )
    elif depth < new_depth:
        expected = np.reshape(np.arange(12 * 4**new_depth), (cell_ids.size, -1))

    if indexing_scheme == "nested":
        zoom_to = healpix_geo.nested.zoom_to

    actual = zoom_to(cell_ids, depth, new_depth)

    np.testing.assert_equal(actual, expected)
