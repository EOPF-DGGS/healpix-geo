import numpy as np
import pytest

import healpix_geo.nested as healpix


@pytest.mark.parametrize(
    ["flat", "expected_ids", "expected_depths", "expected_coverage"],
    (
        pytest.param(
            False,
            np.array(
                [
                    0,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    69,
                    70,
                    71,
                    76,
                    77,
                    79,
                    89,
                    90,
                    91,
                    92,
                    94,
                    95,
                ],
                dtype="uint64",
            ),
            np.array(
                [
                    1,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                ],
                dtype="u8",
            ),
            np.array(
                [
                    True,
                    True,
                    False,
                    True,
                    False,
                    True,
                    True,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                ]
            ),
        ),
        pytest.param(
            True,
            np.array(
                [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    69,
                    70,
                    71,
                    76,
                    77,
                    79,
                    89,
                    90,
                    91,
                    92,
                    94,
                    95,
                ],
                dtype="uint64",
            ),
            np.array(
                [
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                ],
                dtype="u8",
            ),
            np.array(
                [
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                    True,
                    False,
                    True,
                    True,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                ]
            ),
            id="flat",
        ),
    ),
)
def test_zone_coverage(flat, expected_ids, expected_depths, expected_coverage):
    zone = (0.0, 0.0, 90.0, 90.0)
    depth = 2

    cell_ids, depths, fully_covered = healpix.zone_coverage(
        zone, depth, ellipsoid="sphere", flat=flat
    )

    np.testing.assert_equal(cell_ids, expected_ids)
    np.testing.assert_equal(depths, expected_depths)
    np.testing.assert_equal(fully_covered, expected_coverage)


def test_box_coverage():
    center = (45.0, 45.0)
    size = (10.0, 10.0)
    angle = 0.0

    depth = 1

    cell_ids, depths, fully_covered = healpix.box_coverage(
        center, size, angle, depth, ellipsoid="WGS84"
    )

    expected_cell_ids = np.array([0, 1, 2, 3], dtype="uint64")
    expected_depths = np.array([1, 1, 1, 1], dtype="uint8")
    expected_coverage = np.array([False, False, False, False])

    np.testing.assert_equal(cell_ids, expected_cell_ids)
    np.testing.assert_equal(depths, expected_depths)
    np.testing.assert_equal(fully_covered, expected_coverage)


def test_polygon_coverage():
    vertices = np.array(
        [[40.0, 40.0], [50.0, 40.0], [50.0, 50.0], [40.0, 50.0]], dtype="float64"
    )

    depth = 1

    cell_ids, depths, fully_covered = healpix.polygon_coverage(
        vertices, depth, ellipsoid="WGS84"
    )

    expected_cell_ids = np.array([0, 1, 2, 3], dtype="uint64")
    expected_depths = np.array([1, 1, 1, 1], dtype="uint8")
    expected_coverage = np.array([False, False, False, False])

    np.testing.assert_equal(cell_ids, expected_cell_ids)
    np.testing.assert_equal(depths, expected_depths)
    np.testing.assert_equal(fully_covered, expected_coverage)


def test_cone_coverage():
    center = (45.0, 45.0)
    radius = 5.0
    depth = 1

    cell_ids, depths, fully_covered = healpix.cone_coverage(
        center, radius, depth, ellipsoid="WGS84"
    )

    expected_cell_ids = np.array([0, 1, 2, 3], dtype="uint64")
    expected_depths = np.array([1, 1, 1, 1], dtype="uint8")
    expected_coverage = np.array([False, False, False, False])

    np.testing.assert_equal(cell_ids, expected_cell_ids)
    np.testing.assert_equal(depths, expected_depths)
    np.testing.assert_equal(fully_covered, expected_coverage)


def test_elliptical_cone_coverage():
    center = (45.0, 45.0)
    ellipse_geometry = (10.0, 8.0)
    positional_angle = 30.0
    depth = 1

    cell_ids, depths, fully_covered = healpix.elliptical_cone_coverage(
        center, ellipse_geometry, positional_angle, depth, ellipsoid="WGS84"
    )

    expected_cell_ids = np.array([0, 1, 2, 3], dtype="uint64")
    expected_depths = np.array([1, 1, 1, 1], dtype="uint8")
    expected_coverage = np.array([False, False, False, False])

    np.testing.assert_equal(cell_ids, expected_cell_ids)
    np.testing.assert_equal(depths, expected_depths)
    np.testing.assert_equal(fully_covered, expected_coverage)
