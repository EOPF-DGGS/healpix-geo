import cdshealpix
import numpy as np
import pytest

import healpix_geo


class TestHealpixToGeographic:
    @pytest.mark.parametrize(
        ["cell_ids", "depth", "indexing_scheme"],
        (
            pytest.param(np.array([0, 4, 5, 7, 9]), 0, "ring", id="level0-ring"),
            pytest.param(np.array([1, 2, 3, 8]), 0, "nested", id="level0-nested"),
            pytest.param(
                np.array([3, 19, 54, 63, 104, 127]), 4, "ring", id="level4-ring"
            ),
            pytest.param(
                np.array([22, 89, 134, 154, 190]), 4, "nested", id="level4-nested"
            ),
        ),
    )
    def test_spherical(self, cell_ids, depth, indexing_scheme):
        if indexing_scheme == "ring":
            param_cds = 2**depth
            hg_healpix_to_lonlat = healpix_geo.ring.healpix_to_lonlat
            cds_healpix_to_lonlat = cdshealpix.ring.healpix_to_lonlat
        else:
            param_cds = depth
            hg_healpix_to_lonlat = healpix_geo.nested.healpix_to_lonlat
            cds_healpix_to_lonlat = cdshealpix.nested.healpix_to_lonlat

        actual_lon, actual_lat = hg_healpix_to_lonlat(
            cell_ids, depth, ellipsoid="sphere"
        )
        expected_lon_, expected_lat_ = cds_healpix_to_lonlat(cell_ids, param_cds)
        expected_lon = np.asarray(expected_lon_.to("degree"))
        expected_lat = np.asarray(expected_lat_.to("degree"))

        np.testing.assert_allclose(actual_lon, expected_lon)
        np.testing.assert_allclose(actual_lat, expected_lat)

    @pytest.mark.parametrize("ellipsoid", ["unitsphere", "sphere", "WGS84", "bessel"])
    @pytest.mark.parametrize("depth", [0, 1, 9])
    @pytest.mark.parametrize("indexing_scheme", ["ring", "nested"])
    def test_ellipsoidal(self, depth, indexing_scheme, ellipsoid):
        cell_ids = np.arange(12)
        if indexing_scheme == "ring":
            param_cds = 2**depth
            hg_healpix_to_lonlat = healpix_geo.ring.healpix_to_lonlat
            cds_healpix_to_lonlat = cdshealpix.ring.healpix_to_lonlat
        else:
            param_cds = depth
            hg_healpix_to_lonlat = healpix_geo.nested.healpix_to_lonlat
            cds_healpix_to_lonlat = cdshealpix.nested.healpix_to_lonlat

        actual_lon, actual_lat = hg_healpix_to_lonlat(
            cell_ids, depth, ellipsoid="sphere"
        )
        expected_lon_, expected_lat_ = cds_healpix_to_lonlat(cell_ids, param_cds)
        expected_lon = np.asarray(expected_lon_.to("degree"))
        expected_lat = np.asarray(expected_lat_.to("degree"))

        np.testing.assert_allclose(actual_lon, expected_lon)

        diff_lat = actual_lat - expected_lat
        assert np.all(abs(diff_lat) < 0.01)

        signs = np.array([-1, 1])
        actual = signs[(actual_lat >= 0).astype(int)]
        expected_ = np.sign(diff_lat)
        expected = np.where(expected_ == 0, 1, expected_)
        assert np.all(diff_lat == 0) or np.all(actual == expected)
