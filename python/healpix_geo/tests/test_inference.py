import cdshealpix
import numpy as np
import pytest
from astropy.coordinates import Latitude, Longitude

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
            cell_ids, depth, ellipsoid=ellipsoid
        )
        expected_lon_, expected_lat_ = cds_healpix_to_lonlat(cell_ids, param_cds)
        expected_lon = np.asarray(expected_lon_.to("degree"))
        expected_lat = np.asarray(expected_lat_.to("degree"))

        np.testing.assert_allclose(actual_lon, expected_lon)

        diff_lat = actual_lat - expected_lat
        assert np.all(abs(diff_lat) < 0.3)

        signs = np.array([-1, 1])
        actual = signs[(actual_lat >= 0).astype(int)]
        expected_ = np.sign(diff_lat)
        expected = np.where(expected_ == 0, 1, expected_)
        assert np.all(diff_lat == 0) or np.all(actual == expected)


class TestGeographicToHealpix:
    @pytest.mark.parametrize(
        ["lon", "lat", "depth", "indexing_scheme"],
        (
            pytest.param(
                np.array([-170.0, 10.0, 30.0, 124.0, 174.0]),
                np.array([-48.0, -30.0, -5.0, 15.0, 30.0]),
                0,
                "ring",
                id="level0-ring",
            ),
            pytest.param(
                np.array([-170.0, 10.0, 30.0, 124.0, 174.0]),
                np.array([-48.0, -30.0, -5.0, 15.0, 30.0]),
                0,
                "nested",
                id="level0-nested",
            ),
            pytest.param(
                np.array([-70.0, 135.0, 150.0]),
                np.array([-65.0, 0.0, 65.0]),
                4,
                "ring",
                id="level4-ring",
            ),
            pytest.param(
                np.array([-70.0, 135.0, 150.0]),
                np.array([-65.0, 0.0, 65.0]),
                4,
                "nested",
                id="level4-nested",
            ),
        ),
    )
    def test_spherical(self, lon, lat, depth, indexing_scheme):
        if indexing_scheme == "ring":
            param_cds = 2**depth
            hg_lonlat_to_healpix = healpix_geo.ring.lonlat_to_healpix
            cds_lonlat_to_healpix = cdshealpix.ring.lonlat_to_healpix
        else:
            param_cds = depth
            hg_lonlat_to_healpix = healpix_geo.nested.lonlat_to_healpix
            cds_lonlat_to_healpix = cdshealpix.nested.lonlat_to_healpix

        actual = hg_lonlat_to_healpix(lon, lat, depth, ellipsoid="sphere")
        lon_ = Longitude(lon, unit="degree")
        lat_ = Latitude(lat, unit="degree")
        expected = cds_lonlat_to_healpix(lon_, lat_, param_cds)

        np.testing.assert_equal(actual, expected)

    @pytest.mark.parametrize("ellipsoid", ["unitsphere", "sphere", "WGS84", "bessel"])
    @pytest.mark.parametrize("depth", [0, 1, 9])
    @pytest.mark.parametrize("indexing_scheme", ["ring", "nested"])
    def test_ellipsoidal(self, ellipsoid, depth, indexing_scheme):
        lat = np.linspace(-90, 90, 50)
        lon = np.full_like(lat, fill_value=45.0)

        if indexing_scheme == "ring":
            param_cds = 2**depth
            hg = healpix_geo.ring.lonlat_to_healpix
            cds = cdshealpix.ring.lonlat_to_healpix
        else:
            param_cds = depth
            hg = healpix_geo.nested.lonlat_to_healpix
            cds = cdshealpix.nested.lonlat_to_healpix

        actual = hg(lon, lat, depth, ellipsoid=ellipsoid)

        lon_ = Longitude(lon, unit="degree")
        lat_ = Latitude(lat, unit="degree")
        expected = cds(lon_, lat_, param_cds)

        assert actual.dtype == "uint64"
        assert expected.dtype == "uint64"

        # TODO: this is currently a smoke check, try more thorough checks


class TestVertices:
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
            hg_vertices = healpix_geo.ring.vertices
            cds_vertices = cdshealpix.ring.vertices
        else:
            param_cds = depth
            hg_vertices = healpix_geo.nested.vertices
            cds_vertices = cdshealpix.nested.vertices

        actual_lon, actual_lat = hg_vertices(cell_ids, depth, ellipsoid="sphere")
        expected_lon_, expected_lat_ = cds_vertices(cell_ids, param_cds)
        expected_lon = np.asarray(expected_lon_.to("degree"))
        expected_lat = np.asarray(expected_lat_.to("degree"))

        np.testing.assert_allclose(actual_lon, expected_lon)
        np.testing.assert_allclose(actual_lat, expected_lat)
