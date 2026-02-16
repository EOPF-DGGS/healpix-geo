import numpy as np
import pytest
from pyproj import Geod

import healpix_geo


@pytest.mark.parametrize("indexing_scheme", ["nested", "ring"])
@pytest.mark.parametrize(
    ["depth", "from_", "to_"],
    (
        # 1D to_ — same shape as from_
        (
            1,
            np.array([0, 16, 25, 32, 46]),
            np.array([2, 15, 27, 40, 41], dtype="int64"),
        ),
        # 2D to_ — one destination per source (column vector)
        (
            1,
            np.array([0, 16, 25, 32, 46]),
            np.array([[2], [15], [27], [40], [41]], dtype="int64"),
        ),
        # 2D to_ with -1 sentinel (missing destinations → np.nan)
        (
            2,
            np.array([0, 16, 25, 32, 46]),
            np.array([[2, 4], [15, 7], [27, 26], [40, -1], [-1, 41]], dtype="int64"),
        ),
    ),
)
def test_geodesic_distance_shape_and_nan(indexing_scheme, depth, from_, to_):
    """Output shape matches to_, and -1 entries produce np.nan."""
    fn = getattr(healpix_geo, indexing_scheme).geodesic_distance
    result = fn(from_, to_, depth, ellipsoid="WGS84")

    assert result.shape == to_.shape, (
        f"[{indexing_scheme}] Expected shape {to_.shape}, got {result.shape}"
    )
    nan_mask = to_ == -1
    assert np.all(np.isnan(result[nan_mask])), (
        f"[{indexing_scheme}] Expected np.nan for -1 destinations"
    )
    valid_mask = ~nan_mask
    assert np.all(result[valid_mask] > 0), (
        f"[{indexing_scheme}] Geodesic distances must be positive"
    )
    assert np.all(np.isfinite(result[valid_mask])), (
        f"[{indexing_scheme}] Geodesic distances must be finite"
    )


@pytest.mark.parametrize("indexing_scheme", ["nested", "ring"])
def test_geodesic_distance_values(indexing_scheme):
    """Values match a direct pyproj computation on the cell centres."""
    module = getattr(healpix_geo, indexing_scheme)
    depth  = 1
    ellipsoid = "WGS84"
    from_  = np.array([0, 16, 25], dtype="uint64")
    to_    = np.array([2, 15, 27], dtype="int64")

    result = module.geodesic_distance(from_, to_, depth, ellipsoid=ellipsoid)

    # Reference: compute directly with pyproj on the cell centres
    lon1, lat1 = module.healpix_to_lonlat(from_, depth, ellipsoid=ellipsoid)
    lon2, lat2 = module.healpix_to_lonlat(to_.astype("uint64"), depth, ellipsoid=ellipsoid)
    _, _, expected = Geod(ellps=ellipsoid).inv(lon1, lat1, lon2, lat2)

    np.testing.assert_allclose(result, expected, rtol=1e-6)


@pytest.mark.parametrize("indexing_scheme", ["nested", "ring"])
def test_geodesic_distance_symmetry(indexing_scheme):
    """geodesic_distance(A→B) == geodesic_distance(B→A)."""
    fn = getattr(healpix_geo, indexing_scheme).geodesic_distance
    depth = 2
    from_ = np.array([0, 5, 10], dtype="uint64")
    to_   = np.array([3, 8, 20], dtype="int64")

    d_ab = fn(from_, to_,            depth, ellipsoid="WGS84")
    d_ba = fn(to_.astype("uint64"), from_.astype("int64"), depth, ellipsoid="WGS84")

    np.testing.assert_allclose(d_ab, d_ba, rtol=1e-6)


@pytest.mark.parametrize("indexing_scheme", ["nested", "ring"])
def test_geodesic_distance_same_cell_is_zero(indexing_scheme):
    """Distance from a cell to itself is 0."""
    fn = getattr(healpix_geo, indexing_scheme).geodesic_distance
    cells = np.array([0, 10, 47], dtype="uint64")
    result = fn(cells, cells.astype("int64"), depth=2, ellipsoid="WGS84")
    np.testing.assert_allclose(result, 0.0, atol=1e-3)


@pytest.mark.parametrize("indexing_scheme", ["nested", "ring"])
def test_geodesic_distance_error_shape(indexing_scheme):
    """Incompatible from_/to_ shapes raise ValueError."""
    fn = getattr(healpix_geo, indexing_scheme).geodesic_distance
    from_ = np.array([4, 7])
    to_   = np.array([[2, 3], [4, 6], [5, 4]], dtype="int64")

    with pytest.raises(ValueError, match="The shape of `from_` must be compatible"):
        fn(from_, to_, depth=1, ellipsoid="WGS84")


@pytest.mark.parametrize("indexing_scheme", ["nested", "ring"])
def test_geodesic_distance_error_depth(indexing_scheme):
    """Out-of-range depth raises ValueError."""
    fn = getattr(healpix_geo, indexing_scheme).geodesic_distance
    with pytest.raises(ValueError):
        fn(np.array([0]), np.array([1], dtype="int64"), depth=30, ellipsoid="WGS84")
