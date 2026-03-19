from dataclasses import dataclass
from types import ModuleType
from typing import Literal

import numpy as np
import numpy.typing as npt

from healpix_geo.typing import EllipsoidLike


def _dispatch_module(indexing_scheme: str) -> ModuleType:
    from healpix_geo import nested, ring, zuniq

    modules = {
        "nested": nested,
        "ring": ring,
        "zuniq": zuniq,
    }

    module = modules.get(indexing_scheme)
    if module is None:
        raise ValueError(
            f"unknown indexing scheme: {indexing_scheme}."
            f" Available are: {', '.join(modules.keys())}"
        )

    return module


@dataclass(frozen=True)
class Grid:
    level: int | None
    """The refinement level of the grid."""

    indexing_scheme: Literal["nested", "ring", "zuniq"] = "nested"
    """The indexing scheme of the grid."""

    ellipsoid: EllipsoidLike = "sphere"
    """The reference ellipsoid of the grid."""

    def _as_params(self):
        params = {"ellipsoid": self.ellipsoid}
        if self.indexing_scheme != "zuniq":
            params["depth"] = self.level

        return params


def healpix_to_lonlat(
    ipix: npt.NDArray[np.uint64], grid: Grid, *, num_threads: int = 0
) -> (npt.NDArray[np.float64], npt.NDArray[np.float64]):
    r"""Get the longitudes and latitudes of the center of some HEALPix cells.

    Parameters
    ----------
    ipix : `numpy.ndarray`
        The HEALPix cell indexes given as a `np.uint64` numpy array.
    grid : Grid
        The definition of the HEALPix grid.
    num_threads : int, optional
        Specifies the number of threads to use for the computation. Default to 0 means
        it will choose the number of threads based on the RAYON_NUM_THREADS environment variable (if set),
        or the number of logical CPUs (otherwise)

    Returns
    -------
    lon, lat : array-like
        The coordinates of the center of the HEALPix cells given as a longitude, latitude tuple.

    Examples
    --------
    >>> import healpix_geo.auto as hg
    >>> import numpy as np
    >>> ipix = np.array([42, 6, 10])
    >>> grid = hg.Grid(level=3, indexing_scheme="nested", ellipsoid="WGS84")
    >>> lon, lat = hg.healpix_to_lonlat(ipix, grid)
    >>> lon
    array([ 5.625, 50.625, 28.125])
    >>> lat
    array([41.93785391, 19.55202227, 19.55202227])
    """
    module = _dispatch_module(grid.indexing_scheme)
    params = grid._as_params()
    return module.healpix_to_lonlat(ipix, num_threads=num_threads, **params)


def lonlat_to_healpix(
    lon: npt.NDArray[np.float64],
    lat: npt.NDArray[np.float64],
    grid: Grid,
    *,
    num_threads: int = 0,
) -> npt.NDArray[np.uint64]:
    r"""Get the HEALPix indexes that contains specific points.

    Parameters
    ----------
    lon : array-like
        The longitudes of the input points, in degrees.
    lat : array-like
        The latitudes of the input points, in degrees.
    grid : Grid
        The definition of the HEALPix grid.
    num_threads : int, optional
        Specifies the number of threads to use for the computation. Default to 0 means
        it will choose the number of threads based on the RAYON_NUM_THREADS environment variable (if set),
        or the number of logical CPUs (otherwise)

    Returns
    -------
    ipix : `numpy.ndarray`
        A numpy array containing all the HEALPix cell indexes stored as `np.uint64`.

    Examples
    --------
    >>> import healpix_geo.auto as hg
    >>> import numpy as np
    >>> lon = np.array([0, 50, 25], dtype="float64")
    >>> lat = np.array([6, -12, 45], dtype="float64")
    >>> grid = hg.Grid(level=3, indexing_scheme="nested", ellipsoid="WGS84")
    >>> ipix = hg.lonlat_to_healpix(lon, lat, grid)
    >>> ipix
    array([304, 573,  38], dtype=uint64)
    """
    module = _dispatch_module(grid.indexing_scheme)
    params = {"depth": grid.level, "ellipsoid": grid.ellipsoid}

    return module.lonlat_to_healpix(lon, lat, num_threads=num_threads, **params)


def vertices(
    ipix: npt.NDArray[np.uint64], grid: Grid, *, num_threads: int = 0
) -> (npt.NDArray[np.float64], npt.NDArray[np.float64]):
    """Get the longitudes and latitudes of the vertices of some HEALPix cells.

    This method returns the 4 vertices of each cell in `ipix`.

    Parameters
    ----------
    ipix : `numpy.ndarray`
        The HEALPix cell indexes given as a `np.uint64` numpy array.
    grid : Grid
        The definition of the HEALPix grid.
    num_threads : int, optional
        Specifies the number of threads to use for the computation. Default to 0 means
        it will choose the number of threads based on the RAYON_NUM_THREADS environment variable (if set),
        or the number of logical CPUs (otherwise)

    Returns
    -------
    longitude, latitude : array-like
        The coordinates of the 4 vertices of the HEALPix cells.
        `lon` and `lat` are of shape :math:`N` x :math:`4` numpy arrays where N is the number of HEALPix cell given in `ipix`.

    Examples
    --------
    >>> import healpix_geo.auto as hg
    >>> import numpy as np
    >>> ipix = np.array([42, 6, 10])
    >>> grid = hg.Grid(level=12, indexing_scheme="nested", ellipsoid="sphere")
    >>> grid
    Grid(level=12, indexing_scheme='nested', ellipsoid='sphere')
    >>> lon, lat = hg.vertices(ipix, grid)
    >>> np.stack([lon, lat], axis=-1)
    array([[[4.49230957e+01, 6.52784088e-02],
            [4.49340820e+01, 7.46039007e-02],
            [4.49230957e+01, 8.39293945e-02],
            [4.49121094e+01, 7.46039007e-02]],
    <BLANKLINE>
           [[4.50109863e+01, 2.79764560e-02],
            [4.50219727e+01, 3.73019424e-02],
            [4.50109863e+01, 4.66274299e-02],
            [4.50000000e+01, 3.73019424e-02]],
    <BLANKLINE>
           [[4.49670410e+01, 2.79764560e-02],
            [4.49780273e+01, 3.73019424e-02],
            [4.49670410e+01, 4.66274299e-02],
            [4.49560547e+01, 3.73019424e-02]]])
    """
    module = _dispatch_module(grid.indexing_scheme)
    params = grid._as_params()

    return module.vertices(ipix, num_threads=num_threads, **params)


def kth_neighbourhood(
    ipix: npt.NDArray[np.uint64], grid: Grid, *, ring: int, num_threads: int = 0
) -> npt.NDArray[np.int64]:
    """Get the kth ring neighbouring cells of some HEALPix cells.

    This method returns a :math:`N` x :math:`(2 k + 1)^2` `np.uint64` numpy array containing the neighbours of each cell of the :math:`N` sized `ipix` array.
    This method is wrapped around the `kth_neighbourhood <https://docs.rs/cdshealpix/0.1.5/cdshealpix/nested/struct.Layer.html#method.kth_neighbourhood>`__
    method from the `cdshealpix Rust crate <https://crates.io/crates/cdshealpix>`__.

    Parameters
    ----------
    ipix : `numpy.ndarray`
        The HEALPix cell indexes given as a `np.uint64` numpy array.
    grid : Grid
        The definition of the HEALPix grid.
    ring : int
        The number of rings. `ring=0` returns just the input cell ids, `ring=1` returns the 8 (or 7) immediate
        neighbours, `ring=2` returns the 8 (or 7) immediate neighbours plus their immediate neighbours (a total of 24 cells), and so on.
    num_threads : int, default: 0
        Specifies the number of threads to use for the computation. Default to 0 means
        it will choose the number of threads based on the RAYON_NUM_THREADS environment variable (if set),
        or the number of logical CPUs (otherwise)

    Returns
    -------
    neighbours : `numpy.ndarray`
        A :math:`N` x :math:`(2 k + 1)^2` `np.int64` numpy array containing the kth ring neighbours of each cell.
        The :math:`5^{th}` element corresponds to the index of HEALPix cell from which the neighbours are evaluated.
        All its 8 neighbours occup the remaining elements of the line.

    Examples
    --------
    >>> import healpix_geo.auto as hg
    >>> import numpy as np
    >>> ipix = np.array([42, 6, 10])
    >>> grid = hg.Grid(level=12, indexing_scheme="nested", ellipsoid="sphere")
    >>> grid
    Grid(level=12, indexing_scheme='nested', ellipsoid='sphere')
    >>> ring = 3
    >>> neighbours = hg.kth_neighbourhood(ipix, grid, ring=ring)
    >>> neighbours
    array([[       42,  72701309,  72701311,  72701397,       128,       129,
                   43,        41,        40,  72701302,  72701308,  72701310,
             72701396,  72701398,  72701399,  72701303,       130,       131,
                  134,       132,        46,        44,        38,        35,
                   34,  72701297,  72701299,  72701305,  72701307,  72701393,
             72701395,  72701401,  72701404,  72701405,  72701301,  72701300,
                  136,       137,       140,       141,       135,       133,
                   47,        45,        39,        37,        36,        33,
                   32],
           [        6,         1,         3,         9,        12,        13,
                    7,         5,         4,  95070890,  95070906,  95070895,
             95070894,  95070891,         0,         2,         8,        10,
                   11,        14,        15,        26,        24,        18,
                   16, 150994941, 150994943,  72701269,  72701271,  72701277,
             72701279,  72701301,  95070907,  95070905,  95070904,  95070893,
             95070892,  95070889,  95070888,        32,        33,        36,
                   37,        48,        49,        27,        25,        19,
                   17],
           [       10,  72701277,  72701279,  72701301,        32,        33,
                   11,         9,         8,  72701270,  72701276,  72701278,
             72701300,  72701302,  72701303,  72701271,        34,        35,
                   38,        36,        14,        12,         6,         3,
                    2,  72701265,  72701267,  72701273,  72701275,  72701297,
             72701299,  72701305,  72701308,  72701309,  72701269,  72701268,
                   40,        41,        44,        45,        39,        37,
                   15,        13,         7,         5,         4,         1,
                    0]])
    """
    module = _dispatch_module(grid.indexing_scheme)
    params = {}
    if grid.indexing_scheme != "zuniq":
        params["depth"] = grid.level

    return module.kth_neighbourhood(ipix, ring=ring, num_threads=num_threads, **params)
