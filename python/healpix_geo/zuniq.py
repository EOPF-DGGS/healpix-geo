import numpy as np

from healpix_geo import healpix_geo
from healpix_geo.utils import _check_depth, _check_ipixels


def from_nested(ipix, depth, num_threads=0):
    """Convert from nested to zuniq

    Parameters
    ----------
    ipix : `numpy.ndarray`
        The HEALPix cell indexes in the nested scheme given as a `np.uint64` numpy array.
    depth : int or array-like of int
        The HEALPix cell depth given as scalar or a `np.uint8` numpy array.
    num_threads : int, optional
        Specifies the number of threads to use for the computation. Default to 0 means
        it will choose the number of threads based on the RAYON_NUM_THREADS environment variable (if set),
        or the number of logical CPUs (otherwise)

    Returns
    -------
    zuniq : array-like of int
        The cell ids in the zuniq scheme.
    """
    _check_depth(depth)

    ipix = np.atleast_1d(ipix)
    _check_ipixels(data=ipix, depth=depth)
    ipix = ipix.astype(np.uint64)

    depth = depth if isinstance(depth, int) else depth.astype("uint8")
    num_threads = np.uint16(num_threads)

    return healpix_geo.zuniq.from_nested(ipix, depth, num_threads)


def to_nested(ipix, num_threads=0):
    """Convert from zuniq to nested

    Parameters
    ----------
    ipix : `numpy.ndarray`
        The HEALPix cell indexes in the zuniq scheme given as a `np.uint64` numpy array.
    num_threads : int, optional
        Specifies the number of threads to use for the computation. Default to 0 means
        it will choose the number of threads based on the RAYON_NUM_THREADS environment variable (if set),
        or the number of logical CPUs (otherwise)

    Returns
    -------
    nested : array-like of int
        The cell ids in the nested scheme.
    depth : int or array-like of int
        The HEALPix cell depth given as scalar or a `np.uint8` numpy array.
    """
    ipix = np.atleast_1d(ipix).astype(np.uint64)

    num_threads = np.uint16(num_threads)

    return healpix_geo.zuniq.to_nested(ipix, num_threads)


def healpix_to_lonlat(ipix, ellipsoid, num_threads=0):
    r"""Get the longitudes and latitudes of the center of some HEALPix cells.

    Parameters
    ----------
    ipix : `numpy.ndarray`
        The HEALPix cell indexes given as a `np.uint64` numpy array.
    ellipsoid : ellipsoid-like, default: "sphere"
        Reference ellipsoid to evaluate healpix on. If the reference ellipsoid
        is spherical, this will return the same result as
        :py:func:`cdshealpix.nested.healpix_to_lonlat`.
    num_threads : int, optional
        Specifies the number of threads to use for the computation. Default to 0 means
        it will choose the number of threads based on the RAYON_NUM_THREADS environment variable (if set),
        or the number of logical CPUs (otherwise)

    Returns
    -------
    lon, lat : array-like
        The coordinates of the center of the HEALPix cells given as a longitude, latitude tuple.

    Raises
    ------
    ValueError
        When the HEALPix cell indexes given have values out of :math:`[0, 4^{29 - depth}[`.
    ValueError
        When the name of the ellipsoid is unknown.

    Examples
    --------
    >>> from healpix_geo.zuniq import healpix_to_lonlat
    >>> import numpy as np
    >>> cell_ids = np.array([42, 6, 10])
    >>> lon, lat = healpix_to_lonlat(ipix, ellipsoid="WGS84")
    """
    ipix = np.atleast_1d(ipix).astype(np.uint64)

    num_threads = np.uint16(num_threads)

    latitude = np.empty_like(ipix, dtype="float64")
    longitude = np.empty_like(ipix, dtype="float64")

    healpix_geo.zuniq.healpix_to_lonlat(
        ipix, ellipsoid, longitude, latitude, num_threads
    )

    return longitude, latitude


def lonlat_to_healpix(longitude, latitude, depth, ellipsoid="sphere", num_threads=0):
    r"""Get the HEALPix indexes that contains specific points.

    Parameters
    ----------
    lon : array-like
        The longitudes of the input points, in degrees.
    lat : array-like
        The latitudes of the input points, in degrees.
    depth : int or array-like of int
        The HEALPix cell depth given as a `np.uint8` numpy array.
    ellipsoid : ellipsoid-like, default: "sphere"
        Reference ellipsoid to evaluate healpix on. If the reference ellipsoid
        is spherical, this will return the same result as
        :py:func:`cdshealpix.nested.lonlat_to_healpix`.
    num_threads : int, optional
        Specifies the number of threads to use for the computation. Default to 0 means
        it will choose the number of threads based on the RAYON_NUM_THREADS environment variable (if set),
        or the number of logical CPUs (otherwise)

    Returns
    -------
    ipix : `numpy.ndarray`
        A numpy array containing all the HEALPix cell indexes stored as `np.uint64`.

    Raises
    ------
    ValueError
        When the number of longitudes and latitudes given do not match.
    ValueError
        When the name of the ellipsoid is unknown.

    Examples
    --------
    >>> from cdshealpix.ring import lonlat_to_healpix
    >>> import numpy as np
    >>> lon = np.array([0, 50, 25], dtype="float64")
    >>> lat = np.array([6, -12, 45], dtype="float64")
    >>> depth = 3
    >>> ipix = lonlat_to_healpix(lon, lat, depth, ellipsoid="WGS84")
    """
    _check_depth(depth)
    longitude = np.atleast_1d(longitude).astype("float64")
    latitude = np.atleast_1d(latitude).astype("float64")

    num_threads = np.uint16(num_threads)

    ipix = np.empty_like(longitude, dtype="uint64")

    healpix_geo.zuniq.lonlat_to_healpix(
        depth, longitude, latitude, ellipsoid, ipix, num_threads
    )

    return ipix


def vertices(ipix, ellipsoid, num_threads=0):
    """Get the longitudes and latitudes of the vertices of some HEALPix cells in zuniq encoding.

    This method returns the 4 vertices of each cell in `ipix`.

    Parameters
    ----------
    ipix : `numpy.ndarray`
        The HEALPix cell indexes given as a `np.uint64` numpy array.
    ellipsoid : ellipsoid-like, default: "sphere"
        Reference ellipsoid to evaluate healpix on. If the reference ellipsoid
        is spherical, this will return the same result as
        :py:func:`cdshealpix.nested.vertices`.
    num_threads : int, optional
        Specifies the number of threads to use for the computation. Default to 0 means
        it will choose the number of threads based on the RAYON_NUM_THREADS environment variable (if set),
        or the number of logical CPUs (otherwise)

    Returns
    -------
    longitude, latitude : array-like
        The sky coordinates of the 4 vertices of the HEALPix cells.
        `lon` and `lat` are of shape :math:`N` x :math:`4` numpy arrays where N is the number of HEALPix cell given in `ipix`.

    Raises
    ------
    ValueError
        When the HEALPix cell indexes given have values out of :math:`[0, 4^{29 - depth}[`.

    Examples
    --------
    >>> from healpix_geo.nested import vertices
    >>> import numpy as np
    >>> ipix = np.array([42, 6, 10])
    >>> depth = 12
    >>> lon, lat = vertices(ipix, depth, ellipsoid="sphere")
    """
    ipix = np.atleast_1d(ipix).astype(np.uint64)

    num_threads = np.uint16(num_threads)

    shape = ipix.shape + (4,)
    longitude = np.empty(shape=shape, dtype="float64")
    latitude = np.empty(shape=shape, dtype="float64")

    healpix_geo.zuniq.vertices(ipix, ellipsoid, longitude, latitude, num_threads)

    return longitude, latitude
