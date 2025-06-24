import numpy as np

from healpix_geo import healpix_geo


def latitude_authalic_to_geographic(authalic_lat, ellipsoid, num_threads=0):
    r"""Converts authalic latitudes to geographic.

    Parameters
    ----------
    authalic_lat : `numpy.ndarray`
        The authalic latitudes, in radians.
    ellipsoid : str, default: "sphere"
        Reference ellipsoid to evaluate healpix on. If ``"sphere"``, this will return
        the same result as :py:func:`cdshealpix.nested.healpix_to_lonlat`.
    num_threads : int, optional
        Specifies the number of threads to use for the computation. Default to 0 means
        it will choose the number of threads based on the RAYON_NUM_THREADS environment variable (if set),
        or the number of logical CPUs (otherwise)

    Returns
    -------
    geographic_lat : array-like
        The corresponding geographic latitudes, in degrees.

    Raises
    ------
    ValueError
        When the name of the ellipsoid is unknown.

    Examples
    --------
    >>> from healpix_geo.coordinates import latitude_authalic_to_geographic
    >>> import numpy as np
    >>> authalic_lat = np.linspace(-np.pi / 2, np.pi / 2, 5)
    >>> geographic_lat = latitude_authalic_to_geographic(authalic_lat, "WGS84")
    >>> geographic_lat
    array([-90.        , -45.12829693,   0.        ,  45.12829693,   90.        ])
    >>> geographic_lat - np.rad2deg(authalic_lat)
    array([ 0.        , -0.12829693,  0.        ,  0.12829693,  0.        ])
    >>> np.allclose(
    ...     latitude_authalic_to_geographic(authalic_lat, "sphere"),
    ...     np.rad2deg(authalic_lat),
    ... )
    True
    """
    num_threads = np.uint16(num_threads)

    geographic_lat = np.empty_like(authalic_lat, dtype="float64")

    healpix_geo.coordinates.latitude_authalic_to_geographic(
        ellipsoid, authalic_lat, geographic_lat, num_threads
    )

    return geographic_lat


def latitude_geographic_to_authalic(geographic_lat, ellipsoid, num_threads=0):
    r"""Converts geographic latitudes to authalic.

    Parameters
    ----------
    geographic_lat : `numpy.ndarray`
        The geographic latitudes, in degrees.
    ellipsoid : str, default: "sphere"
        Reference ellipsoid to evaluate healpix on. If ``"sphere"``, this will return
        the same result as :py:func:`cdshealpix.nested.healpix_to_lonlat`.
    num_threads : int, optional
        Specifies the number of threads to use for the computation. Default to 0 means
        it will choose the number of threads based on the RAYON_NUM_THREADS environment variable (if set),
        or the number of logical CPUs (otherwise)

    Returns
    -------
    authalic_lat : array-like
        The corresponding authalic latitudes, in radians.

    Raises
    ------
    ValueError
        When the name of the ellipsoid is unknown.

    Examples
    --------
    >>> from healpix_geo.coordinates import latitude_geographic_to_authalic
    >>> import numpy as np
    >>> geographic_lat = np.linspace(-90, 90, 5)
    >>> authalic_lat = latitude_geographic_to_authalic(geographic_lat, "WGS84")
    >>> authalic_lat
    array([-1.57079633, -0.78315896,  0.        ,  0.78315896,  1.57079633])
    >>> geographic_lat - np.rad2deg(authalic_lat)
    array([ 0.        , -0.12829713,  0.        ,  0.12829713,  0.        ])
    >>> np.allclose(
    ...     latitude_geographic_to_authalic(geographic_lat, "sphere"),
    ...     np.deg2rad(geographic_lat),
    ... )
    True
    """
    num_threads = np.uint16(num_threads)

    authalic_lat = np.empty_like(geographic_lat, dtype="float64")

    healpix_geo.coordinates.latitude_geographic_to_authalic(
        ellipsoid, geographic_lat, authalic_lat, num_threads
    )

    return authalic_lat
