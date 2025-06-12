import numpy as np

from healpix_geo import healpix_geo
from healpix_geo.utils import _check_depth, _check_ipixels, _check_ring

RangeMOCIndex = healpix_geo.nested.RangeMOCIndex


def create_empty(depth):
    return RangeMOCIndex.create_empty(depth)


def kth_neighbourhood(ipix, depth, ring, num_threads=0):
    """Get the kth ring neighbouring cells of some HEALPix cells at a given depth.

    This method returns a :math:`N` x :math:`(2 k + 1)^2` `np.uint64` numpy array containing the neighbours of each cell of the :math:`N` sized `ipix` array.
    This method is wrapped around the `kth_neighbourhood <https://docs.rs/cdshealpix/0.1.5/cdshealpix/nested/struct.Layer.html#method.kth_neighbourhood>`__
    method from the `cdshealpix Rust crate <https://crates.io/crates/cdshealpix>`__.

    Parameters
    ----------
    ipix : `numpy.ndarray`
        The HEALPix cell indexes given as a `np.uint64` numpy array.
    depth : int
        The depth of the HEALPix cells.
    ring : int
        The number of rings. `ring=0` returns just the input cell ids, `ring=1` returns the 8 (or 7) immediate
        neighbours, `ring=2` returns the 8 (or 7) immediate neighbours plus their immediate neighbours (a total of 24 cells), and so on.
    num_threads : int, optional
        Specifies the number of threads to use for the computation. Default to 0 means
        it will choose the number of threads based on the RAYON_NUM_THREADS environment variable (if set),
        or the number of logical CPUs (otherwise)

    Returns
    -------
    neighbours : `numpy.ndarray`
        A :math:`N` x :math:`(2 k + 1)^2` `np.int64` numpy array containing the kth ring neighbours of each cell.
        The :math:`5^{th}` element corresponds to the index of HEALPix cell from which the neighbours are evaluated.
        All its 8 neighbours occup the remaining elements of the line.

    Raises
    ------
    ValueError
        When the HEALPix cell indexes given have values out of :math:`[0, 4^{29 - depth}[`.

    Examples
    --------
    >>> from cdshealpix import neighbours_in_kth_ring
    >>> import numpy as np
    >>> ipix = np.array([42, 6, 10])
    >>> depth = 12
    >>> ring = 3
    >>> neighbours = neighbours_in_kth_ring(ipix, depth, ring)
    """
    _check_depth(depth)
    ipix = np.atleast_1d(ipix)
    _check_ipixels(data=ipix, depth=depth)
    ipix = ipix.astype(np.uint64)
    _check_ring(depth, ring)

    # Allocation of the array containing the neighbours
    neighbours = np.full(
        (*ipix.shape, (2 * ring + 1) ** 2), dtype=np.int64, fill_value=-1
    )
    num_threads = np.uint16(num_threads)
    healpix_geo.nested.kth_neighbourhood(depth, ipix, ring, neighbours, num_threads)

    return neighbours


def zoom_to(ipix, depth, new_depth, num_threads=0):
    r"""Change the resolutions the given cell ids

    Parameters
    ----------
    ipix : numpy.ndarray
        The HEALPix cell indexes given as a `np.uint64` numpy array.
    depth : int
        The depth of the HEALPix cells.
    new_depth : int
        The new depth of the HEALPix cells.

    Returns
    -------
    cells : numpy.ndarray
        A :math:`N` (`depth >= new_depth`) or :math:`N` x :math:`4^{\delta d}` `np.int64` numpy array containing the parents or children of the given cells.
        If `depth == new_depth`, returns the input pixels
    """
    _check_depth(depth)
    _check_depth(new_depth)

    if depth == new_depth:
        return ipix

    ipix = np.atleast_1d(ipix)
    _check_ipixels(data=ipix, depth=depth)
    ipix = ipix.astype(np.uint64)

    num_threads = np.uint16(num_threads)
    if depth > new_depth:
        result = np.full_like(ipix, fill_value=0)
    else:
        relative_depth = new_depth - depth
        shape = (*ipix.shape, 4**relative_depth)
        result = np.full(shape, fill_value=0, dtype="uint64")

    healpix_geo.nested.zoom_to(depth, ipix, new_depth, result, num_threads)

    return result


def siblings(ipix, depth, num_threads=0):
    r"""Find the siblings for every cell

    Parameters
    ----------
    ipix : numpy.ndarray
        The HEALPix cell indexes given as a `np.uint64` numpy array.
    depth : int
        The depth of the HEALPix cells.

    Returns
    -------
    cells : numpy.ndarray
        A :math:`N` x :math:`4` or :math:`N` x :math:`12` `np.uint64` numpy array containing the siblings of the given cells.
        If `depth == 0`, the siblings are the base cells.
    """
    _check_depth(depth)
    ipix = np.atleast_1d(ipix)
    _check_ipixels(data=ipix, depth=depth)
    ipix = ipix.astype(np.uint64)

    num_threads = np.uint16(num_threads)

    if depth != 0:
        shape = (*ipix.shape, 4)
    else:
        shape = (*ipix.shape, 12)

    result = np.full(shape, fill_value=0, dtype="uint64")

    healpix_geo.nested.siblings(depth, ipix, result, num_threads)

    return result


def angular_distances(from_, to_, depth, num_threads=0):
    """Compute the angular distances between cell centers

    Parameters
    ----------
    from_ : numpy.ndarray
        The source Healpix cell indexes given as a ``np.uint64`` numpy array. Should be 1D.
    to_ : numpy.ndarray
        The destination Healpix cell indexes given as a ``np.uint64`` numpy array.
        Should be 2D.
    depth : int
        The depth of the Healpix cells.
    num_threads : int, default: 0
        Specifies the number of threads to use for the computation. Default to 0 means
        it will choose the number of threads based on the RAYON_NUM_THREADS environment variable (if set),
        or the number of logical CPUs (otherwise)

    Returns
    -------
    distances : numpy.ndarray
        The angular distances in radians.

    Raises
    ------
    ValueError
        When the Healpix cell indexes given have values out of :math:`[0, 4^{depth}[`.
    """
    _check_depth(depth)

    from_ = np.atleast_1d(from_)
    _check_ipixels(data=from_, depth=depth)
    from_ = from_.astype("uint64")

    mask = to_ != -1
    masked_to = np.where(mask, to_, 0)

    to_ = np.atleast_1d(masked_to)
    _check_ipixels(data=to_, depth=depth)
    to_ = to_.astype("uint64")

    if from_.shape != to_.shape and from_.shape != to_.shape[:-1]:
        raise ValueError(
            "The shape of `from_` must be compatible with the shape of `to_`:\n"
            f"{to_.shape} or {to_.shape[:-1]} must be equal to {from_.shape}."
        )

    if from_.shape == to_.shape:
        intermediate_shape = to_.shape + (1,)
    else:
        intermediate_shape = to_.shape

    distances = np.full(intermediate_shape, dtype="float64", fill_value=np.nan)
    num_threads = np.uint16(num_threads)

    healpix_geo.nested.angular_distances(
        depth, from_, np.reshape(to_, intermediate_shape), distances, num_threads
    )

    return np.where(mask, np.reshape(distances, to_.shape), np.nan)
