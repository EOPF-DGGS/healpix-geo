import numpy as np

from healpix_geo import healpix_geo


def _check_depth(depth):
    ravel_depth = np.ravel(np.atleast_1d(depth))
    if any(ravel_depth < 0) or any(ravel_depth > 29):
        raise ValueError("Depth must be in the [0, 29] closed range")


def _check_ipixels(data, depth):
    npix = 12 * 4 ** (depth)
    if (data >= npix).any() or (data < 0).any():
        raise ValueError(
            f"The input HEALPix cells contains value out of [0, {npix - 1}]"
        )


def neighbours_disk(ipix, depth, ring, num_threads=0):
    """Get the kth ring neighbouring cells of some HEALPix cells at a given depth.

    This method returns a :math:`N` x :math:`(2 k + 1)^2` `np.uint64` numpy array containing the neighbours of each cell of the :math:`N` sized `ipix` array.
    This method is wrapped around the `neighbours_in_kth_ring <https://docs.rs/cdshealpix/0.1.5/cdshealpix/nested/struct.Layer.html#method.neighbours_in_kth_ring>`__
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

    # Allocation of the array containing the neighbours
    neighbours = np.full(
        (*ipix.shape, (2 * ring + 1) ** 2), dtype=np.int64, fill_value=-1
    )
    num_threads = np.uint16(num_threads)
    healpix_geo.nested.neighbours_disk(depth, ipix, ring, neighbours, num_threads)

    return neighbours
