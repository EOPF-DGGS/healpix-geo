use cdshealpix as healpix;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use healpix_geo_core::vectorized::ring::hierarchy as vectorized;

/// Wrapper of `kth_neighbourhood`
/// The given array must be of size (2 * ring + 1)^2
#[pyfunction]
pub(crate) fn kth_neighbourhood<'py>(
    py: Python<'py>,
    depth: u8,
    ipix: &Bound<'py, PyArray1<u64>>,
    ring: u32,
    nthreads: u16,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let layer = healpix::nested::get(depth);
    let ipix_ = ipix.readonly();

    let nside = u64::pow(2, ring);

    let result = vectorized::kth_neighbourhood(ipix_.as_slice()?, &nside, &ring, nthreads as usize);

    Ok(PyArray2::from_vec2(py, &result)?)
}
