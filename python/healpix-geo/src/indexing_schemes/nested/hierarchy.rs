use crate::maybe_parallelize;

use cdshealpix as healpix;
use ndarray::{Array1, Zip, s};
use numpy::{PyArray1, PyArray2, PyArrayDyn, PyArrayMethods};
use pyo3::prelude::*;

use healpix_geo_core::vectorized::nested::hierarchy as vectorized;

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
    let ipix_ = ipix.readonly();
    let layer = healpix::nested::get(depth);

    let result = vectorized::kth_neighbourhood(ipix_.as_slice()?, &layer, &ring, nthreads as usize);

    Ok(PyArray2::from_vec2(py, &result)?)
}

#[pyfunction]
pub(crate) fn zoom_to<'py>(
    _py: Python<'py>,
    depth: u8,
    ipix: &Bound<'py, PyArray1<u64>>,
    new_depth: u8,
    nthreads: u16,
) -> PyResult<Bound<'py, PyArrayDyn<u64>>> {
    use std::cmp::Ordering;

    let ipix_ = ipix.readonly();
    let layer = healpix::nested::get(depth);
    let delta_depth = (depth as i8 - new_depth as i8).abs() as u8;

    let result = match depth.cmp(&new_depth) {
        Ordering::Equal => ipix.to_dyn(),
        Ordering::Less => {
            let result = vectorized::children(ipix_.as_slice()?, delta_depth, nthreads as usize);

            PyArray2::from_vec(py, result)?.to_dyn()
        }
        Ordering::Greater => {
            let result = vectorized::parents(ipix_.as_slice()?, delta_depth, nthreads as usize);

            PyArray1::from_vec(py, result).to_dyn()
        }
    };

    Ok(result)
}

#[pyfunction]
pub(crate) fn siblings<'py>(
    _py: Python<'py>,
    depth: u8,
    ipix: &Bound<'py, PyArrayDyn<u64>>,
    nthreads: u16,
) -> PyResult<Bound<'py, PyArray2<u64>>> {
    let ipix_ = ipix.readonly();
    let layer = healpix::nested::get(depth);

    let siblings = vectorized::siblings(ipix_.as_slice()?, &layer, nthreads as usize);

    Ok(PyArray2::from_vec(py, siblings)?)
}
