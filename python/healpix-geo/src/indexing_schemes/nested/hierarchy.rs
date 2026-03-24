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
pub(crate) fn zoom_to<'a>(
    _py: Python,
    depth: u8,
    ipix: &Bound<'a, PyArrayDyn<u64>>,
    new_depth: u8,
    result: &Bound<'a, PyArrayDyn<u64>>,
    nthreads: u16,
) -> PyResult<()> {
    use crate::hierarchy::nested::{children, parent};
    use std::cmp::Ordering;

    let ipix = unsafe { ipix.as_array() };
    let mut result = unsafe { result.as_array_mut() };

    let layer = healpix::nested::get(depth);

    match depth.cmp(&new_depth) {
        Ordering::Equal => {
            maybe_parallelize!(nthreads, Zip::from(&mut result).and(&ipix), |n, &p| {
                *n = p;
            });
        }
        Ordering::Less => {
            maybe_parallelize!(
                nthreads,
                Zip::from(result.rows_mut()).and(&ipix),
                |mut n, &p| {
                    let map = Array1::from_iter(children(layer, p, new_depth));
                    n.slice_mut(s![..map.len()]).assign(&map);
                },
            );
        }
        Ordering::Greater => {
            maybe_parallelize!(nthreads, Zip::from(&mut result).and(&ipix), |n, &p| {
                *n = parent(layer, p, new_depth);
            });
        }
    };

    Ok(())
}

#[pyfunction]
pub(crate) fn siblings<'a>(
    _py: Python,
    depth: u8,
    ipix: &Bound<'a, PyArrayDyn<u64>>,
    result: &Bound<'a, PyArrayDyn<u64>>,
    nthreads: u16,
) -> PyResult<()> {
    use crate::hierarchy::nested::siblings;

    let ipix = unsafe { ipix.as_array() };
    let mut result = unsafe { result.as_array_mut() };
    let layer = healpix::nested::get(depth);

    maybe_parallelize!(
        nthreads,
        Zip::from(result.rows_mut()).and(&ipix),
        |mut n, &p| {
            let map = Array1::from_iter(siblings(layer, p));
            n.slice_mut(s![..map.len()]).assign(&map);
        },
    );

    Ok(())
}
