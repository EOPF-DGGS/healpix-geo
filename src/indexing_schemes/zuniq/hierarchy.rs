use crate::maybe_parallelize;

use cdshealpix as healpix;
use ndarray::{Array1, Zip, s};
use numpy::{PyArrayDyn, PyArrayMethods};
use pyo3::prelude::*;

use crate::indexing_schemes::nested::hierarchy::kth_neighbourhood_internal;

/// Wrapper of `kth_neighbourhood`
/// The given array must be of size (2 * ring + 1)^2
#[pyfunction]
pub(crate) fn kth_neighbourhood<'a>(
    _py: Python,
    ipix: &Bound<'a, PyArrayDyn<u64>>,
    ring: u32,
    neighbours: &Bound<'a, PyArrayDyn<i64>>,
    nthreads: u16,
) -> PyResult<()> {
    let ipix = unsafe { ipix.as_array() };

    let mut neighbours = unsafe { neighbours.as_array_mut() };

    maybe_parallelize!(
        nthreads,
        Zip::from(neighbours.rows_mut()).and(&ipix),
        |mut n, &p| {
            let (depth, hash) = healpix::nested::from_zuniq(p);

            let layer = healpix::nested::get(depth);
            let map = Array1::from_iter(
                kth_neighbourhood_internal(&hash, layer, &ring)
                    .into_iter()
                    .map(|n| {
                        if n == -1 {
                            n
                        } else {
                            healpix::nested::to_zuniq(depth, n as u64) as i64
                        }
                    }),
            );

            n.slice_mut(s![..map.len()]).assign(&map);
        }
    );

    Ok(())
}
