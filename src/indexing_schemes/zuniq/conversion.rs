use crate::maybe_parallelize;
use cdshealpix as healpix;
use ndarray::{Array, IxDyn, Zip};

use numpy::{IntoPyArray, PyArrayDyn, PyArrayMethods};
use pyo3::prelude::*;

use crate::indexing_schemes::depth::DepthLike;

#[pyfunction]
pub(crate) fn from_nested<'py>(
    py: Python<'py>,
    nested: &Bound<'py, PyArrayDyn<u64>>,
    depth: DepthLike,
    nthreads: u16,
) -> PyResult<Bound<'py, PyArrayDyn<u64>>> {
    let nested = unsafe { nested.as_array() };

    let mut zuniq = Array::<u64, IxDyn>::zeros(nested.shape());
    match depth {
        DepthLike::Constant(d) => {
            maybe_parallelize!(
                nthreads,
                Zip::from(&mut zuniq).and(&nested),
                |result, &hash| {
                    *result = healpix::nested::to_zuniq_unsafe(d, hash);
                }
            );
        }
        DepthLike::Array(depths) => {
            let depths = unsafe { depths.bind(py).as_array() };
            maybe_parallelize!(
                nthreads,
                Zip::from(&mut zuniq).and(&nested).and(&depths),
                |result, &hash, &d| {
                    *result = healpix::nested::to_zuniq_unsafe(d, hash);
                }
            );
        }
    }

    Ok(zuniq.into_pyarray(py))
}

#[allow(clippy::type_complexity)]
#[pyfunction]
pub(crate) fn to_nested<'py>(
    py: Python<'py>,
    zuniq: &Bound<'py, PyArrayDyn<u64>>,
    nthreads: u16,
) -> PyResult<(Bound<'py, PyArrayDyn<u64>>, Bound<'py, PyArrayDyn<u8>>)> {
    let zuniq = unsafe { zuniq.as_array() };

    let mut nested = Array::<u64, IxDyn>::zeros(zuniq.shape());
    let mut depths = Array::<u8, IxDyn>::zeros(zuniq.shape());

    maybe_parallelize!(
        nthreads,
        Zip::from(&mut nested).and(&mut depths).and(&zuniq),
        |n, d, &z| {
            let (d_, n_) = healpix::nested::from_zuniq(z);

            *n = n_;
            *d = d_;
        }
    );

    Ok((nested.into_pyarray(py), depths.into_pyarray(py)))
}
