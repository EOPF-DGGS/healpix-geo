use numpy::{PyArrayDyn, PyArrayMethods};
use pyo3::prelude::*;

use healpix_geo_core::vectorized::depth::DepthLike as Depth;

#[derive(FromPyObject)]
pub(crate) enum DepthLike {
    Constant(u8),
    Array(Py<PyArrayDyn<u8>>),
}

impl DepthLike {
    pub fn into_depth(self, py: Python) -> PyResult<Depth> {
        match self {
            Self::Constant(depth) => Ok(Depth::Scalar(depth)),
            Self::Array(depths) => {
                let bound = depths.bind(py);
                let depths_: Vec<u8> = bound.to_vec()?;
                Ok(Depth::Array(depths_))
            }
        }
    }
}
