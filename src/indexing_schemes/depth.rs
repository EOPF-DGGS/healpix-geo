use numpy::PyArrayDyn;
use pyo3::prelude::*;

#[derive(FromPyObject)]
pub(crate) enum DepthLike {
    Constant(u8),
    Array(Py<PyArrayDyn<u8>>),
}
