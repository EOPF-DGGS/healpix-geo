use moc::moc::range::RangeMOC;
use ndarray::Array1;
use numpy::{PyArray1, PyArrayDyn, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Extract the internal boundary from a list of cells
///
/// Parameters
/// ----------
/// depth : int
///     The cell depth
/// cell_ids : numpy.ndarray
///     The cells describing a connected area
///
/// Returns
/// -------
/// internal_boundary : numpy.ndarray
///     The boundary cells
#[pyfunction]
pub(crate) fn internal_boundary<'py>(
    py: Python<'py>,
    depth: u8,
    ipix: &Bound<'py, PyArrayDyn<u64>>,
) -> PyResult<Bound<'py, PyArray1<u64>>> {
    if depth > 29 {
        return Err(PyValueError::new_err(format!(
            "depth must be between 0 and 29, inclusive (got {})",
            depth
        )));
    }

    let ipix = unsafe { ipix.as_array() };
    let moc = RangeMOC::from_fixed_depth_cells(depth, ipix.iter().copied(), None);

    let border = moc.internal_border();

    let cell_ids = Array1::from_iter(border.flatten_to_fixed_depth_cells());

    Ok(PyArray1::from_owned_array(py, cell_ids))
}
