use ndarray::Array1;
use numpy::{PyArray1, PyArrayDyn, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyType;

use moc::moc::range::RangeMOC;
use moc::qty::Hpx;
use std::cmp::PartialEq;

/// range-based index of healpix cell ids
///
/// Only works with cell ids following the "nested" scheme.
#[derive(PartialEq, Debug)]
#[pyclass]
pub struct RangeMOCIndex {
    moc: RangeMOC<u64, Hpx<u64>>,
}

#[pymethods]
impl RangeMOCIndex {
    #[classmethod]
    fn full_domain(_cls: &Bound<'_, PyType>, depth: u8) -> PyResult<Self> {
        let index = RangeMOCIndex {
            moc: RangeMOC::new_full_domain(depth),
        };

        Ok(index)
    }

    #[classmethod]
    fn from_cell_ids<'a>(
        _cls: &Bound<'a, PyType>,
        _py: Python,
        depth: u8,
        cell_ids: &Bound<'a, PyArrayDyn<u64>>,
    ) -> PyResult<Self> {
        let cell_ids = unsafe { cell_ids.as_array() };
        let index = RangeMOCIndex {
            moc: RangeMOC::from_fixed_depth_cells(depth, cell_ids.iter().copied(), None),
        };

        Ok(index)
    }

    fn union(&self, other: &RangeMOCIndex) -> Self {
        RangeMOCIndex {
            moc: self.moc.union(&other.moc),
        }
    }

    fn intersection(&self, other: &RangeMOCIndex) -> Self {
        RangeMOCIndex {
            moc: self.moc.intersection(&other.moc),
        }
    }

    #[getter]
    fn nbytes(&self) -> u64 {
        self.moc.len() as u64 * 2 * u64::BITS as u64 / 8
    }

    #[getter]
    fn size(&self) -> u64 {
        self.moc.n_depth_max_cells()
    }

    #[getter]
    fn depth(&self) -> u8 {
        self.moc.depth_max()
    }

    fn cell_ids<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyArray1<u64>>> {
        let cell_ids = Array1::from_iter(self.moc.flatten_to_fixed_depth_cells());

        Ok(PyArray1::from_owned_array(py, cell_ids))
    }
}
