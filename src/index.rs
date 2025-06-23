use ndarray::Array1;
use numpy::{PyArray1, PyArrayDyn, PyArrayMethods};
use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PySlice, PyTuple, PyType};

use moc::deser::json::from_json_aladin;
use moc::elemset::range::MocRanges;
use moc::moc::cell::CellMOC;
use moc::moc::range::RangeMOC;
use moc::moc::{
    CellMOCIntoIterator, CellMOCIterator, HasMaxDepth, RangeMOCIntoIterator, RangeMOCIterator,
};
use moc::qty::Hpx;
use std::cmp::PartialEq;
use std::ops::Range;

struct ConcreteSlice {
    pub start: usize,
    pub stop: usize,
    pub step: usize,
}

impl ConcreteSlice {
    pub fn new(
        start: Option<usize>,
        stop: Option<usize>,
        step: Option<usize>,
    ) -> PyResult<ConcreteSlice> {
        let start_ = match start {
            Some(v) => Ok(v),
            None => Err(PyValueError::new_err(
                "Concrete slice expected, found start as None",
            )),
        }?;

        let stop_ = match stop {
            Some(v) => Ok(v),
            None => Err(PyValueError::new_err(
                "Concrete slice expected, found stop as None",
            )),
        }?;

        let step_ = match step {
            Some(v) => Ok(v),
            None => Err(PyValueError::new_err(
                "Concrete slice expected, found step as None",
            )),
        }?;

        Ok(ConcreteSlice {
            start: start_,
            stop: stop_,
            step: step_,
        })
    }
}

trait AsConcreteSlice {
    fn as_concrete_slice(&self) -> PyResult<ConcreteSlice>;
}

impl AsConcreteSlice for Bound<'_, PyTuple> {
    fn as_concrete_slice(&self) -> PyResult<ConcreteSlice> {
        let start = self.get_item(0)?.extract::<Option<usize>>()?;
        let stop = self.get_item(1)?.extract::<Option<usize>>()?;
        let step = self.get_item(2)?.extract::<Option<usize>>()?;

        ConcreteSlice::new(start, stop, step)
    }
}

#[derive(FromPyObject)]
enum OffsetIndexKind<'a> {
    #[pyo3(transparent, annotation = "slice")]
    Slice(Bound<'a, PySlice>),
    #[pyo3(transparent, annotation = "numpy.ndarray")]
    IndexArray(Bound<'a, PyArrayDyn<u64>>),
}

trait Subset {
    fn slice(&self, slice: &ConcreteSlice) -> PyResult<Self>
    where
        Self: Sized;
}

impl Subset for RangeMOC<u64, Hpx<u64>> {
    fn slice(&self, slice: &ConcreteSlice) -> PyResult<RangeMOC<u64, Hpx<u64>>> {
        if slice.step != 1 {
            return Err(PyValueError::new_err(format!(
                "Only step size 1 is supported, got {}",
                slice.step
            )));
        }

        let mut start = slice.start;
        let mut stop = slice.stop;

        let delta_depth = 29 - self.depth_max();
        let shift = delta_depth << 1;

        let ranges = MocRanges::new_from(
            self.moc_ranges()
                .iter()
                .filter_map(|range: &Range<u64>| {
                    let range_size = ((range.end - range.start) >> shift) as usize;

                    if start >= range_size {
                        // range entirely before slice
                        start -= range_size;
                        stop -= range_size;

                        None
                    } else if stop > 0 {
                        // some overlap
                        let new_start = range.start + ((start as u64) << shift);
                        let new_end = if stop >= range_size {
                            range.end
                        } else {
                            range.start + ((stop as u64) << shift)
                        };

                        let new_range = Range {
                            start: new_start,
                            end: new_end,
                        };

                        if start >= range_size {
                            start -= range_size;
                        } else {
                            start = 0;
                        }

                        if stop >= range_size {
                            stop -= range_size;
                        } else {
                            stop = 0;
                        }

                        Some(new_range)
                    } else {
                        // slice exhausted
                        None
                    }
                })
                .collect::<Vec<Range<u64>>>(),
        );

        Ok(RangeMOC::new(self.depth_max(), ranges))
    }
}

/// range-based index of healpix cell ids
///
/// The idea is to compress cell ids at depth 29 based on run-length encoding (RLE).
///
/// Only works with cell ids following the "nested" scheme.
#[derive(PartialEq, Debug, Clone)]
#[pyclass]
#[pyo3(module = "healpix_geo.nested")]
pub struct RangeMOCIndex {
    moc: RangeMOC<u64, Hpx<u64>>,
}

#[pymethods]
impl RangeMOCIndex {
    /// Create a full domain index
    ///
    /// This is a short-cut for creating an index for the entire sphere.
    ///
    /// Parameters
    /// ----------
    /// depth : int
    ///     The cell depth.
    #[classmethod]
    fn full_domain(_cls: &Bound<'_, PyType>, depth: u8) -> PyResult<Self> {
        let index = RangeMOCIndex {
            moc: RangeMOC::new_full_domain(depth),
        };

        Ok(index)
    }

    /// Create an empty index
    ///
    /// Parameters
    /// ----------
    /// depth : int
    ///     The cell depth.
    #[classmethod]
    fn create_empty(_cls: &Bound<'_, PyType>, depth: u8) -> PyResult<Self> {
        let index = RangeMOCIndex {
            moc: RangeMOC::new_empty(depth),
        };

        Ok(index)
    }

    /// Create an index from given cell ids.
    ///
    /// Parameters
    /// ----------
    /// depth : int
    ///     The cell depth.
    /// cell_ids : numpy.ndarray
    ///     The cells to construct the the index from.
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

    /// Compute the set union of two indexes
    ///
    /// Parameters
    /// ----------
    /// other : RangeMOCIndex
    ///     The other index. May have a different depth, in which case the
    ///     result will use the maximum depth between both indexes.
    ///
    /// Returns
    /// -------
    /// result : RangeMOCIndex
    ///     The union of the two indexes.
    fn union(&self, other: &RangeMOCIndex) -> Self {
        RangeMOCIndex {
            moc: self.moc.union(&other.moc),
        }
    }

    /// Compute the set intersection of two indexes
    ///
    /// Parameters
    /// ----------
    /// other : RangeMOCIndex
    ///     The other index. May have a different depth, in which case the
    ///     result will use the maximum depth between both indexes.
    ///
    /// Returns
    /// -------
    /// result : RangeMOCIndex
    ///     The intersection of the two indexes.
    fn intersection(&self, other: &RangeMOCIndex) -> Self {
        RangeMOCIndex {
            moc: self.moc.intersection(&other.moc),
        }
    }

    /// The size of the ranges in bytes, minus any overhead.
    #[getter]
    fn nbytes(&self) -> u64 {
        self.moc.len() as u64 * 2 * u64::BITS as u64 / 8
    }

    /// The number of items in the index.
    #[getter]
    fn size(&self) -> u64 {
        self.moc.n_depth_max_cells()
    }

    /// The depth of the index.
    #[getter]
    fn depth(&self) -> u8 {
        self.moc.depth_max()
    }

    pub fn __setstate__(&mut self, state: &[u8]) -> PyResult<()> {
        // Deserialize the data contained in the PyBytes object
        // and update the struct with the deserialized values.
        // serde+bincode version:
        // *self = deserialize(state).map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

        let cell_moc: CellMOC<u64, Hpx<u64>> = from_json_aladin(
            std::str::from_utf8(state).map_err(|err| PyRuntimeError::new_err(err.to_string()))?,
        )
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        let reconstructed = RangeMOC::from_cells(
            cell_moc.depth_max(),
            cell_moc
                .into_cell_moc_iter()
                .map(|c| -> (u8, u64) { (c.depth, c.idx) }),
            None,
        );
        *self = RangeMOCIndex { moc: reconstructed };

        Ok(())
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        // Serialize the struct and return a PyBytes object
        // containing the serialized data.
        let mut serialized: Vec<u8> = Default::default();
        self.moc
            .clone()
            .into_range_moc_iter()
            .cells()
            .to_json_aladin(None, &mut serialized)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        // let serialized = serialize(&self).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let bytes = PyBytes::new(py, &serialized);
        Ok(bytes)
    }

    pub fn __reduce__(&self, py: Python) -> PyResult<(PyObject, PyObject, PyObject)> {
        let create = py
            .import("healpix_geo")?
            .getattr("nested")?
            .getattr("create_empty")?;
        let args = (self.moc.depth_max(),);
        let state = self.__getstate__(py)?;

        Ok((
            create.into_pyobject(py)?.unbind().into_any(),
            args.into_pyobject(py)?.unbind().into_any(),
            state.into_pyobject(py)?.unbind().into_any(),
        ))
    }

    /// Retrieve the cell ids from the index.
    ///
    /// Returns
    /// -------
    /// cell_ids : numpy.ndarray
    ///     The cell ids contained by the index.
    fn cell_ids<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyArray1<u64>>> {
        let cell_ids = Array1::from_iter(self.moc.flatten_to_fixed_depth_cells());

        Ok(PyArray1::from_owned_array(py, cell_ids))
    }

    /// Subset the index using positions
    ///
    /// Parameters
    /// ----------
    /// indexer : slice of int
    ///     The integer positions. Currently only supports slices.
    ///
    /// Returns
    /// -------
    /// subset : RangeMOCIndex
    ///     The resulting subset.
    fn isel<'a>(&self, _py: Python<'a>, indexer: OffsetIndexKind<'a>) -> PyResult<Self> {
        match indexer {
            OffsetIndexKind::Slice(slice) => {
                let concrete_slice = slice
                    .getattr("indices")?
                    .call1((self.size(),))?
                    .extract::<Bound<'a, PyTuple>>()?
                    .as_concrete_slice()?;

                let subset = self.moc.slice(&concrete_slice)?;

                Ok(RangeMOCIndex { moc: subset })
            }
            OffsetIndexKind::IndexArray(_array) => Err(PyNotImplementedError::new_err(
                "Subsetting using an array is not supported, yet. Please use a slice instead.",
            )),
        }
    }
}
