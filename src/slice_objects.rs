use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyRange, PySlice, PySliceMethods, PyTuple, PyType};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// More powerful version of the built-in slice
///
/// For use with cell ids only.
#[derive(PartialEq, PartialOrd, Debug, Clone)]
pub struct CellIdSlice {
    pub start: Option<u64>,
    pub stop: Option<u64>,
    pub step: Option<u64>,
}

/// More powerful version of the built-in slice
///
/// For positional indexing.
#[derive(PartialEq, PartialOrd, Debug, Clone)]
#[pyclass]
#[pyo3(module = "healpix_geo.slices", name = "Slice", frozen)]
pub struct PositionalSlice {
    #[pyo3(get)]
    pub start: Option<isize>,
    #[pyo3(get)]
    pub stop: Option<isize>,
    #[pyo3(get)]
    pub step: Option<isize>,
}

/// Slice with concrete values
///
/// Note: no `None` values allowed.
#[derive(PartialEq, PartialOrd, Debug, Clone)]
#[pyclass]
#[pyo3(module = "healpix_geo.slices", name = "ConcreteSlice", frozen)]
pub struct ConcreteSlice {
    #[pyo3(get)]
    pub start: isize,
    #[pyo3(get)]
    pub stop: isize,
    #[pyo3(get)]
    pub step: isize,
}

/// Multiple concrete positional slices
#[derive(PartialEq, PartialOrd, Debug, Clone)]
#[pyclass]
#[pyo3(module = "healpix_geo.slices", frozen)]
pub struct MultiConcreteSlice {
    #[pyo3(get)]
    pub slices: Vec<ConcreteSlice>,
}

pub trait AsSlice {
    fn as_positional_slice(&self) -> PyResult<PositionalSlice>;
    fn as_label_slice(&self) -> PyResult<CellIdSlice>;
}

impl AsSlice for Bound<'_, PySlice> {
    fn as_positional_slice(&self) -> PyResult<PositionalSlice> {
        let start = self.getattr("start")?.extract::<Option<isize>>()?;
        let stop = self.getattr("stop")?.extract::<Option<isize>>()?;
        let step = self.getattr("step")?.extract::<Option<isize>>()?;

        Ok(PositionalSlice { start, stop, step })
    }

    fn as_label_slice(&self) -> PyResult<CellIdSlice> {
        let start = self.getattr("start")?.extract::<Option<u64>>()?;
        let stop = self.getattr("stop")?.extract::<Option<u64>>()?;
        let step = self.getattr("step")?.extract::<Option<u64>>()?;

        Ok(CellIdSlice { start, stop, step })
    }
}

#[pymethods]
impl PositionalSlice {
    #[new]
    #[pyo3(signature = (start, stop, step=None, /))]
    fn new(start: Option<isize>, stop: Option<isize>, step: Option<isize>) -> Self {
        PositionalSlice { start, stop, step }
    }

    fn __repr__(&self) -> String {
        let start = match self.start {
            None => "None".to_string(),
            Some(val) => val.to_string(),
        };
        let stop = match self.stop {
            None => "None".to_string(),
            Some(val) => val.to_string(),
        };
        let step = match self.step {
            None => "None".to_string(),
            Some(val) => val.to_string(),
        };

        format!("Slice({start}, {stop}, {step})")
    }

    /// Create a PositionalSlice from a builtin slice object
    #[classmethod]
    fn from_pyslice<'a>(
        _cls: &Bound<'a, PyType>,
        _py: Python<'a>,
        slice: &Bound<'a, PySlice>,
    ) -> PyResult<PositionalSlice> {
        slice.as_positional_slice()
    }

    /// Convert to a builtin slice object
    ///
    /// Note: requires the size as input (the underlying `PySlice` object does
    /// not support `None` arguments)
    pub fn as_pyslice<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PySlice>> {
        let builtins = py.import("builtins")?;

        let result = builtins
            .getattr("slice")?
            .call1((self.start, self.stop, self.step))?;

        result.extract::<Bound<'a, PySlice>>()
    }

    /// Construct concrete indices
    pub fn indices(&self, py: Python<'_>, size: isize) -> PyResult<(isize, isize, isize)> {
        let indices = self.as_pyslice(py)?.indices(size)?;

        Ok((indices.start, indices.stop, indices.step))
    }

    /// Convert to a concrete slice
    ///
    /// This means: no negative start / stop, except if step is negative, in
    /// which case stop may be -1
    pub fn as_concrete(&self, py: Python<'_>, size: isize) -> PyResult<ConcreteSlice> {
        let (start, stop, step) = self.indices(py, size)?;

        Ok(ConcreteSlice { start, stop, step })
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.start.hash(&mut hasher);
        self.stop.hash(&mut hasher);
        self.step.hash(&mut hasher);

        hasher.finish()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }
}

impl ConcreteSlice {
    pub fn join_slices(slices: Vec<Self>) -> Result<Self, String> {
        if slices.is_empty() {
            Err("Empty list".to_string())
        } else if slices.len() == 1 {
            Ok(slices[0].clone())
        } else if !slices
            .windows(2)
            .map(|window| {
                let a = window.first().unwrap();
                let b = window.last().unwrap();

                a.step == b.step
            })
            .reduce(|a, b| a & b)
            .unwrap()
        {
            Err("Step sizes are not equal".to_string())
        } else if !slices
            .windows(2)
            .map(|window| {
                let a = window.first().unwrap();
                let b = window.last().unwrap();
                a.stop == b.start
            })
            .reduce(|a, b| a & b)
            .unwrap()
        {
            Err("Slices are not contiguous".to_string())
        } else {
            let first = slices.first().unwrap();
            let last = slices.last().unwrap();

            Ok(ConcreteSlice {
                start: first.start,
                stop: last.stop,
                step: first.step,
            })
        }
    }
}

#[pymethods]
impl ConcreteSlice {
    fn __repr__(&self) -> String {
        format!(
            "ConcreteSlice({0}, {1}, {2})",
            self.start, self.stop, self.step
        )
    }

    /// Compute the size of the slice
    pub fn size(&self, py: Python<'_>) -> PyResult<usize> {
        let range = PyRange::new_with_step(py, self.start, self.stop, self.step)?;

        range.len()
    }

    /// Extract the elements of the slice
    pub fn indices<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyTuple>> {
        PyTuple::new(py, vec![self.start, self.stop, self.step])
    }

    /// Convert to a python slice
    pub fn as_pyslice<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PySlice>> {
        let builtins = py.import("builtins")?;

        let result = builtins
            .getattr("slice")?
            .call1((self.start, self.stop, self.step))?;

        result.extract::<Bound<'a, PySlice>>()
    }

    #[classmethod]
    pub fn join<'a>(
        _cls: &Bound<'a, PyType>,
        _py: Python<'a>,
        slices: Vec<Self>,
    ) -> PyResult<Self> {
        Self::join_slices(slices).map_err(PyValueError::new_err)
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.start.hash(&mut hasher);
        self.stop.hash(&mut hasher);
        self.step.hash(&mut hasher);

        hasher.finish()
    }
}

#[pyclass]
struct SliceIterator {
    inner: std::vec::IntoIter<ConcreteSlice>,
}

#[pymethods]
impl SliceIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<ConcreteSlice> {
        slf.inner.next()
    }
}

#[pymethods]
impl MultiConcreteSlice {
    fn __repr__(&self) -> String {
        format!(
            "MultiConcreteSlice([{0}])",
            self.slices
                .iter()
                .map(|s| s.__repr__())
                .collect::<Vec<_>>()
                .join(", "),
        )
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<SliceIterator>> {
        let iter = SliceIterator {
            inner: slf.slices.clone().into_iter(),
        };
        Py::new(slf.py(), iter)
    }

    fn size(&self, py: Python<'_>) -> usize {
        self.slices.iter().map(|s| s.size(py).unwrap()).sum()
    }
}
