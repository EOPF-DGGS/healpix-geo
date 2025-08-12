use pyo3::prelude::*;
use pyo3::types::{PyRange, PySlice, PySliceMethods, PyTuple, PyType};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// More powerful version of the built-in slice
#[derive(PartialEq, PartialOrd, Debug, Clone)]
#[pyclass]
#[pyo3(module = "healpix_geo.slices", frozen)]
pub struct Slice {
    #[pyo3(get)]
    start: Option<isize>,
    #[pyo3(get)]
    stop: Option<isize>,
    #[pyo3(get)]
    step: Option<isize>,
}

/// Slice with concrete values
///
/// Note: no `None` values allowed.
#[derive(PartialEq, PartialOrd, Debug, Clone)]
#[pyclass]
#[pyo3(module = "healpix_geo.slices", frozen)]
pub struct ConcreteSlice {
    #[pyo3(get)]
    start: isize,
    #[pyo3(get)]
    stop: isize,
    #[pyo3(get)]
    step: isize,
}

trait AsSlice {
    fn as_slice(&self) -> PyResult<Slice>;
}

impl AsSlice for Bound<'_, PySlice> {
    fn as_slice(&self) -> PyResult<Slice> {
        let start = self.getattr("start")?.extract::<Option<isize>>()?;
        let stop = self.getattr("stop")?.extract::<Option<isize>>()?;
        let step = self.getattr("step")?.extract::<Option<isize>>()?;

        Ok(Slice { start, stop, step })
    }
}

#[pymethods]
impl Slice {
    #[new]
    #[pyo3(signature = (start, stop, step=None, /))]
    fn new(start: Option<isize>, stop: Option<isize>, step: Option<isize>) -> Self {
        Slice { start, stop, step }
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

    /// Create a Slice from a builtin slice object
    #[classmethod]
    fn from_pyslice<'a>(
        _cls: &Bound<'a, PyType>,
        _py: Python<'a>,
        slice: &Bound<'a, PySlice>,
    ) -> PyResult<Slice> {
        slice.as_slice()
    }

    /// Convert to a builtin slice object
    ///
    /// Note: requires the size as input (the underlying `PySlice` object does
    /// not support `None` arguments)
    fn as_pyslice<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PySlice>> {
        let builtins = py.import("builtins")?;

        let result = builtins
            .getattr("slice")?
            .call1((self.start, self.stop, self.step))?;

        result.extract::<Bound<'a, PySlice>>()
    }

    /// Construct concrete indices
    fn indices(&self, py: Python<'_>, size: isize) -> PyResult<(isize, isize, isize)> {
        let indices = self.as_pyslice(py)?.indices(size)?;

        Ok((indices.start, indices.stop, indices.step))
    }

    /// Convert to a concrete slice
    ///
    /// This means: no negative start / stop, except if step is negative, in
    /// which case stop may be -1
    fn as_concrete(&self, py: Python<'_>, size: isize) -> PyResult<ConcreteSlice> {
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
    fn size(&self, py: Python<'_>) -> PyResult<usize> {
        let range = PyRange::new_with_step(py, self.start, self.stop, self.step)?;

        range.len()
    }

    /// Extract the elements of the slice
    fn indices<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyTuple>> {
        PyTuple::new(py, vec![self.start, self.stop, self.step])
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.start.hash(&mut hasher);
        self.stop.hash(&mut hasher);
        self.step.hash(&mut hasher);

        hasher.finish()
    }
}
