use pyo3::prelude::*;
use pyo3::types::{PyTuple, PyType};

/// bounding box
#[derive(PartialEq, PartialOrd, Debug, Clone)]
#[pyclass]
#[pyo3(module = "healpix_geo.geometry", frozen)]
pub struct Bbox {
    #[pyo3(get)]
    lon_min: f64,
    #[pyo3(get)]
    lat_min: f64,
    #[pyo3(get)]
    lon_max: f64,
    #[pyo3(get)]
    lat_max: f64,
}

#[pymethods]
impl Bbox {
    #[new]
    fn new(lon_min: f64, lat_min: f64, lon_max: f64, lat_max: f64) -> Self {
        Self {
            lon_min,
            lat_min,
            lon_max,
            lat_max,
        }
    }

    #[classmethod]
    fn from_tuple<'py>(
        _cls: &Bound<'py, PyType>,
        _py: Python<'py>,
        bbox: &Bound<'py, PyTuple>,
    ) -> PyResult<Self> {
        let lon_min = bbox.get_item(0)?.extract::<f64>()?;
        let lat_min = bbox.get_item(1)?.extract::<f64>()?;
        let lon_max = bbox.get_item(2)?.extract::<f64>()?;
        let lat_max = bbox.get_item(3)?.extract::<f64>()?;

        Ok(Self {
            lon_min,
            lat_min,
            lon_max,
            lat_max,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "Bbox({0}, {1}, {2}, {3})",
            self.lon_min, self.lat_min, self.lon_max, self.lat_max
        )
    }
}
