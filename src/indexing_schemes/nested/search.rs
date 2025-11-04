use cdshealpix as healpix;
use geodesy::Ellipsoid;
use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

fn get_cells(bmoc: healpix::nested::bmoc::BMOC) -> (Array1<u64>, Array1<u8>, Array1<bool>) {
    let len = bmoc.entries.len();
    let mut ipix = Vec::<u64>::with_capacity(len);
    let mut depth = Vec::<u8>::with_capacity(len);
    let mut fully_covered = Vec::<bool>::with_capacity(len);

    for c in bmoc.into_iter() {
        ipix.push(c.hash);
        depth.push(c.depth);
        fully_covered.push(c.is_full);
    }

    depth.shrink_to_fit();
    ipix.shrink_to_fit();
    fully_covered.shrink_to_fit();

    (ipix.into(), depth.into(), fully_covered.into())
}

fn get_flat_cells(bmoc: healpix::nested::bmoc::BMOC) -> (Array1<u64>, Array1<u8>, Array1<bool>) {
    let len = bmoc.deep_size();
    let mut ipix = Vec::<u64>::with_capacity(len);
    let mut depth = Vec::<u8>::with_capacity(len);
    let mut fully_covered = Vec::<bool>::with_capacity(len);

    for c in bmoc.flat_iter_cell() {
        ipix.push(c.hash);
        depth.push(c.depth);
        fully_covered.push(c.is_full);
    }

    depth.shrink_to_fit();
    ipix.shrink_to_fit();
    fully_covered.shrink_to_fit();

    (ipix.into(), depth.into(), fully_covered.into())
}

#[allow(clippy::type_complexity)]
#[pyfunction]
#[pyo3(signature = (depth, bbox, *, ellipsoid = "sphere", flat = true))]
pub(crate) fn bbox_search<'py>(
    py: Python<'py>,
    depth: u8,
    bbox: &Bound<'py, PyTuple>,
    ellipsoid: &str,
    flat: bool,
) -> PyResult<(
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u8>>,
    Bound<'py, PyArray1<bool>>,
)> {
    let ellipsoid_ =
        Ellipsoid::named(ellipsoid).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let coefficients = ellipsoid_.coefficients_for_authalic_latitude_computations();

    let (lon_min, lat_min, lon_max, lat_max) = bbox.extract::<(f64, f64, f64, f64)>()?;

    let bmoc = healpix::nested::zone_coverage(
        depth,
        lon_min.to_radians(),
        ellipsoid_.latitude_geographic_to_authalic(lat_min.to_radians(), &coefficients),
        lon_max.to_radians(),
        ellipsoid_.latitude_geographic_to_authalic(lat_max.to_radians(), &coefficients),
    );

    let (ipix, moc_depth, fully_covered) = if flat {
        get_flat_cells(bmoc)
    } else {
        get_cells(bmoc)
    };

    Ok((
        ipix.into_pyarray(py),
        moc_depth.into_pyarray(py),
        fully_covered.into_pyarray(py),
    ))
}
