use cdshealpix as healpix;
use geodesy::ellps::Ellipsoid;
use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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
pub(crate) fn zone_coverage<'py>(
    py: Python<'py>,
    depth: u8,
    bbox: (f64, f64, f64, f64),
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

    let (lon_min, lat_min, lon_max, lat_max) = bbox;

    let layer = healpix::nested::get(depth);
    let bmoc = layer.zone_coverage(
        lon_min.rem_euclid(360.0).to_radians(),
        ellipsoid_.latitude_geographic_to_authalic(lat_min.to_radians(), &coefficients),
        lon_max.rem_euclid(360.0).to_radians(),
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

#[allow(clippy::type_complexity)]
#[pyfunction]
#[pyo3(signature = (depth, center, size, angle, *, ellipsoid = "sphere", flat = true))]
pub(crate) fn box_coverage<'py>(
    py: Python<'py>,
    depth: u8,
    center: (f64, f64),
    size: (f64, f64),
    angle: f64,
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

    let (lon, lat) = center;
    let (size_lon, size_lat) = size;

    let layer = healpix::nested::get(depth);
    let bmoc = layer.box_coverage(
        lon.rem_euclid(360.0).to_radians(),
        ellipsoid_.latitude_geographic_to_authalic(lat.to_radians(), &coefficients),
        size_lon.rem_euclid(360.0).to_radians(),
        size_lat.to_radians(),
        angle.to_radians(),
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

#[allow(clippy::type_complexity)]
#[pyfunction]
#[pyo3(signature = (depth, vertices, *, ellipsoid = "sphere", exact = false, flat = true))]
pub(crate) fn polygon_coverage<'py>(
    py: Python<'py>,
    depth: u8,
    vertices: &Bound<PyArray2<f64>>,
    ellipsoid: &str,
    exact: bool,
    flat: bool,
) -> PyResult<(
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u8>>,
    Bound<'py, PyArray1<bool>>,
)> {
    let vertices = unsafe { vertices.as_array() };

    let ellipsoid_ =
        Ellipsoid::named(ellipsoid).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let coefficients = ellipsoid_.coefficients_for_authalic_latitude_computations();

    let converted_vertices: Vec<(f64, f64)> = vertices
        .rows()
        .into_iter()
        .map(|row| {
            let lon = row[0];
            let lat = row[1];
            (
                lon.rem_euclid(360.0).to_radians(),
                ellipsoid_.latitude_geographic_to_authalic(lat.to_radians(), &coefficients),
            )
        })
        .collect();

    let layer = healpix::nested::get(depth);
    let bmoc = layer.polygon_coverage(&converted_vertices, exact);

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

#[allow(clippy::type_complexity)]
#[pyfunction]
#[pyo3(signature = (depth, center, radius, *, ellipsoid = "sphere", delta_depth = 0, flat = true))]
pub(crate) fn cone_coverage<'py>(
    py: Python<'py>,
    depth: u8,
    center: (f64, f64),
    radius: f64,
    ellipsoid: &str,
    delta_depth: u8,
    flat: bool,
) -> PyResult<(
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u8>>,
    Bound<'py, PyArray1<bool>>,
)> {
    let ellipsoid_ =
        Ellipsoid::named(ellipsoid).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let coefficients = ellipsoid_.coefficients_for_authalic_latitude_computations();

    if depth > 29 {
        return Err(PyValueError::new_err(
            "depth must be between 0 and 29, inclusive.",
        ));
    } else if depth + delta_depth > 29 {
        return Err(PyValueError::new_err(
            "delta_depth must chosen such that depth + delta_depth <= 29",
        ));
    }

    let (lon, lat) = center;

    let layer = healpix::nested::get(depth);
    let bmoc = layer.cone_coverage_approx_custom(
        delta_depth,
        lon.rem_euclid(360.0).to_radians(),
        ellipsoid_.latitude_geographic_to_authalic(lat.to_radians(), &coefficients),
        radius.to_radians(),
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

#[allow(clippy::type_complexity, clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (depth, center, ellipse_geometry, position_angle, *, ellipsoid = "sphere", delta_depth = 0, flat = true))]
pub(crate) fn elliptical_cone_coverage<'py>(
    py: Python<'py>,
    depth: u8,
    center: (f64, f64),
    ellipse_geometry: (f64, f64),
    position_angle: f64,
    ellipsoid: &str,
    delta_depth: u8,
    flat: bool,
) -> PyResult<(
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u8>>,
    Bound<'py, PyArray1<bool>>,
)> {
    if depth > 29 {
        return Err(PyValueError::new_err(
            "depth must be between 0 and 29, inclusive.",
        ));
    } else if depth + delta_depth > 29 {
        return Err(PyValueError::new_err(
            "delta_depth must chosen such that depth + delta_depth <= 29",
        ));
    }

    let ellipsoid_ =
        Ellipsoid::named(ellipsoid).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let coefficients = ellipsoid_.coefficients_for_authalic_latitude_computations();

    let (lon, lat) = center;
    let (a, b) = ellipse_geometry;

    let layer = healpix::nested::get(depth);
    let bmoc = layer.elliptical_cone_coverage_custom(
        delta_depth,
        lon.rem_euclid(360.0).to_radians(),
        ellipsoid_.latitude_geographic_to_authalic(lat.to_radians(), &coefficients),
        a.to_radians(),
        b.to_radians(),
        position_angle.to_radians(),
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
