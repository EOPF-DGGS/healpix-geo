use crate::ellipsoid::EllipsoidLike;

use cdshealpix as healpix;
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;

use healpix_geo_core::vectorized::nested::coordinates as vectorized;

#[allow(clippy::type_complexity)]
#[pyfunction]
pub(crate) fn healpix_to_lonlat<'py>(
    py: Python<'py>,
    depth: u8,
    ipix: &Bound<'py, PyArray1<u64>>,
    ellipsoid_like: EllipsoidLike,
    nthreads: u16,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let ellipsoid = ellipsoid_like.into_ellipsoid()?;

    let layer = healpix::nested::get(depth);

    let ipix_ = ipix.readonly();

    let (lon, lat): (Vec<f64>, Vec<f64>) =
        vectorized::healpix_to_lonlat(ipix_.as_slice()?, layer, &ellipsoid, nthreads as usize)
            .into_iter()
            .unzip();

    Ok((PyArray1::from_vec(py, lon), PyArray1::from_vec(py, lat)))
}

#[pyfunction]
pub(crate) fn lonlat_to_healpix<'py>(
    py: Python<'py>,
    depth: u8,
    longitude: &Bound<'py, PyArray1<f64>>,
    latitude: &Bound<'py, PyArray1<f64>>,
    ellipsoid_like: EllipsoidLike,
    nthreads: u16,
) -> PyResult<Bound<'py, PyArray1<u64>>> {
    let ellipsoid = ellipsoid_like.into_ellipsoid()?;

    let lon = longitude.readonly();
    let lat = latitude.readonly();
    let coords: Vec<(f64, f64)> = lon
        .as_slice()?
        .iter()
        .zip(lat.as_slice()?)
        .map(|(&lon, &lat)| (lon, lat))
        .collect();

    let layer = healpix::nested::get(depth);

    let ipix = vectorized::lonlat_to_healpix(&coords, layer, &ellipsoid, nthreads as usize);

    Ok(PyArray1::from_vec(py, ipix))
}

#[allow(clippy::type_complexity)]
#[pyfunction]
pub(crate) fn vertices<'py>(
    py: Python<'py>,
    depth: u8,
    ipix: &Bound<'py, PyArray1<u64>>,
    ellipsoid_like: EllipsoidLike,
    nthreads: u16,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let ellipsoid = ellipsoid_like.into_ellipsoid()?;
    let ipix_ = ipix.readonly();

    let layer = healpix::nested::get(depth);

    let vertices: Vec<Vec<(f64, f64)>> =
        vectorized::vertices(ipix_.as_slice()?, layer, &ellipsoid, nthreads as usize);

    let (lon, lat): (Vec<Vec<f64>>, Vec<Vec<f64>>) = vertices
        .into_iter()
        .map(|row: Vec<(f64, f64)>| -> (Vec<f64>, Vec<f64>) { row.into_iter().unzip() })
        .unzip();

    let longitude = PyArray2::from_vec2(py, &lon)?;
    let latitude = PyArray2::from_vec2(py, &lat)?;

    Ok((longitude, latitude))
}

/// Wrapper of `UnitVect3.ang_dist`
/// The given array must be of the same size as `ipix`.
#[pyfunction]
pub(crate) fn angular_distances<'py>(
    py: Python<'py>,
    depth: u8,
    from: &Bound<'py, PyArray1<u64>>,
    to: &Bound<'py, PyArray2<u64>>,
    nthreads: u16,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    use healpix_geo_core::vectorized::nested::distances as vectorized;

    let from_ = from.readonly();
    let to_ = to.readonly();
    let cols = to.shape()[1];

    let layer = cdshealpix::nested::get(depth);
    let result = vectorized::angular_distances(
        from_.as_slice()?,
        to_.as_slice()?,
        cols,
        layer,
        nthreads as usize,
    );

    Ok(PyArray2::from_vec2(py, &result)?)
}
