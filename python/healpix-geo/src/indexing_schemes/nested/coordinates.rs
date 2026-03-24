use crate::ellipsoid::EllipsoidLike;

use cdshealpix as healpix;
use cdshealpix::sph_geom::coo3d::{UnitVec3, UnitVect3, vec3_of};
use ndarray::{Array1, Zip, s};
use numpy::{PyArray1, PyArray2, PyArrayDyn, PyArrayMethods};
use pyo3::prelude::*;

use crate::maybe_parallelize;

use healpix_geo_core::vectorized::nested::coordinates as vectorized;

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
        vectorized::healpix_to_lonlat(ipix_.as_slice()?, &layer, &ellipsoid, nthreads as usize)
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

    let ipix = vectorized::lonlat_to_healpix(&coords, &layer, &ellipsoid, nthreads as usize);

    Ok(PyArray1::from_vec(py, ipix))
}

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
        vectorized::vertices(ipix_.as_slice()?, &layer, &ellipsoid, nthreads as usize);

    let (lon, lat): (Vec<Vec<f64>>, Vec<Vec<f64>>) = vertices
        .into_iter()
        .map(|row: Vec<(f64, f64)>| -> (Vec<f64>, Vec<f64>) { row.into_iter().unzip() })
        .unzip();

    let longitude = PyArray2::from_vec2(py, &lon)?;
    let latitude = PyArray2::from_vec2(py, &lat)?;

    Ok((longitude.into(), latitude.into()))
}

fn to_vec3(depth: u8, cell_id: u64) -> UnitVect3 {
    let (lon, lat) = cdshealpix::nested::center(depth, cell_id);

    vec3_of(lon, lat)
}

/// Wrapper of `UnitVect3.ang_dist`
/// The given array must be of the same size as `ipix`.
#[pyfunction]
pub(crate) fn angular_distances<'a>(
    _py: Python,
    depth: u8,
    from: &Bound<'a, PyArrayDyn<u64>>,
    to: &Bound<'a, PyArrayDyn<u64>>,
    distances: &Bound<'a, PyArrayDyn<f64>>,
    nthreads: u16,
) -> PyResult<()> {
    let from = unsafe { from.as_array() };
    let to = unsafe { to.as_array() };
    let mut distances = unsafe { distances.as_array_mut() };

    maybe_parallelize!(
        nthreads,
        Zip::from(distances.rows_mut()).and(&from).and(to.rows()),
        |mut n, from_, to_| {
            let first = to_vec3(depth, *from_);
            let distances = Array1::from_iter(
                to_.iter()
                    .map(|c| to_vec3(depth, *c))
                    .map(|vec| first.ang_dist(&vec)),
            );

            n.slice_mut(s![..]).assign(&distances);
        }
    );

    Ok(())
}
