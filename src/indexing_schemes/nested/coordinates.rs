use crate::ellipsoid::{EllipsoidLike, IntoGeodesyEllipsoid};
use cdshealpix as healpix;
use cdshealpix::sph_geom::coo3d::{UnitVec3, UnitVect3, vec3_of};
use geodesy::ellps::Latitudes;
use ndarray::{Array1, Zip, s};
use numpy::{PyArrayDyn, PyArrayMethods};
use pyo3::prelude::*;

use crate::maybe_parallelize;

#[pyfunction]
pub(crate) fn healpix_to_lonlat<'py>(
    _py: Python<'py>,
    depth: u8,
    ipix: &Bound<'py, PyArrayDyn<u64>>,
    ellipsoid: EllipsoidLike,
    longitude: &Bound<'py, PyArrayDyn<f64>>,
    latitude: &Bound<'py, PyArrayDyn<f64>>,
    nthreads: u16,
) -> PyResult<()> {
    let is_spherical = ellipsoid.is_spherical();
    let ellipsoid_ = ellipsoid.into_geodesy_ellipsoid()?;

    let ipix = unsafe { ipix.as_array() };
    let mut longitude = unsafe { longitude.as_array_mut() };
    let mut latitude = unsafe { latitude.as_array_mut() };

    let coefficients = ellipsoid_.coefficients_for_authalic_latitude_computations();

    let layer = healpix::nested::get(depth);

    maybe_parallelize!(
        nthreads,
        Zip::from(&mut longitude).and(&mut latitude).and(&ipix),
        |lon, lat, &p| {
            let center = layer.center(p);
            *lon = center.0.to_degrees();
            if is_spherical {
                *lat = center.1.to_degrees();
            } else {
                *lat = ellipsoid_
                    .latitude_authalic_to_geographic(center.1, &coefficients)
                    .to_degrees();
            }
        }
    );
    Ok(())
}

#[pyfunction]
pub(crate) fn lonlat_to_healpix<'a>(
    _py: Python,
    depth: u8,
    longitude: &Bound<'a, PyArrayDyn<f64>>,
    latitude: &Bound<'a, PyArrayDyn<f64>>,
    ellipsoid: EllipsoidLike,
    ipix: &Bound<'a, PyArrayDyn<u64>>,
    nthreads: u16,
) -> PyResult<()> {
    let is_spherical = ellipsoid.is_spherical();
    let ellipsoid_ = ellipsoid.into_geodesy_ellipsoid()?;

    let mut ipix = unsafe { ipix.as_array_mut() };
    let longitude = unsafe { longitude.as_array() };
    let latitude = unsafe { latitude.as_array() };

    let coefficients = ellipsoid_.coefficients_for_authalic_latitude_computations();

    let layer = healpix::nested::get(depth);

    maybe_parallelize!(
        nthreads,
        Zip::from(&longitude).and(&latitude).and(&mut ipix),
        |&lon, &lat, p| {
            let lon_ = lon.to_radians();
            let lat_ = if is_spherical {
                lat.to_radians()
            } else {
                ellipsoid_.latitude_geographic_to_authalic(lat.to_radians(), &coefficients)
            };
            *p = layer.hash(lon_, lat_);
        }
    );

    Ok(())
}

#[pyfunction]
pub(crate) fn vertices<'a>(
    _py: Python,
    depth: u8,
    ipix: &Bound<'a, PyArrayDyn<u64>>,
    ellipsoid: EllipsoidLike,
    longitude: &Bound<'a, PyArrayDyn<f64>>,
    latitude: &Bound<'a, PyArrayDyn<f64>>,
    nthreads: u16,
) -> PyResult<()> {
    let is_spherical = ellipsoid.is_spherical();
    let ellipsoid_ = ellipsoid.into_geodesy_ellipsoid()?;

    let ipix = unsafe { ipix.as_array() };
    let mut longitude = unsafe { longitude.as_array_mut() };
    let mut latitude = unsafe { latitude.as_array_mut() };

    let coefficients = ellipsoid_.coefficients_for_authalic_latitude_computations();

    let layer = healpix::nested::get(depth);

    maybe_parallelize!(
        nthreads,
        Zip::from(longitude.rows_mut())
            .and(latitude.rows_mut())
            .and(&ipix),
        |mut lon, mut lat, &p| {
            let vertices = layer.vertices(p);
            let (vertex_lon, vertex_lat): (Vec<f64>, Vec<f64>) = vertices.into_iter().unzip();
            let vertex_lon_ = Array1::from_iter(
                vertex_lon
                    .into_iter()
                    .map(|l| l.to_degrees() % 360.0)
                    .collect::<Vec<f64>>(),
            );
            lon.slice_mut(s![..]).assign(&vertex_lon_);

            let vertex_lat_ = Array1::from_iter(if is_spherical {
                vertex_lat
                    .into_iter()
                    .map(|l| l.to_degrees())
                    .collect::<Vec<f64>>()
            } else {
                vertex_lat
                    .into_iter()
                    .map(|l| {
                        ellipsoid_
                            .latitude_authalic_to_geographic(l, &coefficients)
                            .to_degrees()
                    })
                    .collect()
            });
            lat.slice_mut(s![..]).assign(&vertex_lat_);
        }
    );

    Ok(())
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
