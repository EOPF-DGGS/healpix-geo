use crate::ellipsoid::{EllipsoidLike, IntoGeodesyEllipsoid};
use cdshealpix as healpix;
use cdshealpix::nested::Layer;
use geodesy::ellps::Latitudes;
use geodesy::{authoring::FourierCoefficients, ellps::Ellipsoid};
use ndarray::{Array1, ArrayViewMut1, Zip, s};
use numpy::{PyArrayDyn, PyArrayMethods};
use pyo3::prelude::*;

use crate::indexing_schemes::depth::DepthLike;
use crate::maybe_parallelize;

#[pyfunction]
pub(crate) fn healpix_to_lonlat<'py>(
    _py: Python<'py>,
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

    maybe_parallelize!(
        nthreads,
        Zip::from(&mut longitude).and(&mut latitude).and(&ipix),
        |lon, lat, &p| {
            let (depth, hash) = cdshealpix::nested::from_zuniq(p);
            let layer = cdshealpix::nested::get(depth);

            let (lon_, lat_) = layer.center(hash);
            *lon = lon_.to_degrees();
            if is_spherical {
                *lat = lat_.to_degrees();
            } else {
                *lat = ellipsoid_
                    .latitude_authalic_to_geographic(lat_, &coefficients)
                    .to_degrees();
            }
        }
    );
    Ok(())
}

fn _convert_to_healpix_inplace(
    lon: &f64,
    lat: &f64,
    ipix: &mut u64,
    layer: &Layer,
    ellipsoid_: &Ellipsoid,
    coefficients: &FourierCoefficients,
    is_spherical: &bool,
) {
    let lon_ = lon.to_radians();
    let lat_ = if *is_spherical {
        lat.to_radians()
    } else {
        ellipsoid_.latitude_geographic_to_authalic(lat.to_radians(), coefficients)
    };
    *ipix = layer.hash(lon_, lat_);
}

#[pyfunction]
pub(crate) fn lonlat_to_healpix<'a>(
    py: Python,
    depth: DepthLike,
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

    match depth {
        DepthLike::Constant(d) => {
            let layer = healpix::nested::get(d);

            maybe_parallelize!(
                nthreads,
                Zip::from(&longitude).and(&latitude).and(&mut ipix),
                |lon, lat, p| _convert_to_healpix_inplace(
                    lon,
                    lat,
                    p,
                    layer,
                    &ellipsoid_,
                    &coefficients,
                    &is_spherical
                ),
            );
        }
        DepthLike::Array(depths) => {
            let bound = depths.bind(py);
            let depths_ = unsafe { bound.as_array() };
            maybe_parallelize!(
                nthreads,
                Zip::from(&longitude)
                    .and(&latitude)
                    .and(&depths_)
                    .and(&mut ipix),
                |lon, lat, &d, p| {
                    let layer = cdshealpix::nested::get(d);

                    _convert_to_healpix_inplace(
                        lon,
                        lat,
                        p,
                        layer,
                        &ellipsoid_,
                        &coefficients,
                        &is_spherical,
                    );
                }
            );
        }
    }

    Ok(())
}

fn _vertices(
    mut lon: ArrayViewMut1<f64>,
    mut lat: ArrayViewMut1<f64>,
    p: u64,
    layer: &Layer,
    ellipsoid: &Ellipsoid,
    coefficients: &FourierCoefficients,
    is_spherical: &bool,
) {
    let vertices = layer.vertices(p);
    let (vertex_lon, vertex_lat): (Vec<f64>, Vec<f64>) = vertices.into_iter().unzip();
    let vertex_lon_ = Array1::from_iter(
        vertex_lon
            .into_iter()
            .map(|l| l.to_degrees() % 360.0)
            .collect::<Vec<f64>>(),
    );
    lon.slice_mut(s![..]).assign(&vertex_lon_);

    let vertex_lat_ = Array1::from_iter(if *is_spherical {
        vertex_lat
            .into_iter()
            .map(|l| l.to_degrees())
            .collect::<Vec<f64>>()
    } else {
        vertex_lat
            .into_iter()
            .map(|l| {
                ellipsoid
                    .latitude_authalic_to_geographic(l, coefficients)
                    .to_degrees()
            })
            .collect()
    });
    lat.slice_mut(s![..]).assign(&vertex_lat_);
}

#[pyfunction]
pub(crate) fn vertices<'a>(
    py: Python,
    depth: DepthLike,
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

    match depth {
        DepthLike::Constant(d) => {
            let layer = healpix::nested::get(d);
            maybe_parallelize!(
                nthreads,
                Zip::from(longitude.rows_mut())
                    .and(latitude.rows_mut())
                    .and(&ipix),
                |lon, lat, &p| {
                    _vertices(
                        lon,
                        lat,
                        p,
                        layer,
                        &ellipsoid_,
                        &coefficients,
                        &is_spherical,
                    );
                }
            );
        }
        DepthLike::Array(depths) => {
            let bound = depths.bind(py);
            let depths_ = unsafe { bound.as_array() };
            maybe_parallelize!(
                nthreads,
                Zip::from(longitude.rows_mut())
                    .and(latitude.rows_mut())
                    .and(&depths_)
                    .and(&ipix),
                |lon, lat, &d, &p| {
                    let layer = healpix::nested::get(d);

                    _vertices(
                        lon,
                        lat,
                        p,
                        layer,
                        &ellipsoid_,
                        &coefficients,
                        &is_spherical,
                    );
                }
            );
        }
    }

    Ok(())
}
