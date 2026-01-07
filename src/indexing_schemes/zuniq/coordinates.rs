use crate::ellipsoid::{EllipsoidLike, IntoGeodesyEllipsoid};
use cdshealpix as healpix;
use geodesy::ellps::Latitudes;
use ndarray::{Zip, s};
use numpy::{PyArrayDyn, PyArrayMethods};
use pyo3::prelude::*;

use crate::indexing_schemes::depth::DepthLike;
use crate::indexing_schemes::nested::coordinates::{
    healpix_to_lonlat_internal, lonlat_to_healpix_internal, vertices_internal,
};
use crate::maybe_parallelize;

#[pyfunction]
pub(crate) fn from_nested<'py>(
    _py: Python<'py>,
    nested: &Bound<'py, PyArrayDyn<u64>>,
    depth: DepthLike,
    nthreads: u16,
) -> PyResult<Bound<'py, PyArray1<u64>>> {
    let nested = unsafe { nested.as_array() };

    let mut zuniq = Array::<u64>::zeros(nested.shape());
    match depth {
        DepthLike::Constant(d) => {
            maybe_parallelize!(
                nthreads,
                Zip::from(&mut zuniq).and(&nested),
                |result, &hash| {
                    *result = healpix::nested::to_zuniq_unsafe(d, hash);
                }
            );
        }
        DepthLike::Array(depths) => {
            maybe_parallelize!(
                nthreads,
                Zip::from(&mut zuniq).and(&nested).and(&depths),
                |result, &hash, &d| {
                    *result = healpix::nested::to_zuniq_unsafe(d, hash);
                }
            );
        }
    }
}

#[pyfunction]
pub(crate) fn to_nested<'py>(
    _py: Python<'py>,
    zuniq: &Bound<'py, PyArrayDyn<u64>>,
    nthreads: u16,
) -> PyResult<(Bound<'py, PyArray1<u64>>, Bound<'py, PyArray1<u8>>)> {
    let zuniq = unsafe { zuniq.as_array() };

    let mut nested = Array::<u64>::zeros(zuniq.shape());
    let mut depths = Array::<u8>::zeros(zuniq.shape());

    maybe_parallelize!(
        nthreads,
        Zip::from(&mut nested).and(&mut depths).and(&zuniq),
        |n, d, &z| {
            let (d_, n_) = healpix::nested::from_zuniq(z);

            *n = n_;
            *d = d_;
        }
    );
}

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

            let (lon_, lat_) =
                healpix_to_lonlat_internal(&hash, layer, &ellipsoid_, &coefficients, &is_spherical);
            *lon = lon_;
            *lat = lat_;
        }
    );
    Ok(())
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
                |lon, lat, p| {
                    *p = lonlat_to_healpix_internal(
                        lon,
                        lat,
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
                Zip::from(&longitude)
                    .and(&latitude)
                    .and(&depths_)
                    .and(&mut ipix),
                |lon, lat, &d, p| {
                    let layer = cdshealpix::nested::get(d);

                    *p = lonlat_to_healpix_internal(
                        lon,
                        lat,
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

#[pyfunction]
pub(crate) fn vertices<'py>(
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
        Zip::from(longitude.rows_mut())
            .and(latitude.rows_mut())
            .and(&ipix),
        |mut lon, mut lat, &p| {
            let (depth, hash) = healpix::nested::from_zuniq(p);
            let layer = healpix::nested::get(depth);

            let (vertices_lon, vertices_lat) =
                vertices_internal(&hash, layer, &ellipsoid_, &coefficients, &is_spherical);
            lon.slice_mut(s![..]).assign(&vertices_lon);
            lat.slice_mut(s![..]).assign(&vertices_lat);
        }
    );

    Ok(())
}
