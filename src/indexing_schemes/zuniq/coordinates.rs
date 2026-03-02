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
                    let hash = lonlat_to_healpix_internal(
                        lon,
                        lat,
                        layer,
                        &ellipsoid_,
                        &coefficients,
                        &is_spherical,
                    );

                    *p = healpix::nested::to_zuniq(d, hash);
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

                    let hash = lonlat_to_healpix_internal(
                        lon,
                        lat,
                        layer,
                        &ellipsoid_,
                        &coefficients,
                        &is_spherical,
                    );

                    *p = healpix::nested::to_zuniq(d, hash);
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
