use crate::ellipsoid::{EllipsoidLike, IntoGeodesyEllipsoid};
use cdshealpix as healpix;
use cdshealpix::nested::Layer;
use geodesy::authoring::FourierCoefficients;
use geodesy::ellps::{Ellipsoid, Latitudes};

pub fn healpix_to_lonlat(
    hash: &u64,
    layer: &Layer,
    ellipsoid: &Ellipsoid,
    coefficients: &FourierCoefficients,
    is_spherical: &bool,
) -> (f64, f64) {
    let center = layer.center(*hash);
    let lon = center.0.to_degrees();

    let lat = if *is_spherical {
        center.1.to_degrees()
    } else {
        ellipsoid
            .latitude_authalic_to_geographic(center.1, coefficients)
            .to_degrees()
    };

    (lon, lat)
}

pub fn lonlat_to_healpix(
    lon: &f64,
    lat: &f64,
    layer: &Layer,
    ellipsoid: &Ellipsoid,
    coefficients: &FourierCoefficients,
    is_spherical: &bool,
) -> u64 {
    let lon_ = lon.to_radians();
    let lat_ = if *is_spherical {
        lat.to_radians()
    } else {
        ellipsoid.latitude_geographic_to_authalic(lat.to_radians(), coefficients)
    };

    layer.hash(lon_, lat_)
}

pub fn vertices(
    hash: &u64,
    layer: &Layer,
    ellipsoid: &Ellipsoid,
    coefficients: &FourierCoefficients,
    is_spherical: &bool,
) -> (Vec<f64>, Vec<f64>) {
    let vertices = layer.vertices(*hash);

    let (vertex_lon, vertex_lat): (Vec<f64>, Vec<f64>) = vertices.into_iter().unzip();

    let vertex_lon_: Vec<f64> = (vertex_lon
        .into_iter()
        .map(|l| l.to_degrees().rem_euclid(360.0))
        .collect());
    let vertex_lat_: Vec<f64> = if *is_spherical {
        vertex_lat.into_iter().map(|l| l.to_degrees()).collect()
    } else {
        vertex_lat
            .into_iter()
            .map(|l| {
                ellipsoid
                    .latitude_authalic_to_geographic(l, coefficients)
                    .to_degrees()
            })
            .collect()
    };

    (vertex_lon_, vertex_lat_)
}
