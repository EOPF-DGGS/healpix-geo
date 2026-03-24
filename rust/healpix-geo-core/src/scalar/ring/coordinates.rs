use crate::ellipsoid::{Ellipsoid, ReferenceBody};

use cdshealpix as healpix;

pub fn healpix_to_lonlat(hash: &u64, nside: &u32, ellipsoid: &Ellipsoid) -> (f64, f64) {
    let center = healpix::ring::center(*nside, *hash);

    let lon = center.0.to_degrees();
    let lat = ellipsoid
        .latitude_authalic_to_geographic(center.1)
        .to_degrees();

    (lon, lat)
}

pub fn lonlat_to_healpix(lon: &f64, lat: &f64, nside: &u32, ellipsoid: &Ellipsoid) -> u64 {
    let lon_ = lon.to_radians();
    let lat_ = ellipsoid.latitude_geographic_to_authalic(lat.to_radians());

    healpix::ring::hash(*nside, lon_, lat_)
}

pub fn vertices(hash: &u64, nside: &u32, ellipsoid: &Ellipsoid) -> Vec<(f64, f64)> {
    let vertices = healpix::ring::vertices(*nside, *hash);

    vertices
        .into_iter()
        .map(|(lon, lat)| {
            (
                lon.to_degrees().rem_euclid(360.0),
                ellipsoid.latitude_authalic_to_geographic(lat).to_degrees(),
            )
        })
        .collect()
}
