use crate::ellipsoid::Ellipsoid;

use cdshealpix::nested::Layer;

pub fn healpix_to_lonlat(hash: &u64, layer: &Layer, ellipsoid: &Ellipsoid) -> (f64, f64) {
    let center = layer.center(*hash);

    let lon = center.0.to_degrees();
    let lat = ellipsoid
        .authalic_to_geographic_latitude(center.1)
        .to_degrees();

    (lon, lat)
}

pub fn lonlat_to_healpix(lon: &f64, lat: &f64, layer: &Layer, ellipsoid: &Ellipsoid) -> u64 {
    let lon_ = lon.to_radians();
    let lat_ = ellipsoid.geographic_to_authalic_latitude(lat.to_radians());

    layer.hash(lon_, lat_)
}

pub fn vertices(hash: &u64, layer: &Layer, ellipsoid: &Ellipsoid) -> (Vec<f64>, Vec<f64>) {
    let vertices = layer.vertices(*hash);

    vertices
        .into_iter()
        .map(|(lon, lat)| {
            (
                lon.to_degrees().rem_euclid(360.0),
                ellipsoid.authalic_to_geographic_latitude(lat).to_degrees(),
            )
        })
        .collect()
}
