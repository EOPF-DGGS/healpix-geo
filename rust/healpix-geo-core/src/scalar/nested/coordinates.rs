use crate::ellipsoid::{Ellipsoid, ReferenceBody};

use cdshealpix::nested::Layer;

pub fn healpix_to_lonlat(hash: &u64, layer: &Layer, ellipsoid: &Ellipsoid) -> (f64, f64) {
    let center = layer.center(*hash);

    let lon = center.0.to_degrees().rem_euclid(360.0);
    let lat = ellipsoid
        .latitude_authalic_to_geographic(center.1)
        .to_degrees();

    (lon, lat)
}

pub fn lonlat_to_healpix(lon: &f64, lat: &f64, layer: &Layer, ellipsoid: &Ellipsoid) -> u64 {
    let lon_ = lon.rem_euclid(360.0).to_radians();
    let lat_ = ellipsoid.latitude_geographic_to_authalic(lat.to_radians());

    layer.hash(lon_, lat_)
}

pub fn vertices(hash: &u64, layer: &Layer, ellipsoid: &Ellipsoid) -> Vec<(f64, f64)> {
    let vertices = layer.vertices(*hash);

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ellipsoid::ReferenceEllipsoid;
    use cdshealpix as healpix;
    use geodesy::ellps::Ellipsoid as GeodesyEllipsoid;

    #[test]
    fn test_lonlat_to_healpix_edge_cases_lon() {
        let layer = healpix::nested::get(0);
        let ellipsoid = Ellipsoid::Ellipsoid(ReferenceEllipsoid::new(
            GeodesyEllipsoid::named("WGS84").unwrap(),
        ));

        let lon: f64 = -180.0;
        let lat: f64 = 75.0;

        let actual = lonlat_to_healpix(&lon, &lat, layer, &ellipsoid);
        assert_eq!(actual, 2);

        let lon: f64 = 180.0;
        let lat: f64 = 75.0;

        let actual = lonlat_to_healpix(&lon, &lat, layer, &ellipsoid);
        assert_eq!(actual, 2);

        let lon: f64 = 0.0;
        let lat: f64 = 75.0;

        let actual = lonlat_to_healpix(&lon, &lat, layer, &ellipsoid);
        assert_eq!(actual, 0);

        let lon: f64 = 360.0;
        let lat: f64 = 75.0;

        let actual = lonlat_to_healpix(&lon, &lat, layer, &ellipsoid);
        assert_eq!(actual, 0);
    }
}
