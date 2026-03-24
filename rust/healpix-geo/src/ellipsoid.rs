use geodesy::authoring::FourierCoefficients;
use geodesy::ellps::{Ellipsoid as GeodesyEllipsoid, EllipsoidBase, Latitudes};

// TODO: create trait for the latitude conversion and split into sphere and ellipsoid
pub struct Ellipsoid {
    ellipsoid: GeodesyEllipsoid,
    coefficients: FourierCoefficients,
    is_spherical: bool,
}

impl Ellipsoid {
    pub fn from_ellipsoid(ellipsoid: GeodesyEllipsoid) -> Self {
        let is_spherical = ellipsoid.flattening() == 0.0;

        let coefficients: FourierCoefficients =
            ellipsoid.coefficients_for_authalic_latitude_computations();

        Self {
            ellipsoid,
            coefficients,
            is_spherical,
        }
    }

    pub fn geographic_to_authalic_latitude(&self, latitude: f64) -> f64 {
        if self.is_spherical {
            latitude
        } else {
            self.ellipsoid
                .latitude_geographic_to_authalic(latitude, &self.coefficients)
        }
    }

    pub fn authalic_to_geographic_latitude(&self, latitude: f64) -> f64 {
        if self.is_spherical {
            latitude
        } else {
            self.ellipsoid
                .latitude_authalic_to_geographic(latitude, &self.coefficients)
        }
    }
}
