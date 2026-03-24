use geodesy::authoring::FourierCoefficients;
use geodesy::{ellps, ellps::Latitudes};

// TODO: create trait for the latitude conversion and split into sphere and ellipsoid
pub struct Ellipsoid {
    ellipsoid: ellps::Ellipsoid,
    coefficients: FourierCoefficients,
    is_spherical: bool,
}

impl Ellipsoid {
    pub fn from_ellipsoid(ellipsoid: ellps::Ellipsoid) -> Self {
        let is_spherical = ellipsoid.flattening() == 0.0;

        let coefficients: FourierCoefficients =
            ellipsoid.coefficients_for_authalic_latitude_computations();

        Self {
            ellipsoid,
            coefficients,
            is_spherical,
        }
    }

    pub fn geographic_to_auxiliary_latitude(&self, latitude: &f64) -> f64 {
        if self.is_spherical {
            latitude
        } else {
            self.ellipsoid.geographic_to_authalic_latitude(latitude)
        }
    }

    pub fn authalic_to_geographic_latitude(&self, latitude: &f64) -> f64 {
        if self.is_spherical {
            latitude
        } else {
            self.ellipsoid.authalic_to_geographic_latitude(latitude)
        }
    }
}
