use geodesy::authoring::FourierCoefficients;
use geodesy::ellps::{Ellipsoid as GeodesyEllipsoid, EllipsoidBase, Latitudes};

pub trait ReferenceBody {
    fn latitude_authalic_to_geographic(&self, latitude: f64) -> f64;
    fn latitude_geographic_to_authalic(&self, latitude: f64) -> f64;
}

pub struct ReferenceSphere {
    #[allow(dead_code)]
    ellipsoid: GeodesyEllipsoid,
}

impl ReferenceBody for ReferenceSphere {
    fn latitude_authalic_to_geographic(&self, latitude: f64) -> f64 {
        latitude
    }

    fn latitude_geographic_to_authalic(&self, latitude: f64) -> f64 {
        latitude
    }
}

pub struct ReferenceEllipsoid {
    ellipsoid: GeodesyEllipsoid,
    coefficients: FourierCoefficients,
}

impl ReferenceBody for ReferenceEllipsoid {
    fn latitude_authalic_to_geographic(&self, latitude: f64) -> f64 {
        self.ellipsoid
            .latitude_geographic_to_authalic(latitude, &self.coefficients)
    }

    fn latitude_geographic_to_authalic(&self, latitude: f64) -> f64 {
        self.ellipsoid
            .latitude_authalic_to_geographic(latitude, &self.coefficients)
    }
}

pub enum Ellipsoid {
    Ellipsoid(ReferenceEllipsoid),
    Sphere(ReferenceSphere),
}

impl ReferenceBody for Ellipsoid {
    fn latitude_authalic_to_geographic(&self, latitude: f64) -> f64 {
        match self {
            Self::Ellipsoid(wrapped) => wrapped.latitude_authalic_to_geographic(latitude),
            Self::Sphere(wrapped) => wrapped.latitude_authalic_to_geographic(latitude),
        }
    }

    fn latitude_geographic_to_authalic(&self, latitude: f64) -> f64 {
        match self {
            Self::Ellipsoid(wrapped) => wrapped.latitude_authalic_to_geographic(latitude),
            Self::Sphere(wrapped) => wrapped.latitude_authalic_to_geographic(latitude),
        }
    }
}

pub fn from_ellipsoid(ellipsoid: GeodesyEllipsoid) -> Ellipsoid {
    let is_spherical = ellipsoid.flattening() == 0.0;

    if is_spherical {
        Ellipsoid::Sphere(ReferenceSphere { ellipsoid })
    } else {
        let coefficients: FourierCoefficients =
            ellipsoid.coefficients_for_authalic_latitude_computations();

        Ellipsoid::Ellipsoid(ReferenceEllipsoid {
            ellipsoid,
            coefficients,
        })
    }
}
