use geodesy::authoring::FourierCoefficients;
use geodesy::ellps::{Ellipsoid as GeodesyEllipsoid, Latitudes};

pub trait ReferenceBody {
    fn latitude_authalic_to_geographic(&self, latitude: f64) -> f64;
    fn latitude_geographic_to_authalic(&self, latitude: f64) -> f64;
}

pub struct ReferenceSphere {
    #[allow(dead_code)]
    ellipsoid: GeodesyEllipsoid,
}

impl ReferenceSphere {
    pub fn new(ellipsoid: GeodesyEllipsoid) -> Self {
        Self { ellipsoid }
    }
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

impl ReferenceEllipsoid {
    pub fn new(ellipsoid: GeodesyEllipsoid) -> Self {
        let coefficients = ellipsoid.coefficients_for_authalic_latitude_computations();

        Self {
            ellipsoid,
            coefficients,
        }
    }
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
