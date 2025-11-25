use geodesy::Ellipsoid as GeoEllipsoid;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[derive(FromPyObject)]
pub(crate) enum EllipsoidLike {
    Named(String),
    EllipsoidParameters {
        #[pyo3(item("semimajor_axis"))]
        semimajor_axis: f64,
        #[pyo3(item("inverse_flattening"))]
        inverse_flattening: f64,
    },
    SphereParameters {
        #[pyo3(item("radius"))]
        radius: f64,
    },
    EllipsoidObject {
        #[pyo3(attribute("semimajor_axis"))]
        semimajor_axis: f64,
        #[pyo3(attribute("inverse_flattening"))]
        inverse_flattening: f64,
    },
    SphereObject {
        #[pyo3(attribute("radius"))]
        radius: f64,
    },
}

impl EllipsoidLike {
    pub fn is_spherical(&self) -> bool {
        match self {
            Self::Named(name) => name.contains("sphere"),
            Self::EllipsoidParameters { .. } | Self::EllipsoidObject { .. } => false,
            Self::SphereParameters { .. } | Self::SphereObject { .. } => true,
        }
    }
}

pub(crate) trait IntoGeodesyEllipsoid {
    fn into_geodesy_ellipsoid(self) -> PyResult<GeoEllipsoid>;
}

impl IntoGeodesyEllipsoid for EllipsoidLike {
    fn into_geodesy_ellipsoid(self) -> PyResult<GeoEllipsoid> {
        match self {
            EllipsoidLike::Named(name) => {
                GeoEllipsoid::named(&name).map_err(|e| PyValueError::new_err(e.to_string()))
            }
            EllipsoidLike::EllipsoidParameters {
                semimajor_axis,
                inverse_flattening,
            }
            | EllipsoidLike::EllipsoidObject {
                semimajor_axis,
                inverse_flattening,
            } => Ok(GeoEllipsoid::new(
                semimajor_axis,
                1.0f64 / inverse_flattening,
            )),
            EllipsoidLike::SphereParameters { radius } | EllipsoidLike::SphereObject { radius } => {
                Ok(GeoEllipsoid::new(radius, 0.0f64))
            }
        }
    }
}
