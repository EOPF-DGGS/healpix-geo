use pyo3::prelude::*;

mod geometry;
mod hierarchy;
mod index;
mod indexing_schemes;
mod slice_objects;

#[pymodule]
mod nested {
    #[pymodule_export]
    use super::index::RangeMOCIndex;

    #[pymodule_export]
    use crate::indexing_schemes::nested::{
        angular_distances, box_coverage, cone_coverage, elliptical_cone_coverage,
        healpix_to_lonlat, kth_neighbourhood, lonlat_to_healpix, polygon_coverage, siblings,
        vertices, zone_coverage, zoom_to,
    };
}

#[pymodule]
mod ring {
    #[pymodule_export]
    use crate::indexing_schemes::ring::{
        angular_distances, healpix_to_lonlat, kth_neighbourhood, lonlat_to_healpix, vertices,
    };
}

#[pymodule]
mod slices {
    #[pymodule_export]
    use crate::slice_objects::{ConcreteSlice, PositionalSlice};
}

#[pymodule(name = "geometry")]
mod geometry_ {
    #[pymodule_export]
    use crate::geometry::Bbox;
}

#[pymodule]
mod healpix_geo {
    #[pymodule_export]
    use super::nested;

    #[pymodule_export]
    use super::ring;

    #[pymodule_export]
    use super::slices;

    #[pymodule_export]
    use crate::geometry_;
}
