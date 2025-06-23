use cdshealpix as healpix;
use cdshealpix::sph_geom::coo3d::{vec3_of, UnitVec3, UnitVect3};
use geodesy::Ellipsoid;
use ndarray::{s, Array1, Zip};
use numpy::{PyArrayDyn, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod hierarchy;
mod index;

#[pymodule]
mod nested {
    #[pymodule_export]
    use super::index::RangeMOCIndex;
    use super::*;

    #[pyfunction]
    pub fn healpix_to_lonlat<'a>(
        _py: Python,
        depth: u8,
        ipix: &Bound<'a, PyArrayDyn<u64>>,
        ellipsoid: &str,
        longitude: &Bound<'a, PyArrayDyn<f64>>,
        latitude: &Bound<'a, PyArrayDyn<f64>>,
        nthreads: u16,
    ) -> PyResult<()> {
        let ellipsoid_ =
            Ellipsoid::named(ellipsoid).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ipix = unsafe { ipix.as_array() };
        let mut longitude = unsafe { longitude.as_array_mut() };
        let mut latitude = unsafe { latitude.as_array_mut() };

        let coefficients = ellipsoid_.coefficients_for_authalic_latitude_computations();

        let layer = healpix::nested::get(depth);
        #[cfg(not(target_arch = "wasm32"))]
        {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(nthreads as usize)
                .build()
                .unwrap();
            pool.install(|| {
                Zip::from(&mut longitude)
                    .and(&mut latitude)
                    .and(&ipix)
                    .par_for_each(|lon, lat, &p| {
                        let center = layer.center(p);
                        *lon = center.0.to_degrees();
                        if ellipsoid == "sphere" {
                            *lat = center.1.to_degrees();
                        } else {
                            *lat = ellipsoid_
                                .latitude_authalic_to_geographic(center.1, &coefficients)
                                .to_degrees();
                        }
                    })
            });
        }
        #[cfg(target_arch = "wasm32")]
        {
            Zip::from(&mut longitude)
                .and(&mut latitude)
                .and(&ipix)
                .par_for_each(|lon, lat, &p| {
                    let center = layer.center(p);
                    if ellipsoid == "sphere" {
                        *lat = center.1.to_degrees();
                    } else {
                        *lat = ellipsoid_
                            .latitude_authalic_to_geographic(center.1, &coefficients)
                            .to_degrees();
                    }
                });
        }
        Ok(())
    }

    #[pyfunction]
    pub fn lonlat_to_healpix<'a>(
        _py: Python,
        depth: u8,
        longitude: &Bound<'a, PyArrayDyn<f64>>,
        latitude: &Bound<'a, PyArrayDyn<f64>>,
        ellipsoid: &str,
        ipix: &Bound<'a, PyArrayDyn<u64>>,
        nthreads: u16,
    ) -> PyResult<()> {
        let ellipsoid_ =
            Ellipsoid::named(ellipsoid).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut ipix = unsafe { ipix.as_array_mut() };
        let longitude = unsafe { longitude.as_array() };
        let latitude = unsafe { latitude.as_array() };

        let coefficients = ellipsoid_.coefficients_for_authalic_latitude_computations();

        let layer = healpix::nested::get(depth);
        #[cfg(not(target_arch = "wasm32"))]
        {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(nthreads as usize)
                .build()
                .unwrap();
            pool.install(|| {
                Zip::from(&longitude)
                    .and(&latitude)
                    .and(&mut ipix)
                    .par_for_each(|&lon, &lat, p| {
                        let lon_ = lon.to_radians();
                        let lat_ = if ellipsoid == "sphere" {
                            lat.to_radians()
                        } else {
                            ellipsoid_
                                .latitude_geographic_to_authalic(lat.to_radians(), &coefficients)
                        };
                        *p = layer.hash(lon_, lat_);
                    })
            });
        }
        #[cfg(target_arch = "wasm32")]
        {
            Zip::from(&longitude)
                .and(&latitude)
                .and(&mut ipix)
                .par_for_each(|&lon, &lat, p| {
                    let lon_ = lon.to_radians();
                    let lat_ = if ellipsoid == "sphere" {
                        lat.to_radians()
                    } else {
                        ellipsoid_.latitude_geographic_to_authalic(lat.to_radians(), &coefficients)
                    };
                    *p = layer.hash(lon_, lat_);
                })
        }
        Ok(())
    }

    #[pyfunction]
    pub fn vertices<'a>(
        _py: Python,
        depth: u8,
        ipix: &Bound<'a, PyArrayDyn<u64>>,
        ellipsoid: &str,
        longitude: &Bound<'a, PyArrayDyn<f64>>,
        latitude: &Bound<'a, PyArrayDyn<f64>>,
        nthreads: u16,
    ) -> PyResult<()> {
        let ellipsoid_ =
            Ellipsoid::named(ellipsoid).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ipix = unsafe { ipix.as_array() };
        let mut longitude = unsafe { longitude.as_array_mut() };
        let mut latitude = unsafe { latitude.as_array_mut() };

        let coefficients = ellipsoid_.coefficients_for_authalic_latitude_computations();

        let layer = healpix::nested::get(depth);
        #[cfg(not(target_arch = "wasm32"))]
        {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(nthreads as usize)
                .build()
                .unwrap();
            pool.install(|| {
                Zip::from(longitude.rows_mut())
                    .and(latitude.rows_mut())
                    .and(&ipix)
                    .par_for_each(|mut lon, mut lat, &p| {
                        let vertices = layer.vertices(p);
                        let (vertex_lon, vertex_lat): (Vec<f64>, Vec<f64>) =
                            vertices.into_iter().unzip();
                        let vertex_lon_ = Array1::from_iter(
                            vertex_lon
                                .into_iter()
                                .map(|l| l.to_degrees() % 360.0)
                                .collect::<Vec<f64>>(),
                        );
                        lon.slice_mut(s![..]).assign(&vertex_lon_);

                        let vertex_lat_ = Array1::from_iter(if ellipsoid == "sphere" {
                            vertex_lat
                                .into_iter()
                                .map(|l| l.to_degrees())
                                .collect::<Vec<f64>>()
                        } else {
                            vertex_lat
                                .into_iter()
                                .map(|l| {
                                    ellipsoid_
                                        .latitude_authalic_to_geographic(l, &coefficients)
                                        .to_degrees()
                                })
                                .collect()
                        });
                        lat.slice_mut(s![..]).assign(&vertex_lat_);
                    })
            });
        }
        #[cfg(target_arch = "wasm32")]
        {
            Zip::from(longitude.rows_mut())
                .and(latitude.rows_mut())
                .and(&ipix)
                .par_for_each(|mut lon, mut lat, &p| {
                    let vertices = layer.vertices(p);
                    let (vertex_lon, vertex_lat): (Vec<f64>, Vec<f64>) =
                        vertices.into_iter().unzip();
                    let vertex_lon_ = Array1::from_iter(
                        vertex_lon
                            .into_iter()
                            .map(|l| l.to_degrees() % 360.0)
                            .collect::<Vec<f64>>(),
                    );
                    lon.slice_mut(s![..]).assign(&vertex_lon_);

                    let vertex_lat_ = Array1::from_iter(if ellipsoid == "sphere" {
                        vertex_lat
                            .into_iter()
                            .map(|l| l.to_degrees())
                            .collect::<Vec<f64>>()
                    } else {
                        vertex_lat
                            .into_iter()
                            .map(|l| {
                                ellipsoid_
                                    .latitude_authalic_to_geographic(l, &coefficients)
                                    .to_degrees()
                            })
                            .collect()
                    });
                    lat.slice_mut(s![..]).assign(&vertex_lat_);
                });
        }
        Ok(())
    }

    /// Wrapper of `kth_neighbourhood`
    /// The given array must be of size (2 * ring + 1)^2
    #[pyfunction]
    fn kth_neighbourhood<'a>(
        _py: Python,
        depth: u8,
        ipix: &Bound<'a, PyArrayDyn<u64>>,
        ring: u32,
        neighbours: &Bound<'a, PyArrayDyn<i64>>,
        nthreads: u16,
    ) -> PyResult<()> {
        let ipix = unsafe { ipix.as_array() };
        let mut neighbours = unsafe { neighbours.as_array_mut() };
        let layer = healpix::nested::get(depth);
        #[cfg(not(target_arch = "wasm32"))]
        {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(nthreads as usize)
                .build()
                .unwrap();
            pool.install(|| {
                Zip::from(neighbours.rows_mut())
                    .and(&ipix)
                    .par_for_each(|mut n, &p| {
                        let map = Array1::from_iter(
                            layer
                                .kth_neighbourhood(p, ring)
                                .into_iter()
                                .map(|v| v as i64),
                        );

                        n.slice_mut(s![..map.len()]).assign(&map);
                    })
            });
        }
        #[cfg(target_arch = "wasm32")]
        {
            Zip::from(neighbours.rows_mut())
                .and(&ipix)
                .for_each(|mut n, &p| {
                    let map = Array1::from_iter(
                        layer
                            .kth_neighbourhood(p, ring)
                            .into_iter()
                            .map(|v| v as i64),
                    );

                    n.slice_mut(s![..]).assign(&map);
                });
        }
        Ok(())
    }

    #[pyfunction]
    fn zoom_to<'a>(
        _py: Python,
        depth: u8,
        ipix: &Bound<'a, PyArrayDyn<u64>>,
        new_depth: u8,
        result: &Bound<'a, PyArrayDyn<u64>>,
        nthreads: u16,
    ) -> PyResult<()> {
        use super::hierarchy::nested::{children, parent};
        use std::cmp::Ordering;

        let ipix = unsafe { ipix.as_array() };
        let mut result = unsafe { result.as_array_mut() };
        let layer = healpix::nested::get(depth);

        #[cfg(not(target_arch = "wasm32"))]
        {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(nthreads as usize)
                .build()
                .unwrap();

            match depth.cmp(&new_depth) {
                Ordering::Equal => {
                    pool.install(|| {
                        Zip::from(&mut result).and(&ipix).par_for_each(|n, &p| {
                            *n = p;
                        })
                    });
                }
                Ordering::Less => {
                    pool.install(|| {
                        Zip::from(result.rows_mut())
                            .and(&ipix)
                            .par_for_each(|mut n, &p| {
                                let map = Array1::from_iter(children(layer, p, new_depth));
                                n.slice_mut(s![..map.len()]).assign(&map);
                            })
                    });
                }
                Ordering::Greater => {
                    pool.install(|| {
                        Zip::from(&mut result).and(&ipix).par_for_each(|n, &p| {
                            *n = parent(layer, p, new_depth);
                        })
                    });
                }
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            match depth.cmp(&new_depth) {
                Ordering::Equal => {
                    Zip::from(&mut result).and(&ipix).par_for_each(|n, &p| {
                        *n = p;
                    });
                }
                Ordering::Less => {
                    Zip::from(result.rows_mut())
                        .and(&ipix)
                        .par_for_each(|mut n, &p| {
                            let map = Array1::from_iter(children(layer, p, new_depth));
                            n.slice_mut(s![..map.len()]).assign(&map);
                        });
                }
                Ordering::Greater => {
                    Zip::from(&mut result).and(&ipix).par_for_each(|n, &p| {
                        *n = parent(layer, p, new_depth);
                    });
                }
            }
        }

        Ok(())
    }

    #[pyfunction]
    fn siblings<'a>(
        _py: Python,
        depth: u8,
        ipix: &Bound<'a, PyArrayDyn<u64>>,
        result: &Bound<'a, PyArrayDyn<u64>>,
        nthreads: u16,
    ) -> PyResult<()> {
        use super::hierarchy::nested::siblings;

        let ipix = unsafe { ipix.as_array() };
        let mut result = unsafe { result.as_array_mut() };
        let layer = healpix::nested::get(depth);

        #[cfg(not(target_arch = "wasm32"))]
        {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(nthreads as usize)
                .build()
                .unwrap();

            pool.install(|| {
                Zip::from(result.rows_mut())
                    .and(&ipix)
                    .par_for_each(|mut n, &p| {
                        let map = Array1::from_iter(siblings(layer, p));
                        n.slice_mut(s![..map.len()]).assign(&map);
                    })
            });
        }
        #[cfg(target_arch = "wasm32")]
        {
            Zip::from(result.rows_mut())
                .and(&ipix)
                .par_for_each(|mut n, &p| {
                    let map = Array1::from_iter(siblings(layer, p));
                    n.slice_mut(s![..map.len()]).assign(&map);
                });
        }
        Ok(())
    }

    fn to_vec3(depth: u8, cell_id: u64) -> UnitVect3 {
        let (lon, lat) = cdshealpix::nested::center(depth, cell_id);

        vec3_of(lon, lat)
    }

    /// Wrapper of `UnitVect3.ang_dist`
    /// The given array must be of the same size as `ipix`.
    #[pyfunction]
    fn angular_distances<'a>(
        _py: Python,
        depth: u8,
        from: &Bound<'a, PyArrayDyn<u64>>,
        to: &Bound<'a, PyArrayDyn<u64>>,
        distances: &Bound<'a, PyArrayDyn<f64>>,
        nthreads: u16,
    ) -> PyResult<()> {
        let from = unsafe { from.as_array() };
        let to = unsafe { to.as_array() };
        let mut distances = unsafe { distances.as_array_mut() };
        #[cfg(not(target_arch = "wasm32"))]
        {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(nthreads as usize)
                .build()
                .unwrap();
            pool.install(|| {
                Zip::from(distances.rows_mut())
                    .and(&from)
                    .and(to.rows())
                    .par_for_each(|mut n, from_, to_| {
                        let first = to_vec3(depth, *from_);
                        let distances = Array1::from_iter(
                            to_.iter()
                                .map(|c| to_vec3(depth, *c))
                                .map(|vec| first.ang_dist(&vec)),
                        );

                        n.slice_mut(s![..]).assign(&distances);
                    })
            });
        }
        #[cfg(target_arch = "wasm32")]
        {
            Zip::from(distances.rows_mut())
                .and(&from)
                .and(to.rows())
                .for_each(|mut n, from_, to_| {
                    let first = to_vec3(depth, *from_);
                    let distances = Array1::from_iter(
                        to_.iter()
                            .map(|c| to_vec3(depth, *c))
                            .map(|vec| first.ang_dist(&vec)),
                    );

                    n.slice_mut(s![..]).assign(&distances);
                })
        }
        Ok(())
    }
}

#[pymodule]
mod ring {
    use super::*;

    #[pyfunction]
    pub fn healpix_to_lonlat<'a>(
        _py: Python,
        depth: u8,
        ipix: &Bound<'a, PyArrayDyn<u64>>,
        ellipsoid: &str,
        longitude: &Bound<'a, PyArrayDyn<f64>>,
        latitude: &Bound<'a, PyArrayDyn<f64>>,
        nthreads: u16,
    ) -> PyResult<()> {
        let ellipsoid_ =
            Ellipsoid::named(ellipsoid).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ipix = unsafe { ipix.as_array() };
        let mut longitude = unsafe { longitude.as_array_mut() };
        let mut latitude = unsafe { latitude.as_array_mut() };

        let coefficients = ellipsoid_.coefficients_for_authalic_latitude_computations();

        let nside = healpix::nside(depth);
        #[cfg(not(target_arch = "wasm32"))]
        {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(nthreads as usize)
                .build()
                .unwrap();
            pool.install(|| {
                Zip::from(&mut longitude)
                    .and(&mut latitude)
                    .and(&ipix)
                    .par_for_each(|lon, lat, &p| {
                        let center = healpix::ring::center(nside, p);
                        *lon = center.0.to_degrees();
                        if ellipsoid == "sphere" {
                            *lat = center.1.to_degrees();
                        } else {
                            *lat = ellipsoid_
                                .latitude_authalic_to_geographic(center.1, &coefficients)
                                .to_degrees();
                        }
                    })
            });
        }
        #[cfg(target_arch = "wasm32")]
        {
            Zip::from(&mut longitude)
                .and(&mut latitude)
                .and(&ipix)
                .par_for_each(|lon, lat, &p| {
                    let center = healpix::ring::center(nside, p);
                    if ellipsoid == "sphere" {
                        *lat = center.1.to_degrees();
                    } else {
                        *lat = ellipsoid_
                            .latitude_authalic_to_geographic(center.1, &coefficients)
                            .to_degrees();
                    }
                });
        }
        Ok(())
    }

    #[pyfunction]
    pub fn lonlat_to_healpix<'a>(
        _py: Python,
        depth: u8,
        longitude: &Bound<'a, PyArrayDyn<f64>>,
        latitude: &Bound<'a, PyArrayDyn<f64>>,
        ellipsoid: &str,
        ipix: &Bound<'a, PyArrayDyn<u64>>,
        nthreads: u16,
    ) -> PyResult<()> {
        let ellipsoid_ =
            Ellipsoid::named(ellipsoid).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut ipix = unsafe { ipix.as_array_mut() };
        let longitude = unsafe { longitude.as_array() };
        let latitude = unsafe { latitude.as_array() };

        let coefficients = ellipsoid_.coefficients_for_authalic_latitude_computations();

        let nside = healpix::nside(depth);
        #[cfg(not(target_arch = "wasm32"))]
        {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(nthreads as usize)
                .build()
                .unwrap();
            pool.install(|| {
                Zip::from(&longitude)
                    .and(&latitude)
                    .and(&mut ipix)
                    .par_for_each(|&lon, &lat, p| {
                        let lon_ = lon.to_radians();
                        let lat_ = if ellipsoid == "sphere" {
                            lat.to_radians()
                        } else {
                            ellipsoid_
                                .latitude_geographic_to_authalic(lat.to_radians(), &coefficients)
                        };
                        *p = healpix::ring::hash(nside, lon_, lat_);
                    })
            });
        }
        #[cfg(target_arch = "wasm32")]
        {
            Zip::from(&longitude)
                .and(&latitude)
                .and(&mut ipix)
                .par_for_each(|&lon, &lat, p| {
                    let lon_ = lon.to_radians();
                    let lat_ = if ellipsoid == "sphere" {
                        lat.to_radians()
                    } else {
                        ellipsoid_.latitude_geographic_to_authalic(lat.to_radians(), &coefficients)
                    };
                    *p = healpix::ring::hash(nside, lon_, lat_);
                })
        }
        Ok(())
    }

    #[pyfunction]
    pub fn vertices<'a>(
        _py: Python,
        depth: u8,
        ipix: &Bound<'a, PyArrayDyn<u64>>,
        ellipsoid: &str,
        longitude: &Bound<'a, PyArrayDyn<f64>>,
        latitude: &Bound<'a, PyArrayDyn<f64>>,
        nthreads: u16,
    ) -> PyResult<()> {
        let ellipsoid_ =
            Ellipsoid::named(ellipsoid).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ipix = unsafe { ipix.as_array() };
        let mut longitude = unsafe { longitude.as_array_mut() };
        let mut latitude = unsafe { latitude.as_array_mut() };

        let coefficients = ellipsoid_.coefficients_for_authalic_latitude_computations();

        let nside = healpix::nside(depth);
        #[cfg(not(target_arch = "wasm32"))]
        {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(nthreads as usize)
                .build()
                .unwrap();
            pool.install(|| {
                Zip::from(longitude.rows_mut())
                    .and(latitude.rows_mut())
                    .and(&ipix)
                    .par_for_each(|mut lon, mut lat, &p| {
                        let vertices = healpix::ring::vertices(nside, p);
                        let (vertex_lon, vertex_lat): (Vec<f64>, Vec<f64>) =
                            vertices.into_iter().unzip();
                        let vertex_lon_ = Array1::from_iter(
                            vertex_lon
                                .into_iter()
                                .map(|l| l.to_degrees() % 360.0)
                                .collect::<Vec<f64>>(),
                        );
                        lon.slice_mut(s![..]).assign(&vertex_lon_);

                        let vertex_lat_ = Array1::from_iter(if ellipsoid == "sphere" {
                            vertex_lat
                                .into_iter()
                                .map(|l| l.to_degrees())
                                .collect::<Vec<f64>>()
                        } else {
                            vertex_lat
                                .into_iter()
                                .map(|l| {
                                    ellipsoid_
                                        .latitude_authalic_to_geographic(l, &coefficients)
                                        .to_degrees()
                                })
                                .collect()
                        });
                        lat.slice_mut(s![..]).assign(&vertex_lat_);
                    })
            });
        }
        #[cfg(target_arch = "wasm32")]
        {
            Zip::from(longitude.rows_mut())
                .and(latitude.rows_mut())
                .and(&ipix)
                .par_for_each(|mut lon, mut lat, &p| {
                    let vertices = healpix::ring::vertices(nside, p);
                    let (vertex_lon, vertex_lat): (Vec<f64>, Vec<f64>) =
                        vertices.into_iter().unzip();
                    let vertex_lon_ = Array1::from_iter(
                        vertex_lon
                            .into_iter()
                            .map(|l| l.to_degrees() % 360.0)
                            .collect::<Vec<f64>>(),
                    );
                    lon.slice_mut(s![..]).assign(&vertex_lon_);

                    let vertex_lat_ = Array1::from_iter(if ellipsoid == "sphere" {
                        vertex_lat
                            .into_iter()
                            .map(|l| l.to_degrees())
                            .collect::<Vec<f64>>()
                    } else {
                        vertex_lat
                            .into_iter()
                            .map(|l| {
                                ellipsoid_
                                    .latitude_authalic_to_geographic(l, &coefficients)
                                    .to_degrees()
                            })
                            .collect()
                    });
                    lat.slice_mut(s![..]).assign(&vertex_lat_);
                });
        }
        Ok(())
    }

    /// Wrapper of `kth_neighbourhood`
    /// The given array must be of size (2 * ring + 1)^2
    #[pyfunction]
    fn kth_neighbourhood<'a>(
        _py: Python,
        depth: u8,
        ipix: &Bound<'a, PyArrayDyn<u64>>,
        ring: u32,
        neighbours: &Bound<'a, PyArrayDyn<i64>>,
        nthreads: u16,
    ) -> PyResult<()> {
        let ipix = unsafe { ipix.as_array() };
        let mut neighbours = unsafe { neighbours.as_array_mut() };
        let layer = healpix::nested::get(depth);
        #[cfg(not(target_arch = "wasm32"))]
        {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(nthreads as usize)
                .build()
                .unwrap();
            pool.install(|| {
                Zip::from(neighbours.rows_mut())
                    .and(&ipix)
                    .par_for_each(|mut n, &p| {
                        let p_nested = layer.from_ring(p);
                        let map = Array1::from_iter(
                            layer
                                .kth_neighbourhood(p_nested, ring)
                                .into_iter()
                                .map(|v| layer.to_ring(v) as i64),
                        );

                        n.slice_mut(s![..map.len()]).assign(&map);
                    })
            });
        }
        #[cfg(target_arch = "wasm32")]
        {
            Zip::from(neighbours.rows_mut())
                .and(&ipix)
                .for_each(|mut n, &p| {
                    let p_nested = layer.from_ring(p);
                    let map = Array1::from_iter(
                        layer
                            .kth_neighbourhood(p_nested, ring)
                            .into_iter()
                            .map(|v| layer.to_ring(v) as i64),
                    );

                    n.slice_mut(s![..]).assign(&map);
                });
        }
        Ok(())
    }

    fn to_vec3(nside: u32, cell_id: u64) -> UnitVect3 {
        let (lon, lat) = cdshealpix::ring::center(nside, cell_id);

        vec3_of(lon, lat)
    }

    /// Wrapper of `UnitVect3.ang_dist`
    /// The given array must be of the same size as `ipix`.
    #[pyfunction]
    fn angular_distances<'a>(
        _py: Python,
        depth: u8,
        from: &Bound<'a, PyArrayDyn<u64>>,
        to: &Bound<'a, PyArrayDyn<u64>>,
        distances: &Bound<'a, PyArrayDyn<f64>>,
        nthreads: u16,
    ) -> PyResult<()> {
        let from = unsafe { from.as_array() };
        let to = unsafe { to.as_array() };
        let mut distances = unsafe { distances.as_array_mut() };
        let nside = cdshealpix::nside(depth);
        #[cfg(not(target_arch = "wasm32"))]
        {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(nthreads as usize)
                .build()
                .unwrap();
            pool.install(|| {
                Zip::from(distances.rows_mut())
                    .and(&from)
                    .and(to.rows())
                    .par_for_each(|mut n, from_, to_| {
                        let first = to_vec3(nside, *from_);
                        let distances = Array1::from_iter(
                            to_.into_iter()
                                .map(|c| to_vec3(nside, *c))
                                .map(|vec| first.ang_dist(&vec)),
                        );

                        n.slice_mut(s![..]).assign(&distances);
                    })
            });
        }
        #[cfg(target_arch = "wasm32")]
        {
            Zip::from(distances.rows_mut())
                .and(&from)
                .and(to.rows())
                .for_each(|mut n, from_, to_| {
                    let first = to_vec3(nside, from_);
                    let distances = Array1::from_iter(
                        cell_ids
                            .into_iter()
                            .map(|c| to_vec3(nside, c))
                            .map(|vec| first.ang_dist(vec)),
                    );

                    n.slice_mut(s![..]).assign(&distances);
                })
        }
        Ok(())
    }
}

#[pymodule]
mod healpix_geo {
    #[pymodule_export]
    use super::nested;

    #[pymodule_export]
    use super::ring;
}
