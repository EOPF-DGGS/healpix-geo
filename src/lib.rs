use cdshealpix as healpix;
use cdshealpix::sph_geom::coo3d::{vec3_of, UnitVec3, UnitVect3};
use ndarray::{s, Array1, Zip};
use numpy::{PyArrayDyn, PyArrayMethods};
use pyo3::prelude::*;

mod hierarchy;

#[pymodule]
mod nested {
    use super::*;

    /// Wrapper of `kth_neighbourhood`
    /// The given array must be of size (2 * ring + 1)^2
    #[pyfunction]
    fn neighbours_disk<'a>(
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
    unsafe fn siblings<'a>(
        _py: Python,
        depth: u8,
        ipix: &Bound<'a, PyArrayDyn<u64>>,
        result: &Bound<'a, PyArrayDyn<u64>>,
        nthreads: u16,
    ) -> PyResult<()> {
        use super::hierarchy::nested::siblings;

        let ipix = ipix.as_array();
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
