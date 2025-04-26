use pyo3::prelude::*;

mod healpix;

#[pymodule]
mod nested {
    use super::*;
    use cdshealpix as healpix;
    use ndarray::{s, Array1, Zip};
    use numpy::{PyArrayDyn, PyArrayMethods};

    /// Wrapper of `neighbours_disk`
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
                            layer.neighbours_disk(p, ring).into_iter().map(|v| v as i64),
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
                        layer.neighbours_disk(p, ring).into_iter().map(|v| v as i64),
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
        use super::healpix::nested::{children, parent};
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
        use super::healpix::nested::siblings;

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
}

#[pymodule]
mod ring {
    use super::*;
    use cdshealpix as healpix;
    use ndarray::{s, Array1, Zip};
    use numpy::{PyArrayDyn, PyArrayMethods};

    /// Wrapper of `neighbours_disk`
    /// The given array must be of size (2 * ring + 1)^2
    #[pyfunction]
    unsafe fn neighbours_disk<'a>(
        _py: Python,
        depth: u8,
        ipix: &Bound<'a, PyArrayDyn<u64>>,
        ring: u32,
        neighbours: &Bound<'a, PyArrayDyn<i64>>,
        nthreads: u16,
    ) -> PyResult<()> {
        let ipix = ipix.as_array();
        let mut neighbours = neighbours.as_array_mut();
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
                                .neighbours_disk(p_nested, ring)
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
                            .neighbours_disk(p_nested, ring)
                            .into_iter()
                            .map(|v| layer.to_ring(v) as i64),
                    );

                    n.slice_mut(s![..]).assign(&map);
                });
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
