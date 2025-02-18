use pyo3::prelude::*;

#[pymodule]
mod nested {
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

    fn zoom_to<'a>(
        _py: Python,
        depth: u8,
        ipix: &Bound<'a, PyArrayDyn<u64>>,
        new_depth: u8,
        result: &Bound<'a, PyArrayDyn<u64>>,
        nthreads: u16,
    ) -> PyResult<()> {
        use cdshealpix::nested::HashParts;
        let ipix = ipix.as_array();
        let mut result = unsafe { result.as_array_mut() };
        let layer = healpix::nested::get(depth);
        let delta_depth: i16 = new_depth as i16 - depth as i16;

        #[cfg(not(target_arch = "wasm32"))]
        {
            match delta_depth {
                0 => {
                    // TODO: copy cell ids
                }
                delta_depth if delta_depth < 0 => {
                    // TODO: compute parent cell ids
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(nthreads as usize)
                        .build()
                        .unwrap();
                    pool.install(|| {
                        Zip::from(result.rows_mut())
                            .and(&ipix)
                            .par_for_each(|mut n, &p| {
                                HashParts { d0h, i, j } = layer.decode_hash(p);

                                let shift = -delta_depth;

                                let i = i >> shift;
                                let j = j >> shift;

                                let parent = layer.build_hash_from_parts(d0h, i, j);

                                n[0] = parent;
                            })
                    });
                }
                delta_depth if delta_depth > 0 => {
                    // TODO: compute child cell ids. Make sure `result` has enough space.
                }
            }
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
