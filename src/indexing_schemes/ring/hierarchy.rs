use cdshealpix as healpix;
use ndarray::{s, Array1, Zip};
use numpy::{PyArrayDyn, PyArrayMethods};
use pyo3::prelude::*;

/// Wrapper of `kth_neighbourhood`
/// The given array must be of size (2 * ring + 1)^2
#[pyfunction]
pub(crate) fn kth_neighbourhood<'a>(
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
