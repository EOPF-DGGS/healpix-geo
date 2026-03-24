use rayon::prelude::*;

use cdshealpix::nested::Layer;

use crate::maybe_parallelize;
use crate::scalar::nested::hierarchy as scalar;

pub fn kth_neighbourhood(
    ipix: &[u64],
    layer: &Layer,
    ring: &u32,
    nthreads: usize,
) -> Vec<Vec<i64>> {
    let mut result = Vec::<Vec<i64>>::with_capacity(ipix.len());

    maybe_parallelize!(nthreads, ipix, result, |hash| scalar::kth_neighbourhood(
        hash, layer, ring
    ));

    result
}
