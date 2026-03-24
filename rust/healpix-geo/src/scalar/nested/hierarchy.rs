use cdshealpix as healpix;
use cdshealpix::nested::Layer;

pub fn kth_neighbourhood(hash: &u64, layer: &Layer, ring: &u32) -> Vec<i64> {
    layer
        .kth_neighbourhood(*hash, *ring)
        .into_iter()
        .map(|v| v as i64)
        .collect()
}
