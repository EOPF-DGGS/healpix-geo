use cdshealpix as healpix;

pub fn kth_neighbourhood(hash: &u64, nside: &u32, ring: &u32) -> Vec<i64> {
    let depth = healpix::depth(nside);
    let layer = healpix::nested::get(depth);

    let hash_nested = layer.from_ring(hash);

    layer
        .kth_neighbourhood(hash_nested, *ring)
        .into_iter()
        .map(|v| v as i64)
        .map(|v| if v == -1 { v } else { layer.to_ring(v) })
        .collect()
}
