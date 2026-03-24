use cdshealpix as healpix;

pub fn kth_neighbourhood(hash: &u64, ring: &u32) -> Vec<i64> {
    let (depth, hash_nested) = healpix::nested::from_zuniq(*hash);
    let layer = healpix::nested::get(depth);

    layer
        .kth_neighbourhood(hash_nested, *ring)
        .into_iter()
        .map(|v| v as i64)
        .map(|v| {
            if v == -1 {
                v
            } else {
                cdshealpix::nested::to_zuniq(depth, v as u64) as i64
            }
        })
        .collect()
}
