use cdshealpix as healpix;

pub fn from_zuniq(hash: &u64) -> (u64, u8) {
    let (depth, hash_nested) = healpix::nested::from_zuniq(*hash);
    let hash_ring = healpix::nested::get(depth).to_ring(hash_nested);

    (hash_ring, depth)
}

pub fn from_nested(hash: &u64, depth: &u8) -> u64 {
    healpix::nested::get(*depth).to_ring(*hash)
}

pub fn to_zuniq(hash: &u64, depth: &u8) -> u64 {
    let hash_nested = healpix::nested::get(*depth).from_ring(*hash);

    healpix::nested::to_zuniq_unsafe(*depth, hash_nested)
}

pub fn to_nested(hash: &u64, depth: &u8) -> u64 {
    healpix::nested::get(*depth).from_ring(*hash)
}
