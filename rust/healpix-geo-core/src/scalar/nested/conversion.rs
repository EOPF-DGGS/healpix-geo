use cdshealpix as healpix;

pub fn from_zuniq(hash: &u64) -> (u64, u8) {
    let (depth, hash_nested) = healpix::nested::from_zuniq(*hash);

    (hash_nested, depth)
}

pub fn from_ring(hash: &u64, depth: &u8) -> u64 {
    healpix::nested::get(*depth).from_ring(*hash)
}

pub fn to_zuniq(hash: &u64, depth: &u8) -> u64 {
    healpix::nested::to_zuniq_unsafe(*depth, *hash)
}

pub fn to_ring(hash: &u64, depth: &u8) -> u64 {
    healpix::nested::get(*depth).to_ring(*hash)
}
