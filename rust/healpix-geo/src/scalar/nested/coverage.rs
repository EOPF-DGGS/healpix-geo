use cdshealpix::nested::bmoc::BMOC;

fn get_cells(bmoc: BMOC) -> (Vec<u64>, Vec<u8>, Vec<bool>) {
    let len = bmoc.entries.len();

    let mut ipix = Vec::<u64>::with_capacity(len);
    let mut depth = Vec::<u8>::with_capacity(len);
    let mut fully_covered = Vec::<bool>::with_capacity(len);

    for c in bmoc.into_iter() {
        ipix.push(c.hash);
        depth.push(c.depth);
        fully_covered.push(c.is_full);
    }

    depth.shrink_to_fit();
    ipix.shrink_to_fit();
    fully_covered.shrink_to_fit();

    (ipix, depth, fully_covered)
}

fn get_flat_cells(bmoc: BMOC) -> (Vec<u64>, Vec<u8>, Vec<bool>) {
    let len = bmoc.deep_size();
    let mut ipix = Vec::<u64>::with_capacity(len);
    let mut depth = Vec::<u8>::with_capacity(len);
    let mut fully_covered = Vec::<bool>::with_capacity(len);

    for c in bmoc.flat_iter_cell() {
        ipix.push(c.hash);
        depth.push(c.depth);
        fully_covered.push(c.is_full);
    }

    depth.shrink_to_fit();
    ipix.shrink_to_fit();
    fully_covered.shrink_to_fit();

    (ipix, depth, fully_covered)
}

pub fn zone_coverage(
    bbox: (f64, f64, f64, f64),
    ellipsoid: &Ellipsoid,
    is_spherical: &bool,
    flat: &bool
) -> (Vec<u64>, Vec<u64>, Vec<u64>) {

}
