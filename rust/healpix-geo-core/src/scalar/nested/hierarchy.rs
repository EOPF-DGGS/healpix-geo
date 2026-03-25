use cdshealpix::nested::Layer;

pub fn kth_neighbourhood(hash: &u64, layer: &Layer, ring: &u32) -> Vec<i64> {
    let mut neighbours: Vec<i64> = layer
        .kth_neighbourhood(*hash, *ring)
        .into_iter()
        .map(|v| v as i64)
        .collect();

    let expected_size = usize::pow((2 * ring + 1) as usize, 2);
    if neighbours.len() < expected_size {
        neighbours.resize(expected_size, -1);
    }

    neighbours
}
