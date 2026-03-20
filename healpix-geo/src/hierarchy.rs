pub mod nested {
    use cdshealpix::nested::Layer;
    use std::cmp::Ordering;

    pub fn parent(layer: &Layer, ipix: u64, depth: u8) -> u64 {
        match layer.depth().cmp(&depth) {
            Ordering::Less => {
                panic!(
                    "Parent depth must be smaller than the current depth, but got {} < {}",
                    layer.depth(),
                    depth
                )
            }
            Ordering::Equal => ipix,
            Ordering::Greater => {
                let relative_depth = layer.depth() - depth;
                ipix >> (2 * relative_depth)
            }
        }
    }

    pub fn siblings(layer: &Layer, ipix: u64) -> Vec<u64> {
        let range = if layer.depth() == 0 {
            0..12
        } else {
            let parent = parent(layer, ipix, layer.depth() - 1);

            parent << 2..(parent + 1) << 2
        };

        range.collect::<Vec<u64>>()
    }

    pub fn children(layer: &Layer, ipix: u64, depth: u8) -> Vec<u64> {
        if layer.depth() >= depth {
            panic!(
                "Child depth must be greater than the current depth, but got {} and {}",
                depth,
                layer.depth()
            )
        } else {
            let relative_depth = depth - layer.depth();
            let first = ipix << (2 * relative_depth);
            let last = (ipix + 1) << (2 * relative_depth);
            let range = first..last;

            range.collect::<Vec<u64>>()
        }
    }
}
