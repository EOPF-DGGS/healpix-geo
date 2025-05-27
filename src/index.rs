use moc::moc::range::RangeMOC;
use moc::qty::Hpx;
use std::cmp::PartialEq;

#[derive(PartialEq, Debug)]
pub struct RangeMOCIndex {
    moc: RangeMOC<u64, Hpx<u64>>,
}

impl RangeMOCIndex {
    pub fn full_domain(depth: u8) -> Self {
        RangeMOCIndex {
            moc: RangeMOC::new_full_domain(depth),
        }
    }

    pub fn from_cell_ids<T: Iterator<Item = u64>>(depth: u8, cell_ids: T) -> Self {
        RangeMOCIndex {
            moc: RangeMOC::from_fixed_depth_cells(depth, cell_ids, None),
        }
    }

    pub fn union(&self, other: RangeMOCIndex) -> Self {
        RangeMOCIndex {
            moc: self.moc.union(&other.moc),
        }
    }

    pub fn intersection(&self, other: RangeMOCIndex) -> Self {
        RangeMOCIndex {
            moc: self.moc.intersection(&other.moc),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_domain() {
        let depth = 20;
        let index = RangeMOCIndex::full_domain(depth);

        assert_eq!(index.moc.depth_max(), depth);
        assert_eq!(index.moc.n_depth_max_cells(), 12 * 4u64.pow(depth as u32));
    }

    #[test]
    fn test_from_cell_ids() {
        let depth: u8 = 1;
        let cell_ids: Vec<u64> = vec![0, 5, 8, 13, 32, 33, 35, 41, 43, 44, 45];
        let index = RangeMOCIndex::from_cell_ids(depth, cell_ids.iter().copied());

        assert_eq!(index.moc.depth_max(), depth);
        assert_eq!(index.moc.n_depth_max_cells(), cell_ids.len() as u64);
    }

    #[test]
    fn test_union() {
        let depth: u8 = 4;
        let n_cells_per_base_cell: u64 = 4u64.pow(depth as u32);

        let index1 = RangeMOCIndex::from_cell_ids(depth, 0..6 * n_cells_per_base_cell);
        let index2 = RangeMOCIndex::from_cell_ids(
            depth,
            6 * n_cells_per_base_cell..12 * n_cells_per_base_cell,
        );

        let actual = index1.union(index2);
        let expected = RangeMOCIndex::from_cell_ids(depth, 0..12 * n_cells_per_base_cell);

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_intersection() {
        let depth: u8 = 3;
        let n_cells_per_base_cell: u64 = 4u64.pow(depth as u32);
        let cell_ids1 = 2 * n_cells_per_base_cell..4 * n_cells_per_base_cell;
        let cell_ids2 = 3 * n_cells_per_base_cell..5 * n_cells_per_base_cell;

        let index1 = RangeMOCIndex::from_cell_ids(depth, cell_ids1);
        let index2 = RangeMOCIndex::from_cell_ids(depth, cell_ids2);

        let actual = index1.intersection(index2);

        let expected = RangeMOCIndex::from_cell_ids(
            depth,
            2 * n_cells_per_base_cell..3 * n_cells_per_base_cell,
        );

        assert_eq!(actual, expected);
    }
}
