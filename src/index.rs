use ndarray::{Array1, Ix1};
use numpy::{PyArray1, PyArrayDyn, PyArrayMethods};
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::type_object::PyTypeInfo;
use pyo3::types::{PyBytes, PySlice, PyType};

use ndarray::parallel::prelude::*;
use rayon::iter::ParallelIterator;

use cdshealpix::nested;

use moc::deser::json::from_json_aladin;
use moc::elemset::range::MocRanges;
use moc::moc::cell::CellMOC;
use moc::moc::range::{CellSelection, RangeMOC};
use moc::moc::{
    CellMOCIntoIterator, CellMOCIterator, HasMaxDepth, RangeMOCIntoIterator, RangeMOCIterator,
};
use moc::qty::Hpx;
use moc::ranges::SNORanges;
use std::cmp::PartialEq;
use std::ops::Range;

use crate::geometry::GeometryTypes;
use crate::slice_objects::{AsSlice, CellIdSlice, ConcreteSlice, MultiConcreteSlice};

#[derive(FromPyObject, IntoPyObject)]
enum IndexKind<'py> {
    #[pyo3(transparent, annotation = "slice")]
    Slice(Bound<'py, PySlice>),
    #[pyo3(transparent, annotation = "numpy.ndarray")]
    Array(Bound<'py, PyArrayDyn<u64>>),
}

trait Overlap {
    fn overlap(
        &self,
        range: &Range<u64>,
        depth: u8,
        offset: usize,
    ) -> Option<(ConcreteSlice, Range<u64>)>;
}

impl Overlap for CellIdSlice {
    fn overlap(
        &self,
        range: &Range<u64>,
        depth: u8,
        offset: usize,
    ) -> Option<(ConcreteSlice, Range<u64>)> {
        let relative_depth = 29 - depth;

        let range_start = range.start >> (relative_depth << 1);
        let range_end = range.end >> (relative_depth << 1);

        let start = self.start.unwrap_or(range_start);
        let stop = self.stop.unwrap_or(range_end - 1);
        let step = self.step.unwrap_or(1);

        if (start < range_end) && (stop >= range_start) {
            let pos_slice = ConcreteSlice {
                start: start.saturating_sub(range_start) as isize + offset as isize,
                stop: (stop + 1).min(range_end).saturating_sub(range_start) as isize
                    + offset as isize,
                step: step as isize,
            };

            let new_range = Range {
                start: start.max(range_start) << (relative_depth << 1),
                end: (stop + 1).min(range_end) << (relative_depth << 1),
            };

            Some((pos_slice, new_range))
        } else {
            None
        }
    }
}

trait Subset {
    fn slice(&self, slice: &ConcreteSlice) -> PyResult<Self>
    where
        Self: Sized;

    fn subset(&self, indexer: &Bound<'_, PyArrayDyn<u64>>) -> PyResult<Self>
    where
        Self: Sized;
}

impl Subset for RangeMOC<u64, Hpx<u64>> {
    fn slice(&self, slice: &ConcreteSlice) -> PyResult<RangeMOC<u64, Hpx<u64>>> {
        if slice.step != 1 {
            return Err(PyValueError::new_err(format!(
                "Only step size 1 is supported, got {}",
                slice.step
            )));
        }

        let mut start = slice.start;
        let mut stop = slice.stop;

        let delta_depth = 29 - self.depth_max();
        let shift = delta_depth << 1;

        let ranges = MocRanges::new_from(
            self.moc_ranges()
                .iter()
                .filter_map(|range: &Range<u64>| {
                    let range_size = ((range.end - range.start) >> shift) as isize;

                    if start >= range_size {
                        // range entirely before slice
                        start -= range_size;
                        stop -= range_size;

                        None
                    } else if stop > 0 {
                        // some overlap
                        let new_start = range.start + ((start as u64) << shift);
                        let new_end = if stop >= range_size {
                            range.end
                        } else {
                            range.start + ((stop as u64) << shift)
                        };

                        let new_range = Range {
                            start: new_start,
                            end: new_end,
                        };

                        if start >= range_size {
                            start -= range_size;
                        } else {
                            start = 0;
                        }

                        if stop >= range_size {
                            stop -= range_size;
                        } else {
                            stop = 0;
                        }

                        Some(new_range)
                    } else {
                        // slice exhausted
                        None
                    }
                })
                .collect::<Vec<Range<u64>>>(),
        );

        Ok(RangeMOC::new(self.depth_max(), ranges))
    }

    fn subset(&self, array: &Bound<'_, PyArrayDyn<u64>>) -> PyResult<RangeMOC<u64, Hpx<u64>>> {
        let array = unsafe { array.as_array() };
        let delta_depth = 29 - self.depth_max();
        let shift = delta_depth << 1;

        let slice_offsets = self
            .moc_ranges()
            .iter()
            .map(|range| ((range.end - range.start) >> shift) as usize)
            .scan(0, |acc, x| {
                let cur = *acc;
                *acc += x;
                Some((cur, *acc))
            })
            .collect::<Vec<(usize, usize)>>();
        let slice_starts = self
            .moc_ranges()
            .iter()
            .map(|range| range.start)
            .collect::<Vec<u64>>();

        let cell_ids: Vec<u64> = array
            .iter()
            .map(|&index| {
                let position = index as usize;
                if index >= self.n_depth_max_cells() {
                    Err(PyValueError::new_err("{index} is out of bounds"))
                } else {
                    let slice_index = slice_offsets
                        .iter()
                        .position(|x| position >= x.0 && position < x.1)
                        .unwrap_or(slice_offsets.len() - 1);
                    let slice_start = slice_starts[slice_index] >> shift;
                    let selected = slice_start + (index - (slice_offsets[slice_index].0 as u64));

                    Ok(selected)
                }
            })
            .collect::<PyResult<Vec<u64>>>()?;

        Ok(RangeMOC::from_fixed_depth_cells(
            self.depth_max(),
            cell_ids.into_iter(),
            None,
        ))
    }
}

trait SizedRanges {
    fn range_sizes(&self) -> Vec<usize>;
}

impl SizedRanges for RangeMOC<u64, Hpx<u64>> {
    fn range_sizes(&self) -> Vec<usize> {
        let relative_depth = 29 - self.depth_max();

        self.moc_ranges()
            .0
            .par_iter()
            .map(|r| ((r.end - r.start) >> (relative_depth << 1)) as usize)
            .collect()
    }
}

fn range_offsets(sizes: Vec<usize>) -> Vec<usize> {
    sizes
        .iter()
        .scan(0, |state, x| {
            let val = *state;
            *state += x;
            Some(val)
        })
        .collect()
}

trait IndexSetOps {
    fn index_intersection(&self, other: Self) -> PyResult<(Vec<ConcreteSlice>, Self)>
    where
        Self: Sized;
}

impl IndexSetOps for RangeMOC<u64, Hpx<u64>> {
    fn index_intersection(&self, other: Self) -> PyResult<(Vec<ConcreteSlice>, Self)> {
        let depth = self.depth_max();
        let relative_depth = 29 - depth;
        let shift = relative_depth << 1;

        let range_sizes = self.range_sizes();
        let offsets: Vec<usize> = range_offsets(range_sizes);

        let new_moc = self.intersection(&other);

        let slices = other
            .moc_ranges()
            .par_iter()
            .filter_map(|range_o| {
                let start_o = range_o.start >> shift;
                let end_o = range_o.end >> shift;
                let slices: Vec<_> = self
                    .moc_ranges()
                    .iter()
                    .enumerate()
                    .filter_map(|(index, range_s)| {
                        let start_s = range_s.start >> shift;
                        let end_s = range_s.end >> shift;
                        let offset = offsets[index];

                        if (start_o <= end_s) && (end_o >= start_s) {
                            let pos_slice = ConcreteSlice {
                                start: start_o.saturating_sub(start_s) as isize + offset as isize,
                                stop: end_o.min(end_s).saturating_sub(start_s) as isize
                                    + offset as isize,
                                step: 1,
                            };

                            Some(pos_slice)
                        } else {
                            None
                        }
                    })
                    .collect();

                if !slices.is_empty() {
                    Some(ConcreteSlice::join_slices(slices).map_err(PyValueError::new_err))
                } else {
                    None
                }
            })
            .collect::<PyResult<Vec<ConcreteSlice>>>()?;

        Ok((slices, new_moc))
    }
}

/// range-based index of healpix cell ids
///
/// The idea is to compress cell ids at depth 29 based on run-length encoding (RLE).
///
/// Only works with cell ids following the "nested" scheme.
#[derive(PartialEq, Debug, Clone)]
#[pyclass]
#[pyo3(module = "healpix_geo.nested")]
pub struct RangeMOCIndex {
    moc: RangeMOC<u64, Hpx<u64>>,
}

#[pymethods]
impl RangeMOCIndex {
    /// Create a full domain index
    ///
    /// This is a short-cut for creating an index for the entire sphere.
    ///
    /// Parameters
    /// ----------
    /// depth : int
    ///     The cell depth.
    #[classmethod]
    fn full_domain(_cls: &Bound<'_, PyType>, depth: u8) -> PyResult<Self> {
        let index = RangeMOCIndex {
            moc: RangeMOC::new_full_domain(depth),
        };

        Ok(index)
    }

    /// Create an empty index
    ///
    /// Parameters
    /// ----------
    /// depth : int
    ///     The cell depth.
    #[classmethod]
    fn create_empty(_cls: &Bound<'_, PyType>, depth: u8) -> PyResult<Self> {
        let index = RangeMOCIndex {
            moc: RangeMOC::new_empty(depth),
        };

        Ok(index)
    }

    /// Create an index from given cell ids.
    ///
    /// Parameters
    /// ----------
    /// depth : int
    ///     The cell depth.
    /// cell_ids : numpy.ndarray
    ///     The cells to construct the the index from.
    #[classmethod]
    fn from_cell_ids<'a>(
        _cls: &Bound<'a, PyType>,
        _py: Python,
        depth: u8,
        cell_ids: &Bound<'a, PyArrayDyn<u64>>,
    ) -> PyResult<Self> {
        let cell_ids = unsafe { cell_ids.as_array() };
        let index = RangeMOCIndex {
            moc: RangeMOC::from_fixed_depth_cells(depth, cell_ids.iter().copied(), None),
        };

        Ok(index)
    }

    /// Compute the set union of two indexes
    ///
    /// Parameters
    /// ----------
    /// other : RangeMOCIndex
    ///     The other index. May have a different depth, in which case the
    ///     result will use the maximum depth between both indexes.
    ///
    /// Returns
    /// -------
    /// result : RangeMOCIndex
    ///     The union of the two indexes.
    fn union(&self, other: &RangeMOCIndex) -> Self {
        RangeMOCIndex {
            moc: self.moc.union(&other.moc),
        }
    }

    /// Compute the set intersection of two indexes
    ///
    /// Parameters
    /// ----------
    /// other : RangeMOCIndex
    ///     The other index. May have a different depth, in which case the
    ///     result will use the maximum depth between both indexes.
    ///
    /// Returns
    /// -------
    /// result : RangeMOCIndex
    ///     The intersection of the two indexes.
    fn intersection(&self, other: &RangeMOCIndex) -> Self {
        RangeMOCIndex {
            moc: self.moc.intersection(&other.moc),
        }
    }

    /// The size of the ranges in bytes, minus any overhead.
    #[getter]
    fn nbytes(&self) -> u64 {
        self.moc.len() as u64 * 2 * u64::BITS as u64 / 8
    }

    /// The number of items in the index.
    #[getter]
    fn size(&self) -> u64 {
        self.moc.n_depth_max_cells()
    }

    /// The depth of the index.
    #[getter]
    fn depth(&self) -> u8 {
        self.moc.depth_max()
    }

    pub fn __setstate__(&mut self, state: &[u8]) -> PyResult<()> {
        // Deserialize the data contained in the PyBytes object
        // and update the struct with the deserialized values.
        // serde+bincode version:
        // *self = deserialize(state).map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

        let cell_moc: CellMOC<u64, Hpx<u64>> = from_json_aladin(
            std::str::from_utf8(state).map_err(|err| PyRuntimeError::new_err(err.to_string()))?,
        )
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        let reconstructed = RangeMOC::from_cells(
            cell_moc.depth_max(),
            cell_moc
                .into_cell_moc_iter()
                .map(|c| -> (u8, u64) { (c.depth, c.idx) }),
            None,
        );
        *self = RangeMOCIndex { moc: reconstructed };

        Ok(())
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        // Serialize the struct and return a PyBytes object
        // containing the serialized data.
        let mut serialized: Vec<u8> = Default::default();
        self.moc
            .clone()
            .into_range_moc_iter()
            .cells()
            .to_json_aladin(None, &mut serialized)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        // let serialized = serialize(&self).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let bytes = PyBytes::new(py, &serialized);
        Ok(bytes)
    }

    pub fn __reduce__(&self, py: Python) -> PyResult<(PyObject, PyObject, PyObject)> {
        let create = py
            .import("healpix_geo")?
            .getattr("nested")?
            .getattr("create_empty")?;
        let args = (self.moc.depth_max(),);
        let state = self.__getstate__(py)?;

        Ok((
            create.into_pyobject(py)?.unbind().into_any(),
            args.into_pyobject(py)?.unbind().into_any(),
            state.into_pyobject(py)?.unbind().into_any(),
        ))
    }

    /// Retrieve the cell ids from the index.
    ///
    /// Returns
    /// -------
    /// cell_ids : numpy.ndarray
    ///     The cell ids contained by the index.
    fn cell_ids<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyArray1<u64>>> {
        let cell_ids = Array1::from_iter(self.moc.flatten_to_fixed_depth_cells());

        Ok(PyArray1::from_owned_array(py, cell_ids))
    }

    /// Subset the index using positions
    ///
    /// Parameters
    /// ----------
    /// indexer : slice of int
    ///     The integer positions. Currently only supports slices.
    ///
    /// Returns
    /// -------
    /// subset : RangeMOCIndex
    ///     The resulting subset.
    fn isel<'a>(&self, py: Python<'a>, indexer: IndexKind<'a>) -> PyResult<Self> {
        match indexer {
            IndexKind::Slice(slice) => {
                let concrete_slice = slice
                    .as_positional_slice()?
                    .as_concrete(py, self.size() as isize)?;

                let subset = self.moc.slice(&concrete_slice)?;

                Ok(RangeMOCIndex { moc: subset })
            }
            IndexKind::Array(array) => {
                let subset = self.moc.subset(&array)?;

                Ok(RangeMOCIndex { moc: subset })
            }
        }
    }

    /// Subset the index using positions
    ///
    /// Parameters
    /// ----------
    /// indexer : slice of int or array-like of uint64
    ///     The cell ids or ranges of cell ids to find.
    ///
    /// Returns
    /// -------
    /// subset : RangeMOCIndex
    ///     The resulting subset.
    /// indexer : slice of int or array-like of uint64
    ///     The integer
    fn sel<'a>(&self, py: Python<'a>, indexer: IndexKind<'a>) -> PyResult<(IndexKind<'a>, Self)> {
        let depth = self.moc.depth_max();
        let range_sizes = self.moc.range_sizes();
        let offsets = range_offsets(range_sizes);

        match indexer {
            IndexKind::Slice(pyslice) => {
                // algorithm:
                // - compute the length of each internal range
                // - perform a cumulative sum to get the offsets for each range
                // - for each slice:
                //   - find all overlapping ranges
                //   - map the slice onto all overlapping ranges
                //   - assemble the sliced range and the resulting index
                //
                // Overlap check:
                //
                // (
                //   (s.start is None or s.start < r.end)
                //   and (s.stop is None or s.stop >= r.start)
                // )
                let slice = pyslice.as_label_slice()?;
                let (slices, ranges): (Vec<_>, Vec<_>) = self
                    .moc
                    .moc_ranges()
                    .iter()
                    .enumerate()
                    .filter_map(|(index, range)| slice.overlap(range, depth, offsets[index]))
                    .unzip();

                let cls = <ConcreteSlice as PyTypeInfo>::type_object(py);
                let joined_slice = ConcreteSlice::join(&cls, py, slices)?;

                let new_moc: RangeMOC<u64, Hpx<u64>> =
                    RangeMOC::new(self.moc.depth_max(), MocRanges::new_from(ranges));
                let new_index = RangeMOCIndex { moc: new_moc };

                Ok((IndexKind::Slice(joined_slice.as_pyslice(py)?), new_index))
            }
            IndexKind::Array(array) => {
                // algorithm:
                // - compute the length of each range
                // - compute range offsets
                // - for each value in the array:
                //   - find the range with value >= start and value <= end
                //   - if not found, raise
                //   - if found, compute the integer offset as range_offset + (value - start) / step
                let array = unsafe { array.as_array() }
                    .into_owned()
                    .into_dimensionality::<Ix1>()
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                let delta_depth = 29 - depth;
                let shift = delta_depth << 1;

                let ranges = self
                    .moc
                    .moc_ranges()
                    .par_iter()
                    .map(|x| Range {
                        start: x.start >> shift,
                        end: x.end >> shift,
                    })
                    .collect::<Vec<_>>();

                let (positions, cell_ids): (Vec<_>, Vec<_>) = array
                    .par_iter()
                    .map(|&hash| {
                        let range_index = ranges
                            .par_iter()
                            .position_first(|r| r.contains(&hash))
                            .ok_or(hash);

                        range_index
                            .map(|idx| {
                                let position = (hash - ranges[idx].start) + offsets[idx] as u64;

                                (position, hash)
                            })
                            .map_err(|err| {
                                PyKeyError::new_err(format!("Cannot find {hash} ({})", err))
                            })
                    })
                    .collect::<Result<Vec<(_, _)>, _>>()?
                    .into_iter()
                    .unzip();

                let new_moc: RangeMOC<u64, Hpx<u64>> = RangeMOC::from_fixed_depth_cells(
                    self.moc.depth_max(),
                    cell_ids.into_iter(),
                    None,
                );
                let new_index = RangeMOCIndex { moc: new_moc };

                Ok((
                    IndexKind::Array(PyArrayDyn::from_owned_array(
                        py,
                        Array1::from_iter(positions).into_dyn(),
                    )),
                    new_index,
                ))
            }
        }
    }

    /// Query by geometry
    ///
    /// Parameters
    /// ----------
    /// geometry : Bbox or shapely.Geometry
    ///     The geometry to query by. Supported are:
    ///     - Bbox for true bounding box queries (planar geometry)
    ///     - shapely objects for spherical geometry queries
    ///
    /// Returns
    /// -------
    /// slices : MultiConcreteSlice
    ///     The slices necessary for extract the subdomain.
    /// moc : RangeMOCIndex
    ///     The index for the queried cell ids.
    fn query<'py>(
        &self,
        py: Python<'py>,
        geometry: &Bound<'py, PyAny>,
    ) -> PyResult<(MultiConcreteSlice, Self)> {
        let depth = self.moc.depth_max();
        let layer = nested::get(depth);

        let geom = GeometryTypes::from_pyobject(py, geometry)?;

        let geometry_moc = match geom {
            GeometryTypes::Point(lon, lat) => {
                let hash = layer.hash(lon.rem_euclid(360.0).to_radians(), lat.to_radians());

                RangeMOC::from_fixed_depth_cells(depth, vec![hash].into_iter(), None)
            }
            GeometryTypes::LineString(coords) => {
                let hashes = coords
                    .into_iter()
                    .map(|(lon, lat)| {
                        layer.hash(lon.rem_euclid(360.0).to_radians(), lat.to_radians())
                    })
                    .collect::<Vec<u64>>();

                RangeMOC::from_fixed_depth_cells(depth, hashes.into_iter(), None)
            }
            GeometryTypes::Polygon(exterior, _interiors) => {
                let converted = exterior
                    .into_iter()
                    .map(|r| (r.0.rem_euclid(360.0).to_radians(), r.1.to_radians()))
                    .collect::<Vec<(_, _)>>();

                RangeMOC::from_polygon(&converted, false, depth, CellSelection::All)
            }
            GeometryTypes::Bbox(lon_min, lat_min, lon_max, lat_max) => RangeMOC::from_zone(
                lon_min.rem_euclid(360.0).to_radians(),
                lat_min.to_radians(),
                lon_max.rem_euclid(360.0).to_radians(),
                lat_max.to_radians(),
                depth,
                CellSelection::All,
            ),
        };

        let (slices, moc) = self.moc.index_intersection(geometry_moc)?;

        let multi_slice = MultiConcreteSlice { slices };

        Ok((multi_slice, RangeMOCIndex { moc }))
    }
}
