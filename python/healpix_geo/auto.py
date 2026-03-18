from dataclasses import dataclass
from types import ModuleType
from typing import Literal

import numpy as np
import numpy.typing as npt

from healpix_geo.typing import EllipsoidLike


def _dispatch_module(indexing_scheme: str) -> ModuleType:
    from healpix_geo import nested, ring, zuniq

    modules = {
        "nested": nested,
        "ring": ring,
        "zuniq": zuniq,
    }

    module = modules.get(indexing_scheme)
    if module is None:
        raise ValueError(
            f"unknown indexing scheme: {indexing_scheme}."
            f" Available are: {', '.join(modules.keys())}"
        )

    return module


@dataclass(frozen=True)
class Grid:
    level: int | None
    """The refinement level of the grid."""

    indexing_scheme: Literal["nested", "ring", "zuniq"] = "nested"
    """The indexing scheme of the grid."""

    ellipsoid: EllipsoidLike = "sphere"
    """The reference ellipsoid of the grid."""

    def _as_params(self):
        params = {"ellipsoid": self.ellipsoid}
        if self.indexing_scheme != "zuniq":
            params["depth"] = self.level

        return params


def healpix_to_lonlat(
    ipix: npt.NDArray[np.uint64], grid: Grid, *, num_threads: int = 0
) -> (npt.NDArray[np.float64], npt.NDArray[np.float64]):
    module = _dispatch_module(grid.indexing_scheme)
    params = grid._as_params()
    return module.healpix_to_lonlat(ipix, num_threads=num_threads, **params)


def lonlat_to_healpix(
    lon: npt.NDArray[np.float64],
    lat: npt.NDArray[np.float64],
    grid: Grid,
    *,
    num_threads: int = 0,
) -> npt.NDArray[np.uint64]:
    module = _dispatch_module(grid.indexing_scheme)
    params = grid._as_params()
    return module.lonlat_to_healpix(lon, lat, num_threads=num_threads, **params)


def vertices(
    ipix: npt.NDArray[np.uint64], grid: Grid, *, num_threads: int = 0
) -> (npt.NDArray[np.float64], npt.NDArray[np.float64]):
    module = _dispatch_module(grid.indexing_scheme)
    params = grid._as_params()

    return module.vertices(ipix, num_threads=num_threads, **params)


def kth_neighbourhood(
    ipix: npt.NDArray[np.uint64], grid: Grid, *, ring: int, num_threads: int = 0
) -> npt.NDArray[np.int64]:
    module = _dispatch_module(grid.indexing_scheme)
    params = grid._as_params()

    return module.kth_neighbourhood(ipix, ring=ring, num_threads=num_threads, **params)
