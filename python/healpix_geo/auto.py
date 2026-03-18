from dataclasses import dataclass
from typing import Literal

from healpix_geo.typing import EllipsoidLike


@dataclass(frozen=True)
class Grid:
    level: int | None
    """The refinement level of the grid."""

    indexing_scheme: Literal["nested", "ring", "zuniq"] = "nested"
    """The indexing scheme of the grid."""

    ellipsoid: EllipsoidLike = "sphere"
    """The reference ellipsoid of the grid."""
