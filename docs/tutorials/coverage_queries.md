# Cover requests

This tutorial explains how to find all the HEALPix cells which intersect a geographic region.

## Type of requests

### 1. Cone Coverage

Find all cells within a given radius around a point.
```python
import numpy as np
from healpix_geo.nested import cone_coverage

# Center and radius
lon_center = 2.3522  # Paris
lat_center = 48.8566
radius_deg = 1.0  # radius in degrees (~111 km)
depth = 8

cells = cone_coverage(
    (lon_center, lat_center), radius_deg,
    depth, ellipsoid="WGS84"
)

print(f"Number of cells in the radius: {len(cells)}")
```

### 2. Box Coverage

Find all cells in a box coverage.

```python
from healpix_geo.nested import box_coverage

# Coverage box
lon_min, lat_min = 2.0, 48.5
lon_max, lat_max = 3.0, 49.0

center = (
    0.5 * (lon_min + lon_max),
    0.5 * (lat_min + lat_max),
)

size = (
    lon_max - lon_min,  
    lat_max - lat_min,  
)

angle = 0.0   
depth = 8

cells = box_coverage(center, size, angle, depth, ellipsoid="WGS84", flat=True)

print(f"Cells number : {len(cells)}")
```

### 3. Polygon coverage

Find all cells in a polygon coverage.

```python
from healpix_geo.nested import polygon_coverage
import numpy as np

vertices = np.array([[2.0, 48.5], [3.0, 48.5], [2.5, 49.0]])

depth = 8

cells = polygon_coverage(
    vertices,depth,
    ellipsoid="WGS84",
    flat=True
)

print(f"Cells in the polygon : {len(cells)}")
```

## Resume

### Principal functions

| Function | Usage | Key parameters |
|----------|-------|-----------------|
| `zone_coverage` | Zone request | bbox, depth |
| `cone_coverage` | Circular request | center, radius, depth |
| `box_coverage` | Rectangular request | center, size, angle, depth |
| `elliptical_cone_coverage` | Ellipticall cone request| center, ellipse_geometry, position_angle, depth |
| `polygon_coverage` | Polygonal request| vertices, depth |
| `internal_boundary` | Boundaries | depth, ipix |

## Next Steps

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} Working MOC
:link: working_with_moc
:link-type: doc
:::

:::{grid-item-card} Performance
:link: performance_optimization
:link-type: doc

:::

:::{grid-item-card} Hierarchy
:link: ../user-guide/hierarchical_indexing
:link-type: doc
:::

:::{grid-item-card} Api reference
:link: api
:link-type: doc

:::

::::
