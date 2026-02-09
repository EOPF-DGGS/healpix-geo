# Quickstart

This tutorial presents you the fundamental concepts of `healpix-geo` through practical examples.

## First example : Coordinates conversion

The most common task is to convert geographic coordinates into HEALPix cell identifiers:

```python
import numpy as np
from healpix_geo.nested import lonlat_to_healpix, healpix_to_lonlat

# Define some points of interest (longitude, latitude in degrees)
paris = (2.3522, 48.8566)
tokyo = (139.6917, 35.6895)
new_york = (-74.0060, 40.7128)

lon = np.array([paris[0], tokyo[0], new_york[0]])
lat = np.array([paris[1], tokyo[1], new_york[1]])

# Convert in HEALPix cells level 8
depth = 8
ipix = lonlat_to_healpix(lon, lat, depth, ellipsoid="WGS84")

print(f"HEALPix cells (depth={depth}):")
print(f"  Paris: {ipix[0]}")
print(f"  Tokyo: {ipix[1]}")
print(f"  New York: {ipix[2]}")
```

:::{note}
Cell indices are unique integers that identify each cell. Each city is in a different cell at level 8.
:::

## Understand `depth` parameter

The `depth` parameter (or level) controls the tiling resolution :

- `depth=0` : 12 cells around the world (~4700 km per cell)
- `depth=8` : 3 145 728 cells (~18 km per cell)
- `depth=12` : 201 326 592 cells (~1.1 km per cell)

:::{seealso}
See {doc}`HEALPix cell statistics <healpix/levels>` for a complete table of levels with their resolutions.
:::

### Example: Same point at different resolutions

```python
# Same point at different resolutions
lon, lat = np.array([2.3522]), np.array([48.8566])  # Paris

for depth in [0, 4, 8, 12]:
    ipix = lonlat_to_healpix(lon, lat, depth, ellipsoid="WGS84")
    print(f"Depth {depth:2d}: cell {ipix[0]:10d}")
```

:::{tip}
**General rule**: Increasing the `depth` by 1 multiplies the number of cells by 4 and divides the cell size by 2.
:::

## Reverse conversion: cell → coordinates

Recover the coordinates of the **center** of a cell:

```python
# Obtain coordinates of the cell center
ipix = np.array([349440])  # Paris
depth = 8

lon_center, lat_center = healpix_to_lonlat(ipix, depth, ellipsoid="WGS84")
print(f"Cell center {ipix[0]}:")
print(f"  Longitude: {lon_center[0]:.4f}°")
print(f"  Latitude:  {lat_center[0]:.4f}°")
```

:::{note}
The center coordinates may be slightly different from the original coordinates, because we have "rounded" to the nearest cell.
:::

## Research of neighbour

HEALPix makes it easy to find **neighboring** cells:

```python
from healpix_geo.nested import kth_neighbourhood

ipix = np.array([100])
depth = 5
k = 1  # direct neighbour (distance 1)

neighbours = kth_neighbourhood(ipix, depth, k)
print(f"Direct neighbourhoods of the cell {ipix[0]}:")
print(neighbours)
```

:::{seealso}
For more details on hierarchy and neighbourhood, see {doc}`../user-guide/hierarchical_indexing`.
:::

## Hierarchic navigation

```python
from healpix_geo.nested import zoom_to

parent_ipix = np.array([100])
parent_depth = 5

# Obtain the 4 children cells
children_ipix = zoom_to(parent_ipix, parent_depth, parent_depth + 1)

print(f"Parent cell:  {parent_ipix[0]} (depth={parent_depth})")
print(f"Children cell: {children_ipix} (depth={parent_depth + 1})")
```

## Next Steps

Now you know basics, you can explore :

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} Ellipsoids
:link: ellipsoid_basics
:link-type: doc

:::

:::{grid-item-card} Advanced Conversions
:link: coordinate_conversion
:link-type: doc

:::

:::{grid-item-card} Cover Requests
:link: coverage_queries
:link-type: doc

:::

:::{grid-item-card} Performance
:link: performance_optimization
:link-type: doc

:::

::::
