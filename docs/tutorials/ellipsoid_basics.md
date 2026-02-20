# Ellipsoids basics

This tutorial explain why and how to use ellipsoids of reference in `healpix-geo`.

## Supported Ellipsoids

```python
import numpy as np
from healpix_geo.nested import lonlat_to_healpix

# Test Point
lon, lat = np.array([0.0]), np.array([45.0])
depth = 10

ellipsoids = {
    "sphere",
    "WGS84",
    "GRS80",
    "WGS72",
}

print(f"HEALPix cell at point (0°, 45°) at depth={depth}:\n")
for ellipsoid, description in ellipsoids.items():
    ipix = lonlat_to_healpix(lon, lat, depth, ellipsoid=ellipsoid)
    print(f"{ellipsoid:10s}: {ipix[0]:10d}  # {description}")
```

:::{seealso}
For a list of supported named ellipsoids, see [here](https://github.com/busstoptaktik/geodesy/blob/f9090b8c91f401892a93979f100fa4d987eb0836/src/ellipsoid/constants.rs#L6-L54).
:::

### Which ellipsoid choose ?

| Ellipsoid | When to use it                                          |
| --------- | ------------------------------------------------------- |
| **WGS84** | **Recommended by default** - International Standard GPS |
| GRS80     | Almost equal to WGS84                                   |
| WGS72     | Only for compatibility with old data                    |
| sphere    | Quick tests                                             |

:::{important}
**Golden rule**: Use **WGS84** unless you have a specific reason to use a different ellipsoid.
:::

## Typical use case

### GPS data

```python
# GPS trajectory data
gps_points = np.array(
    [
        [2.3522, 48.8566],  # Paris
        [2.3532, 48.8576],  # 100m further
        [2.3542, 48.8586],  # 100m further
    ]
)

lon_gps = gps_points[:, 0]
lat_gps = gps_points[:, 1]
depth = 16  # High resolution

# Always use WGS84 for GPS
ipix = lonlat_to_healpix(lon_gps, lat_gps, depth, ellipsoid="WGS84")
print(f"HEALPix cells of GPS trajectory:")
print(ipix)
```

## Next steps

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} Reference systems
:link: ../reference-system
:link-type: doc

:::

:::{grid-item-card} Advanced conversion
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
