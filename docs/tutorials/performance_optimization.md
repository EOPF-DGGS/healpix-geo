# Performance optimisations

This tutorial resents techniques for optimised performances of HEALPix calculations.

## Vectorisation

Always prefer vectorised operations :

```python
import numpy as np
from healpix_geo.nested import lonlat_to_healpix

# Good : vectorised
lon = np.random.uniform(-180, 180, 1000000)
lat = np.random.uniform(-90, 90, 1000000)
ipix = lonlat_to_healpix(lon, lat, depth=10, ellipsoid="WGS84")

# Bad : loop
ipix_list = []
for i in range(len(lon)):
    ipix_list.append(lonlat_to_healpix(lon[i], lat[i], 10, "WGS84"))
```

## Multi-threading

The `num_threads` parameter controls parallel execution.

```python
# Automatic (use all available CPU cores)
ipix = lonlat_to_healpix(lon, lat, depth=10, ellipsoid="WGS84", num_threads=0)

# Use 4 threads
ipix = lonlat_to_healpix(lon, lat, depth=10, ellipsoid="WGS84", num_threads=4)

# Sequential execution (single thread)
ipix = lonlat_to_healpix(lon, lat, depth=10, ellipsoid="WGS84", num_threads=1)
```

:::{tip}

- num_threads=0 uses all available CPU cores.
- num_threads=1 disables parallelism.
- num_threads>1 uses the specified number of threads.
:::
