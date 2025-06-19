# Towards a lookup list for HEALPIX levels

![](healpix_global_indexing.png)

## Calculating cell areas for HEALPIX levels

TODO: replace spherical calculation (here from pyresample) with [spherely](https://github.com/benbovy/spherely)

TODO: generate larger subset of cell per level and compute average cell size

TODO: compare cds-healpix and healpy cell geometries and sizes

## Overview

- generate a cell boundary coordinates (via ipix number) from cds-healpix
- creates a shapely Polygon
- calculates area in different ways: projected centered LAEA, spherical area, "guesstimate" subdivision from given Earth's surface by max number of cells in given level

## Packages used

- geopandas
- shapely
- cds-healpix-python
- pyresample (could be replaced with )
- tabulate (for markdown output)

## Cell stats

````{table} Healpix Level Overview ($R = 6371km$)

```{include} healpix_levels_table.md
```

````
