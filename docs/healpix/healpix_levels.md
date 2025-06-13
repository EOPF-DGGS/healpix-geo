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

```{jupyter-execute}
import pyresample
from pyresample.geometry import AreaDefinition
# import pyresample.spherical_geometry
from pyresample import spherical
```

## Packages used

- geopandas
- shapely
- cds-healpix-python
- pyresample (could be replaced with )
- tabulate (for markdown output)

## Levels until 25

```{jupyter-execute}
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import cdshealpix as cds
import astropy.units as u

from shapely.geometry import Polygon, Point

# A=4πr2

# authalic 6371 km
# eqatorial 6378
radius = 6371

# 510064472 km²
# 510072000
earth_a = 510072000
earth_a_sq = math.pi * 4 * (radius * radius)

print(earth_a , earth_a_sq)
print(earth_a / 12)

def polygons_for_ipix(cell_ids, depth):
    lon, lat = cds.nested.vertices(cell_ids, depth, step=3)
    lon = lon.wrap_at(180 * u.deg)
    polygons = []
    for i, cell_id in enumerate(cell_ids):
        paired_2d = np.column_stack((lon.deg[i], lat.deg[i]))
        points = [ Point(x, y) for (x, y) in paired_2d ]
        poly = Polygon(points)
        polygons.append({'geometry': poly, 'ipix': cell_id})
    return gpd.GeoDataFrame(polygons, crs=4326)

zone_col = []

for l in range(0, 26):
    max_cells = 12 * np.power(4, l)

    g = polygons_for_ipix([1], l)
    p = g.geometry[0].centroid



    laea_crs_def = f'+proj=laea +lat_0={p.y} +lon_0={p.y} +x_0=4321000 +y_0=3210000 +ellps=GRS80 +units=m +no_defs'
    heal_crs_def = '+proj=healpix'

    laea_proj = g.to_crs(laea_crs_def)

    heal_proj = g.to_crs(heal_crs_def)

    coords = list(g.geometry[0].exterior.coords)[:-1]
    coords.reverse()

    sph_poly = spherical.SphPolygon(np.deg2rad( np.array(coords)), radius=6371)

    sphere_area_km2 = sph_poly.area()
    sphere_area_m2 = sphere_area_km2 * 1000000

    laea_projected_area_m2 = laea_proj.geometry[0].area
    laea_projected_area_km2 = laea_projected_area_m2 / 1000000

    heal_projected_area_m2 = heal_proj.geometry[0].area
    heal_projected_area_km2 = heal_projected_area_m2 / 1000000

    subdivided_area_km2 = earth_a / max_cells
    subdivided_area_m2 = subdivided_area_km2 * 1000000

    zone_col.append({'level': l,
                     'num_cells': max_cells,
                     # 'heal_projected_area_m2' : np.round(heal_projected_area_m2, 4),
                     'laea_projected_area_m2' : np.round(laea_projected_area_m2, 4),
                     'sphere_area_m2' : np.round(sphere_area_m2, 4),
                     'subdivided_area_m2' : np.round(subdivided_area_m2, 4),

                     # 'heal_projected_area_km2': np.round(heal_projected_area_km2, 4),
                     'laea_projected_area_km2': np.round(laea_projected_area_km2, 4),
                     'sphere_area_km2': np.round(sphere_area_km2, 4),
                     'subdivided_area_km2': np.round(subdivided_area_km2, 4),

                    })

pd.options.display.float_format = '{:20,.3f}'.format

level_info = pd.DataFrame(zone_col)
level_info
```

|     | level | num_cells                  | laea_projected_area_m2 | sphere_area_m2         | subdivided_area_m2     | laea_projected_area_km2 | sphere_area_km2 | subdivided_area_km2 |
| --: | ----: | :------------------------- | :--------------------- | :--------------------- | :--------------------- | :---------------------- | :-------------- | :------------------ |
|   0 |     0 | 12.000                     | 41,469,891,629,958.164 | 42,062,053,840,411.023 | 42,506,000,000,000.000 | 41,469,891.630          | 42,062,053.840  | 42,506,000.000      |
|   1 |     1 | 48.000                     | 10,565,571,996,934.770 | 10,581,664,199,360.223 | 10,626,500,000,000.000 | 10,565,571.997          | 10,581,664.199  | 10,626,500.000      |
|   2 |     2 | 192.000                    | 2,649,873,227,961.956  | 2,660,167,722,993.080  | 2,656,625,000,000.000  | 2,649,873.228           | 2,660,167.723   | 2,656,625.000       |
|   3 |     3 | 768.000                    | 661,468,605,823.771    | 664,340,488,270.517    | 664,156,250,000.000    | 661,468.606             | 664,340.488     | 664,156.250         |
|   4 |     4 | 3,072.000                  | 165,312,492,831.922    | 166,048,419,803.430    | 166,039,062,500.000    | 165,312.493             | 166,048.420     | 166,039.062         |
|   5 |     5 | 12,288.000                 | 41,324,801,815.315     | 41,509,886,436.981     | 41,509,765,625.000     | 41,324.802              | 41,509.886      | 41,509.766          |
|   6 |     6 | 49,152.000                 | 10,330,994,456.104     | 10,377,334,018.246     | 10,377,441,406.250     | 10,330.995              | 10,377.334      | 10,377.441          |
|   7 |     7 | 196,608.000                | 2,582,735,770.530      | 2,594,324,921.505      | 2,594,360,351.562      | 2,582.736               | 2,594.325       | 2,594.360           |
|   8 |     8 | 786,432.000                | 645,683,140.621        | 648,580,694.120        | 648,590,087.891        | 645.683                 | 648.581         | 648.590             |
|   9 |     9 | 3,145,728.000              | 161,420,735.047        | 162,145,140.291        | 162,147,521.973        | 161.421                 | 162.145         | 162.148             |
|  10 |    10 | 12,582,912.000             | 40,355,180.631         | 40,536,283.054         | 40,536,880.493         | 40.355                  | 40.536          | 40.537              |
|  11 |    11 | 50,331,648.000             | 10,088,794.962         | 10,134,070.547         | 10,134,220.123         | 10.089                  | 10.134          | 10.134              |
|  12 |    12 | 201,326,592.000            | 2,522,198.728          | 2,533,517.565          | 2,533,555.031          | 2.522                   | 2.534           | 2.534               |
|  13 |    13 | 805,306,368.000            | 630,549.681            | 633,379.427            | 633,388.758            | 0.630                   | 0.633           | 0.633               |
|  14 |    14 | 3,221,225,472.000          | 157,637.420            | 158,344.821            | 158,347.189            | 0.158                   | 0.158           | 0.158               |
|  15 |    15 | 12,884,901,888.000         | 39,409.355             | 39,586.133             | 39,586.797             | 0.039                   | 0.040           | 0.040               |
|  16 |    16 | 51,539,607,552.000         | 9,852.339              | 9,896.389              | 9,896.699              | 0.010                   | 0.010           | 0.010               |
|  17 |    17 | 206,158,430,208.000        | 2,463.085              | 2,474.242              | 2,474.175              | 0.003                   | 0.003           | 0.003               |
|  18 |    18 | 824,633,720,832.000        | 615.771                | 618.633                | 618.544                | 0.001                   | 0.001           | 0.001               |
|  19 |    19 | 3,298,534,883,328.000      | 153.943                | 154.586                | 154.636                | 0.000                   | 0.000           | 0.000               |
|  20 |    20 | 13,194,139,533,312.000     | 38.486                 | 38.791                 | 38.659                 | 0.000                   | 0.000           | 0.000               |
|  21 |    21 | 52,776,558,133,248.000     | 9.621                  | 9.662                  | 9.665                  | 0.000                   | 0.000           | 0.000               |
|  22 |    22 | 211,106,232,532,992.000    | 2.405                  | 2.307                  | 2.416                  | 0.000                   | 0.000           | 0.000               |
|  23 |    23 | 844,424,930,131,968.000    | 0.601                  | 0.865                  | 0.604                  | 0.000                   | 0.000           | 0.000               |
|  24 |    24 | 3,377,699,720,527,872.000  | 0.150                  | 0.288                  | 0.151                  | 0.000                   | 0.000           | 0.000               |
|  25 |    25 | 13,510,798,882,111,488.000 | 0.038                  | 0.000                  | 0.038                  | 0.000                   | 0.000           | 0.000               |

```{jupyter-execute}
# !pip install tabulate
# import tabulate

for l in level_info.map('{:20,.3f}'.format).to_markdown().split("\n"):
    print(l)
```
