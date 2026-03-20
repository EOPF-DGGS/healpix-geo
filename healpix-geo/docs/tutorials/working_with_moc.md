# Work with MOC

This tutorial explains how to use Multi-Order Coverage maps (MOC) to represent efficiently complex regions.

## zuniq scheme

The MOC is using the **zuniq** scheme which encode depth + ipix in one integer 64-bit.

```python
from healpix_geo.zuniq import from_nested, to_nested
import numpy as np

# Cells nested
ipix_nested = np.array([100, 200, 300])
depth = 8

# Convert in zuniq
zuniq_ids = from_nested(ipix_nested, depth)
print(f"Zuniq IDs: {zuniq_ids}")

# Convert return in nested
ipix_back, depth_back = to_nested(zuniq_ids)
print(f"Nested: depth={depth_back}, ipix={ipix_back}")
```

## Creates a MOC

```python
from healpix_geo.nested import RangeMOCIndex

moc = RangeMOCIndex
```
