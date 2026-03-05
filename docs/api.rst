API reference
=============

This page contains the complete documentation of all the functions and classes available in `healpix-geo`.

Overview
========

`healpix-geo` is organised in several modules following the **HEALPix indexing scheme** used :

- `healpix_geo.nested` : nested scheme
- `healpix_geo.ring` : ring scheme
- `healpix_geo.zuniq` : Zuniq scheme for Multi-Order Coverage (MOC)

.. tip::
   **For most of the applications**, use module `healpix_geo.nested`. It offers the best support for hierarchical operations.


Module nested
==============

The module `healpix_geo.nested` gives functions for the nested indexing scheme.

.. currentmodule:: healpix_geo.nested

Coordinates Conversions
~~~~~~~~~~~~~~~~~~~~~~~~

Conversions between geographic coordinates and HEALPix indices.

.. autosummary::
   :toctree: generated/

   healpix_to_lonlat
   lonlat_to_healpix
   vertices

.. seealso::
   Tutorial complete : :doc:`tutorials/coordinate_conversion`

Hierarchy and neighborhood
~~~~~~~~~~~~~~~~~~~~~~~~~~

Navigation in the hierarchical structure of HEALPix.

.. autosummary::
   :toctree: generated/

   kth_neighbourhood
   zoom_to
   siblings

.. seealso::
   Complete tutorial : :doc:`user-guide/hierarchical_indexing`

Cover Requests
~~~~~~~~~~~~~~~

Find all the cells which intersect a region.

.. autosummary::
   :toctree: generated/

   zone_coverage
   box_coverage
   polygon_coverage
   cone_coverage
   elliptical_cone_coverage
   internal_boundary


.. seealso::
   Complete tutorial : :doc:`tutorials/coverage_queries`

Distance Calculations
~~~~~~~~~~~~~~~~~~~~~

Calculate distances between HEALPix cells.

.. autosummary::
   :toctree: generated/

   angular_distances


Indexes and data structure
~~~~~~~~~~~~~~~~~~~~~~~~~~

Classes to manipulate HEALPix cell sets.

.. autosummary::
   :toctree: generated/

   RangeMOCIndex


Module ring
===========

The module `healpix_geo.ring` gives for the ring indexation scheme.

.. currentmodule:: healpix_geo.ring

.. note::
   The ring scheme is principally given for **compatibility**. For new applications, prefer `healpix_geo.nested`.

Coordinates Conversions
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   healpix_to_lonlat
   lonlat_to_healpix
   vertices

Hierarchy
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   kth_neighbourhood

Distance Calculations
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   angular_distances

Module zuniq
============

The module `healpix_geo.zuniq` gives functions for the zuniq scheme, utilised for MOC (Multi-Order Coverage).

.. currentmodule:: healpix_geo.zuniq

Conversions
~~~~~~~~~~~

Conversions between schemes nested et zuniq.

.. autosummary::
   :toctree: generated/

   from_nested
   to_nested

.. seealso::
   Complete tutorial : :doc:`tutorials/coordinate_conversion`


Coordinates Conversions
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   healpix_to_lonlat
   lonlat_to_healpix
   vertices

Helpers
~~~~~~~
.. currentmodule:: healpix_geo

.. autosummary::
   :toctree: generated/

   geometry.Bbox
   slices.Slice
   slices.ConcreteSlice
   slices.MultiConcreteSlice

Common Parameters
==================

healpix-geo functions shares common parameters :

Geographical coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~

- **lon** (*array-like*) : Longitude(s) in degrees [-180, 180]
- **lat** (*array-like*) : Latitude(s) in degrees [-90, 90]

HEALPix indices
~~~~~~~~~~~~~~~

- **ipix** (*array-like of int*) : Indice(s) of HEALPix cell
- **depth** (*int*) : resolution level [0, 29]

  - 0 = coarser (12 cells)
  - 29 = finer (~1.2 cm)
  - See :doc:`healpix/levels` for the complete table

Ellipsoids
~~~~~~~~~~

- **ellipsoid** (*str*) : Ellipsoidal Model to use

  - ``"WGS84"`` : Standard GPS
  - ``"GRS80"`` : Geodesic System of reference 1980
  - ``"WGS72"`` : Old standard GPS
  - ``"sphere"`` : Perfect sphere

.. tip::
   **Always use** ``ellipsoid="WGS84"`` for real geospatial applications.
   See :doc:`tutorials/ellipsoid_basics` for more details.


Conventions
===========

Return types
~~~~~~~~~~~~

- Functions returns **numpy arrays**
- Coordinates are always in **decimal degrees**
- Indices are always **unsigned integers** (uint64)

Vectorisation
~~~~~~~~~~~~~

All functions are **vectorized** and accept:

- Scalars (automatically converted to arrays)
- 1D NumPy arrays
- Multidimensional NumPy arrays (depending on the function)
