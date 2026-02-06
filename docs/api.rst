API reference
=============

This page contains the complete documentation of all the functions and classes available in `healpix-geo`.

Overview
============== 

`healpix-geo` is organised in several modules following the **HEALPix indexing scheme** used :

- :mod:`healpix_geo.nested` : nested scheme
- :mod:`healpix_geo.ring` : ring scheme 
- :mod:`healpix_geo.zuniq` : Zuniq scheme for Multi-Order Coverage (MOC)

.. tip::
   **For most of the applications**, use module :mod:`healpix_geo.nested`. It offers the best support for hierarchical operations.

Guide of module choice
=========================

Choose your module following your needs :

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - Module
     - When to use it
     - Principal functionnalities
   * - **nested**
     - - General applications 
       - Hierarchical requests
       - Multi-resolution
     - - Coordinates conversions 
       - Coverage
       - Hierarchy
       - Distances
   * - **ring**
     - - Compatibility legacy
       - Specific order requested
     - - Coordinates conversions 
       - Distances
       - Limited neighbours
   * - **zuniq**
     - - Work with MOC
       - Interoperability
     - - Conversions nested ↔ zuniq
       - Cell Coordinates


Module nested 
==============

The module :mod:`healpix_geo.nested` gives functions for the nested indexing scheme.

.. currentmodule:: healpix_geo.nested

Coordinates Conversions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Conversions between geographic coordinates and HEALPix indices.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   healpix_to_lonlat
   lonlat_to_healpix
   vertices

.. seealso::
   Complete tutorial : :doc:`tutorials/coordinate_conversion`

Hierarchy and neighborhood
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Navigation in the hierarchical structure of HEALPix.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   kth_neighbourhood
   zoom_to
   siblings

**Quick Examples** :

.. code-block:: python

   from healpix_geo.nested import kth_neighbourhood, zoom_to

   # Direct neighbours
   neighbours = kth_neighbourhood(ipix, depth=8, k=1)

       # Parents and children
   children = zoom_to(parent_ipix, parent_depth, parent_depth + 1)

.. seealso::
   Complete tutorial : :doc:`user-guide/hierarchical_indexing`

Cover Requests
~~~~~~~~~~~~~~~

Find all the cells which intersect a region.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   zone_coverage
   box_coverage
   polygon_coverage
   cone_coverage
   elliptical_cone_coverage
   internal_boundary

.. seealso::
   Complete tutorial : :doc:`tutorials/coverage_queries`

Distance Calculations
~~~~~~~~~~~~~~~~~~~~~~~

Calculate distances between HEALPix cells.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   angular_distances


Indexes and data structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Classes to manipulate HEALPix cell sets.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   RangeMOCIndex

.. seealso::
   Complete tutorial : :doc:`tutorials/working_with_moc`

Module ring
===========

The module :mod:`healpix_geo.ring` gives for the ring indexation scheme.

.. currentmodule:: healpix_geo.ring

.. note::
   The ring scheme is principally given for **compatibility**. For ne applications, prefer :mod:`healpix_geo.nested`.

Coordinates Conversions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   healpix_to_lonlat
   lonlat_to_healpix
   vertices

Hierarchy
~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   kth_neighbourhood

Distance Calculations
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   angular_distances

Module zuniq
============

The module :mod:`healpix_geo.zuniq` gives functions for the zuniq scheme, utilised for MOC (Multi-Order Coverage).

.. currentmodule:: healpix_geo.zuniq

Conversions
~~~~~~~~~~~

Conversions between schemes nested et zuniq.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   from_nested
   to_nested

**Quick Example** :

.. code-block:: python

   from healpix_geo.zuniq import from_nested, to_nested

   # nested → zuniq
   zuniq_id = from_nested(ipix, depth)

   # zuniq → nested
   ipix, depth = to_nested(zuniq_id)

Coordinates Conversions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   healpix_to_lonlat
   lonlat_to_healpix
   vertices

.. seealso::
   Tutoriel MOC : :doc:`tutorials/working_with_moc`

Common Parameters
==================

healpix-geo functions shares common parameters :

Geographical coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~~

- **lon** (*array-like*) : Longitude(s) in degrees [-180, 180]
- **lat** (*array-like*) : Latitude(s) in degrees [-90, 90]

HEALPix indices
~~~~~~~~~~~~~~~

- **ipix** (*array-like of int*) : Indice(s) of HEALPix cell
- **depth** (*int*) : resolution level [0, 29]

  - 0 = coarser (12 cells)
  - 29 = finer (~0.2 mm)
  - See :doc:`healpix/levels` pour le tableau complet

Ellipsoids
~~~~~~~~~~~

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
~~~~~~~~~~~~~~~

- Functions returns **numpy arrays**
- Coordinates are always in **decimal degrees**
- Indices are always **unsigned integers** (uint64)

Vectorisation
~~~~~~~~~~~~~

All functions are **vectorized** and accept:

- Scalars (automatically converted to arrays)
- 1D NumPy arrays
- Multidimensional NumPy arrays (depending on the function)


