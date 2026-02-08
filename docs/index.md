# HEALPix-geo: Geo-algorithms for HEALPix

`healpix-geo` is a Python library which contains geo-specific algorithms for healpix. It is built on top of the [cdshealpix rust crate](https://crates.io/crates/cdshealpix), but unlike the [cdshealpix python bindings](https://cds-astro.github.io/cds-healpix-python/index.html) it does not require `astropy`.

- {doc}`HEALPix cell statistics <healpix/levels>` - Tab of levels and resolutions

## Install

::::{tab-set}

:::{tab-item} conda

```bash
conda install -c conda-forge healpix-geo
```

:::

:::{tab-item} pip

```bash
pip install healpix-geo
```

:::

:::{tab-item} pixi

```bash
pixi add healpix-geo
```

:::

:::{tab-item} uv

```bash
uv add healpix-geo
```

:::

:::{tab-item} From source

```bash
pixi run build-all-wheels # option 1: build wheels for all supported python versions
pixi run -e py313 build-wheel # option 2: build wheel for python=3.13

# then install the appropriate wheel:
pip install ./target/wheels/healpix-geo-<version>-cp313-cp313-<wheel-version>.whl
```

:::
::::

## Principal Functionalities

- **Ellipsoid Support** : `healpix-geo` supports reference ellipsoids such as WGS84 for optimal geodetic accuracy.

- **Performances** : Rust implementation with Python bindings for fast computation, including native multi-threading support.

- **Geo-algorithms** : Coverage calculations, neighbor search, distance calculations...

- **Easy Integration** :Compatible with NumPy, compatible with visualization tools such as Matplotlib.

## Start

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} Quickstart
:link: tutorials/quickstart
:link-type: doc

Learn basics in 10 minutes.
:::

:::{grid-item-card} User Guide
:link: user-guide/index
:link-type: doc

Fundamental concepts and detailed guides.
:::

:::{grid-item-card} Gallery of Examples
:link: examples/index
:link-type: doc

See concrete examples with visualisations.
:::

:::{grid-item-card} API Reference
:link: api
:link-type: doc

Complete documentation of all functions.
:::

::::

## Resources

- {doc}`HEALPix cell statistics <healpix/levels>` - Tab of levels and resolutions
- {doc}`reference-system` - Reference systems and ellipsoids
- {doc}`terminology` - Glossary of using terms

```{toctree}
---
maxdepth: 2
caption: Guide utilisateur
hidden: true
---
installation
user-guide/index
tutorials/index
examples/index
```

```{toctree}
---
maxdepth: 2
caption: Référence
hidden: true
---
api
healpix/index
reference-system
terminology
```
