# Installing

`healpix-geo` is available on `conda-forge`:

```{sh}
conda install -c conda-forge healpix-geo
pixi add healpix-geo
```

or on PyPI:

```{sh}
# pip
pip install healpix-geo
# uv
uv add healpix-geo
```

## From source

To install from source, run

```{sh}
pixi run build-all-wheels  # build wheels for all supported python versions
pixi run -e py313 build-wheel  # build wheel for python=3.13
```

then install the appropriate wheel:

```{sh}
pip install ./target/wheels/healpix-geo-<version>-cp313-cp313-<wheel-version>.whl
```
