# Terminology

```{glossary}

ellipsoid-like
    An ellipsoid specification. Can be either:

    - the name of the ellipsoid as a {py:class}`str`. For a complete list of known ellipsoids, see [the `geodesy` crate](https://github.com/busstoptaktik/geodesy/blob/f9090b8c91f401892a93979f100fa4d987eb0836/src/ellipsoid/constants.rs#L6-L54).
    - a {py:class}`dict`, with either a `"radius"` item for spheres or `"semimajor_axis"` and `"inverse_flattening"` for ellipsoids. All items need to be {py:class}`float`s.
    - a class with a `"radius"` attribute for spheres or `"semimajor_axis"` and `"inverse_flattening"` attributes for ellipsoids. All attributes need to be {py:class}`float`s.

    If an object or {py:class}`dict` could be interpreted as both a sphere and an ellipsoid, the ellipsoid will be preferred.
```
