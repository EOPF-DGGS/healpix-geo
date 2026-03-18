import pytest

import healpix_geo
from healpix_geo import auto


@pytest.mark.parametrize(
    ["scheme", "expected"],
    (
        ("nested", healpix_geo.nested),
        ("ring", healpix_geo.ring),
        ("zuniq", healpix_geo.zuniq),
    ),
    ids=["nested", "ring", "zuniq"],
)
def test_dispatch_module(scheme, expected):
    actual = auto._dispatch_module(scheme)

    assert actual is expected
