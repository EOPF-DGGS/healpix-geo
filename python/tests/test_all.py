import healpix_geo


def test_sum_as_string():
    assert healpix_geo.sum_as_string(1, 1) == "2"
