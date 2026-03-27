import pytest

from healpix_geo import slices

missing = object()


class TestSlice:
    @pytest.mark.parametrize("start", [0, None, 10])
    @pytest.mark.parametrize("stop", [3, None, 10])
    @pytest.mark.parametrize("step", [1, None, 2, missing])
    def test_init(self, start, stop, step):
        if step is missing:
            vals = (start, stop)
            step = None
        else:
            vals = (start, stop, step)

        actual = slices.Slice(*vals)

        assert actual.start == start
        assert actual.stop == stop
        assert actual.step == step

    @pytest.mark.parametrize("start", [0, None, 10])
    @pytest.mark.parametrize("stop", [3, None, 10])
    @pytest.mark.parametrize("step", [1, None, 2, -1])
    def test_repr(self, start, stop, step):
        slice_ = slices.Slice(start, stop, step)
        actual = repr(slice_)

        expected = repr(slice(start, stop, step)).title()

        assert actual == expected

    @pytest.mark.parametrize(
        "pyslice",
        (
            slice(None),
            slice(None, 2),
            slice(3, None),
            slice(None, None, -1),
            slice(10, None, -1),
            slice(3, -2),
        ),
    )
    def test_from_pyslice(self, pyslice):
        actual = slices.Slice.from_pyslice(pyslice)

        assert actual.start == pyslice.start
        assert actual.start == pyslice.start
        assert actual.start == pyslice.start

    @pytest.mark.parametrize(
        "pyslice",
        (
            slice(None),
            slice(None, 2),
            slice(3, None),
            slice(None, None, -1),
            slice(10, None, -1),
            slice(3, -2),
        ),
    )
    @pytest.mark.parametrize("size", (3, 8, 13))
    def test_roundtrip_pyslice(self, pyslice, size):
        slice_ = slices.Slice.from_pyslice(pyslice)

        actual = slice_.as_pyslice()
        assert actual == pyslice

    @pytest.mark.parametrize(
        "pyslice",
        (
            slice(None),
            slice(None, 2),
            slice(3, None),
            slice(None, None, -1),
            slice(10, None, -1),
            slice(3, -2),
        ),
    )
    @pytest.mark.parametrize("size", (3, 8, 13))
    def test_as_concrete(self, pyslice, size):
        slice_ = slices.Slice(pyslice.start, pyslice.stop, pyslice.step)

        actual = slice_.as_concrete(size)
        expected = slice(*pyslice.indices(size))

        assert actual.start == expected.start
        assert actual.stop == expected.stop
        assert actual.step == expected.step

    def test_in_dict(self):
        slice1 = slices.Slice(0, 4)
        slice2 = slices.Slice(4, 6)

        d = {slice1: 1, slice2: 2}

        assert d[slice1] == 1
        assert d[slice2] == 2

    @pytest.mark.parametrize(
        ["vals", "expected"],
        (
            ((0, 4), True),
            ((1, 4), False),
            ((0, 4, 1), False),
        ),
    )
    def test_compare(self, vals, expected):
        slice_ = slices.Slice(0, 4)
        other = slices.Slice(*vals)

        actual = slice_ == other
        assert actual == expected
