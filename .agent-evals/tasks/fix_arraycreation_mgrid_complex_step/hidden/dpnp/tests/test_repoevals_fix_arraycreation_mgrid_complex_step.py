import numpy
import pytest
from numpy.testing import assert_allclose

import dpnp


def _check(result, expected):
    if isinstance(result, (list, tuple)):
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert r.shape == e.shape
            assert_allclose(dpnp.asnumpy(r), e)
    else:
        assert result.shape == expected.shape
        assert_allclose(dpnp.asnumpy(result), expected)


@pytest.mark.parametrize("grid", ["mgrid", "ogrid"])
@pytest.mark.parametrize(
    "sl",
    [
        slice(0, 10, 2.5j),
        slice(0, 10, 3.5j),
        slice(-2, 7, 4.9j),
        slice(1, 5, 5j),
        slice(0, 5, 1j),
    ],
)
def test_single_slice_complex_step(grid, sl):
    result = getattr(dpnp, grid)[sl]
    expected = getattr(numpy, grid)[sl]
    _check(result, expected)


@pytest.mark.parametrize("grid", ["mgrid", "ogrid"])
@pytest.mark.parametrize("step", [2.5j, 3.5j])
def test_single_slice_matches_tuple_form(grid, step):
    g = getattr(dpnp, grid)
    single = g[0:10:step]
    in_tuple = g[0:10:step,]
    _check(single, dpnp.asnumpy(in_tuple[0]))


@pytest.mark.parametrize("grid", ["mgrid", "ogrid"])
def test_md_slice_complex_step(grid):
    sl = (slice(0, 10, 2.5j), slice(0, 10, 3.5j))
    result = getattr(dpnp, grid)[sl]
    expected = getattr(numpy, grid)[sl]
    _check(result, expected)
