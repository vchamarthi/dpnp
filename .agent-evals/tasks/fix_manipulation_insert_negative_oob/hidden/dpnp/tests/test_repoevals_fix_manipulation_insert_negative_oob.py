import numpy
import pytest
from numpy.testing import assert_array_equal

import dpnp


@pytest.mark.parametrize(
    "idx, values",
    [
        ([-6, 0], [9, 8]),
        ([0, -6], [9, 8]),
        ([-4, -5], [3, 4]),
        ([-4, 1, 2], [3, 4, 5]),
        ([0, 6], [9, 8]),
        ([4, 4], [3, 4]),
    ],
)
@pytest.mark.parametrize("kind", ["list", "numpy", "dpnp"])
def test_index_out_of_bounds_multi(idx, values, kind):
    a = dpnp.array([0, 1, 2])
    if kind == "numpy":
        idx = numpy.array(idx)
    elif kind == "dpnp":
        idx = dpnp.array(idx)
    with pytest.raises(IndexError, match="out of bounds"):
        dpnp.insert(a, idx, values)


@pytest.mark.parametrize("axis", [0, 1])
def test_index_out_of_bounds_negative_ndim(axis):
    a = dpnp.ones((3, 3))
    with pytest.raises(IndexError, match="out of bounds"):
        dpnp.insert(a, [-5, 0], 9, axis=axis)


@pytest.mark.parametrize(
    "idx",
    [[-3, 0], [-3, 3], [3, 3], [-1, -2], [0, 1, 2, 3], [-3, -3, 3]],
)
def test_in_bounds_multi_matches_numpy(idx):
    a = numpy.array([0, 1, 2])
    values = numpy.arange(10, 10 + len(idx))
    expected = numpy.insert(a, idx, values)
    result = dpnp.insert(dpnp.array(a), idx, dpnp.array(values))
    assert_array_equal(dpnp.asnumpy(result), expected)
