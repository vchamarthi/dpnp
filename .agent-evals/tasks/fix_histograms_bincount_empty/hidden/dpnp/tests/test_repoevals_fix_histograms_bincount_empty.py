import numpy
import pytest
from numpy.testing import assert_array_equal

import dpnp


@pytest.mark.parametrize("dtype", ["i1", "i4", "i8", "u1", "u4"])
@pytest.mark.parametrize("minlength", [0, 2, 5])
def test_bincount_empty(dtype, minlength):
    a = numpy.array([], dtype=dtype)
    ia = dpnp.array(a)

    expected = numpy.bincount(a, minlength=minlength)
    result = dpnp.bincount(ia, minlength=minlength)
    assert result.shape == expected.shape
    assert result.dtype == dpnp.intp
    assert_array_equal(result.asnumpy(), expected)


@pytest.mark.parametrize("dtype", ["i4", "i8"])
@pytest.mark.parametrize("w_dtype", ["i4", "f4"])
def test_bincount_empty_with_weights(dtype, w_dtype):
    a = numpy.array([], dtype=dtype)
    w = numpy.array([], dtype=w_dtype)
    ia, iw = dpnp.array(a), dpnp.array(w)

    expected = numpy.bincount(a, weights=w, minlength=2)
    result = dpnp.bincount(ia, weights=iw, minlength=2)
    assert result.shape == expected.shape
    assert_array_equal(result.asnumpy(), expected)


def test_bincount_nonempty_unchanged():
    # a non-empty input still works the same way
    a = numpy.array([0, 1, 1, 3], dtype="i8")
    result = dpnp.bincount(dpnp.array(a), minlength=6)
    assert_array_equal(result.asnumpy(), numpy.bincount(a, minlength=6))
