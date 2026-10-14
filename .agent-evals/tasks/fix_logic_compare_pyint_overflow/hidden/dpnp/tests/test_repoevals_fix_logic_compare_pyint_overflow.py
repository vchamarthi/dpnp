import operator

import numpy
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import dpnp

from .helper import has_support_aspect64

_FUNCS = [
    ("equal", operator.eq),
    ("not_equal", operator.ne),
    ("less", operator.lt),
    ("less_equal", operator.le),
    ("greater", operator.gt),
    ("greater_equal", operator.ge),
]


def _values(dtype):
    info = numpy.iinfo(dtype)
    return [int(info.min), 0, 1, int(info.max)]


@pytest.mark.parametrize("dtype", ["i1", "i8", "u1", "u8"])
@pytest.mark.parametrize("scalar", [-1, 0, 2**32, 2**63, 2**64 - 1])
@pytest.mark.parametrize("name, pyop", _FUNCS, ids=[f[0] for f in _FUNCS])
def test_compare_array_with_pyint(dtype, scalar, name, pyop):
    vals = _values(dtype)
    a = dpnp.array(vals, dtype=dtype)
    func = getattr(dpnp, name)

    res1 = func(a, scalar)
    res2 = func(scalar, a)
    assert res1.dtype == dpnp.bool
    assert_array_equal(dpnp.asnumpy(res1), [pyop(v, scalar) for v in vals])
    assert_array_equal(dpnp.asnumpy(res2), [pyop(scalar, v) for v in vals])


@pytest.mark.parametrize("dtype", ["u1", "u8"])
@pytest.mark.parametrize("name, pyop", _FUNCS, ids=[f[0] for f in _FUNCS])
def test_compare_operator_negative_pyint(dtype, name, pyop):
    vals = _values(dtype)
    a = dpnp.array(vals, dtype=dtype)
    res = pyop(a, -1)
    assert_array_equal(dpnp.asnumpy(res), [pyop(v, -1) for v in vals])


@pytest.mark.skipif(not has_support_aspect64(), reason="needs fp64")
@pytest.mark.parametrize(
    "dtype, scalar",
    [("u8", -1), ("u8", -7), ("u4", -3), ("i8", 2**63), ("i4", 2**40)],
)
def test_divide_by_out_of_range_pyint(dtype, scalar):
    vals = [0, 2, 100, int(numpy.iinfo(dtype).max)]
    a = dpnp.array(vals, dtype=dtype)

    expected = numpy.array(vals, dtype=numpy.float64) / float(scalar)
    for res in (dpnp.divide(a, scalar), dpnp.true_divide(a, scalar), a / scalar):
        assert dpnp.issubdtype(res.dtype, dpnp.floating)
        assert_allclose(dpnp.asnumpy(res), expected, rtol=1e-6)


@pytest.mark.skipif(not has_support_aspect64(), reason="needs fp64")
@pytest.mark.parametrize("dtype, scalar", [("u8", -1), ("i8", 2**63)])
def test_divide_pyint_by_array(dtype, scalar):
    vals = [1, 2, 100]
    a = dpnp.array(vals, dtype=dtype)
    expected = float(scalar) / numpy.array(vals, dtype=numpy.float64)
    res = dpnp.divide(scalar, a)
    assert_allclose(dpnp.asnumpy(res), expected, rtol=1e-6)
