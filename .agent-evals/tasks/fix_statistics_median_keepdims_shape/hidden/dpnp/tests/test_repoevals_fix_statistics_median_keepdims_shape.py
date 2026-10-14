import numpy
import pytest
from numpy.testing import assert_allclose

import dpnp


@pytest.mark.usefixtures("suppress_mean_empty_slice_numpy_warnings")
@pytest.mark.parametrize("func", ["median", "nanmedian"])
@pytest.mark.parametrize(
    "shape, axis, keepdims, out_shape",
    [
        ((0, 3, 4), (1, 2), False, (0,)),
        ((0, 3, 4), (1, 2), True, (0, 1, 1)),
        ((3, 0, 4), (0, 2), False, (0,)),
        ((2, 0, 3, 4), (0, 3), True, (1, 0, 3, 1)),
    ],
)
def test_empty_kept_dim(func, shape, axis, keepdims, out_shape):
    ia = dpnp.empty(shape, dtype=dpnp.default_float_type())
    result = getattr(dpnp, func)(ia, axis=axis, keepdims=keepdims)
    assert result.shape == out_shape


@pytest.mark.parametrize(
    "shape, axis",
    [((1, 5), 1), ((3, 1, 4), 2), ((2, 1, 5), (0, 2)), ((1, 1, 6), -1)],
)
@pytest.mark.parametrize("keepdims", [True, False])
def test_nanmedian_size1_kept_dim(shape, axis, keepdims):
    rng = numpy.random.default_rng(1234)
    a = rng.uniform(-5, 5, size=shape).astype(dpnp.default_float_type())
    a.flat[0] = numpy.nan
    ia = dpnp.array(a)

    result = dpnp.nanmedian(ia, axis=axis, keepdims=keepdims)
    expected = numpy.nanmedian(a, axis=axis, keepdims=keepdims)
    assert result.shape == expected.shape
    assert_allclose(dpnp.asnumpy(result), expected, rtol=1e-6)


@pytest.mark.parametrize(
    "shape, axis",
    [((1, 5), 1), ((3, 1, 4), 2), ((2, 1, 5), (0, 2))],
)
@pytest.mark.parametrize("keepdims", [True, False])
def test_median_size1_kept_dim(shape, axis, keepdims):
    rng = numpy.random.default_rng(4321)
    a = rng.uniform(-5, 5, size=shape).astype(dpnp.default_float_type())
    ia = dpnp.array(a)

    result = dpnp.median(ia, axis=axis, keepdims=keepdims)
    expected = numpy.median(a, axis=axis, keepdims=keepdims)
    assert result.shape == expected.shape
    assert_allclose(dpnp.asnumpy(result), expected, rtol=1e-6)
