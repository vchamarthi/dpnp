import numpy
import pytest
from numpy.testing import assert_allclose

import dpnp


@pytest.mark.parametrize("dtype", [dpnp.float32, dpnp.complex64])
@pytest.mark.parametrize("hermitian", [False, True])
def test_pinv_default_tolerance_single_precision(dtype, hermitian):
    # 1e-9 is far below single-precision rounding noise relative to 1.0,
    # so it must be treated as zero by the default cutoff
    a = dpnp.array(numpy.diag([1.0, 1e-9]), dtype=dtype)

    result = dpnp.linalg.pinv(a, hermitian=hermitian)
    assert result.dtype == dtype
    assert_allclose(
        dpnp.asnumpy(result), numpy.diag([1.0, 0.0]), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("dtype", [dpnp.float32, dpnp.complex64])
def test_pinv_default_tolerance_single_precision_batch(dtype):
    a = numpy.stack([numpy.diag([2.0, 1e-9, 1.0]), numpy.diag([1e-10, 4.0, 1.0])])
    ia = dpnp.array(a, dtype=dtype)

    expected = numpy.stack(
        [numpy.diag([0.5, 0.0, 1.0]), numpy.diag([0.0, 0.25, 1.0])]
    )
    result = dpnp.linalg.pinv(ia)
    assert_allclose(dpnp.asnumpy(result), expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dtype", [dpnp.float32, dpnp.complex64])
def test_pinv_explicit_tolerance_unchanged(dtype):
    a = dpnp.array(numpy.diag([1.0, 1e-3]), dtype=dtype)
    result = dpnp.linalg.pinv(a, rtol=1e-6)
    assert_allclose(
        dpnp.asnumpy(result), numpy.diag([1.0, 1e3]), rtol=1e-4
    )


@pytest.mark.parametrize(
    "dtype", [dpnp.int32, dpnp.int64, dpnp.uint8, dpnp.bool]
)
@pytest.mark.parametrize("shape", [(0, 0), (0, 3), (3, 0), (2, 0, 3), (2, 3, 0)])
def test_pinv_empty_non_inexact_dtype(dtype, shape):
    a = dpnp.empty(shape, dtype=dtype)
    result = dpnp.linalg.pinv(a)
    assert result.shape == shape[:-2] + (shape[-1], shape[-2])
    assert result.dtype == dpnp.default_float_type(a.sycl_queue)


@pytest.mark.parametrize("dtype", [dpnp.int32, dpnp.int64])
def test_pinv_non_empty_non_inexact_dtype(dtype):
    a = numpy.array([[1, 2, 3], [4, 1, 1], [2, 3, 1]]).astype(dtype)
    result = dpnp.linalg.pinv(dpnp.array(a))
    assert result.dtype == dpnp.default_float_type()
    assert_allclose(
        dpnp.asnumpy(result),
        numpy.linalg.pinv(a.astype(numpy.float64)),
        rtol=1e-4,
        atol=1e-5,
    )
