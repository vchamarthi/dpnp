import numpy
import pytest
from numpy.testing import assert_allclose

import dpnp

from .helper import has_support_aspect64


def _dtypes():
    if has_support_aspect64():
        return [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128]
    return [numpy.float32, numpy.complex64]


def _check_unitary(x):
    x = dpnp.asnumpy(x)
    n = x.shape[-1]
    prod = x @ numpy.swapaxes(x, -1, -2).conj()
    eye = numpy.broadcast_to(numpy.eye(n, dtype=prod.dtype), prod.shape)
    assert_allclose(prod, eye, atol=1e-5)


_MATRICES = [
    [[1, 0], [0, 0]],
    [[0, 0], [0, 0]],
    [[2, 0, 0], [0, 0, 0], [0, 0, -3]],
    [[0, 0, 0], [0, 5, 0], [0, 0, 0]],
]


@pytest.mark.parametrize("dtype", _dtypes())
@pytest.mark.parametrize("mat", _MATRICES)
def test_hermitian_singular_vh_unitary(dtype, mat):
    a = numpy.array(mat, dtype=dtype)
    ia = dpnp.array(a)

    u, s, vh = dpnp.linalg.svd(ia, hermitian=True)
    _check_unitary(u)
    _check_unitary(vh)

    # the decomposition still reconstructs the input
    rec = (dpnp.asnumpy(u) * dpnp.asnumpy(s)[..., None, :]) @ dpnp.asnumpy(vh)
    assert_allclose(rec, a.astype(rec.dtype), atol=1e-5)

    # singular values match NumPy
    assert_allclose(
        dpnp.asnumpy(s),
        numpy.linalg.svd(a, compute_uv=False).astype(dpnp.asnumpy(s).dtype),
        atol=1e-5,
    )


@pytest.mark.parametrize("dtype", _dtypes())
def test_hermitian_singular_batch(dtype):
    a = numpy.array(
        [[[1, 0], [0, 0]], [[0, 0], [0, 4]], [[3, 0], [0, -2]]], dtype=dtype
    )
    ia = dpnp.array(a)

    u, s, vh = dpnp.linalg.svd(ia, hermitian=True)
    _check_unitary(vh)
    rec = (dpnp.asnumpy(u) * dpnp.asnumpy(s)[..., None, :]) @ dpnp.asnumpy(vh)
    assert_allclose(rec, a.astype(rec.dtype), atol=1e-5)


def test_hermitian_singular_int_input():
    ia = dpnp.array([[1, 0], [0, 0]])
    u, s, vh = dpnp.linalg.svd(ia, hermitian=True)
    _check_unitary(vh)
