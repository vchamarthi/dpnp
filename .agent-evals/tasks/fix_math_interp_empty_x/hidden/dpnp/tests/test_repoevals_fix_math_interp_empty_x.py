import numpy
import pytest

import dpnp

from .helper import has_support_aspect64


def _float():
    return dpnp.default_float_type()


def _complex():
    return dpnp.complex128 if has_support_aspect64() else dpnp.complex64


@pytest.mark.parametrize("x_shape", [(0,), (0, 3), (2, 0)])
def test_empty_x_empty_xp(x_shape):
    x = dpnp.empty(x_shape, dtype=_float())
    xp = dpnp.array([], dtype=_float())
    fp = dpnp.array([], dtype=_float())

    result = dpnp.interp(x, xp, fp)
    assert isinstance(result, dpnp.ndarray)
    assert result.shape == x_shape
    assert result.dtype == _float()


def test_empty_x_empty_xp_complex_fp():
    x = dpnp.array([], dtype=_float())
    xp = dpnp.array([], dtype=_float())
    fp = dpnp.array([], dtype=_complex())

    result = dpnp.interp(x, xp, fp)
    assert result.shape == (0,)
    assert result.dtype == _complex()


def test_empty_x_nonempty_xp():
    x = dpnp.array([], dtype=_float())
    xp = dpnp.array([1, 2], dtype=_float())
    fp = dpnp.array([3, 4], dtype=_float())

    result = dpnp.interp(x, xp, fp)
    assert result.shape == (0,)
    assert result.dtype == _float()


def test_empty_x_complex_fp():
    x = dpnp.empty((0, 2), dtype=_float())
    xp = dpnp.array([1, 2], dtype=_float())
    fp = dpnp.array([3 + 4j, 5 + 6j], dtype=_complex())

    result = dpnp.interp(x, xp, fp)
    assert result.shape == (0, 2)
    assert result.dtype == _complex()


def test_nonempty_x_empty_xp_still_raises():
    x = dpnp.array([0.5], dtype=_float())
    xp = dpnp.array([], dtype=_float())
    fp = dpnp.array([], dtype=_float())
    with pytest.raises(ValueError):
        dpnp.interp(x, xp, fp)


def test_values_unchanged():
    x = numpy.array([0, 1, 1.5, 2.72, 3.14])
    xp = numpy.array([1.0, 2.0, 3.0])
    fp = numpy.array([3.0, 2.0, 0.0])
    result = dpnp.interp(
        dpnp.array(x, dtype=_float()),
        dpnp.array(xp, dtype=_float()),
        dpnp.array(fp, dtype=_float()),
    )
    numpy.testing.assert_allclose(
        dpnp.asnumpy(result), numpy.interp(x, xp, fp), rtol=1e-5
    )
