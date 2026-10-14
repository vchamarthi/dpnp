import numpy
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import dpnp


def _np(x):
    return dpnp.asnumpy(x)


def test_nonzero_offset():
    a = numpy.arange(10, dtype=numpy.int32)
    ia = dpnp.array(a)

    assert_array_equal(_np(ia[3:].view()), a[3:].view())
    assert_array_equal(
        _np(ia[3:].view(dpnp.uint32)), a[3:].view(numpy.uint32)
    )


def test_nonzero_offset_2d_itemsize_change():
    a = numpy.arange(12, dtype=numpy.int32).reshape(3, 4)
    ia = dpnp.array(a)

    assert_array_equal(_np(ia[1:].view()), a[1:].view())
    assert_array_equal(_np(ia[1:].view(dpnp.int16)), a[1:].view(numpy.int16))
    assert_array_equal(_np(ia[1:].view(dpnp.int64)), a[1:].view(numpy.int64))


def test_nonzero_offset_negative_strides():
    a = numpy.arange(10, dtype=numpy.int32)
    ia = dpnp.array(a)

    assert_array_equal(_np(ia[::-1].view()), a[::-1].view())
    assert_array_equal(_np(ia[8:2:-2].view()), a[8:2:-2].view())


@pytest.mark.parametrize("dt", ["i1", "i2", "i4", "i8", "u4", "f4"])
@pytest.mark.parametrize(
    "sl",
    [
        slice(3, None),
        slice(1, 8),
        slice(2, None, 3),
        slice(None, None, -1),
        slice(8, 2, -2),
        slice(10, None),
    ],
)
def test_nonzero_offset_dtypes(dt, sl):
    a = numpy.arange(10).astype(dt)
    ia = dpnp.array(a)
    assert_array_equal(_np(ia[sl].view()), a[sl].view())


def test_nonzero_offset_0d():
    a = numpy.arange(10, dtype=numpy.int32)
    ia = dpnp.array(a)
    assert_array_equal(_np(ia[5, ...].view()), a[5, ...].view())


def test_nonzero_offset_chained():
    a = numpy.arange(20, dtype=numpy.int32)
    ia = dpnp.array(a)

    assert_array_equal(_np(ia[2:][3:].view()), a[2:][3:].view())
    assert_array_equal(
        _np(ia[4:].view().view(dpnp.uint32)),
        a[4:].view().view(numpy.uint32),
    )


def test_nonzero_offset_3d():
    a = numpy.arange(24, dtype=numpy.int32).reshape(2, 3, 4)
    ia = dpnp.array(a)
    for sl in [
        numpy.s_[1:],
        numpy.s_[:, 1:, :],
        numpy.s_[:, :, 2:],
        numpy.s_[1:, 1:, 1:],
    ]:
        assert_array_equal(_np(ia[sl].view()), a[sl].view())


def test_nonzero_offset_writes_reach_parent():
    a = numpy.arange(10, dtype=numpy.int32)
    ia = dpnp.array(a)

    iv = ia[3:].view()
    iv[0] = -7
    a[3] = -7
    assert_array_equal(_np(ia), a)

    b = numpy.arange(6, dtype=numpy.int32)
    ib = dpnp.array(b)
    bv, ibv = b[2:].view(numpy.int16), ib[2:].view(dpnp.int16)
    bv[0] = 99
    ibv[0] = 99
    assert_array_equal(_np(ib), b)


def test_nonzero_offset_compute():
    a = numpy.arange(24, dtype=numpy.float32).reshape(2, 3, 4)
    ia = dpnp.array(a)

    av, iav = a[:, 1:, :].view(), ia[:, 1:, :].view()
    assert_allclose(_np(dpnp.sum(iav)), numpy.sum(av))
    assert_allclose(_np(iav * 2), av * 2)


def test_nonzero_offset_complex_real():
    a = numpy.arange(8).astype(numpy.complex64)
    ia = dpnp.array(a)
    assert_array_equal(
        _np(ia[2:].view(dpnp.float32)), a[2:].view(numpy.float32)
    )

    b = numpy.arange(8).astype(numpy.float32)
    ib = dpnp.array(b)
    assert_array_equal(
        _np(ib[2:].view(dpnp.complex64)), b[2:].view(numpy.complex64)
    )


def test_misaligned_offset_raises():
    # the element offset (in bytes) is not a multiple of the new itemsize
    ia = dpnp.arange(10, dtype=dpnp.int16)
    with pytest.raises(ValueError):
        ia[1:9].view(dpnp.int32)

    ia = dpnp.arange(8, dtype=dpnp.float32)
    with pytest.raises(ValueError):
        ia[1:7].view(dpnp.complex64)


def test_subclass_nonzero_offset():
    class MyArray(dpnp.ndarray):
        pass

    x = dpnp.arange(10, dtype=dpnp.int32)
    view = x[3:].view(type=MyArray)
    assert type(view) is MyArray
    assert_array_equal(_np(view), numpy.arange(3, 10, dtype=numpy.int32))
