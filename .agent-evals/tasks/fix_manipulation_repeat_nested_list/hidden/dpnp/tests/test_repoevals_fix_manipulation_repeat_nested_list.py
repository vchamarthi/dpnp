import numpy
import pytest
from numpy.testing import assert_array_equal

import dpnp


@pytest.mark.parametrize(
    "x, repeats",
    [
        ([3], [[4]]),
        ([3], ([4],)),
        ([1, 2, 3, 4, 5], [[1, 2, 3, 4, 5]]),
        ([1, 2, 3, 4, 5], [[1], [2], [3], [4], [5]]),
        ([1, 2, 3, 4, 5, 6], [[1, 2, 3], [1, 2, 3]]),
    ],
)
def test_nested_sequence_repeats_raises_value_error(x, repeats):
    # NumPy raises ValueError for these
    with pytest.raises(ValueError):
        numpy.repeat(numpy.array(x), repeats)

    a = dpnp.array(x)
    with pytest.raises(ValueError):
        dpnp.repeat(a, repeats)
    with pytest.raises(ValueError):
        a.repeat(repeats)


@pytest.mark.parametrize(
    "x, repeats, axis",
    [
        ([3], [4], None),
        ([3], (4,), None),
        ([1, 2, 3], [2], None),
        ([1, 2, 3], range(3), None),
        ([1, 2, 3], [0, 1, 2], None),
        ([[1, 2], [3, 4]], [1, 2], 0),
        ([[1, 2], [3, 4]], [3], 1),
        ([[1, 2], [3, 4]], (2,), None),
    ],
)
def test_flat_sequence_repeats_match_numpy(x, repeats, axis):
    a = numpy.array(x)
    expected = numpy.repeat(a, repeats, axis=axis)
    result = dpnp.repeat(dpnp.array(a), repeats, axis=axis)
    assert_array_equal(dpnp.asnumpy(result), expected)


def test_negative_sequence_repeats_raise():
    a = dpnp.array([1, 2, 3])
    with pytest.raises(ValueError):
        dpnp.repeat(a, [-1])
    with pytest.raises(ValueError):
        dpnp.repeat(a, [1, -1, 2])
