import pytest

import dpnp
from dpnp.dpnp_utils.dpnp_utils_pad import _pad_normalize_dict_width


def test_empty_dict_gives_zero_pairs():
    assert _pad_normalize_dict_width({}, 3) == [(0, 0), (0, 0), (0, 0)]


def test_int_width_is_symmetric():
    assert _pad_normalize_dict_width({1: 2}, 3) == [(0, 0), (2, 2), (0, 0)]


def test_tuple_width_kept_as_is():
    assert _pad_normalize_dict_width({0: (1, 3)}, 2) == [(1, 3), (0, 0)]


def test_mixed_and_negative_axis():
    res = _pad_normalize_dict_width({0: 1, -1: (2, 0)}, 3)
    assert res == [(1, 1), (0, 0), (2, 0)]


@pytest.mark.parametrize(
    "width", [(1,), (1, 2, 3), [1, 2], (1.0, 2), 1.5, None, "2"]
)
def test_invalid_width_raises(width):
    with pytest.raises(TypeError, match="Invalid pad width"):
        _pad_normalize_dict_width({0: width}, 2)


def test_dpnp_pad_with_dict():
    a = dpnp.ones((2, 3), dtype="i4")
    res = dpnp.pad(a, {1: (1, 2)})
    assert res.shape == (2, 6)
    assert int(res.sum()) == 6
    res = dpnp.pad(a, {0: 1})
    assert res.shape == (4, 3)
