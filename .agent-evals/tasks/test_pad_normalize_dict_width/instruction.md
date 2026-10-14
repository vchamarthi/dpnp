`dpnp.pad` accepts `pad_width` as a dictionary mapping axis indices to pad widths (for example
`dpnp.pad(a, {1: (1, 2)})`). That dictionary form is normalized by the private helper
`_pad_normalize_dict_width(pad_width, ndim)` in `dpnp/dpnp_utils/dpnp_utils_pad.py`, which is barely tested.

Write pytest tests for the observable behaviour of `_pad_normalize_dict_width` (and the dictionary form of
`dpnp.pad` where useful): axes that are not mentioned, single-integer widths, `(before, after)` tuples,
negative axes, and the errors raised for invalid widths. Put the tests in a new file
`dpnp/tests/test_pad_dict_width.py`; they are run with
`python -m pytest -q -p no:randomly dpnp/tests/test_pad_dict_width.py` on a CPU-only machine
(`ONEAPI_DEVICE_SELECTOR=opencl:cpu`). Do not change any library code under `dpnp/` other than tests.

End your answer with ```json {"targets": ["dpnp/tests/test_pad_dict_width.py"]}```.
