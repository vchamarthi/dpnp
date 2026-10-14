`dpnp.bincount` fails on an empty input array, while NumPy handles it.

```python
import dpnp
x = dpnp.array([], dtype="i8")
dpnp.bincount(x)
# ValueError: reduction cannot be performed over zero-size axes
dpnp.bincount(x, minlength=3)                        # same error
dpnp.bincount(x, weights=dpnp.array([]), minlength=2)  # same error
```

NumPy returns an all-zero integer (`intp`) array of length `minlength` (an empty array when
`minlength` is 0), including when `weights` are passed:

```python
>>> import numpy
>>> numpy.bincount(numpy.array([], dtype="i8"), minlength=3)
array([0, 0, 0])
```

Please make `dpnp.bincount` match NumPy for empty input (result values, shape, and the integer
result dtype), without changing its behaviour for non-empty input.

Do not weaken or delete existing tests.
