Passing a nested sequence as `repeats` to `dpnp.repeat` produces a confusing error instead of a
proper `ValueError`:

```python
import dpnp
x = dpnp.array([3])
dpnp.repeat(x, [[4]])
# TypeError: '<' not supported between instances of 'list' and 'int'
```

NumPy rejects such `repeats` with a `ValueError`, because a `repeats` sequence must be 0- or
1-dimensional. `dpnp.repeat` (and `dpnp.ndarray.repeat`) should raise `ValueError` for any nested
list/tuple `repeats` (e.g. `[[4]]`, `([4],)`, `[[1, 2, 3, 4, 5]]`, `[[1], [2], [3], [4], [5]]` for
a 5-element array), while flat sequences such as `[2]`, `(2,)`, `range(3)` or `[0, 1, 2]` keep
working and matching NumPy.

Do not weaken or delete existing tests.
