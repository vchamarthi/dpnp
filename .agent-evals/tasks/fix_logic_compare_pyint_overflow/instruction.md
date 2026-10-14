Comparing an integer array with a Python `int` that does not fit the array's dtype raises
`OverflowError`, while NumPy returns the mathematically correct answer:

```python
import dpnp
a = dpnp.array([0, 1, 2**63, 2**64 - 1], dtype="u8")
dpnp.less(a, -1)
# OverflowError: Python integer -1 out of bounds for uint64
a == -1                                           # same error
dpnp.greater(dpnp.array([1, 2], dtype="i1"), 2**32)  # same error
```

NumPy compares against the true value of the scalar (`-1` is smaller than every `uint64`, so
`less` gives all `False`). The same affects `equal`, `not_equal`, `less_equal`, `greater` and
`greater_equal`, with the scalar on either side, and the comparison operators of `dpnp.ndarray`.

True division has the same problem:

```python
dpnp.divide(dpnp.array([0, 2, 2**63], dtype="u8"), -1)  # OverflowError
```

NumPy returns a floating-point result here. Please make these functions give the correct boolean
(for comparisons) or floating-point (for `divide`/`true_divide`/`/`) results for out-of-range Python
integer scalars. `floor_divide` should stay as it is (NumPy raises there too).

Do not weaken or delete existing tests.
