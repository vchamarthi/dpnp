`dpnp.interp` raises for an empty array of sample points even when there is nothing to
interpolate (empty `x`):

```python
import dpnp
x = dpnp.array([])
dpnp.interp(x, dpnp.array([]), dpnp.array([]))
# ValueError: array of sample points is empty
```

Recent NumPy returns an empty array in this case. When `x` is empty, `dpnp.interp` should return an
empty array with the same shape as `x` (e.g. `(0,)` or `(0, 3)`) and the usual result dtype (the
default floating type, or a complex type if `fp` is complex), whether or not `xp`/`fp` are empty.
A non-empty `x` with empty `xp` must still raise `ValueError`, and existing results for non-empty
input must not change.

Do not weaken or delete existing tests.
