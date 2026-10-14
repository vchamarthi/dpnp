`dpnp.ndarray.view()` returns wrong data for an array that does not start at the beginning of its
memory allocation (e.g. a slice):

```python
import dpnp
a = dpnp.arange(10, dtype=dpnp.int32)
a[3:].view()
# array([0, 1, 2, 3, 4, 5, 6])      expected array([3, 4, 5, 6, 7, 8, 9])
a[3:].view(dpnp.uint32)            # same problem
a[::-1].view()                     # wrong as well
```

The view should refer to exactly the same elements as the array it is taken from, as in NumPy:
for any slicing (positive or negative steps, multi-dimensional, chained slices, 0-d), with or
without a change of dtype (same, smaller or larger itemsize, real/complex), and for subclasses
(`view(type=...)`). Writes through the view must land in the corresponding elements of the
original array.

If the requested dtype cannot address the array's starting position in memory (its byte offset is
not a multiple of the new itemsize, e.g. `dpnp.arange(10, dtype=dpnp.int16)[1:9].view(dpnp.int32)`),
a `ValueError` should be raised instead of returning a view of the wrong memory.

Do not weaken or delete existing tests.
