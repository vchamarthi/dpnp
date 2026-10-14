`dpnp.insert` silently accepts out-of-bounds negative indices when `obj` has more than one element,
and produces a wrong result:

```python
import dpnp
dpnp.insert(dpnp.array([0, 1, 2]), [-6, 0], [9, 8])   # returns an array, no error
dpnp.insert(dpnp.array([0, 1, 2]), [-4, -5], [3, 4])  # returns an array, no error
```

With a single index the bounds are checked:

```python
dpnp.insert(dpnp.array([0, 1, 2]), [-4], [3])
# IndexError: index -4 is out of bounds for axis 0 with size 3
```

Like NumPy (2.6+), `dpnp.insert` should raise `IndexError` with a message of the form
`index <i> is out of bounds for axis <axis> with size <n>` whenever any index in `obj` is `< -n` or
`> n`, for multi-element `obj` given as a list, a NumPy array, or a dpnp array, and for any `axis`.
Valid indices (including `-n` and `n`) must keep working as before.

Do not weaken or delete existing tests.
