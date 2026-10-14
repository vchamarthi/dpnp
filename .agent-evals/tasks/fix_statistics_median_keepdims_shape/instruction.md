`dpnp.median` and `dpnp.nanmedian` mishandle kept (non-reduced) dimensions in two ways.

1. With a tuple/list `axis`, a kept dimension of size 0 makes them raise instead of returning an
   empty result:

   ```python
   import dpnp
   a = dpnp.empty((0, 3, 4))
   dpnp.median(a, axis=(1, 2))                  # ValueError; expected shape (0,)
   dpnp.median(a, axis=(1, 2), keepdims=True)   # ValueError; expected shape (0, 1, 1)
   dpnp.nanmedian(a, axis=(1, 2))               # ValueError
   ```

   Reducing the same array over a single axis already works.

2. `dpnp.nanmedian` drops kept dimensions of size 1 when the input contains NaN:

   ```python
   import dpnp, numpy
   a = numpy.random.rand(1, 5); a[0, 0] = numpy.nan
   dpnp.nanmedian(dpnp.array(a), axis=1).shape   # ()
   numpy.nanmedian(a, axis=1).shape              # (1,)
   ```

   e.g. a `(3, 1, 4)` array reduced over `axis=2` should give shape `(3, 1)`.

Result shapes (and values) should match NumPy for both functions, with and without `keepdims`.

Do not weaken or delete existing tests.
