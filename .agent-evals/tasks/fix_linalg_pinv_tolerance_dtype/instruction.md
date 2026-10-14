Two problems with `dpnp.linalg.pinv` around data types:

1. With the default tolerance, single-precision input on a device that supports float64 keeps
   singular values that are pure rounding noise:

   ```python
   import dpnp
   a = dpnp.diag(dpnp.array([1.0, 1e-9], dtype=dpnp.float32))
   dpnp.linalg.pinv(a)
   # array([[1.e+00, 0.e+00],
   #        [0.e+00, 1.e+09]], dtype=float32)
   ```

   The documented default cutoff is `max(M, N) * eps` of the precision the computation runs in, so
   for `float32`/`complex64` input the `1e-9` singular value should be discarded and the result
   should be `[[1, 0], [0, 0]]` (also with `hermitian=True` and for stacked matrices). This matches
   NumPy.

2. For empty integer or boolean input the result keeps the input dtype, while non-empty integer
   input gives a floating-point result:

   ```python
   dpnp.linalg.pinv(dpnp.empty((0, 3), dtype=dpnp.int64)).dtype   # int64
   dpnp.linalg.pinv(dpnp.ones((2, 3), dtype=dpnp.int64)).dtype    # float64
   ```

   Empty input should return the same floating-point dtype that non-empty input of that dtype would
   (the device's default floating type for integer/boolean input), with the transposed shape as
   before.

Results with an explicit `rtol`/`rcond` must not change.

Do not weaken or delete existing tests.
