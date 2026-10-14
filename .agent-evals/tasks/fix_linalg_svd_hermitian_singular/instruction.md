`dpnp.linalg.svd(a, hermitian=True)` returns a `vh` that is not unitary when `a` is singular:

```python
import dpnp
a = dpnp.array([[1.0, 0.0], [0.0, 0.0]])
u, s, vh = dpnp.linalg.svd(a, hermitian=True)
vh
# array([[ 1.,  0.],
#        [ 0.,  0.]])     <- a row of zeros
vh @ vh.conj().T          # not the identity
```

`u` and `vh` returned by an SVD must be unitary, also for singular (and zero) Hermitian matrices,
for real and complex dtypes and for stacked inputs, and `u @ diag(s) @ vh` must still reconstruct
`a`. Recent NumPy returns a unitary `vh` in this case.

Do not weaken or delete existing tests.
