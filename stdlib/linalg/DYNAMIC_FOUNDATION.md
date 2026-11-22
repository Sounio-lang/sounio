# Dynamic numerical foundation

Native typed owners in `collections::numeric_buffer` and
`collections::integer_buffer` separate requested payload budgets from allocator
pages and process RSS. Owners are linear, raw payload pointers remain private,
allocation failure is explicit, reserve preserves the previous allocation on
failure, and close consumes the owner. Matrix descriptors validate affine
bounds and checked access revalidates against the current owner's length.
The caller supplies all kernel output and scratch buffers.

`linalg::dynamic_least_squares` uses column equilibration, column-pivoted Givens
QR and LASSQ norms. It avoids normal equations and has no small fixed row/column
ceiling. The available owner sizes and checked workspace arithmetic set limits.
Nonzero normalized terms that disappear are reported as arithmetic-range errors.
The coefficient output is transactional. The reported R-diagonal ratio is a
diagnostic, not a certified condition number or covariance estimate.

`dynamic_qr_factorize` exports upper-triangular R in pivoted coordinates,
original-column scales and an exact integer permutation:
A[:,permutation] = Q R diag(scales[permutation]).
The response and matrix workspace are scratch; R/scales/permutation commit only
on success. The independent C++ oracle verifies the Gram identity from actual
native output on a 512 by 12 full-rank structured design, and failure preserves
all three public outputs.

Dynamic LU and Cholesky provide checked factorizations and solves with typed
integer pivots. LU skips only exact-zero multipliers, preserving the nonzero
update order; signed-zero bit identity is not promised. Dynamic SVD uses direct
one-sided Jacobi rotations, supports tall/wide inputs, and returns explicit
iteration/orthogonality/range status. See the module-specific documents.

The bounded symmetric eigensolver now rejects invalid dimensions, asymmetric
or nonfinite inputs, uses scale-aware normalization and relative deflation,
and protects direct tridiagonal-QR entry points against overflow and false
convergence. Bounded legacy APIs remain explicit.

Slurm measurements on the same structured 8192 by 64 QR fixture were 43.50 s
before optimization and 3.14 s after, both 106496 KiB peak RSS. These observations
do not characterize general dense-matrix performance.

Lean 4.33 proofs in SounioNumericBuffer cover an abstract affine-index/payload
model and step/reserve state properties. They do not prove the whole allocator,
IEEE operations, compiler or generated executable correct. Native rollback,
ownership, rank, factor reconstruction, and range witnesses remain separate.
