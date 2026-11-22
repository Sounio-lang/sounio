# Dynamic direct SVD

## Scope

`linalg::dynamic_svd` is a caller-workspace one-sided Jacobi thin SVD.
It does not form A-transpose-A, invoke the legacy Matrix8 implementation or
allocate memory internally. Tall, square and wide finite real matrices are
accepted through a checked NumericMatrix descriptor borrowing a NumericBuffer.
Wide input is copied virtually transposed into the same tall engine; original
thin U and V roles are swapped on output.

Source prepared pending central current-source compilation/native concordance.
No numerical accuracy, full-domain representability or comparative-performance
claim is made. The independent reference sources are not reported as executed.

## API and storage

`dynamic_svd_workspace(rows,cols)` returns status and required element counts:
work_len=rows*cols+k*k+2*k, u_len=rows*k, v_len=cols*k, sigma_len=k,
where k=min(rows,cols). Positive dimensions and every product/sum are checked
against the numerical buffer's maximum payload-element limit before arithmetic.
Invalid query returns1 with zero lengths.

`dynamic_svd(a,layout,u,v,sigma_scaled,work,max_sweeps,tolerance,rcond)`
borrows input/descriptor and exclusively borrows four distinct output/scratch
owners. Effects Mut,Div,Panic; no Alloc. Outputs need at least the queried lengths;
only active prefixes are written. The input layout may be strided, transposed
or repeated through its checked descriptor. Inactive cells are not inspected.
No raw-pointer view or owner copy is used.

All public U/V/sigma outputs remain unchanged until convergence, completion and
orthogonality checks pass. Workspace contents become scratch after valid
computation starts, including failure. There is no per-sweep allocation or
trajectory/history storage. Caller creates/releases owners and supplies budgets.

## Scaled singular values

The output sigma_scaled ALWAYS contains dimensionless singular values of
A/sigma_scale, sorted descending. sigma_scale is the maximum absolute active
input element; it is1 for an exactly zero matrix. Thus mathematically:

```
A = U * diag(sigma_scale * sigma_scaled) * transpose(V)
```

U is rows-by-k and V is cols-by-k, both contiguous row-major.
The product sigma_scale*sigma_scaled may overflow even when the represented
factorization is useful. No raw singular-value payload or implicit infinity is
returned. Consumers needing products must use appropriate checked scaling;
the descriptor is not a certificate that each raw singular value is representable.

## Numerical method and termination

Each pair uses LASSQ-style column norms and a compensated dot of individually
normalized columns. This avoids squaring a tiny column norm or overflowing
products of large norms. Rotations are computed from bounded norm ratios and
a scaled hypot, without constructing the potentially overflowing Jacobi tau.
B and the right factor receive identical column rotations.

A complete post-sweep pass measures max absolute normalized column correlation.
Convergence requires this actual represented criterion<=tolerance; rotating the
last pair or exhausting the budget never implies success. max_sweeps1..10000;
finite tolerance in[8*epsilon,0.25]. Large tolerances explicitly request loose
orthogonality; status0 is not a stronger accuracy guarantee.

After sorting, every positive singular column is normalized, including columns
below the diagnostic rank threshold. Only EXACT zero columns receive two-pass
modified Gram-Schmidt completion from canonical basis vectors. Candidates too
small relative to the trace-based1/sqrt(m) existence bound are skipped.
Both resulting thin factors are then checked for unit norms and mutual
orthogonality against the requested tolerance. An unavailable completion fails
rather than publishing zero vectors as an orthonormal factor.

rcond must be finite0<=rcond<1. rcond0 selects max(rows,cols)*epsilon, requiring
that default remain<1. Rank counts sigma_i/sigma_max>rcond_used. This is numerical
rank metadata, not proof of exact rank, conditioning or inferential degrees of
freedom. Positive singular values are never truncated solely by rcond.

## States and limits

- 0 complete; all public factors valid under the stated floating-point criterion.
- 1 invalid dimensions, descriptors, buffers, output lengths or options.
- 2 nonfinite active input.
- 3 sweep budget exhausted.
- 4 unavailable intermediate arithmetic or scaling.
- 6 orthonormal completion/final verification unavailable.

rows/cols and diagnostic progress can remain populated on failure; public
factor buffers are unavailable and unchanged. Callers must inspect status.

Global scaling rejects any nonzero input coordinate that rounds to zero.
A required rotation with norm ratio below DBL_MIN returns4, avoiding a false
convergence claim from silently dropped coupling. This is intentionally
conservative and excludes some mathematically representable extreme matrices.
Other binary64 cancellation and norm/rotation rounding remain possible.
No blanket epsilon zeroing of positive singular values is performed.

A complete sweep costs O(max(rows,cols)*k*k), with another correlation pass;
completion can cost more for many zero columns. Metadata is not a process-RSS
bound, and caller payload budgets do not reserve scheduler memory.

## Verification sources

`tests/run-pass/stdlib_deep_svd_status.sio` covers invalid/overflowing dimensions,
transactional rejection, active NaN, diagonal rank, exact zero/rank deficiency,
wide orientation, meaningful small singular values without squaring their norm,
and explicit global-normalization underflow rejection.

`tests/reference/deep_svd_reference.cpp` is a C++23 analytic fixture emitter:
closed-form2x2 singular values, wide shape, rank-deficient/zero completion,
strided poisoned inactive cells and a1024x32 disjoint-support matrix with known
singular values. Generated Sounio checks every reconstructed active matrix cell
and both thin Gram matrices; it does not require a particular basis/sign within
degenerate singular spaces. A dense one-sweep case checks budget failure and
unchanged public outputs.

Compile/run the emitter only on the authorized cluster lane; generate
`tests/run-pass/stdlib_deep_svd_reference.sio` and perform Sounio check/compile/run
through central Slurm with exact source/compiler identities. No production
Jacobi rotations are copied into the oracle's expected-value calculation.
