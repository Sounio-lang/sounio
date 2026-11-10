# Checked dynamic LU and Cholesky

## Contract and scope

`linalg::dynamic_factorize` provides scaled square LU with partial row pivoting,
strict-SPD Cholesky, and single-RHS triangular solves. Buffers and workspaces are
caller-owned; the hot kernels have Mut,Div,Panic and no Alloc. No owner copy,
raw-pointer view, inverse or determinant is produced. Legacy factorize.sio is
unchanged. Source is prepared pending central current-source native validation;
no convergence, accuracy, corpus or performance acceptance is claimed here.

`dynamic_factor_workspace(n)` checks positive square dimensions and compound
n*n+n and3*n element requirements before arithmetic. It returns:
factor_len=n*n, numeric_work=n*n, integer_work=n, lu_solve_work=3*n,
cholesky_solve_work=2*n. Counts respect the shared numerical payload limit;
there is no fixed8x8 capacity or hidden dimension clamp.

## Factor APIs

`dynamic_lu(a,layout,factor,pivots,work,pivot_work,rcond)` borrows input and
checked square NumericMatrix descriptor, and exclusively borrows separate
NumericBuffer factor/work and IntegerBuffer public/private pivot owners.
Pivots are exact typed integers, never f64 row labels. P[i] is the ORIGINAL row
index now at factor rowi. Public outputs are written only on complete success.
All active input values are finite-preflighted before numerical pivot failures.

The input is normalized by scale=maxabs(A), using scale1 for an all-zero matrix.
Any nonzero coordinate that rounds to zero during normalization returns4.
The compact factor stores unit-diagonal L implicitly below the diagonal and
scaled U on/above it:

```
P*A/scale = L*U_scaled
```

The factorization searches each active column for its largest absolute remaining
pivot. A normalized pivot<=relative_threshold is unavailable(status3).
rcond is finite0<=rcond<1; rcond0 uses n*epsilon. This threshold is a declared
numerical decision, not an exact-singularity or rank certificate.

LU skips the Schur-update inner loop only when the computed multiplier is
EXACTLY zero. This is not a tolerance-based sparsification: every nonzero
multiplier uses the existing update order and finite checks. Untouched cells
need not retain the signed-zero bits that a subtraction of a zero product
would have produced. No measured speed or memory improvement is claimed.
The original factor/status reference baseline passed centrally at job16091.
After the exact-zero skip, the same generated factor reference again completed
natively with exit0 at job16098. This focused concordance does not establish
full corpus acceptance, a performance comparison or ROS4 recovery.

`dynamic_cholesky(a,layout,factor,work,symmetry_tol,rcond)` uses the same scale.
symmetry_tol0 requires EXACT original active offdiagonal equality.
A positive symmetry tolerance<=0.25 accepts the global-scale-relative difference
and explicitly factors the symmetric MIDPOINT of the normalized matrix:
(Bij+Bji)/2, B=A/scale. The normalized sum is bounded before halving, preserving
subnormal midpoint contributions better than separately halved operands.
This is a declared change of analyzed matrix, not silent correction of data.
A nonpositive computed Schur pivot returns6; a positive pivot below the declared
relative threshold returns3. Canonical sqrt is used; upper output entries are0.

```
sym(B) = L_scaled * transpose(L_scaled)
```

The raw Cholesky factor would be sqrt(scale)*L_scaled. It is not returned and
its elementwise representability is not certified. With exact symmetry this
models A/scale; with tolerance it models the stated staged midpoint.

## Metadata and states

DynamicFactorResult exposes status,n,method(0LU/1Cholesky),completed_pivots,
swaps,scale,relative_threshold,diagonal_ratio,growth_factor,symmetry_error.
completed_pivots is NOT rank. diagonal_ratio is min/max absolute factor diagonal,
not a condition estimate. LU growth_factor is the maximum of1 and encountered
normalized Schur-update magnitudes; it is an observed elimination growth
diagnostic, not a bound on solution error. Cholesky growth_factor0 means not
computed. symmetry_error is the measured maximum global-scale relative mismatch
when that scan completes; error metadata can be partial.

- 0 all public factor outputs available.
- 1 invalid dimensions, descriptor, buffers, options or solve permutation.
- 2 nonfinite active matrix/RHS/factor inputs.
- 3 zero/below-threshold pivot.
- 4 unavailable staged arithmetic or underflow of a required nonzero quantity.
- 5 symmetry requirement failed.
- 6 nonpositive computed Cholesky pivot.

Factor/pivot outputs remain untouched on every failure. Scratch may change.
No result certifies a user-modified factor, pivot owner or fabricated metadata.
Arithmetic failures are conservative: a rearranged exact expression might remain
representable. Finite rounding and elimination error remain possible.

## Solve APIs

`dynamic_lu_solve(factor,pivots,fit,b,x,work)` and
`dynamic_cholesky_solve(factor,fit,b,x,work)` validate method/status, buffer lengths,
finite factors/RHS and scale. LU verifies every pivot in[0,n) and uniqueness
before accessing marker indices, using O(n)0/1 markers in its numeric workspace.
No dangerous access is protected only by a logical &&/|| expression.

The RHS is normalized by its maximum absolute element (zero RHS uses1).
Any required nonzero normalization that rounds to zero returns4.
Compensated triangular dot products solve in normalized units. A shared balanced
least_squares_restore rescales z*b_scale/a_scale, preserving useful results
without an overflowing raw scale ratio or premature product underflow.
A nonzero solution rounding entirely to zero is unavailable.
The public x prefix is committed only after all restored coordinates pass.

DynamicSolveResult exposes status,n,method. Success does not assess residual,
conditioning, physical units or statistical variance. With symmetry_tol>0,
Cholesky solves refer to the explicitly analyzed symmetric midpoint model.
No multi-RHS batching or iterative refinement is provided in this increment.

## Verification sources

`stdlib_deep_factor_status.sio` covers invalid size, active NaN, transactional
singularity, required row pivoting, solve permutation duplication, strict and
tolerated symmetry, indefiniteness and strict unrepresentable solve underflow.

`tests/reference/deep_factor_reference.cpp` emits native fixtures using C++23
quad arithmetic for independently constructed RHS/known solutions. Cases cover
required pivoting,128x128 strided SPD systems through BOTH LU and Cholesky, and
a tiny-scale system whose useful solution requires balanced restoration.
The fixture reconstructs PA/scale=LU or A/scale=LL-transpose and checks every
solution component with relative tolerance, without a unit-sized absolute floor.
Padding cells are poisoned and must not participate in the factorization.

Compile emitter on authorized cluster with -std=c++23 -O2 -Wall -Wextra -pedantic
and -lquadmath, then generate
`tests/run-pass/stdlib_deep_factor_reference.sio`.
Typecheck, compile and run through centrally coordinated Slurm with exact
source/compiler identities. No execution was performed by the implementation agent.
