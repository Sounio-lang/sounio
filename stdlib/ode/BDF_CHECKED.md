<!-- docs:meta
topic_id: repo.stdlib.ode.bdf-checked
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.stdlib.ode.bdf-checked
-->

# Checked BDF1 integration

Import ode::bdf_checked and obtain bdf_workspace(n) before allocating. The returned lengths and payload_bytes cover seven numeric vectors (including state), Jacobian, LU factor, factor workspace and two integer pivot buffers. All owners are supplied by the caller; solve_bdf1_checked performs no allocation. Distinct linear owners must back every argument. Close all owners after use. See tests/stdlib/bdf/scale128.sio for the complete allocation/callback/solve/close example.

This is implicit Euler with adaptive step doubling and an analytic row-major Jacobian. Each attempted interval computes a coarse step and two represented substeps. Only an accepted fine result replaces state. The error estimate compensates for unequal represented substep lengths; it is a local estimate, not a global or certified bound. Integration is forward only. A zero span succeeds without callbacks. Endpoints and the midpoint must permit representable progress; an interval with no representable interior point is refused.

Callbacks return zero on success and must fill their complete output buffer. RHS receives time, state and length-n output; the Jacobian receives time, state and length-n*n output. Buffers are poisoned before calls, so partial writes or nonfinite outputs refuse the attempted step. Callback effects are not rolled back; callbacks must represent the same mathematical function for identical inputs.

atol is strictly positive; rtol is nonnegative. newton_tol is a dimensionless residual threshold in (0,0.1], with residual scaled by atol+rtol*max(abs(old),abs(trial)). dt_min and dt_init are positive, dt_max>=dt_init>=dt_min; a terminal interval below dt_min may finish if both substeps progress. max_attempts is 1..1000000; max_newton is 1..1000. rcond is in [0,1) and is passed to canonical checked LU. Every input must be finite. Use the returned status rather than inferring success from the final vector.

| Status | Meaning |
| --- | --- |
| 0 | Target reached |
| 1 | Invalid layout, options or domain |
| 2 | Representable time progress unavailable |
| 3 | Attempt budget exhausted |
| 4 | Nonfinite arithmetic or incomplete/nonfinite callback output |
| 5 | Error-control step floor reached |
| 6 | Nonzero callback status; callback_kind distinguishes RHS=1/Jacobian=2 |
| 7 | Newton update budget exhausted |
| 8 | Canonical linear algebra refused; inspect linear_status |

On failure state and t_final retain the last accepted solution, while scratch may change. accepted, rejected, attempts, nfeval, njeval, nfactorizations, nlinear and newton_updates count actual work, including failed attempts. last_error and last_residual are diagnostics, not uncertainty estimates.

The older private bdf64 module remains bounded and experimental. Singular elimination now refuses; partial callback writes are detected; Newton uses its requested tolerance; histories reset when step size changes; bounded traces retain the final endpoint. Its legacy variance path is restricted to first-order independent diagonal propagation and refuses coupled uncertainty rather than silently dropping covariance. It is not a general variable-order, variable-step BDF implementation.
