1. [BLOCKER] The Picard self-map theorem is applied incorrectly because `picard_box` uses the same `time_interval = arb(step/2, step/2)` for both the initial box construction and the fixed-point iteration, but the interval arithmetic in `field(box)` evaluates the vector field over the entire box simultaneously, which does not guarantee that the image is contained in the box for the actual ODE solution.
   - worker.py: `picard_box` function, lines ~110-135
   - This is the core validation claim. If the Picard iteration doesn't actually prove existence/uniqueness via the contraction mapping theorem (which requires the Lipschitz constant times step < 1), the entire enclosure is unproven.
   - Use a rigorous Picard-Lindelöf theorem with explicit Lipschitz bounds on the box, not just interval arithmetic iteration.

2. [BLOCKER] The logarithmic norm bound `logarithmic_norm_bound` is computed incorrectly: the first row `-y + abs(4*y - x)` is not the infinity-norm row sum of the Jacobian; it appears to be a heuristic that doesn't match the actual Jacobian structure.
   - worker.py: `logarithmic_norm_bound` function, lines ~140-150
   - The Gronwall recurrence `next_radius = (amplification * radius + local_remainder + rounding_radius)` relies on this bound being a true upper bound on the logarithmic norm. If it's wrong, the error propagation is invalid.
   - Compute the actual Jacobian matrix and its infinity-norm row sums rigorously.

3. [BLOCKER] The Taylor remainder term `remainder_coefficients = flow_coefficients(box, TAYLOR_ORDER + 1)` uses the Picard box as the initial condition, but the coefficients are computed assuming the initial condition is a point, not an interval; the interval arithmetic in the convolution sums is not guaranteed to bound the true remainder.
   - worker.py: `advance` function, lines ~160-175
   - The remainder bound `local_remainder` may be an underestimate, invalidating the error control.
   - Use a rigorous Taylor remainder theorem with explicit interval bounds on the (TAYLOR_ORDER+1)-th derivative over the box.

4. [BLOCKER] The event bisection `locate_event` does not handle the case where the interval `[low, high]` contains zero (i.e., `sign` returns 0 for both endpoints), and the bisection loop can terminate with `middle_sign == 0` without ever narrowing the bracket.
   - worker.py: `locate_event` function, lines ~190-215
   - If the event time is not isolated or the interval contains a zero-crossing with sign changes on both sides, the bisection may fail to converge or produce a spurious result.
   - Add explicit handling for the `sign == 0` case: if the interval contains zero, either refine with more bisections or fail closed.

5. [BLOCKER] The event-state enclosure `event_box = picard_box(low_center, low_radius, high - low)` is computed using the low-center state and the full time step `high - low`, but the actual event state could be anywhere in the interval, and the Picard box may not contain the true event state.
   - worker.py: `locate_event` function, line ~216
   - The determinant formula uses `second_event[3]` (the ell component) and `final_normal = second_event[0]*second_event[1] - ZS`, but these are evaluated at the Picard box's center, not guaranteed to be the actual event state.
   - Compute the event state enclosure rigorously, e.g., by intersecting the Picard box with the section condition.

6. [MAJOR] The fixed decimal parameters (`ZS`, `ORIGIN_X`, etc.) are converted to `arb` types using decimal strings, but the conversion is not guaranteed to be exact; the resulting balls may not contain the true decimal values.
   - worker.py: lines ~30-40
   - If the decimal-to-ball conversion introduces rounding errors, the entire computation is based on approximate parameters, undermining the "validated" claim.
   - Convert decimals to exact rationals (e.g., using `Fraction`) and then to `arb` with explicit error bounds.

7. [MAJOR] The accumulated rounding error is not tracked: `STATS.accumulated_mu_h` sums `mu * step` but this is not used in the error bound; the radius propagation `next_radius` only accounts for local remainder and rounding, not accumulated floating-point errors in the coefficient recursion.
   - worker.py: `advance` function, lines ~165-185
   - Over 1686 steps, even small per-step rounding errors can accumulate significantly, invalidating the global error bound.
   - Use a rigorous interval arithmetic library that tracks all rounding errors, or add explicit error accumulation terms.

8. [MAJOR] The determinant formula `determinant = second_event[3].exp() * initial_normal / final_normal * q0_area` uses the ell component at the event time, but the ell component is not the integral of the divergence over the trajectory; it's a state variable that must be propagated correctly.
   - worker.py: `main` function, lines ~270-280
   - The formula assumes `ell(T)` is the integral of the divergence, but the ODE for `ell` is `x - y - (w + ZS)/2 - 1`, which is not the divergence of the vector field (the divergence is `2y - x + x - (w+ZS)/2 - 1`).
   - Verify the determinant formula against the actual variational equations; the current formula may be incorrect.

9. [MAJOR] The verifier `verify.py` checks `det_width > Fraction(1, 10**22)` and `MAX_GLOBAL_RADIUS_UPPER_Q > Fraction(1, 10**12)`, but these thresholds are not derived from the contract or any theoretical bound; they are arbitrary and may be too loose or too tight.
   - verify.py: lines ~180-200
   - If the thresholds are too loose, the certificate may pass even when the computation is invalid; if too tight, it may fail for valid computations.
   - Derive thresholds from the contract's stated obligations (e.g., the Picard self-map theorem and Gronwall recurrence).

10. [MAJOR] The verifier does not independently check the Picard containment obligation; it only checks that `PICARD_CALLS == PICARD_CONTAINMENTS`, which is a self-reported count from the worker, not an independent verification.
    - verify.py: lines ~160-170
    - A malicious or buggy worker could report `PICARD_CONTAINMENTS` equal to `PICARD_CALLS` even if the containment failed.
    - Re-run the Picard iteration independently on the reported centers and radii to verify containment.

11. [MINOR] The `inflate` function uses `arb(value.mid(), value.rad() * factor + epsilon)` where `epsilon = 1/2^220`, but the inflation factor `17/16` may not be sufficient to cover all rounding errors in the interval operations.
    - worker.py: `inflate` function, lines ~95-100
    - If the inflation is insufficient, the Picard box may not contain the true solution, invalidating the enclosure.
    - Use a rigorous inflation that guarantees inclusion, e.g., by computing the image with interval arithmetic and then inflating by a factor that provably covers all rounding errors.

12. [MINOR] The `sign` function returns 0 if the interval contains zero, but the bisection logic treats `middle_sign == 0` as a reason to break, which can cause premature termination with an ambiguous event.
    - worker.py: `sign` function and `locate_event`, lines ~185-215
    - If the event time is very close to zero, the interval may contain zero for many bisections, leading to an unvalidated result.
    - Continue bisecting until the interval is narrow enough to determine the sign, or fail closed if the interval cannot be resolved.

13. [MINOR] The verifier checks that the determinant interval is contained in the CAPD interval, but the CAPD comparison is not proof of correctness; it's only a sanity check. The verifier should not rely on CAPD as a substitute for rigorous validation.
    - verify.py: `capd_intervals` function, lines ~130-150
    - If the CAPD data is wrong or the comparison is flawed, the verifier may pass an invalid certificate.
    - Remove the CAPD comparison from the verification logic; only use it as an optional cross-check.

14. [MINOR] The `flow_coefficients` recursion for the `ell` component (index 3) uses `coefficients[2][n] / 2` and `(ZS / 2 + 1) if n == 0`, but the actual ODE is `x - y - (w + ZS)/2 - 1`, which should have `(w + ZS)/2` not `w/2 + ZS/2`; the code appears to be correct, but the `ZS` handling for `n == 0` is suspicious.
    - worker.py: `flow_coefficients` function, lines ~75-90
    - If the constant term is wrong, the Taylor coefficients for `ell` are incorrect, invalidating the determinant.
    - Verify the recursion against the ODE by expanding the Taylor series manually.

15. [MINOR] The `main` function does not check that the initial conditions satisfy the ODE's constraints (e.g., `initial_normal` is positive), but the certificate checks `initial_normal.lower() > 0`; this is a runtime check, not a static one, and could fail if the initial conditions are outside the valid domain.
    - worker.py: `main` function, lines ~250-270
    - If the initial conditions are invalid, the Picard iteration may fail or produce a spurious result.
    - Add a static check on the initial conditions before starting the computation.

16. [MINOR] The `verify.py` script requires `PYTHON_FLINT_VERSION == "0.8.0"` but does not verify that the worker actually used this version; it only checks the reported version string, which could be spoofed.
    - verify.py: line ~170
    - A malicious worker could report a different version than the one actually used.
    - Check the actual installed version of python-flint at verification time, or use a cryptographic attestation of the worker's environment.

17. [NIT] The `exact_fraction` function converts an `arb` to a `Fraction` using `value.fmpq()`, but this may not be exact for all `arb` values; the conversion could round the interval endpoints.
    - worker.py: `exact_fraction` function, lines ~50-60
    - If the conversion is not exact, the printed fractions may not be the true interval endpoints, invalidating the certificate.
    - Use the `arb`'s lower and upper bounds directly, or ensure the conversion is exact.

18. [NIT] The `max_upper` function uses `candidate > result` to compare `arb` values, but `arb` comparisons are not always total; this could lead to incorrect maximum selection.
    - worker.py: `max_upper` function, lines ~65-75
    - If two intervals are incomparable, the maximum may be chosen arbitrarily, leading to an incorrect bound.
    - Use `max` with a total order on `arb` values, or compute the maximum of the upper bounds explicitly.

19. [NIT] The `validate_hex` function checks that the hex string is exactly 64 characters, but does not check that it's a valid SHA-256 hash (e.g., all zeros is technically a valid hex string but not a valid hash).
    - worker.py: `validate_hex` function, lines ~220-230
    - A malicious input could pass the hex check but not be a valid hash, leading to a false binding.
    - Add a check that the hex string is not all zeros or other invalid patterns.

20. [NIT] The verifier checks `stderr` is empty, but a worker could write to stderr without affecting the exit code; this is a minor consistency check but not a proof of correctness.
    - verify.py: line ~150
    - A worker could print warnings to stderr and still produce a valid certificate; the check is too strict and may reject valid runs.
    - Allow warnings on stderr but check that they don't indicate errors.

21. [NIT] The `locate_event` function uses `low_center, low_radius = center, radius` initially, but after the first bisection, `low_center` and `low_radius` are set to the results of `advance(center, radius, middle)`, which may not be the correct state at the low endpoint.
    - worker.py: `locate_event` function, lines ~195-210
    - If the state at the low endpoint is not correctly propagated, the event box may not contain the true event state.
    - Use the actual state at the low endpoint, not the initial state, for the Picard box.

22. [NIT] The `sign` function uses `arb(component, radius)` to construct an interval, but this assumes the center and radius are exact; if they are not, the interval may not contain the true state.
    - worker.py: `sign` function, lines ~185-190
    - If the center or radius has rounding errors, the sign determination may be incorrect.
    - Use the actual interval from the propagation, not a reconstructed one.

23. [NIT] The `main` function prints `MAX_GLOBAL_RADIUS_UPPER_Q` but the verifier checks it against a threshold; however, the threshold is not derived from the contract and may be too loose.
    - worker.py: line ~290, verify.py: line ~185
    - A radius of 1e-12 may be too large for the claimed precision, but the verifier accepts it.
    - Derive the threshold from the Taylor remainder and Gronwall bounds.

24. [NIT] The verifier checks `EVENT2_TIME_WIDTH_Q` against a threshold of `2^-40`, but the worker's bisection uses `EVENT_BISECTIONS = 60`, which should give a width of `STEP / 2^60 = 2^-68`, so the threshold is much looser than necessary.
    - verify.py: line ~190
    - The loose threshold may hide errors in the bisection or event location.
    - Tighten the threshold to match the bisection precision, or justify the looser bound.

25. [NIT] The `expected_bindings` function in `verify.py` uses `digest(root / CONTRACT_REL)` but the contract file is not included in the challenge domain; this is a minor inconsistency in the binding scheme.
    - verify.py: `expected_bindings` function, lines ~110-125
    - The challenge should bind to the contract to ensure the contract is not modified after the challenge is issued.
    - Include the contract hash in the challenge domain.

26. [NIT] The worker's `fail` function uses `SystemExit` with a string, which prints the message to stderr and exits with code 1; this is acceptable but not ideal for a rigorous validation system.
    - worker.py: `fail` function, lines ~45-50
    - A more structured error handling would be better for debugging and auditing.
    - Use a custom exception class with detailed error codes.

27. [NIT] The `STATS.max_global_radius` is updated using `max_upper([STATS.max_global_radius, next_radius])`, but `max_upper` returns an `arb` which may have rounding errors; this could overestimate the maximum radius.
    - worker.py: `advance` function, line ~170
    - The maximum radius is used in the certificate, so an overestimate could cause a false pass.
    - Use the upper bound of the maximum explicitly.

28. [NIT] The `q0_area` is computed as `(UNSTABLE_X * STABLE_Y - STABLE_X * UNSTABLE_Y) * RADIUS_U * RADIUS_S`, but this assumes the unstable and stable directions are orthogonal; if they are not, the area is not the true area of the initial parallelogram.
    - worker.py: `main` function, line ~260
    - If the directions are not orthogonal, the area formula is incorrect, invalidating the determinant.
    - Use the actual area of the parallelogram spanned by the unstable and stable vectors.

29. [NIT] The `initial_normal` is computed as `initial_box[0] * initial_box[1] - ZS`, but the section is defined as `w = 0`, and the normal velocity is `x*y - zs`; this is correct, but the sign convention should be verified.
    - worker.py: `main` function, line ~265
    - If the sign convention is wrong, the event detection may be inverted.
    - Verify the sign convention against the ODE and the section definition.

30. [NIT] The `verify.py` script checks that `PICARD_CALLS >= 1686`, but this is a weak lower bound; the actual number of calls depends on the number of steps and events, which could be higher.
    - verify.py: line ~170
    - A worker could report a lower number of calls than actually performed, hiding errors.
    - Compute the expected number of calls from the step count and event bisections, and check the reported value against it.

31. [NIT] The `advance` function returns `[component.mid() for component in polynomial]` as the new center, but the `mid()` method may not be exact; it could introduce rounding errors.
    - worker.py: `advance` function, line ~180
    - If the midpoint is not exact, the next step's propagation may be inaccurate.
    - Use the actual polynomial value at the step time, not its midpoint.

32. [NIT] The `picard_box` function uses `arb(step / 2, step / 2)` as the time interval, but this assumes the step is a power of two; for other step sizes, the interval may not be correctly represented.
    - worker.py: `picard_box` function, line ~115
    - The step is hardcoded as `2^-8`, but the function should handle arbitrary steps.
    - Use a general interval representation for the time step.

33. [NIT] The `flow_coefficients` recursion for the `w` component uses `ZS if n == 0` as a constant term, but this is only correct if `ZS` is constant; if `ZS` were a variable, the recursion would be wrong.
    - worker.py: `flow_coefficients` function, line ~80
    - The code assumes `ZS` is constant, which is stated in the contract, but this should be verified.
    - Add a comment or assertion that `ZS` is constant.

34. [NIT] The `logarithmic_norm_bound` function uses `max_upper(rows)` which returns the maximum of the upper bounds, but the true logarithmic norm is the maximum of the row sums, not the maximum of the upper bounds of the rows.
    - worker.py: `logarithmic_norm_bound` function, line ~145
    - If the rows are intervals, the maximum of the upper bounds may not be the true maximum of the row sums.
    - Compute the row sums as intervals and then take the maximum of their upper bounds.

35. [NIT] The `verify.py` script checks that `stderr` is empty, but a worker could write to stderr without affecting the exit code; this is a minor consistency check but not a proof of correctness.
    - verify.py: line ~150
    - A worker could print warnings to stderr and still produce a valid certificate; the check is too strict and may reject valid runs.
    - Allow warnings on stderr but check that they don't indicate errors.

36. [NIT] The `expected_bindings` function in `verify.py` uses `digest(root / CONTRACT_REL)` but the contract file is not included in the challenge domain; this is a minor inconsistency in the binding scheme.
    - verify.py: `expected_bindings` function, lines ~110-125
    - The challenge should bind to the contract to ensure the contract is not modified after the challenge is issued.
    - Include the contract hash in the challenge domain.

37. [NIT] The worker's `fail` function uses `SystemExit` with a string, which prints the message to stderr and exits with code 1; this is acceptable but not ideal for a rigorous validation system.
    - worker.py: `fail` function, lines ~45-50
    - A more structured error handling would be better for debugging and auditing.
    - Use a custom exception class with detailed error codes.

38. [NIT] The `STATS.max_global_radius` is updated using `max_upper([STATS.max_global_radius, next_radius])`, but `max_upper` returns an `arb` which may have rounding errors; this could overestimate the maximum radius.
    - worker.py: `advance` function, line ~170
    - The maximum radius is used in the certificate, so an overestimate could cause a false pass.
    - Use the upper bound of the maximum explicitly.

39. [NIT] The `q0_area` is computed as `(UNSTABLE_X * STABLE_Y - STABLE_X * UNSTABLE_Y) * RADIUS_U * RADIUS_S`, but this assumes the unstable and stable directions are orthogonal; if they are not, the area is not the true area of the initial parallelogram.
    - worker.py: `main` function, line ~260
    - If the directions are not orthogonal, the area formula is incorrect, invalidating the determinant.
    - Use the actual area of the parallelogram spanned by the unstable and stable vectors.

40. [NIT] The `initial_normal` is computed as `initial_box[0] * initial_box[1] - ZS`, but the section is defined as `w = 0`, and the normal velocity is `x*y - zs`; this is correct, but the sign convention should be verified.
    - worker.py: `main` function, line ~265
    - If the sign convention is wrong, the event detection may be inverted.
    - Verify the sign convention against the ODE and the section definition.
