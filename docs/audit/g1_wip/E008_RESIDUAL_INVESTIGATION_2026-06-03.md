# E008 residual (5 progs) — investigation + B-array verdict + D correct-fix design — 2026-06-03

After the literal-narrowing fixes (int + float), 5 E008 progs remain. NONE is a clean
mechanical bug; all are semantically-loaded. Investigated each:

## B — int-width (ffi_ctypes, fft_spectral): NOT a clean "safe return extension"
The ffi_ctypes array case is a **let/var binding**, not a return:
`var data: [i32; 256] = [0; 256]` — `[0;256]` is `[i64;256]`. The existing let-narrowing
(check.sio:4533/4589) gates on `is_integer_type(ann)`, which is FALSE for an array type, so
it doesn't fire. And `last_literal_kind` is a BAD signal for arrays: the array-repeat handler
checks the count literal (`256`) last, so `last_literal_kind==1` even when the ELEMENTS are
non-literal → gating on it would wrongly allow `[somevar;N]→[i32;N]` truncation. A SAFE fix
needs structural detection (returned/bound expr is an array-literal whose elements are int
literals), spanning the let AND return checks — more than the "safe return extension" it
appeared. Deferred as its own scoped task; NOT done (avoids unsafe truncation allowance).

## C — unit (unit_energy_explicit_conversion): leave mc strict
`fn takes_ev(energy: eV) -> f64 { energy }` returns eV as f64. Coercing eV→f64 implicitly
drops dimensional checking. mc is arguably CORRECT to require `energy as f64`; the test
(named *explicit_conversion*) likely should write the explicit cast. Design call — NOT coerced.

## D — epistemic (observe_with_effect, unobserved_basic): correct fix DESIGNED (not a coercion)
The intent (test comment: "boundaries satisfied because `with Observe`"): Unobserved<T> <-> T
is allowed ONLY when the `Observe` effect (id 13) is in scope — an effect-gated observation
boundary, NOT a blanket coercion (preserves the safety: no observe without `with Observe`).
The correct fix unwraps Unobserved<T> at ALL observation boundaries, each Observe-gated:
- return statement (checker_check_return_expr_inplace) — E008
- fn-body tail expr (the 2529 check) — E008
- `let y: f64 = x` binding — E001
- `consume(u)` call argument — E009
- `reading > 36.0` comparison — E004 (the EXISTING handler at 2758 checks the Observe effect
  but then `check_binary_op_types(Unobserved<f64>, f64)` still rejects — it must UNWRAP the
  Unobserved to its inner before typing the op).
So D is a genuine, multi-site epistemic FEATURE (unwrap-at-boundary, effect-gated), touching
the language's core. Tried return+tail with an `unobserved_boundary_compatible` helper
(effect-gated) — the right shape, but it does not fully connect because the other sites
(let/call/compare) and the binary-op unwrap are also needed; reverted to keep the branch
clean. Implementing it properly is the next scoped task and should be done carefully
(it defines observation semantics), not rushed as a coercion.

## Net
E008 133 → 5 (literal classes resolved, validated). The 5 residual = 1 deferred (B, needs
array-literal structural detection) + 1 design-leave (C, dimensional safety) + 3 needing the
D epistemic-boundary feature. None should be "fixed" by blanket coercion.
