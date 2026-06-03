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

## D feature — IMPLEMENTED & logically CORRECT, but BLOCKED on the by-value-struct codegen bug (2026-06-03)

Implemented the full Observe-gated observation-boundary feature and it WORKS logically:
- Keystone: unified Unobserved<T> lowering (checker_lower_type_expr_mut TypeNamed arm now
  produces TyUnobserved with inner, matching the by-value path — the in-place path was
  dropping te.type_args → ty_named("Unobserved"), retkind=6=TyNamed, the reason the boundary
  helper (keyed on TyUnobserved) never matched).
- Observe-gated (effect id 13) Unobserved<T> <-> T at: return-stmt, fn-body tail, let/var
  binding, call argument, and comparison (unwrap operands to inner for op-typing).
- RESULT: both D progs (unobserved_basic, observe_with_effect) check CLEAN; E008 5 → 1.

**But the full census REGRESSED catastrophically: PASS 209 → 89, CRASH 0 → 226.** Bisected:
NOT the keystone (reverting it left CRASH=226). The cause is the boundary HELPERS themselves:
`unobserved_boundary_compatible` / `unobserved_inner_or_self` do `*inner` on a
`Box<TypeEntry>` — i.e. a **272-byte TypeEntry passed/returned BY VALUE from a Box deref**.
That is EXACTLY the by-value-large-struct codegen bug we root-caused as the dominant crasher
(BLK-20260603-byval-arg-crasher-deref). These helpers run in hot checker paths
(every return/let/call/binary), so bin/souc miscompiles those functions → 226 crashes.

### Conclusion: D is BLOCKED on the by-value-struct-return/arg codegen fix
D's logic is correct and complete (proven: the D progs pass). It cannot LAND until the
large-by-value-struct codegen bug is fixed — any code that unwraps Unobserved<T> (`*box`)
hits it. So the dependency is concrete: **fix the by-value-struct codegen → then D lands as
written.** Reverted all D edits to the safe state (PASS 209, CRASH 0, E008 5 — no regression
shipped). The full D implementation is recorded here for re-application post-codegen-fix.

## Attacking the struct-by-value codegen bug DIRECTLY: NOT repro-isolable (2026-06-03)

Tried to reproduce the bug with the exact D-helper pattern in isolation: `fn deref_return(b:
Box<Big272>) -> Big272 { *b }` and `take_by_val(*box)` (deref a Box<272-byte struct> used
BY VALUE as arg AND return). On the current bin/souc: **compiles and runs CORRECTLY**
(r1=12, g2.a=7, g2.cc=5). So `*box` large-struct-by-value is NOT a localized miscompile.

Combined with the earlier evidence (the byval-arg-crasher lane: faulting fn had a 672KB
frame, rcx=-1 AND return-addr=0 = frame corruption; raising stack 12MB→1GB cleared 39
crashers; G1 got CRASH→0 by *mut-transcribing away the by-value Checker copies), the
conclusion is firm: **this is NOT a fixable localized codegen instruction bug — it is a
large-frame / stack-scale problem.** Functions that copy large structs (Checker 164KB,
TypeEntry 272B) by value build 0.5MB+ frames; at that scale the frame setup / deep recursion
corrupts (args read as −1, return address 0). It is layout-sensitive (any substantial
check.sio change shifts the binary and amplifies it — that is why adding D's boundary code
took CRASH 0→226 even with the keystone reverted).

### Consequence for "fix the codegen bug"
A direct codegen-instruction fix is intractable with available means: no isolable repro, and
the faulting function VARIES with layout, so editing core arg-passing/frame codegen blind =
2:36-rebuild guessing (and it already produced a 226-crash regression once). The PROVEN fix
is FRAME REDUCTION — G1's *mut transcription (no by-value large-struct copies) drove CRASH→0.
So the realistic path is NOT "fix the miscompile" but "don't build >0.5MB frames":
- For D: rewrite the boundary helpers FRAME-LIGHT — compare via the Box-taking
  `types_compatible_inner(a.inner, b.inner)` instead of `types_compatible(*box, …)`, and
  avoid returning `*box` by value — so D lands without ever materializing a large by-value
  struct. (Same avoidance strategy G1 used for the crashers.)
- A true codegen fix would be the "heap-allocated large locals" refactor (frames stay small
  regardless of by-value struct size) — a large architectural change, separate lane.
