<!-- docs:meta
topic_id: repo.docs.audit.pbpk-rapamycin-triage-2026-06-02
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.pbpk-rapamycin-triage-2026-06-02
-->

# dissertation_pbpk_rapamycin — 5-failure triage (2026-06-02)

**Verdict: COMPILER bug (struct-return zeroing), NOT clinical-value drift.** All 5 failing
tests trace to one cause: `rk4_step_pbpk` returns a ZEROED `PBPKState`.

Run on `bin/souc 9d4ef541`: `Passed: 2  Failed: 5`.
- T2 PASS (gut "empty" at 24h — but only because it was 0 from step 0, not from absorption).
- T7 PASS (study/Hypothesis compile-time block).
- **T1/T3/T4/T5/T6 FAIL**: no plasma, mass-balance violated, AUC≤0, no uncertainty — all
  downstream of a zeroed state.

## Empirical trace (instrumented the real demo)
- **Initial state correct**: `var state = PBPKState { gut: DOSE()*1000.0, ... }` → gut=2000.0
  (verified; the struct-literal-with-expr-field is fine here).
- **After the 1st `state = rk4_step_pbpk(&state, dt)`: gut=0.0 AND plasma=0.0** (verified).
  The entire state is zeroed by the first integrator step.
- **via-temp does NOT fix it** (`let ns = rk4_step_pbpk(&state,dt); state = ns` → still 0):
  the struct-RETURN itself zeros, not the assignment / self-aliasing.
- The numeric model is sound (KA=1.2, F=0.14, Vd=1200 → plasma derivative >0 initially).

## Why it's the codegen bug, not the model (5 falsified hypotheses, all by minimal probe)
Minimal probes on `bin/souc` that all WORKED (so the bug is NOT these): 2-field
`s = step(&s)` self-aliasing; 3-field `let k = rhs(...)` reading k.d0/k.d1/k.d2; struct
literal with constant fields; param-into-literal (in small f64 structs). The bug only
appears at `rk4_step_pbpk`'s complexity: a struct-returning fn that builds its result
struct-literal from MANY inner struct-returns (k1..k4 = `pbpk_rhs(...)`) and complex
field expressions. Same family as `SRET_FORWARDING_BUG_2026-06-02.md` (gdb-pinned: caller
sret pointer dropped) and the olanzapine smoker bug (`a3ea42082`).

## This is the 3rd dissertation/stdlib manifestation of the struct-return codegen bug
1. `olanzapine_pbpk_params_smoker` — copy-then-reassign → cl_hepatic=0 (fixed at source,
   constant literal, `a3ea42082`).
2. SRET-forwarding minimal repro — `return ctor()` zeros (gdb-pinned).
3. **pbpk_rapamycin `rk4_step_pbpk`** — complex struct-literal return zeros the whole state.

→ Strengthens the case that the **bin/souc large-struct/struct-return codegen bug is
actively blocking real dissertation research demos**, not just the modular compiler. The
durable fix is the codegen (the SRET-lowering owner / G1 lane); see SRET handoff.

## Possible source workaround (NOT applied — flagged, may relocate the bug)
Refactor `rk4_step_pbpk` to write the result through a mutable out-parameter
(`fn rk4_step_pbpk(s: &PBPKState, dt, out: &! PBPKState)`) instead of returning a struct —
arg-passing/out-params are in the WORKING set of the bug family. Caveats: must use a
separate `var next` to avoid `s`/`out` aliasing, requires `with Mut` field stores, and may
just relocate the miscompile. Not done here; the triage (code-vs-clinical) was the task.
