<!-- docs:meta
topic_id: repo.docs.audit.madaros-fo-call-boundary-dispatch-2026-08-18
authority: repo_only
audience: users
last_validated: 2026-08-18
validated_by: grok-cli2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-fo-call-boundary-dispatch-2026-08-18
-->

# Dispatch — first-order GUM transfer dies at a user function boundary

**Date:** 2026-08-18  
**Lane:** grok-cli2 / `fo-call-boundary`  
**Status:** OPEN (dispatch; **do not fix in this wave**)  
**Symptom people will search:** Madaros `variance_of` prints `0.000000` after a call, lean_single does not. That was filed as “Family A fabrication” / `#1792`.  
**Disease:** first-order sensitivity is dropped at the call, not at the printer. Fabrication is the observable.

`lower.sio` is **not claimed** by this lane. empryo-1 / gen2 raise may take it.

## 0. Why this name

A `0.000000` variance is a plausible exact measurement. It will not look like `2^63`. Anyone grepping “fabrication” or “Family A” a month from now will miss the compiler object. The object is:

> Madaros FO transfer across a user `fn` is a closed catalog. Shapes outside the catalog arrive with variance **exactly 0**.

Related, not the same:

| Name | What it is |
|---|---|
| Family B / `#1896` | `print_f64` saturates `\|x\| ≳ 1e19` → `2^63`. Value is correct. |
| April ζ | deep-chain `variance_of` slot OOB → `2^63`. Depth-dependent. |
| `#1543` / `gum_cross_function` | same-file `let r = x+y; return r` **already fixed**. Still PASS. |
| **This dispatch** | catalog miss / arity>2 / import / method. Value is **0**. |

## 1. Calc, not print

Measured 2026-08-18, dual-engine, source Madaros ELF `127ebcc7…` (post-KCONF, no E230) and `SOUNIO_SOUC_ENGINE=lean_single`.

A live GUM var on the thesis surfaces is ~`1e-4` (visible at 6 decimal places). On the failing CALL path, `v == 0` and `v * 1e12` still prints 0. Family B’s emitter is not involved.

## 1b. OpSub / negation — measured, not inferred (REFUTED as language-wide)

Sounio has no unary minus. Every `-k*C` is written `0.0 - k*C`, so it was reasonable to ask: if OpSub is a catalog hole, does **every negation in the language** drop uncertainty?

Deciding probe (`docs/audit/repro/fo_opsub_inline_vs_call.sio`): one pair of seeds (`measure(10,u=1)`, `measure(4,u=1)`), three expressions **side by side**, then the same three through a user `fn`.

| # | Expression | Through | Madaros | lean | GUM want |
|---|---|---|---|---|---|
| T1 | `a - b` | **inline** | **2 LIVE** | 2 LIVE | 2 |
| T2 | `0.0 - a` | **inline** | **1 LIVE** | 1 LIVE | 1 |
| T3 | `a * b` | **inline** | **116 LIVE** | 116 LIVE | 116 |
| T4 | `sub2(a, b)` | **call** | **0 ZERO** | 2 LIVE | 2 |
| T5 | `neg1(a)` → `0.0 - a` | **call** | **0 ZERO** | 1 LIVE | 1 |
| T6 | `mul2(a, b)` | **call** | **116 LIVE** | 2 LIVE | 116 |

**Verdict:** `0.0 - a` does **not** have a private path that skips FO, and it does **not** die inline. Inline OpSub uses `fo_combine_sens_addsub` (the same combine as add). The only special-case in the inline lowerer is `a - a` (same ident) → variance 0, which is correct.

So: **negation does not lose uncertainty**. Wrapping negation (or any OpSub) in a user `fn` does, because the **call catalog** has no OpSub. The title of this dispatch stays “FO dies at a user function boundary”. It does **not** become “negation loses uncertainty”. That claim is false and larger than the evidence.

Pin: `tests/run-pass/fo_opsub_inline.sio` (`expect-stdout: OP_SUB_INLINE_LIVE`) so a later call-catalog patch cannot “fix” OpSub by breaking the inline combine. Madaros inline is GUM-correct (2 / 1 / 116). lean is LIVE on all three (magnitude may peel-forward; that is not this dispatch).

## 2. Boundary — not every call

`fo_register_pure_fn_transfer` / `fo_classify_expr_transfer` in `self-hosted/ir/lower.sio` register only:

- 1 or 2 parameters (a third param **returns without registering**)
- identity of `p0`
- `p0 * lit`, `lit * p0`, `p0 * p1`
- `p0 + p1`
- forward `g(p0)` / `g(p0, p1)` with args in order

No `OpSub`, no `OpDiv`, no nested binary, no `p2`.

Same-file matrix (`docs/audit/repro/fo_call_boundary_matrix.sio`), `measure(10, u=1)` so peel var = 1; second seed `measure(4, u=1)`:

| # | Shape | Madaros value | lean value | Catalog? |
|---|---|---|---|---|
| C0 | peel `.value` | **1 LIVE** | 1 LIVE | n/a |
| C1 | `id1(x)` | **1 LIVE** | 1 LIVE | yes (id p0) |
| C2 | `x * 2` | **4 LIVE** (GUM `2²σ²`) | 1 LIVE | yes (scale) |
| C3 | `0.0 - x` | **0 ZERO** | 1 LIVE | no OpSub |
| C4 | `x / 5` | **0 ZERO** | 1 LIVE | no OpDiv |
| C5 | `x + y` | **2 LIVE** | 2 LIVE | yes (add) |
| C6 | `x * y` | **116 LIVE** (GUM) | 2 LIVE | yes (mul p0·p1) |
| C7 | `x - y` | **0 ZERO** | 2 LIVE | no OpSub |
| C8 | `id3(x,y,0)` 3 params | **0 ZERO** | LIVE (1.0 isolated; 2.0 in the two-seed matrix) | arity>2 abort |
| C9 | method `ops.mscale(x)` | **0 ZERO** | 1 LIVE | recv is p0; body uses p1 |

July control `gum_cross_function.sio` (same-file `let`/`return` add + scale): **PASS both engines**, `var(sum)=5`, `var(scaled)=16`. The catalog that `#1543` closed is still closed.

Imported `epistemic::fo` (`docs/audit/repro/fo_call_boundary_import.sio`) — **same shapes that are LIVE same-file**:

| Import | Madaros | lean |
|---|---|---|
| `fo_scale(x, 2)` | **0 ZERO** | 1 LIVE |
| `fo_add2(x, y)` | **0 ZERO** | 2 LIVE |
| `fo_mul2(x, y)` | **0 ZERO** | 2 LIVE |
| `fo_div2(x, y)` | **0 ZERO** | 2 LIVE |

Direct vs method vs imported:

- **Direct, same file, catalog, ≤2 params:** Madaros keeps FO and often the **GUM magnitude**.
- **Direct, same file, outside catalog or arity>2:** Madaros **0**.
- **Method (recv + f64):** Madaros **0** even when the body is `x * 2`.
- **Imported stdlib helper:** Madaros **0** even for `fo_scale` / `fo_add2` (catalog shapes).

It is not “every call”. It is “every call the catalog does not fire for”, plus “every imported call” on this instrument.

## 3. Which engine is the oracle

lean_single **never fabricated 0** on this matrix. It is the honesty oracle.

It is **not** a GUM-magnitude oracle. On `x * 2` lean reports 1 (the peel) where GUM wants 4; Madaros reports 4. On `x * y` lean reports 2; Madaros reports 116 (`10²·1 + 4²·1`). Copying lean’s transfer blindly would re-break the catalog that already matches GUM.

Fix direction: **extend Madaros transfer** (arity, `OpSub`/`OpDiv`, method recv, import prepass) so a miss fails closed or expands. Do not replace Madaros’s catalog with lean’s “forward the peel”.

## 4. Minimal repro

`tests/run-pass/fo_call_boundary_arity3.sio` — two functions, one variance, value is 0 on the far side:

```sounio
fn id3(a: f64, b: f64, c: f64) -> f64 { a }

fn main() -> i64 with IO, Epistemic {
    let k: Knowledge<f64> = measure(10.0, uncertainty: 1.0)
    let v = variance_of(id3(k.value, 0.0, 0.0))
    // Madaros: v == 0, rc=1, FO_CALL_BOUNDARY_ZERO
    // lean_single: v live, FO_CALL_BOUNDARY_LIVE
}
```

Op pin (same disease, other hole): `tests/run-pass/fo_call_boundary_neg.sio` (`fn neg1(x) { 0.0 - x }`).

Both are `//@ known-failure` with `expect-stdout: FO_CALL_BOUNDARY_LIVE` and exit 1 on zero. Honest pins, not passing tests with a tag.

Thesis RHS (`rhs(c, cl, fu)` + `0.0 - … / 5.0`) is arity 3 **and** OpSub **and** OpDiv. Triple miss **at the call**. The same `0.0 - k*C` written inline keeps FO (T2). `#1889` inlined it so adaptive/rk4 stay live; any new helper of that shape will under-report again.

## 4b. Arity ceiling — why 2, what a raise costs, does it cover the thesis

The skip is **not a regression**. It is the original line in `990168abc2` (2026-07-26), the commit that *created* the transfer table:

```
} else {
    // >2 params: skip (unsupported transfer)
    return lo
}
```

That commit stored **two names** (`p0`, `p1`) and five kinds that only mention those two (`id / lit·x / x+y / x·y / forward`). The skip is the table width, written as “unsupported”, not “we tried 3 and it crashed”. Bisect will not find a later introducing commit (grok-cli5, independent).

Call-site apply on **today’s main** still only reads arg0 (kinds 1, 2, 6) and arg1 (kinds 3, 4). There is no kind for `a+b+c` and no p2 in `fo_param_index`. Raising the skip without widening names + kinds + apply is a no-op for any body that is not already a 1- or 2-param catalog shape.

Measured identity-of-first-arg (`docs/audit/repro/fo_arity_ceiling.sio`), peel var = 1:

| Arity | Madaros | lean |
|---|---|---|
| 2 | **1 LIVE** | 1 LIVE |
| 3 | **0** | 1 LIVE |
| 4 | **0** | 1 LIVE |
| 8 | **0** | 1 LIVE |

grok-cli5’s additive matrix (do not restage): same-file `add3` expected 14 → Madaros 0; `add4` expected 14.25 → 0. Import/div stay on that lane (preserved branch `fix/fo-multimod-import-20260728` already has named commits for those two; this lane does not port them).

What the FO programme itself did when it outgrew 2 — **read, not ported**: it did not bump the skip. It added bytecode for nested `+/−/·/÷` at ≤4 (`4f8e7dcf23`), then a hash-indexed param table 4→8 (`8f0291ef63`), then 8→16 (`0b94cdeb0f`). That branch no longer contains `>2 params: skip`. Titles claim the lifts; they are a trail, not a receipt (main is hundreds of commits ahead).

### Does skip → 4 or → 8 cover the dissertation?

**No.** Two layers, both measured:

**Layer A — files that lose variance today (Knowledge / `variance_of` after a call).** After the #1889 inline, `rapamycin_epistemic_adaptive` and `rapamycin_rk4_budget` are LIVE (`FAMILY_A_VAR_LIVE`). `rapamycin_iso_budget` peels `.value` on purpose; Budget64 is the ISO path. `rapamycin_epistemic_pbpk` / `rapamycin_gum_vs_mc` do not call `variance_of`. The only remaining FO-call zero on this slice is `gum_fo_across_call.sio`: **one helper, arity 3**, body nest+`−`+`/`. skip=4 would *admit* it to classification; the classifier still would not register the body. A ceiling bump does not turn that 0 into a live GUM var.

**Layer B — helpers sitting in those files if the model is written as functions again.** In `rapamycin_rk4_budget.sio` the named RHS (called only from the plain finite-diff sim today) are `rhs_brain` **4**, `rhs_periph` **4**, `rhs_blood` **11**. `examples/dissertation_pbpk_rapamycin.sio` has `pbpk_rhs(y0,y1,y2)` arity **3**, but its uncertainty is hand-rolled finite-diff, not `variance_of`. Max arity on the Knowledge-shaped RHS is **11**. skip=8 still skips `rhs_blood`. Even skip=16 would still miss the body (nest / OpSub / OpDiv).

| Thesis helper (Knowledge path vs plain) | Arity | Body | skip=4 | skip=8 |
|---|---:|---|---|---|
| adaptive `ep_*` / `cube_root` | 1–2 | catalog or unused for FO | already in | already in |
| adaptive RHS | — | **inlined** in `main` (#1889) | n/a | n/a |
| rk4 Knowledge loop | — | **inlined** in `main` (#1889) | n/a | n/a |
| `rhs_brain` / `rhs_periph` (only called from plain finite-diff) | 4 | nest + OpSub + OpDiv | arity yes, body no | arity yes, body no |
| `rhs_blood` (plain finite-diff only) | **11** | nest + OpSub + OpDiv | **still skipped** | **still skipped** (11>8) |
| `gum_fo` `rhs(c,cl,fu)` | 3 | nest + OpSub + OpDiv | arity yes, body no | arity yes, body no |

Census of `fn` with arity > 2 in 41 thesis-ish `tests/run-pass` files (rapamycin / dissertation / gum_fo / pbpk / …): **58 helpers** (38×3, 15×4, 4×5–8, **1×11** = `rhs_blood`). None of those 58 have a catalog body (`id` / `p0+p1` / `p0*p1` / `ℓ·p0`). 37 contain `−` or `/`. A skip-only bump therefore **changes the behaviour of zero thesis helpers**. The 58 stay unregistered because of body shape (and `rhs_blood` also because 11 > 8).

Today’s thesis **numbers** are live because of inline, not because of the ceiling. If the model is written as functions again: a ceiling of 4 or 8 still misses `rhs_blood` (11) and still misses every body that is not id/scale/add/mul. A small skip bump is **not** a small fix with large thesis value. The large value needs the bytecode path (≤4 then 8 then 16) that the preserved FO programme already named — grok-cli5’s viability triage, not this lane.

Naive lift risk: changing `return lo` to “collect 2 names and classify anyway” would make `id3` look like `id1` (correct) and `add3` look like `add2` (silent **under**-estimate). Worse than an honest 0 if nobody is looking.

## 4c. How many thesis surfaces **traverse** arity ≥ 3 today (not: how many fail)

Measured 2026-08-19 on `origin/main` `91be3a0959`. Method: parse `fn` defs + call sites, BFS from `main`, follow same-file calls **and** resolved `use` imports. A surface counts if any reachable helper has arity ≥ 3. That is **traverse**, not fail — a surface can traverse and still print a plausible number if the lost term is small or if uncertainty is not `variance_of` FO.

Universe: the 51 entries in `TESTS` + `TESTS_SMOKE` of `scripts/ci/dissertation_pbpk_suite_gate.sh`.

Filter “FO-shaped”: arity ≥ 3, not `print*` / `estate_set` / `ode_params_set` / `t_test*` / string params. Printers with 3 args are not the ceiling hole.

| | Count |
|---|---:|
| Surfaces in the gate | **51** |
| Traverse FO-shaped arity ≥ 3 from `main` (**transitive**) | **46** |
| Do not traverse | **5** |

The five that do **not** traverse: `rapamycin_epistemic_adaptive` (RHS inlined), `dissertation_tirzepatide_demo`, `dissertation_vancomycin_demo`, `halo_pgx_gate_pass`, `rapamycin_kaxi_fuse_prior`.

Max arity reached on a traversing surface: **11** (`rhs_blood` on `rapamycin_rk4_budget`; `mc_auc_sd` on `gum_vs_mc`). Typical stack is tsit5 / BBB / oral runner (4–8).

Re-derive:

```bash
python3 scripts/dev/thesis_fo_arity_traverse_census.py
```

**This is not a fail count.** Most of the 46 do not print `variance_of=0` today (Budget64, finite-diff, Knightian, hand-rolled GUM). They still **cross** a helper the FO catalog cannot see. If that helper ever carries peeled Knowledge, the 0 is silent.

**Urgency:** 46/51 is not two surfaces. The bytecode port is not optional before September if any of those call chains is, or becomes, a Knowledge path. Adaptive is the exception (inlined), not the rule.

Limitations of the walker: line comments only; method recv counted in arity; `use` resolved by path heuristic under `stdlib/`; ambiguous callee names flagged, not guessed.

## 5. What this does to a thesis

The unconditional reading — every uncertainty that crosses any function is underestimated — is **false**. Same-file 1- and 2-arg catalog helpers are live and often GUM-correct.

The conditional reading is not: every Madaros number that crosses a ≥3-arg helper, or a non-catalog body (`OpSub`/`OpDiv`/nest), or (on this instrument) any imported helper, can die, and the 0 looks like an exact measurement.

`#1792` stays OPEN until CALL of a thesis-shaped helper is live. Do not close it with another inline or with a skip bump that leaves `rhs_blood` at 11.

## 6. Do not fix here

This is interprocedural FO in `self-hosted/ir/lower.sio` (`fo_register_pure_fn_transfer`, `fo_classify_expr_transfer`, import prepass, method recv). It is larger than a one-shape patch. A partial catalog extension that makes arity3 live and leaves `OpSub` dead will still zero every Euler `0.0 - k*c`.

**Acceptance for a later fix** (not this PR):

1. `fo_call_boundary_arity3` and `fo_call_boundary_neg` PASS under default Madaros (untag known-failure).
2. Same-file catalog cases stay GUM-correct (`gum_cross_function` still 5 / 16).
3. Dual-engine: Madaros must not go to 0 where lean is live, **and** must not regress catalog magnitudes to lean’s peel-forward.
4. Import of `fo_scale` / `fo_add2` re-measured; if still 0, file as a child of this dispatch, do not pretend the same-file fix closed import.

## 7. Reproduce

```bash
export MADAROS_STACK_KB=524288 SOUNIO_STDLIB_PATH=$(pwd)/stdlib
./bin/souc run tests/run-pass/fo_call_boundary_arity3.sio
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run tests/run-pass/fo_call_boundary_arity3.sio
./bin/souc run tests/run-pass/gum_cross_function.sio   # still PASS both
./bin/souc run docs/audit/repro/fo_call_boundary_matrix.sio
./bin/souc run docs/audit/repro/fo_call_boundary_import.sio
```

## 8. AI disclosure

Characterization and dispatch by grok-cli2 under human direction. GAIDeT-ICMJE 2025. Numbers re-derived this session; no compiler edit.
