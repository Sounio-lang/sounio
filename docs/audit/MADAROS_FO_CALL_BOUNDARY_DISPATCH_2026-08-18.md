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

Thesis RHS (`rhs(c, cl, fu)` + `0.0 - … / 5.0`) is arity 3 **and** OpSub **and** OpDiv. Triple miss. `#1889` inlined it so adaptive/rk4 stay live; any new helper of that shape will under-report again.

## 5. What this does to a thesis

Any model whose right-hand side is a function — that is, every model written as a model — reports **less uncertainty than it has** under default `souc` (Madaros). The number is plausible. The only control is the other engine, which almost nobody runs.

`#1792` stays OPEN until CALL of a non-catalog shape is live. Do not close it with another inline.

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
