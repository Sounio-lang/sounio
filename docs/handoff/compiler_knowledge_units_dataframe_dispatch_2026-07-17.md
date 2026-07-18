<!-- docs:meta
topic_id: repo.docs.handoff.compiler-knowledge-units-dataframe-dispatch-2026-07-17
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.compiler-knowledge-units-dataframe-dispatch-2026-07-17
-->

# Dispatch to CODEX-2 — Compile-time dimensional units for the measurement DataFrame (`Knowledge<T>` binding)

**Date:** 2026-07-17
**Owner of the surface under change:** CODEX-2 (compiler; `self-hosted/`, `bootstrap/`)
**Author of this dispatch:** stdlib / data-science lane (does not edit the compiler)
**Status:** proposal — awaiting CODEX-2 review & scheduling

---

## 0. TL;DR

The compiler **already enforces dimensional units at compile time** — `check_binary_units`
(`self-hosted/check/check.sio:18696`) rejects `metre + second`, and `Knowledge<T>` type
expressions already compute `UnitDim` vectors (`knowledge_context.sio:183`). We are ~80% of the way
to a capability **pandas structurally cannot have**: a DataFrame whose columns carry SI dimensions
checked at compile time.

The one structural gap is small and precise: **products and quotients of united quantities drop
their unit** (two `// For now, result is f64 with no named unit` TODOs at `check.sio:18728` and the
`OpDiv` branch), because `UnitRegistry` can only be keyed by *name*, not by *dimension vector*. Fixing
that (~1 new method + ~8 changed lines) makes derived quantities (`velocity = m/s`,
`energy = kg·m²/s²`) compose end-to-end. Everything else is stdlib + a base-ordering alignment we can
do on our side.

This dispatch specifies the change set (P1–P4), acceptance tests, and the runtime artifacts already on
`main` that pin the intended semantics.

---

## 1. Why (the product intent)

We shipped the "measurement DataFrame" — the differentiator that makes Sounio *better* than pandas,
not just comparable. All merged to `main`, all run-proofed, no compiler changes:

- **Uncertainty (GUM-native):** `data::uncertain_frame` (#1102), `data::uncertain_groupby` (#1104),
  `data::uncertain_combine` (#1110) — column/group/fused reductions returning `epistemic::gum::GUMResult`
  (value ± U95, Welch–Satterthwaite dof).
- **Units (runtime):** `data::quantity` (#1105) — SI dimensional analysis, 7-exponent vectors,
  `q_add` panics on dimension mismatch.
- **Proofs:** `formal/SounioDataFrame.lean` (#1106) — machine-checked verb invariants.
- **Exactness:** `data::exact` (#1111) — zero-float-error rational analytics.

`data::quantity` gives **runtime** dimensional safety. The categorical win — the thing no float64
dataframe library can retrofit — is moving that check to **compile time**, so a pipeline that adds an
`mg/L` column to an `mmol/L` column simply does not compile. The machinery for this already lives in the
compiler; this dispatch is about finishing and exposing it.

---

## 2. What already exists (verified against source, 2026-07-17)

| Capability | Location | State |
|---|---|---|
| `struct UnitDim { exponents:[i64;7], scale_num, scale_den }` | `self-hosted/check/units.sio:25` | ✅ base order `[mass, length, time, temp, amount, current, luminosity]` |
| `unit_dim_mul` / `unit_dim_div` / `unit_dim_dimensionless` / `unit_dim_is_dimensionless` | `units.sio:112…` | ✅ exponent algebra |
| `struct UnitRegistry` + `register(name, dim)` / `find(name)->i64` / `get_dim(idx)->UnitDim` | `units.sio:178–228` | ⚠️ **name-keyed only — no intern-by-dim** |
| `register_builtin_units` | `units.sio:228` | ✅ base units pre-registered |
| `TypeEntry.unit_id: i64` (−1 = dimensionless) threaded through the checker | `check.sio` (many) | ✅ |
| `check_binary_units` — enforces `+`/`−`/comparisons need equal `unit_id`; `*`/`/` combine dims | `check.sio:18696` | ⚠️ **enforcement real; product/quotient unit dropped** |
| `report_unit_mismatch` (diagnostic) | `check.sio:18771` | ✅ |
| Referencing a declared unit yields `ty_with_unit(ty_f64(), uid)` | `check.sio:14511–14514` | ✅ so `5.0 * metre` already carries a unit |
| Unit declarations registered as `SymUnit` | `resolve/resolve.sio:413–415` | ✅ surface `unit` decls resolve |
| Function-arg + cast unit checking | `check.sio:21901`, `7008`, `22691` | ✅ |
| `Knowledge<T>` is `TypeKind::TyKnowledge`; type-exprs compute `UnitDim` (incl. `*`,`/`) | `epistemic.sio`, `knowledge_context.sio:183` | ✅ **Knowledge already carries unit dimensions at the type level** |

**Conclusion:** compile-time unit checking on unit-annotated scalars is *real today*, and `Knowledge<T>`
is already a unit-carrying type. The gaps are (a) derived-unit propagation through `*`/`/`, (b) a shared
base-ordering with the runtime `data::quantity`, and (c) attaching units to DataFrame columns.

---

## 3. The gap, precisely

### 3.1 Products/quotients drop the unit (the core blocker)

In `check_binary_units` (`check.sio:18728`), the both-sides-united `OpMul` branch computes the product
dimension and then throws it away:

```
BinaryOp::OpMul => {
    if lu >= 0 && ru >= 0 {
        let ld = c.units.get_dim(lu)
        let rd = c.units.get_dim(ru)
        let prod = unit_dim_mul(ld, rd)
        // For now, result is f64 with no named unit (product unit)
        // A future pass could register the product unit
        (c, result)                       // <-- unit_id lost
    } else { ... }
}
```

`OpDiv` has the identical shape. Consequence: `length * length` type-checks but the result is
dimensionless-typed, so `area + length` is **not** caught, and no derived quantity (velocity, energy,
force, pressure) can be dimensionally tracked. This is the single change that unlocks composable
dimensional algebra.

**Root cause:** `UnitRegistry` can only be keyed by `Name` (`register`/`find`), so there is nowhere to
put an *anonymous* product dimension and get a stable `unit_id` back.

### 3.2 Base-ordering mismatch with `data::quantity`

Runtime `data::quantity::Dim` uses `[m(length), kg(mass), s(time), amp(current), k(temp), mol(amount), cd(lum)]`.
Compiler `UnitDim` uses `[mass, length, time, temp, amount, current, luminosity]`. Only indices 2 (time)
and 6 (luminosity) align. The permutation is `compiler[i] = quantity[perm[i]]`, `perm = [1,0,2,4,5,3,6]`.
We would prefer to **re-order `data::quantity` to the compiler's canonical order** (a stdlib-only change
on our side) so there is one convention; recorded here so the two never silently diverge.

### 3.3 DataFrame columns carry no unit

The stdlib `data::dataframe::DataFrame` stores `f64` cells; columns have integer *name* ids but no
`unit_id`. So today the compile-time check only reaches unit-annotated *scalars* (including the
`GUMResult`/reductions we return), not whole columns.

---

## 4. Proposed change set

Ordered by leverage. P1 is the only strictly-required compiler change; P2 is ours; P3/P4 are the payoff.

### P1 — Intern product/quotient dimensions (REQUIRED, ~1 method + ~8 lines) — CODEX-2

Add an intern-by-dimension path to `UnitRegistry` (`units.sio`):

```
// Return a stable unit_id for a raw dimension vector, registering an anonymous
// entry (deduped by exponent equality) if none exists yet.
fn find_or_register_dim(self, dim: UnitDim) -> (UnitRegistry, i64) with Mut, Panic, Div { ... }
```

Dedup by comparing `exponents` (and scale) against existing entries; append if new. Then in
`check_binary_units`, both-united `OpMul`/`OpDiv` branches:

```
let prod = unit_dim_mul(ld, rd)               // or unit_dim_div for OpDiv
let (u2, uid) = c.units.find_or_register_dim(prod)
c = c with { units: u2 }
(c, ty_with_unit(result, uid))                // <-- keep the derived unit
```

For `OpDiv`, keep the existing `unit_dim_is_dimensionless(quot)` short-circuit → `result` (units cancel).
After this, velocity/area/energy propagate and downstream `+`/`−`/comparison checks fire on them.

**Acceptance (check-fail tests, new `tests/compiler/units_*`):**
1. `metre * metre` then `+ metre` ⇒ **unit mismatch error** (area ≠ length).
2. `metre / second` used where a `metre` is expected ⇒ error; used where the same `m/s` is expected ⇒ OK.
3. `metre * second / second` ⇒ dimensionally `metre` (units cancel), assignable to a `metre`.
4. Regression: existing base-unit `+`/`−` tests still pass; no false positives on dimensionless code
   (`lu < 0 && ru < 0` fast path unchanged).

### P2 — Shared canonical base order + a `std` units module (OURS, stdlib) — data lane

- Re-order `data::quantity::Dim` fields to the compiler's canonical
  `[mass, length, time, temp, amount, current, luminosity]`; update `data::quantity` run-proof
  accordingly. (Keeps a single convention; no permutation glue.)
- Add a stdlib units surface (`unit` decls for the 7 SI base + common derived: `newton`, `pascal`,
  `joule`, `watt`, `volt`, `hertz`, `mole_per_litre`, …) so user code and columns annotate against one
  registry. We draft; CODEX-2 confirms the `unit` declaration form and any parser constraints.

### P3 — Unit-typed measurement results & columns (PAYOFF) — joint

Two increments, smallest first:

- **P3a (small):** have the uncertainty reductions return a **unit-carrying** result. Since a value
  referencing a unit is already `ty_with_unit(ty_f64(), uid)`, `uframe_mean_type_a(...)` can be wrapped
  so its value participates in unit checks — e.g. a mean concentration typed `mg/L` cannot be added to a
  mean typed `mmol/L`. Mostly stdlib; needs P1 for any derived-unit results.
- **P3b (fuller):** per-column unit ids on the DataFrame. Two designs for CODEX-2 to weigh:
  (i) a `Quantity`-typed / `Column<unit>` storage, or
  (ii) **`Knowledge<f64>` columns** — preferred, see P4.

### P4 — Unify units with uncertainty via `Knowledge<T>` (VISION) — CODEX-2 + us

`TyKnowledge` already layers epsilon/validity/provenance **and** `unit_id`, and
`knowledge_context_unit_dim_from_type_expr` already computes dimensions for `Knowledge<…>` type
expressions. So a measurement column can be **one type** carrying *value + uncertainty + unit*, all
checked at compile time:

```
// a column of concentrations, each a measured value with GUM uncertainty, in mg/L
col: Knowledge<f64 in mg_per_L>
```

Then `uframe_mean` over a `Knowledge<f64 in mg_per_L>` column *is* a `Knowledge<f64 in mg_per_L>`, and
adding it to an `mmol/L` mean is a **compile error** — uncertainty and dimensional safety unified in the
type. This is the end state that makes the measurement-DataFrame claim airtight. Needs: confirming the
`Knowledge<T in unit>` surface syntax, and P1 so derived-unit measurements (e.g. a flux `mg/L/s`) type
correctly.

---

## 5. Acceptance criteria (definition of done)

- **P1:** the four check-fail/positive tests in §4 P1 pass; full existing compiler test suite green;
  `#print`-style dump shows product/quotient expressions carrying a non-negative `unit_id`.
- **P2:** `data::quantity` run-proof still `EXACT`-green after the base re-order; the new `std` units
  module `check`s and a smoke program annotating `5.0 * metre` type-checks.
- **P3a:** a run-proof/compile-fail pair: `uframe_mean` in `mg/L` + `uframe_mean` in `mmol/L` fails to
  compile with `report_unit_mismatch`; same-unit adds compile.
- **P4:** a `Knowledge<f64 in mg_per_L>` column example type-checks; a cross-unit add is rejected.

Prefer the existing gate style: a check-*fail* corpus (programs that MUST NOT compile, asserted by the
expected diagnostic) alongside positive run-proofs.

---

## 6. Scope / non-goals / risk

- **In:** derived-unit propagation (P1), base-order alignment + `std` units (P2), unit-typed reductions
  and columns (P3), `Knowledge<T>` unification (P4).
- **Out (for now):** unit *conversion* arithmetic (the `scale_num/scale_den` fields exist but automatic
  rescaling — `km` vs `m` — is a later pass); full derived-unit *naming* (anonymous ids are enough for
  checking); non-SI/affine units (°C, dB).
- **Risk:** P1's intern-by-dim must dedup correctly or the registry grows per-expression; a linear
  exponent-vector compare is fine at current scale. Guard the dimensionless fast path so scalar-heavy
  code sees zero overhead. Low blast radius — the change is additive and behind the existing
  `lu>=0 && ru>=0` branch.

---

## 7. Pointers

- Compiler: `self-hosted/check/units.sio` (UnitDim, UnitRegistry), `self-hosted/check/check.sio:18696`
  (`check_binary_units`), `self-hosted/check/knowledge_context.sio:183`
  (`knowledge_context_unit_dim_from_type_expr`), `self-hosted/resolve/resolve.sio:413` (SymUnit).
- Runtime semantics this should match: `stdlib/data/quantity.sio` (#1105) and its run-proof
  `tests/stdlib/data/test_quantity_stdlib.sio`; the measurement lane
  `stdlib/data/uncertain_frame.sio` (#1102), `uncertain_groupby.sio` (#1104),
  `uncertain_combine.sio` (#1110).
- Thesis / context: the "better than pandas" measurement-DataFrame lane (uncertainty + units + proofs +
  exactness). Compile-time dimensional safety is the one pillar that requires the compiler.
