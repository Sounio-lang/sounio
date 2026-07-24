<!-- docs:meta
topic_id: repo.docs.handoff.compiler-units-bigframe-binding-dispatch-2026-07-23
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.compiler-units-bigframe-binding-dispatch-2026-07-23
-->

# Dispatch to CODEX-2 — Finish compile-time dimensional UNITS binding (the one differentiator pandas can never have)

**Date:** 2026-07-23
**Owner of the surface under change:** CODEX-2 (compiler; `self-hosted/`)
**Author:** stdlib / data-science lane (does not edit the compiler)
**Status:** proposal — supersedes `compiler_knowledge_units_dataframe_dispatch_2026-07-17.md` (re-verified against current source; line numbers drifted — see §2)

---

## 0. TL;DR

The compiler **already rejects `metre + second` at compile time** — `check_binary_units`
(`self-hosted/check/check.sio:18961`), `OpAdd` branch at `:18971` calls `report_unit_mismatch`
(`:19036`, diagnostic `E041`). That is a categorical capability **pandas structurally cannot have**:
a float64 column can never carry an SI dimension the type checker enforces. We are ~80% there.

Two precise, cited gaps remain:

1. **Products/quotients drop their derived unit** — `OpMul` (`check.sio:18994-18995`) and `OpDiv`
   (`:19013-19014`) compute the product/quotient `UnitDim` and then **discard it**, because
   `UnitRegistry` (`units.sio:178-214`) is **keyed by name only** — there is nowhere to intern an
   anonymous derived dimension and get a stable `unit_id` back. Fix: one `find_or_register_dim` method
   + ~8 lines at the two TODOs.

2. **`Knowledge<T>` values are NOT unit-checked at all today.** This corrects the prior dispatch's
   central claim. `ty_knowledge` (`types.sio:911`) hardcodes `unit_id: -1` on the outer wrapper, so
   every `Knowledge<…>` value reaches the binary check with `unit_id < 0` and hits the
   `lu < 0 && ru < 0` fast-path (`check.sio:4889`) that returns **before** `check_binary_units` runs.
   The dimension math for Knowledge type-exprs exists (`knowledge_context.sio:183`) but is
   **disconnected** from any value's `unit_id`. Fix: propagate a unit_id onto the wrapper in
   `checker_lower_knowledge_type_mut` (`check.sio:1517`).

P1 (intern derived dims) serves **both** paths. The scalar path works after P1 alone. The
Knowledge/column path works only after P1 **plus** the wrapper-propagation fix. That dependency is the
spine of this dispatch.

---

## 1. Why this is the single highest-leverage differentiator

The "better than pandas" measurement-DataFrame lane already shipped uncertainty (GUM-native
`uncertain_frame`/`uncertain_groupby`), runtime units (`data::quantity`), Lean proofs, and exact
rational analytics. Every one of those is a *runtime or library* property a determined pandas user
could approximate.

**Compile-time dimensional safety is the only pillar that is structurally impossible for pandas.**
A pandas `Series` is float64; there is no type in the language to attach `[mass, -length·3]` to, and no
type checker to reject `density_col + mass_col`. In Sounio the machinery is already inside the
compiler — `TypeEntry.unit_id` (`types.sio:150`) threads a registry index through every expression,
and addition of mismatched units is already a hard error. Finishing this makes a whole class of
scientific-data bug (adding `mg/L` to `mmol/L`, treating a flux as a concentration) a **compile
error**, not a silent wrong number. That is the airtight version of the pitch.

It **composes** with the layers already merged: the epistemic layer (`gpu/numerical.sio:992-1014`
maps roundoff/stability → the `epsilon`/`validity`/`provenance` layers of `Knowledge<T>`) and the
~325-verb bigframe surface (`stdlib/data/bigframe_ops.sio`, 325 `pub fn`s). The end state is one type —
`Knowledge<f64 in mg_per_L>` — carrying **value + GUM uncertainty + SI dimension**, all checked at
compile time.

---

## 2. What the source actually shows (re-verified 2026-07-23)

| Fact | Location | State |
|---|---|---|
| `struct UnitDim { exponents:[i64;7], scale_num, scale_den }`, order `[mass, length, time, temp, amount, current, luminosity]` | `units.sio:25-29` (comment `:26`) | ✅ |
| `unit_dim_mul` (add exps) / `unit_dim_div` (sub exps) / `unit_dim_is_dimensionless` | `units.sio:112 / 125 / 151` | ✅ exponent algebra |
| `unit_dim_mass_concentration` = `[1,-3,0,0,0,0,0]` (density) | `units.sio:79-85` | ✅ (acceptance anchor, §5) |
| `UnitRegistry` + `register(name,dim)` / `find(name)->i64` / `get_dim(idx)->UnitDim` | `units.sio:178-214` | ⚠️ **name-keyed only — no intern-by-dim** |
| `register_builtin_units` (base SI + a few derived pre-registered) | `units.sio:228` | ✅ |
| `check_binary_units` — `+`/`−`/comparisons require equal `unit_id` | `check.sio:18961`; `OpAdd` `:18971`; `OpSub` `:18979`; cmp `:19025-19030` | ✅ enforcement real |
| `OpMul` computes `prod` then returns bare `result` | `check.sio:18988-19002`; TODO `:18994-18995` | ⚠️ **product unit dropped** |
| `OpDiv` computes `quot`; dimensionless cancels, else returns bare `result` | `check.sio:19004-19022`; drop `:19013-19014` | ⚠️ **quotient unit dropped** |
| `report_unit_mismatch` → `E041` | `check.sio:19036` | ✅ |
| Referencing a declared unit → `ty_with_unit(ty_f64(), unit_idx)` | `check.sio:14776-14779` | ✅ scalars carry units today |
| `checker_finish_binary_units_inplace` reads `left.unit_id`/`right.unit_id`, returns early if both `<0` | `check.sio:4886-4895` (fast-path `:4889`) | ✅ (this is why Knowledge values skip the check) |
| Knowledge binary path calls the finish fn with the **outer** `left_ty`/`right_ty` | `check.sio:4950-4973` (call `:4961`) | ⚠️ |
| `checker_lower_knowledge_type_mut` → `ty_knowledge(inner_ty, eps_val)` | `check.sio:1517-1541` | ⚠️ **does not set a unit_id** |
| `ty_knowledge` hardcodes `unit_id: -1` on the wrapper | `types.sio:899-918` (`:911`) | ⚠️ **the core P4 gap** |
| `knowledge_context_unit_dim_from_type_expr` computes a `UnitDim` but callers use it only for named-unit registration / field-constraint checks | `knowledge_context.sio:183-236`, `246-252` | ⚠️ **dim math disconnected from value unit_id** |
| Runtime `data::quantity::Dim`, order `[m,kg,s,amp,k,mol,cd]` = `[length,mass,time,current,temp,amount,lum]` | `stdlib/data/quantity.sio:25-35` | ⚠️ **different base order** |

Note: the 2026-07-17 dispatch cited `check.sio:18696`/`18728`; those have drifted to `18961`/`18994`.
Line numbers in this doc are freshly read.

---

## 3. The asks

### P1 — Intern derived (product/quotient) dimensions — REQUIRED, CODEX-2 (~1 method + ~8 lines)

`UnitRegistry` can only be keyed by `Name` (`units.sio:191` `register`, `:200` `find`), so an anonymous
product/quotient dimension has no home. Add an intern-by-dimension path in `units.sio`:

```
// Return a stable unit_id for a raw dimension vector, registering an anonymous
// entry (deduped by exponent equality) if none exists. Reuses UnitEntry with a
// blank Name. entries[] cap is 32 (units.sio:179) — bump if the corpus needs it.
impl UnitRegistry {
    fn find_or_register_dim(self, dim: UnitDim) -> (UnitRegistry, i64) with Mut, Panic, Div {
        var i: i64 = 0
        while i < self.count {
            if unit_dim_compatible(self.entries[i].dim, dim) { return (self, i) }  // units.sio:139
            i = i + 1
        }
        var reg = self
        if reg.count < 32 {
            reg.entries[reg.count] = UnitEntry { name: Name { buf: [0;128], len: 0 }, dim: dim }
            let id = reg.count
            reg.count = reg.count + 1
            (reg, id)
        } else { (reg, -1) }
    }
}
```

Then at the two TODOs in `check_binary_units`:

```
// OpMul, both united — check.sio:18994-18995 (replace the two TODO lines + the bare `result`)
let prod = unit_dim_mul(ld, rd)
let ip = c.units.find_or_register_dim(prod)
c.units = ip.0
(c, ty_with_unit(result, ip.1))               // <- keep the derived unit (types.sio:1507)

// OpDiv, both united, non-dimensionless — check.sio:19012-19014
let iq = c.units.find_or_register_dim(quot)
c.units = iq.0
(c, ty_with_unit(result, iq.1))
```

Keep the `unit_dim_is_dimensionless(quot)` short-circuit (`check.sio:19009`) → bare `result` (units
cancel). After this, `length*length`, `metre/second`, `mass/volume` all propagate a `unit_id`, and the
existing `OpAdd`/`OpSub`/comparison checks (`:18971-19030`) fire on the derived results.

### P2 — Align `data::quantity` base order to the compiler's canonical order — OURS, data lane

Compiler order (`units.sio:26`) is `[mass, length, time, temp, amount, current, luminosity]`.
`data::quantity::Dim` (`quantity.sio:25-35`) is `[length, mass, time, current, temp, amount, lum]`. The
permutation is `compiler[i] = quantity[perm[i]]`, `perm = [1,0,2,4,5,3,6]` (only time@2 and lum@6
already align). We re-order the `Dim` fields and `dim_make`/`dim_mul`/`dim_div` (`quantity.sio:46/58/71`)
to the compiler order so there is **one** convention and no silent divergence between the runtime library
and the compile-time check. Stdlib-only; re-run the `data::quantity` proof after.

### P3 — Surface: attach unit annotations to bigframe columns → `Knowledge<f64 in unit>` — JOINT (needs the P4 wrapper fix)

Give a bigframe column a unit so the whole column is dimensionally typed. Two coupled compiler asks:

- **P3a — propagate unit_id onto the `Knowledge` wrapper.** In `checker_lower_knowledge_type_mut`
  (`check.sio:1517`), compute the inner/annotation dimension via the already-present
  `knowledge_context_unit_dim_from_type_expr` (`knowledge_context.sio:183`), intern it with
  `find_or_register_dim` (P1), and pass the resulting `unit_id` into `ty_knowledge`. Today
  `ty_knowledge` (`types.sio:911`) forces `unit_id: -1`; either add a unit_id parameter or wrap its
  result with `ty_with_unit`. **Without this, no Knowledge value is ever unit-checked** — it always hits
  the `lu<0 && ru<0` fast path (`check.sio:4889`).
- **P3b — surface syntax `Knowledge<f64 in mg_per_L>`.** CODEX-2 confirms the annotation form and the
  parser hook that fills `KnowledgeTypeInfo` with the unit name; the checker resolves it against the
  registry (`units.find`) exactly as scalar unit refs do (`check.sio:14776-14779`).

Once P3a lands, the Knowledge binary path (`check.sio:4961`) already routes through
`check_binary_units` — so unit checking on Knowledge columns comes "for free" from P1's machinery.

### (Vision, no new ask) Unified measurement type

With P1 + P3, a bigframe column is one type carrying value + GUM uncertainty + SI dimension.
`uframe_mean` over a `Knowledge<f64 in mg_per_L>` column *is* `Knowledge<f64 in mg_per_L>`; adding it to
an `mmol/L` mean is a compile error. This unifies the epistemic layer (`gpu/numerical.sio:992-1014`)
with dimensional safety in a single type.

---

## 4. Dependency spine

```
P1 (intern derived dims, check_binary_units)
 ├─ scalar derived units work immediately          (metre/second, mass/volume)
 └─ P3a (propagate unit_id onto TyKnowledge wrapper, check.sio:1517 → types.sio ty_knowledge)
      └─ P3b (surface `Knowledge<f64 in unit>`)
           └─ bigframe columns + uncertainty reductions are unit-checked
P2 (quantity base-order align) — independent, stdlib-only, do anytime
```

---

## 5. Acceptance criteria

**Regression guards (already true today — must stay true):**
- `metre + second` fails to typecheck with `E041` (`OpAdd`, `check.sio:18972 → 19036`). This already
  works; P1 must not break it.
- Dimensionless-only code sees zero behavior change (the `lu<0 && ru<0` fast path,
  `check.sio:18966`/`4889`, is untouched).

**P1 unlocks (new check-fail / positive corpus, `tests/compiler/units_*`):**
- `metre * metre` → area `[0,2,0,0,0,0,0]`; then `+ metre` ⇒ **`E041`** (area ≠ length). *Before P1 this
  compiles silently — the bug this fixes.*
- **Density anchor (chains mul + div, exercises `find_or_register_dim` on both paths):**
  `kg / (m * m * m)` → `m*m*m` = length³ `[0,3,0,0,0,0,0]`, then mass − length³ =
  `[1,-3,0,0,0,0,0]`, which is exactly `unit_dim_mass_concentration` (`units.sio:79-85`). The result
  must (a) type as a density-carrying `unit_id`, and (b) be **non-addable** to a plain `kg` mass ⇒
  `E041`.
- `metre * second / second` ⇒ units cancel to `metre` (`unit_dim_is_dimensionless` short-circuit,
  `check.sio:19009`), assignable to a `metre`-typed binding.

**P3 unlocks:**
- A `Knowledge<f64 in mg_per_L>` value/column type-checks; adding it to a `Knowledge<f64 in mmol_L>`
  value fails with `E041` (proves the wrapper now carries a unit_id and routes through
  `check_binary_units`).
- Two same-unit Knowledge means add cleanly (positive case).

**P2:**
- `data::quantity` run-proof still green after the base re-order; a smoke program annotating
  `5.0 * metre` type-checks.

Prefer the existing gate style: a **check-fail** corpus (programs that MUST NOT compile, asserted by the
expected `E041` diagnostic) alongside positive run-proofs.

---

## 6. Scope / non-goals / risk

- **In:** derived-unit propagation (P1), base-order alignment (P2), Knowledge-wrapper unit_id +
  column surface (P3).
- **Out:** automatic scale conversion (`scale_num`/`scale_den` exist, `units.sio:27-28`, but `km`↔`m`
  rescaling is a later pass); human-readable derived-unit *names* (anonymous interned ids suffice for
  checking); affine/non-SI units (°C, dB).
- **Risk:** `find_or_register_dim` must dedup by `unit_dim_compatible` (`units.sio:139`) or the registry
  grows per-expression; the `entries[32]` cap (`units.sio:179`) may need raising for a real pipeline —
  size it and guard the overflow (`return -1`) so an exhausted registry degrades to "unchecked", never
  miscompiles. Blast radius is low: every change is additive and behind the existing `lu>=0 && ru>=0`
  branches.

---

## 7. Pointers

- `self-hosted/check/units.sio` — `UnitDim` (`:25`), algebra (`:112`,`:125`,`:139`,`:151`),
  `UnitRegistry` (`:178-214`), `register_builtin_units` (`:228`), `unit_dim_mass_concentration` (`:79`).
- `self-hosted/check/check.sio` — `check_binary_units` (`:18961`), OpMul TODO (`:18994`), OpDiv drop
  (`:19013`), `report_unit_mismatch` (`:19036`), `checker_finish_binary_units_inplace` (`:4886`),
  Knowledge binary path (`:4950-4973`), `checker_lower_knowledge_type_mut` (`:1517`), scalar unit ref
  (`:14776`).
- `self-hosted/check/types.sio` — `TypeEntry.unit_id` (`:150`), `ty_with_unit` (`:1507`),
  `ty_knowledge` (`:899`, the `unit_id: -1` at `:911`).
- `self-hosted/check/knowledge_context.sio` — `knowledge_context_unit_dim_from_type_expr` (`:183`).
- `self-hosted/gpu/numerical.sio` — epistemic-layer bridge (`:992-1014`).
- `stdlib/data/quantity.sio` — runtime `Dim` (`:25`), `dim_make`/`dim_mul`/`dim_div` (`:46`/`:58`/`:71`).
- `stdlib/data/bigframe_ops.sio` — the ~325-verb column surface these units attach to.
