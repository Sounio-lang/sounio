<!-- docs:meta
topic_id: repo.docs.audit.type-archaeology-family-g-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: grok-cli5
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.type-archaeology-family-g-2026-08-19
-->

# Type archaeology family G — privacy and justice

**This is a census.** It does not reclassify `docs/internal/concepts/registry.tsv`. The founder decides promotions.

**SHA of `origin/main` at measurement:** `98eb2b4f41a3cecfa1eccae2f635ade3c62f653f` (`98eb2b4f41`)  
**Engine:** this worktree `bin/souc` → Madaros v0.80.0. Inherited `SOUC_BIN` / `SOUNIO_SOUC_BIN` unset. `SOUNIO_STDLIB_PATH` pinned here. lean_single was not used as a semantic authority.

**Table:** [`TYPE_ARCHAEOLOGY_FAMILY_G_2026-08-19.tsv`](TYPE_ARCHAEOLOGY_FAMILY_G_2026-08-19.tsv)

**Cross table (the spec skeleton):** [`TYPEKIND_CONCEPT_CROSS_2026-08-19.md`](TYPEKIND_CONCEPT_CROSS_2026-08-19.md)

**Assigned kinds (4):** DiffPrivate, DPBudget, FairPrediction, FairnessGap.

FairnessGap has no documentation. The only non-archive, non-training, non-compiler mention is `docs/audit/MODULAR_PIPE_COVERAGE_MAP_2026-06-01.md` listing it as a handler with 0 call sites.

## Method

One position per kind. Evidence **run this turn**, not read from `epistemic.sio`.

| Position | What had to be run this turn |
|---|---|
| Garden | Name assigned to this lane; this turn produced **no** diagnostic that distinguishes the TypeKind from a ghost name, and no program constructed a value of it. |
| Hypothesis | A checker rule **fired** that names the kind; no program constructed a value. |
| Executable | A program **constructed** a value **and** the checker imposed something. Not the final rung when Claim-ready also holds. |
| Claim-ready | A **wrong** program was **refused because of this type** — the diagnostic names the type's *meaning*, not just its wrapper wall. |

Hard rule applied: a type that only accepts is a label. `let x: Kind<f64> = 1.0` → `E001` is **not** Claim-ready. `fn id(x: Kind<f64>) -> Kind<f64> { x }` checking OK is **not** construction (`NoSuchType` does the same). `as Kind<f64>` checking OK is construction **only** when a later diagnostic names that TypeKind (DiffPrivate / DPBudget are lexer keywords; FairPrediction is not).

Forbidden moves not taken: classify FairPrediction as Hypothesis because `check_fair_prediction_type` exists unread; promote DPBudget to Claim-ready because E076 is printed in `check.sio`; treat `diff_private_basic.sio` checking OK as inhabiting DiffPrivate (it is f64 arithmetic with comments).

## Positive controls (zero Claim-ready is not an empty claim)

| Control | What would refute it | Result this turn |
|---|---|---|
| Mention is not a type | `fn id(x: NoSuchType<f64>) -> NoSuchType<f64> { x }` fails while FairPrediction / FairnessGap pass | **All three check OK.** Mention of `Kind<f64>` is worthless. |
| `as` of a ghost is not construction | `1.0 as NoSuchType<f64>` fails while `1.0 as FairPrediction<f64>` / `FairnessGap<f64>` pass | **All three check OK.** `as Fair*` does not construct TyFairPrediction / TyFairnessGap. |
| `as` of a keyword is construction | `1.0 as DiffPrivate<f64>` then pass to `fn takes(x: f64)` names DiffPrivate | **E009** `expected f64` / `found DiffPrivate`. Same shape for DPBudget. |
| E001 bind is a label | `let r: DiffPrivate<f64> = mech(42.0)` names DiffPrivate in a Claim-ready way | **E001** `expected DiffPrivate` / `found f64`. Same shape for DPBudget, FairPrediction, FairnessGap, and for `i64 = true`. |
| Claim-ready is distinguishable | A family-G kind produces E075 / E076 / E080 / E081 / E082 | **None fired.** Those printers exist; no user program reached them. |
| A real type can refuse | `let x: i64 = true` | **E001** `expected i64` / `found bool`. Claim-ready control for primitives, not for family G. |
| Knowledge is not a label | `let k: Knowledge<f64> = measure(...)` then `let x: f64 = k` | **E001** `expected f64` / `found Knowledge<f64>`. Inhabited wrapper with a wall. DiffPrivate has the wall without a `measure`. |

## Positions

### DiffPrivate — Executable

SHA `98eb2b4f41`.

Constructed this turn: `let a = 1.0 as DiffPrivate<f64>` checks OK, and passing `a` to `fn takes(x: f64)` is **E009 found DiffPrivate**. The annotation is a lexer keyword (`TokenKind::DiffPrivate`) lowered by `lower_diffprivate_type` to `ty_diff_private(_, 1000, -1)` — ε is hardcoded to 1000 milli-ratio. The user cannot write ε.

Imposition this turn: nominal wall only.

| Probe | rc | diagnostic |
|---|---:|---|
| `fn id(x: DiffPrivate<f64>) -> DiffPrivate<f64> { x }` | 0 | check OK |
| `1.0 as DiffPrivate<f64>` | 0 | check OK |
| `let r: DiffPrivate<f64> = mech(42.0)` (`mech -> f64`) | 1 | E001 expected DiffPrivate found f64 |
| `fn mech(x: f64) -> DiffPrivate<f64> { x }` | 1 | E008 expected DiffPrivate found f64 |
| `takes(a)` after `as DiffPrivate` | 1 | E009 expected f64 found DiffPrivate |
| `DiffPrivate<f64, f64>` (try to write ε) | 1 | E001 expected DiffPrivate found f64 — extra arg ignored, still no ε |
| tracked `tests/frontend/annotation_diffprivate_basic.sio` | 1 | E001 expected DiffPrivate found f64 (`//@ run-pass` is a lying tag) |
| tracked `tests/frontend/diff_private_basic.sio` | 0 | check OK — **no DiffPrivate value**; Laplace/Gaussian/composition are f64 comments |
| tracked `tests/compile-fail/annotation_diffprivate_mismatch.sio` | 1 | E001 expected DiffPrivate found i64 (`//@ ignore`) |

Not Claim-ready. The refuse is the wrapper wall. No program this turn was refused **because ε was wrong, because a mechanism was uncalibrated, or because sequential composition was violated**. E075 did not fire. `dp_sequential_compose` / `dp_parallel_compose` have **zero** call sites in `self-hosted/check/check.sio`.

### DPBudget — Executable

SHA `98eb2b4f41`.

Same inhabit path as DiffPrivate: `1.0 as DPBudget<f64>` checks OK. `lower_dp_budget_type` always builds `ty_dp_budget(_, 1000, 0)` — ε_total hardcoded 1000, **ε_spent hardcoded 0**.

| Probe | rc | diagnostic |
|---|---:|---|
| `fn id(x: DPBudget<f64>) -> DPBudget<f64> { x }` | 0 | check OK |
| `1.0 as DPBudget<f64>` | 0 | check OK |
| `let b: DPBudget<f64> = init(1000.0)` | 1 | E001 expected DPBudget found f64 |
| tracked `tests/frontend/annotation_dp_budget_basic.sio` | 1 | E001 expected DPBudget found f64 (`//@ run-pass` is a lying tag) |
| two sequential queries (below) | 0 | check OK |

Not Claim-ready. E076 did not fire.

### FairPrediction — Garden

SHA `98eb2b4f41`.

No lexer token. No `TypeExprKind`. `fn id(x: FairPrediction<f64>)`, `1.0 as FairPrediction<f64>`, and the same two programs with `NoSuchType` all check OK. `let p: FairPrediction<f64> = 1.0` is E001 expected FairPrediction found f64 — the same shape as a ghost name.

No diagnostic this turn distinguished TyFairPrediction from TyNamed("FairPrediction"). Checker rules (`check_fair_prediction_type`, E080, E082, `counterfactual_fairness_from_potential_outcome`) were not reached. They are not evidence under the run-not-read rule.

`tests/frontend/fairness_basic.sio` checks OK and never writes the type. It is a metrics notebook.

### FairnessGap — Garden

SHA `98eb2b4f41`.

Same surface as FairPrediction: no lexer token, no TypeExpr, ghost-identical probes. `ty_fairness_gap` is not even `pub`. `check_fairness_gap_type` (E081) did not fire.

No documentation. The coverage map already said 0 call sites. This turn confirms the user cannot name the TypeKind.

## The question that matters: does DPBudget spend?

A differential-privacy budget is a quantity that is **spent**. Sequential composition (Dwork 2006, Thm 3.14) says two queries on the same dataset cost ε₁+ε₂, which is more than one query.

**Run this turn:**

```
fn q1(b: DPBudget<f64>) -> DPBudget<f64> { b }
fn q2(b: DPBudget<f64>) -> DPBudget<f64> { b }
fn main() -> i64 {
    let b0 = 1000.0 as DPBudget<f64>
    let b1 = q1(b0)
    let b2 = q2(b1)
    0
}
```

`souc check` → **rc=0**. Two queries type-check as the identity. The type of `b2` is the type of `b0`.

`dp_budget_consume` exists in `self-hosted/check/epistemic.sio` and is called from **nowhere** in the live checker. `rg` this turn: definitions in `epistemic.sio` / `types.sio` comments / archived sprint-21 grep-gate. Zero call sites that would increment `array_size`.

So: two consultations do **not** spend more than one. They spend the same as zero. DPBudget is a number with a pretty name plus a nominal wall against bare `f64`. It is not a budget.

The tracked "composition" file `tests/frontend/diff_private_basic.sio` adds `f64` epsilons in user arithmetic (`spent = spent + query_epsilons[i]`). That is a comment with a loop, not type-level composition.

## What the compiler is ahead of

The TypeKind enum has the four names. The checker has E075–E082 printers and composition helpers. The parser has two of the four keywords. The language, at `98eb2b4f41` under Madaros, has two inhabited labels and two ghosts.

A spec that declared DiffPrivate / DPBudget / FairPrediction / FairnessGap as language types would be writing the enum, not the language.

## Counts

Garden=2  Hypothesis=0  Executable=2  Claim-ready=0

Hypothesis is empty on purpose. The positive control that would fill it is a diagnostic that names the kind without a constructed value (family B's E097 `deferral_policy` item). Family G did not produce one.

Claim-ready is empty on purpose. The positive control is `let x: i64 = true` (E001 expected i64 found bool) and family B's E108/E109/… named-kind refuses. Family G produced no E075/E076/E080/E081/E082.
