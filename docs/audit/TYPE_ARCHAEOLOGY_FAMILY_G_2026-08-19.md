<!-- docs:meta
topic_id: repo.docs.audit.type-archaeology-family-g-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: grok-cli5
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.type-archaeology-family-g-2026-08-19
-->

# Type archaeology family G — privacy and justice (protocol v2)

**This is a census.** It does not reclassify `docs/internal/concepts/registry.tsv`. The founder decides promotions.

**SHA of `origin/main` at this re-evaluation:** `2b4d217a043ca061f8d1156384de029b4847e872` (`2b4d217a04`)  
**Engine:** this worktree `bin/souc` → Madaros v0.80.0. Inherited `SOUC_BIN` unset for the run. `SOUNIO_STDLIB_PATH` pinned here. lean_single was not used as a semantic authority.

**Do not read a position from this file.** Positions are the output of
`bash scripts/ci/kind_ladder_gate.sh`. Index (no position column):
[`tests/archaeology/kind_ladder/index.tsv`](../../tests/archaeology/kind_ladder/index.tsv).

**Table (historical v2 notes, not the ladder):** [`TYPE_ARCHAEOLOGY_FAMILY_G_2026-08-19.tsv`](TYPE_ARCHAEOLOGY_FAMILY_G_2026-08-19.tsv)

**Cross table:** [`TYPEKIND_CONCEPT_CROSS_2026-08-19.md`](TYPEKIND_CONCEPT_CROSS_2026-08-19.md)

**Assigned kinds (4):** DiffPrivate, DPBudget, FairPrediction, FairnessGap.

v1 (same files, SHA `98eb2b4f41`) is superseded as a *protocol* reading, not as a measurement. The defect was the ladder, not the runs. Positions did **not** move: nobody had marked Claim-ready without a passing construction.

## Protocol v2 (applied)

1. **Monotone.** Claim-ready ⇒ Executable ⇒ Hypothesis ⇒ Garden. No program that constructs the *kind* and **passes** ⇒ maximum is Hypothesis, no matter how many refusals fire.
2. **Reserva (off the ladder).** The compiler **actively** refuses every use with a **named** diagnostic, and no use passes. Name occupied, fail-closed, semantics unimplemented. Not Hypothesis (there the compiler is silent). Not Claim-ready (refusing everything is not discriminating). Honest, and worth more than Hypothesis. Control this turn: `f128` bind and signature-only are both **E218**. Family G is not Reserva — uses pass.
3. **The two-program test.** One program that **must** pass, one that **must** fail. Both fail ⇒ Reserva. Certo passes and errado fails ⇒ Claim-ready (the only case that is). Certo passes and errado passes ⇒ Executable. Without both programs ⇒ no position above Hypothesis.
4. **Layer (founder rule 3).** Every type must exist in every layer. Each row records the deepest layer that still **names** the kind: parser, checker, HLIR, IR, codegen. A checker kind the IR does not name is erasure, not design.

Format: `kind | posição | camada_mais_profunda | prog_certo | prog_errado | sha_main`.

## Corrections from v1

| kind | v1 | v2 | why |
|---|---|---|---|
| DiffPrivate | Executable | **Executable** | Certo (`as` + id) passes. The errado that tests the declared meaning (sequential composition / ε) also **passes**. Not Claim-ready. Not Reserva. |
| DPBudget | Executable | **Executable** | Certo (`as` + id) passes. Two queries, which **must** spend, pass with spent still 0. |
| FairPrediction | Garden | **Garden** | No program constructs `TyFairPrediction`. Surface is ghost-identical to `NoSuchType`. Compiler silent on the TypeKind. |
| FairnessGap | Garden | **Garden** | Same. `ty_fairness_gap` is not even `pub`. |

No row was a false Claim-ready. F128-style refuse-only did not occur in this family.

## Why the nominal pair is not Claim-ready

This pair exists for every name, including ghosts:

| program | DiffPrivate | DPBudget | FairPrediction | FairnessGap | NoSuchType |
|---|---|---|---|---|---|
| `1.0 as Kind<f64>` | rc=0 | rc=0 | rc=0 | rc=0 | rc=0 |
| `fn id(x: Kind<f64>)` | rc=0 | rc=0 | rc=0 | rc=0 | rc=0 |
| `let x: Kind<f64> = 1.0` | E001 expected DiffPrivate | E001 expected DPBudget | E001 expected FairPrediction | E001 expected FairnessGap | E001 expected NoSuchType |
| `takes(a): f64` after `as` | E009 found DiffPrivate | E009 found DPBudget | E009 found FairPrediction | E009 found FairnessGap | E009 found NoSuchType |

If (as, coerce) were enough, `NoSuchType` would be Claim-ready. It is not a type. That pair is therefore **not** the test. Control this turn: `let x: i64 = 1` passes and `let x: i64 = true` is E001 — that pair *is* Claim-ready, because `i64` is not ghost-identical to `NoSuchType`.

`as DiffPrivate` / `as DPBudget` still count as constructing the **kind** (lexer keyword → `TypeExprKind` → `ty_diff_private` / `ty_dp_budget`). `as Fair*` does **not**: there is no TypeExpr; the checker never calls `ty_fair_prediction` / `ty_fairness_gap` from `check.sio`.

## Tabela v2

| kind | posição | camada_mais_profunda | prog_certo | prog_errado | sha_main |
|---|---|---|---|---|---|
| DiffPrivate | **Executable** | **checker** | `/tmp/archaeology-g-v2/dp_certo_id.sio` — `1.0 as DiffPrivate<f64>` then `id` — **rc=0** | `/tmp/archaeology-g-v2/dp_errado_compose.sio` — two sequential identity queries **must** compose ε or refuse — **rc=0** (they do neither) | 2b4d217a04 |
| DPBudget | **Executable** | **checker** | `/tmp/archaeology-g-v2/bud_certo_id.sio` — `1000.0 as DPBudget<f64>` then `id` — **rc=0** | `/tmp/archaeology-g-v2/bud_errado_twoq.sio` — two queries **must** spend more than one — **rc=0** (spend stays 0) | 2b4d217a04 |
| FairPrediction | **Garden** | **checker** | *none* that constructs `TyFairPrediction` (`as` / `id` are the NoSuchType twin) | *none* that names the TypeKind (`E001 expected FairPrediction` is the ghost wall) | 2b4d217a04 |
| FairnessGap | **Garden** | **checker** | *none* | *none* | 2b4d217a04 |

### Layer evidence (rule 3)

| kind | parser | checker | HLIR | IR | codegen | deepest named |
|---|---|---|---|---|---|---|
| DiffPrivate | yes — `TokenKind::DiffPrivate`, `TypeExprKind::TypeDiffPrivate` | yes — `TyDiffPrivate`, `lower_diffprivate_type` → `ty_diff_private(_, 1000, -1)` | **no** | **no** | **no** | checker |
| DPBudget | yes — `TokenKind::DPBudget`, `TypeExprKind::TypeDPBudget` | yes — `TyDPBudget`, `lower_dp_budget_type` → `ty_dp_budget(_, 1000, 0)` | **no** | **no** | **no** | checker |
| FairPrediction | **no** TypeExpr | yes — `TyFairPrediction` in the enum; ctor never called from `check.sio` | **no** | **no** | **no** | checker |
| FairnessGap | **no** TypeExpr | yes — `TyFairnessGap`; `ty_fairness_gap` is not `pub`; ctor never called from `check.sio` | **no** | **no** | **no** | checker |

`souc compile` of the three `as` programs (DiffPrivate, DPBudget, FairPrediction) all succeeded this turn and emitted **8648-byte** ELFs. The name is erased. The backend never sees the kind. That is the checker→HLIR debt on a live program, not a missing file.

## Runs this turn (Madaros v0.80.0, SHA `2b4d217a04`)

| Probe | rc | diagnostic |
|---|---:|---|
| `ctrl_i64_certo` (`let x: i64 = 1; x+x`) | 0 | check OK — Claim-ready control, certo |
| `ctrl_i64_errado` (`let x: i64 = true`) | 1 | E001 expected i64 found bool |
| `ctrl_f128_bind` / `ctrl_f128_sig` | 1 / 1 | **E218** both — Reserva control |
| `ctrl_knowledge_certo` (`measure`) | 0 | check OK |
| `ctrl_knowledge_errado` (coerce to f64) | 1 | E001 expected f64 found Knowledge\<f64\> |
| `dp_certo_as` / `dp_certo_id` | 0 / 0 | constructs `TyDiffPrivate` |
| `dp_errado_bind` (mech→f64 into DiffPrivate) | 1 | E001 expected DiffPrivate found f64 |
| `dp_errado_return` (`fn mech -> DiffPrivate { x }`) | 1 | E008 expected DiffPrivate found f64 |
| `dp_errado_coerce` | 1 | E009 expected f64 found DiffPrivate — **ghost-shaped** (see above) |
| `dp_errado_compose` (two identity queries) | 0 | **the meaning-errado passes** |
| `bud_certo_as` / `bud_certo_id` | 0 / 0 | constructs `TyDPBudget` |
| `bud_errado_bind` | 1 | E001 expected DPBudget found f64 |
| `bud_errado_coerce` | 1 | E009 expected f64 found DPBudget — ghost-shaped |
| `bud_errado_twoq` | 0 | **two queries spend the same as zero** |
| `fp_*` / `fg_*` / `ghost_*` as, id, bind, coerce | identical | Fair* is TyNamed |
| `tests/frontend/annotation_diffprivate_basic.sio` (`//@ run-pass`) | 1 | E001 expected DiffPrivate found f64 — official certo **fails** |
| `tests/frontend/annotation_dp_budget_basic.sio` (`//@ run-pass`) | 1 | E001 expected DPBudget found f64 — official certo **fails** |

E075 / E076 / E080 / E081 / E082 did not fire. `dp_budget_consume`, `dp_sequential_compose`, `dp_parallel_compose`, `check_fair_prediction_type`, `check_fairness_gap_type` have **zero** call sites in `self-hosted/check/check.sio`.

## Does DPBudget spend?

A differential-privacy budget is a quantity that is **spent**. Sequential composition (Dwork 2006, Thm 3.14) says two queries on the same dataset cost ε₁+ε₂.

`bud_errado_twoq.sio` type-checks as the identity. `lower_dp_budget_type` always builds `ty_dp_budget(_, 1000, 0)` — ε_total hardcoded 1000, **ε_spent hardcoded 0**. `dp_budget_consume` is never called.

Two consultations do not spend more than one. They spend the same as zero. DPBudget is a number with a pretty name plus a nominal wall. It is not a budget.

That is why the position is Executable and not Claim-ready: the program that **must** fail (or at least change the type) **passes**.

## Counts v2

| posição | n | kinds |
|---|---:|---|
| Garden | 2 | FairPrediction, FairnessGap |
| Hypothesis | 0 | — |
| Executable | 2 | DiffPrivate, DPBudget |
| **Reserva** | **0** | Family G occupies names that still accept `as` / `id`. That is not fail-closed. |
| Claim-ready | 0 | no pair that constructs the kind **and** refuses the meaning-errado |

Hypothesis remains empty on purpose. Promoting Fair* by reading `check_fair_prediction_type` would violate run-not-read. v2's ceiling without both programs is Hypothesis; Garden is still the honest floor when the compiler is silent on the TypeKind.

## Dívida de camadas (Regra 3)

Re-measured this turn on this worktree's `self-hosted/check/types.sio` + `self-hosted/hlir/ir.sio`:

| measure | n | note |
|---|---:|---|
| TypeKind (checker) | 99 | including Family G |
| HlirTypeKind variants in source | 44 | `Contest` and `Robust` each appear twice |
| HlirTypeKind unique | 42 | |
| same stem (`TyX` ↔ `HlirTypeX`) | **19** | Array Bool Contest Counterfactual F32 F64 I128 I32 I64 I8 Intervention Knowledge Robust Tuple U128 U32 U64 U8 Validated |
| HLIR-only unique | **23** | founder brief said 24; measured 23. Dual Function I16 Mat2 Mat3 Mat4 Octonion Ptr Quat QuatConv2d QuatGate QuatLinear QuatRnnState Sedenion Struct U16 Vec2 Vec2d Vec3 Vec3d Vec4 Vec4d Void |
| of which algebra (Octonion, Sedenion, Quat*, Dual, Vec*, Mat*) | **17** | backend knows them; checker TypeKind does not name them |

Both directions are debt. Family G is the first direction: four checker names, zero HLIR names. The algebra set is the inverse: HLIR-only.

A spec that declared DiffPrivate / DPBudget / FairPrediction / FairnessGap as language types would be writing the enum, not the language. A spec that declared Octonion as absent because it is not a TypeKind would be writing the checker, not the backend.

## What a spec may not say

- That Sounio has ε-differential privacy as a type. DiffPrivate is an Executable nominal wrapper. ε is hardcoded. E075 is silent.
- That Sounio has a privacy budget. DPBudget does not spend.
- That Sounio has fairness as a type. FairPrediction and FairnessGap are Garden.
- That refusing `f64` as `DiffPrivate` is Claim-ready. `NoSuchType` does the same.

The decision to add Concept-IDs for privacy and justice, or to delete the TypeKinds, is the founder's. This census does not do either.
