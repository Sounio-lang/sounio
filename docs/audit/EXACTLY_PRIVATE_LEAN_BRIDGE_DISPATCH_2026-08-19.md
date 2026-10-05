<!-- docs:meta
topic_id: repo.docs.audit.exactly-private-lean-bridge-dispatch-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: grok-cli4
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.exactly-private-lean-bridge-dispatch-2026-08-19
-->

# ExactlyPrivate&lt;T&gt; ↔ Lean theorem bridge — forensic dispatch

**Date:** 2026-08-19  
**sha measured:** `6ce6e4dafd` (worktree tip; Lean CI evidence from run on `bd361fd0923b`)  
**Host (souc check):** Slurm `cpu-ops` / job 10421 / `cpuops-t560-proxmox`  
**Host (git / Lean CI log pull):** login worktree `sounio-workspace-control-0`  
**Write set:** this document + `docs/audit/exactly_private_lean/**` receipts only  
**Forbidden this round:** any edit under `self-hosted/` or `formal/`

Receipts: `docs/audit/exactly_private_lean/`

---

## Semantic lane declaration

```text
Semantic-Lane-ID: exactly-private-lean-bridge-20260819
Owner: grok-cli4
Concept-IDs: none claimed (surgical wrappers are type-system surface, not registry Status rows)
Intent-Preserved: Lean algebra stays the claim-oracle for ZD kernels; checker remains ceremony until founder chooses a bridge
Transformation: none — measurement + proposed correction with cost
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced:
  - ExactlyPrivate<T> checker path is effect-id ceremony (ZD=18) only; no annihilation obligation
  - Lean proves generic 4-annihilators for every primitive AND a concrete A=e3+e10 alice_kernel
  - G9 Audited "emit Lean obligation" is a fixed Lean term, not a compiler emit path for surgical ops
  - Madaros default engine does not fire E201 on missing ZD; lean_single does
  - Eight-wrapper Lean table: six algebraic partial, Forgettable necessity, Interpretable basis-only
Claims-Forbidden:
  - "ExactlyPrivate is proven at use sites" (ceremony only)
  - "Audited emits Lean per op today" (no surgical emit path measured)
  - "type is generic-over-A with full theorem coverage" without distinguishing generic vs A-only theorems
  - any schedule or self-hosted patch as done work
Assumptions: type-position use counted as `\bWrapper\s*<` in .sio outside self-hosted/ and compile-fail/
Write-Set: docs/audit/EXACTLY_PRIVATE_LEAN_BRIDGE_DISPATCH_2026-08-19.md, docs/audit/exactly_private_lean/**
Read-Set: formal/lean4/SounioSurgical*.lean, SounioImpossibilityChain.lean, SounioInterpBasis.lean,
          SounioZeroDivisorBridge.lean, lakefile.lean, self-hosted/check/check.sio (read-only),
          examples/zd_*, stdlib/{privacy,epistemic,safety}/*, .github/workflows/ci.yml, tmp ceremony witnesses
Positive-Witness: lake build of SounioSurgicalInterventions in CI Lean Proofs job (success)
Negative-Witness: ExactlyPrivate<f64> + with ZD + empty body typechecks (ceremony_default_rc=0)
Acceptance-Gate: re-run dual-engine ceremony on Slurm; re-pull Lean CI job log for default_target build
Integration-Target: founder decision only
Authoritative-Only-If: engines named per cell; no self-hosted/formal edits in the PR
```

---

## Executive finding (three layers that do not speak)

| Layer | What exists | What it does **not** do |
|---|---|---|
| **1. Lean** | `formal/lean4/SounioSurgicalInterventions.lean` names `ExactlyPrivate<T>` in comments and proves ZD kernel facts (`unlearning_kernel_exact`, `every_primitive_has_4_annihilators_restated`, …). Module is `@[default_target]`; CI `lake build` succeeds. | Does not type-check Sounio programmes. Does not bind a Sounio type use-site to a theorem name. Full type semantics deliberately out of scope (file header). |
| **2. Example convention** | `examples/zd_machine_unlearning.sio` applies `A = e3+e10` product and cites the Lean kernel in comments. `stdlib/privacy/exactly_private.sio` implements `forget_contribution` on `[f64;16]` with `with ZD`. | Neither file puts a live value in type position `ExactlyPrivate<…>` that the checker could discharge against Lean. Compliance is human convention. |
| **3. Checker** | Eight `lower_*` wrappers in `self-hosted/check/check.sio` (~155 lines, 16705–16859): each requires effect ID(s) then **returns `inner_ty`**. E200–E207 messages name the theorems’ *intent*. | No check that any annihilation, fibre locality, complement preservation, Lean term, or Temporal window is present in the body. **Ceremony, not proof.** |

This is the SOUNIO-TYPE-INTERROGATION third-type failure: the name of a theorem on a type, without a machine link from use-site to theorem.

---

## 1. Minimum proposition for non-ceremony

### Phrase (candidate obligation)

> **At every use of `ExactlyPrivate<T>` that constructs or transforms a value claimed to be “forgotten”, the checker (or an emitted obligation discharged by Lean) must establish that the contribution subspace is contained in a right-annihilator kernel of a declared ZD primitive `A`, and that the programme applies an operator that maps that kernel to zero (algebraically, not ε-approximately).**

Unpacking:

1. **Declared `A`** — which primitive (or linear combination) is the unlearning operator.
2. **Kernel membership** — the forgotten contribution lives in `annihilatorsOf(A)` (4-D for every primitive).
3. **Action** — some operation realises `A · w` (or equivalent projection) so kernel components become zero.
4. **Discharge** — either static proof on a decidable fragment, or a Lean obligation that CI / `lake env lean` re-checks.

### What is statically decidable today (finite PrimSed model)

Already proved in Lean with `native_decide` / computational facts (no Mathlib, no `sorry`):

| Fact | Theorem / def | Scope |
|---|---|---|
| Every primitive has exactly 4 right-annihilators | `every_primitive_has_4_annihilators` / `_restated` | **All 84** primitives |
| 168 projective ZD classes | `zd_classes_168` / `basis_168_completeness` | Global graph |
| Concrete kernel of `A = e3+e10` has length 4 and annihilates | `alice_kernel_is_4d`, `unlearning_kernel_exact`, `alice_kernel_fully_annihilated` | **One** `A` (the example’s Alice operator) |
| Fibre locality of annihilation | `editing_locality_kernel_bound` | All primitives |
| Complement size 79 | `capability_removal_preserves_complement` | All primitives |

So: **kernel dimension and pair-hood are decidable on the finite primitive basis.** A checker that only ever talks about `PrimSed` indices and enumerated annihilator lists could, in principle, verify “this weight is a linear combination of these four basis vectors” and “this op is multiplication by this `A`”.

### What is not decidable (or not wired) as a Sounio static check

| Obligation fragment | Why it escapes today’s checker |
|---|---|
| Arbitrary `T` payload is “the contribution of user U” | Semantic, not algebraic; needs provenance / encoding convention. |
| Runtime `[f64;16]` weights equal a stated kernel combination after float ops | Floating-point; at best residual bounds, not exact Lean equality. |
| “Some annihilating op appears in the CFG” without a named primitive API | Undecidable pattern search over general code; needs a **closed surface** (e.g. only `forget_contribution` / `sed_mul(A,·)` builtins count). |
| Full `ExactlyPrivate<T>` denotational semantics | Explicitly out of scope in the Lean file’s propositional scaffold. |

### Evaluation of founder candidates

| Candidate | Verdict | Notes |
|---|---|---|
| **(a)** Value constructed from the annihilator kernel of a declared ZD element | **Partial / best static core.** Matches `alice_kernel` + generic 4-annihilator theorems. Requires a type parameter or ghost `A: PrimSed` (or closed set of named operators). Decidable on PrimSed; not on opaque `T`. |
| **(b)** Body contains an operation `A ∘ x` with `A` tied to the theorem | **Syntactic heuristic unless `A` and `∘` are builtins.** Today’s example uses sedenion product by convention; checker does not see it. Could be a **dataflow fact** if product is a single intrinsic. |
| **(c)** Proof obligation emitted and discharged | **Architectural target; machinery exists for a different domain.** `madaros emit-lean-obligations` records **Equivalence Theory / Invariant** sites, not surgical wrappers (see §2). G9’s “witness” in Lean is a **fixed term** `audit_witness_is_derivation`, not a per-call-site emit. |

**Recommended minimum for a non-ceremony `ExactlyPrivate` (founder choice, not implemented):**  
parameterise the wrapper as `ExactlyPrivate<T, A>` (or attach `A` via a ghost/effect payload), accept only values produced by a **closed constructor set** whose Lean image is “in `annihilatorsOf(A)`”, and require the forgetting path to call a **single intrinsic** whose Lean image is `unlearning_kernel_exact`-style action for that `A`. Generic `A` is covered by `every_primitive_has_4_annihilators`; the concrete Alice example remains a specialisation, not the whole type.

---

## 2. Theorem citation and Audited&lt;T&gt; (G9)

### How a type could name a Lean theorem verifiably

Viable patterns (none fully wired for surgical types today):

1. **String/stable theorem ID in metadata** + CI job that `lake env lean` checks `#check` / `#print axioms` for that ID (claim-oracle style; Madaros is the Sounio clock, Lean is the algebra clock).
2. **Emit a `.lean` obligation per use-site** that imports `Sounio.SurgicalInterventions` and proves a concrete goal (kernel membership / action), then `lake build` that file in CI.
3. **Dependent/ghost index** `ExactlyPrivate A T` elaborated to a Lean term of type `isZeroPair A v` for each basis component.

### What G9 actually is today

**Lean side** (`SounioSurgicalInterventions.lean` §7):

```lean
def audit_witness_is_derivation :
    alice_kernel.all (fun v => isZeroPair primA v) :=
  unlearning_kernel_exact
```

- This is a **constructive term of a fixed proposition about `alice_kernel` / `primA`**.
- It is **not** generated from a Sounio AST.
- CI even warns: definition is a proposition; prefer `theorem` (see receipt `lean_ci_surgical_snippet.txt`).
- Comments claim the compiler “can serialize it” and re-check in ~50ms — **aspirational prose**, not a measured emit path for `Audited<T>`.

**Checker side** (`lower_audited_type`):

- Requires effect IDs **18 (ZD) and 19 (Witness)**.
- Returns `inner_ty`.
- **No** call to obligation recording, **no** write of a `.lean` file, **no** link to `audit_witness_is_derivation`.

**Emit path that does exist** (different domain):

- `madaros emit-lean-obligations` → Equivalence Theory invariants (`docs/audit/EQUIVALENCE_THEORY_LEAN_*`).
- Measured: recording gated for invariant/chaotic sites, **not** for `TypeAudited` / surgical wrappers.

**Verdict:** G9 is the **same class of promise** as G3: Lean has an algebraic artefact; the type is effect ceremony; the bridge is unbuilt. The name “Audited” does not currently mean “Lean re-checked this call”.

---

## 3. Is the ZD element fixed or a parameter?

| Question | Measurement |
|---|---|
| Does Lean prove 4 annihilators for **every** primitive? | **Yes.** `every_primitive_has_4_annihilators` / `_restated`: `validPrims.all (… length == 4)`. |
| Does Lean prove a fully enumerated kernel for **one** concrete `A`? | **Yes.** `alice_kernel` / `annihilatorsOfA` for `A = e3 + e10` (matches `examples/zd_machine_unlearning.sio`). |
| Does the Sounio type `ExactlyPrivate<T>` parameterise `A`? | **No.** Single type argument `T`; lowering ignores any operator identity. |
| Can the type be honestly generic over `A`? | **Algebra yes, wrapper no.** Generic kernel **existence** is proved; generic **enumerated basis + action at a use-site** would need `A` in the type or in a closed intrinsic table. Treating the Alice theorems as the definition of the whole type **over-claims**. |

**Decision implication:** a generic `ExactlyPrivate<T>` can at most mean “some ZD kernel exists and ZD effect is in scope” (still ceremony). A claim-ready type needs either:

- `ExactlyPrivate<T; A>` with `A` drawn from the finite PrimSed set, or  
- a small family of named operators (`AliceUnlearn`, …) each backed by a concrete Lean kernel def.

---

## 4. Table of eight wrappers

| Wrapper | Gate (effect ceremony) | Lean algebraic backing | Lean ↔ type bridge | Classification |
|---|---|---|---|---|
| **Forgettable&lt;T&gt;** | E200 / ZD=18 | `SounioImpossibilityChain`: `zd_effect_is_necessary_for_forgettable`, `forgettable_type_soundness_condition` (ZD necessary below sedenion level — **meta** necessity, not a kernel action theorem) | None | **Partial** (necessity only) |
| **ExactlyPrivate&lt;T&gt;** (G3) | E201 / ZD=18 | `unlearning_kernel_exact`, `alice_kernel_*`, `every_primitive_has_4_annihilators_restated`, `surgical_trilogy` / `surgical_hexad` G3 conjunct | None (comments + stdlib cite) | **Partial** (algebra yes; type no) |
| **Editable&lt;T&gt;** (G5) | E202 / ZD=18 | `editing_locality_fiber9`, `editing_locality_kernel_bound` | None | **Partial** |
| **CapabilityGated&lt;T&gt;** (G7) | E203 / ZD=18 | `capability_complement_is_large`, `capability_removal_preserves_complement` | None | **Partial** |
| **Composable&lt;T&gt;** (G8) | E204 / ZD=18 | `composition_preserves_orthogonal_complement`, `composition_cardinality_balance` | None | **Partial** |
| **Audited&lt;T&gt;** (G9) | E205 / ZD=18 + Witness=19 | `audit_witness_is_derivation` (fixed Alice term), `audit_witness_shape` | **No surgical emit**; Equiv-Theory emit is a different feature | **Partial / promise** |
| **Revivable&lt;T&gt;** (G10) | E206 / ZD=18 + Temporal=20 | `revive_inverse_property`, `revive_window_well_defined` (kernel-level; not a Temporal window model) | None | **Partial** |
| **Interpretable&lt;T&gt;** (AMI) | E207 / ZD=18 | `SounioInterpBasis`: `basis_168_*`, `ami_canonical_basis` (168-class completeness — **basis**, not a type-use theorem named Interpretable) | None; **no** `tests/compile-fail/*interpretable*` | **Partial** (basis yes; type name absent in Lean API) |

**Compile-fail fixtures present:**  
`exactly_private_requires_zd`, `editable_requires_zd`, `capability_gated_requires_zd`, `composable_requires_zd`, `audited_requires_witness`, `revivable_requires_temporal`.  
**Absent:** Forgettable-only and Interpretable-only refuse fixtures (Forgettable may ride on shared ZD messaging; Interpretable has **zero** surface `\bInterpretable\s*<` hits outside compiler internals — see §5).

Lean file honesty (header of `SounioSurgicalInterventions.lean`): full formalization of wrapper **semantics** would need sedenion-module theory; current content is computational ZD graph + **propositional scaffold** that *names* the types.

---

## 5. Cost map (measurement only — no patch)

### Checker surface (read-only LOC)

| Item | Size |
|---|---|
| Eight `lower_*` bodies + comments | **155 lines** (`check.sio` 16705–16859) |
| Per-wrapper logic | ~15–20 lines: effect ID test → `report_error_at` → return `inner_ty` |
| Dispatch arms | `TypeForgettable` … `TypeInterpretable` near 16263–16270 |
| E200–E207 message strings | ~8 lines near 13367–13374 |

A **minimal non-ceremony** step that stayed inside the existing effect pattern would still be ceremony. A real bridge is a **new subsystem**: operator parameters, constructor allow-list, and/or obligation emit — order-of-magnitude **larger than 155 lines**, touching check + (likely) parser/AST if `A` is a type argument + CI Lean consume job. Exact LOC **not estimated as a schedule**; founder scopes after choosing (a)/(b)/(c).

### What breaks if ceremony becomes real

| Risk | Detail |
|---|---|
| Stdlib / examples that only declare `with ZD` | Would fail until they use closed constructors / intrinsics. |
| Regulatory façade types (`gdpr.sio`, `eu_aiact.sio`, `hipaa.sio`) | Use wrappers in type position as documentation-shaped APIs; would need either EDNC-style carve-out or real kernels. |
| Madaros vs lean_single split | **Already broken for E201** (see §6): default Madaros accepts missing ZD; lean_single refuses. Tightening Madaros to match lean_single is a **prerequisite** before any stronger obligation, or compile-fail tests are engine-dependent lies. |
| Float residual demos | `zd_machine_unlearning` proves numerical residual ≈ 0 by convention; a Lean bridge must not pretend f64 equality is `native_decide` on PrimSed. |

### Type-position usage (`\bWrapper\s*<` in `.sio`, excluding `self-hosted/` and `tests/compile-fail/`)

| Wrapper | Files | Paths (surface) |
|---:|---:|---|
| ExactlyPrivate | 6 | `stdlib/privacy/exactly_private.sio`, `stdlib/regulatory/{gdpr,eu_aiact}.sio`, `stdlib/clinical/biomarker.sio`, `stdlib/epistemic/revocable.sio`, `artifacts/zd-ssm/model.sio` |
| Forgettable | 1 | `examples/zd_bptt_ssm.sio` |
| Editable | 5 | `stdlib/epistemic/editable.sio`, `stdlib/clinical/biomarker.sio`, `examples/{zd_model_editing_locality,meta_self_editing}.sio`, `artifacts/zd-ssm/benchmarks/zsre_eval.sio` |
| CapabilityGated | 4 | `stdlib/safety/capability.sio`, `stdlib/regulatory/eu_aiact.sio`, `stdlib/clinical/biomarker.sio`, `artifacts/zd-ssm/benchmarks/wmdp_eval.sio` |
| Composable | 3 | `stdlib/epistemic/composable.sio`, `stdlib/regulatory/hipaa.sio`, `examples/zd_model_composition.sio` |
| Audited | 5 | `stdlib/epistemic/audited.sio`, `stdlib/regulatory/{gdpr,hipaa,eu_aiact}.sio`, `examples/zd_audit_witness.sio` |
| Revivable | 2 | `stdlib/epistemic/revivable.sio`, `examples/zd_revivable_edit.sio` |
| Interpretable | **0** | (parser/lexer/compiler only) |

**Note:** `examples/zd_machine_unlearning.sio` is the operational G3 story and **does not** use `ExactlyPrivate<` in type position — pure convention + comments citing Lean.

---

## 6. Mandatory controls

### Positive control — Lean theorem verified in CI

| Field | Value |
|---|---|
| Workflow | `.github/workflows/ci.yml` job **`lean-proofs`** / display name **Lean Proofs** |
| Command | `cd formal/lean4 && lake build` (step “Build all default-target proof modules”) |
| Module | `lean_lib «SounioSurgicalInterventions»` with `@[default_target]` in `formal/lean4/lakefile.lean` |
| Evidence run | [actions/runs/32280011465](https://github.com/Sounio-lang/sounio/actions/runs/32280011465) · job id `96157045272` · conclusion **success** · headSha `bd361fd0923b` |
| Log extract | `⚠ [92/225] Replayed SounioSurgicalInterventions` then later modules `✔ Built …`; full job green. Receipt: `docs/audit/exactly_private_lean/lean_ci_surgical_snippet.txt` |
| Local lake on compute | **INDETERMINATE / absent** — no `lake`/`elan` on workspace control or cpu-ops measured this round; **CI is the positive control**, not a local rebuild. |

Theorems of record for G3 (same module, no `sorry`):  
`alice_kernel_is_4d`, `unlearning_kernel_exact`, `alice_kernel_fully_annihilated`, plus generic `every_primitive_has_4_annihilators_restated`.

### Negative control — ceremony compiles without annihilation

Witnesses (also under `docs/audit/exactly_private_lean/`):

**`witness_ceremony_with_zd.sio`**

```sounio
// Ceremony: ExactlyPrivate param + with ZD, body never annihilates
fn observe(p: ExactlyPrivate<f64>) -> i32 with ZD {
    0
}
fn main() -> i32 with ZD {
    0
}
```

**`witness_nozd.sio`**

```sounio
fn observe(p: ExactlyPrivate<f64>) -> i32 {
    0
}
fn main() -> i32 { 0 }
```

**Slurm dual-engine receipt** (`slurm_ceremony_dual_engine.txt`, job **10421**, host `cpuops-t560-proxmox`):

| Cell | Engine | Command shape | rc | Notes |
|---|---|---|---:|---|
| Ceremony (with ZD, no annihilate) | **Madaros v0.80.0** (default `bin/souc`) | `souc check witness_ceremony_with_zd.sio` | **0** | Accepts empty body — **negative control holds** |
| Missing ZD | **Madaros v0.80.0** | `souc check witness_nozd.sio` | **0** | **E201 not fired** — Madaros weaker than lean_single |
| Ceremony | **lean_single** (`SOUNIO_SOUC_ENGINE=lean_single`) | same | **0** | Ceremony also accepted |
| Missing ZD | **lean_single** | same | **1** | `error[E201]: parameter uses ExactlyPrivate<T> without \`with ZD\` effect` |

**Interpretation:**

1. **Ceremony claim is measured, not rhetorical:** a programme may name `ExactlyPrivate<f64>`, declare `with ZD`, never annihilate, and typecheck on **both** engines (rc=0).
2. **Even the ceremony gate is split:** default user-facing Madaros does **not** enforce E201 on this witness; lean_single does. Compile-fail tests that expect E201 are **lean_single-shaped** unless Madaros is brought to parity (out of scope here; separate repair dispatch).

Earlier failed witnesses that tried to **construct** `ExactlyPrivate` values hit `error[E008]` (return type ExactlyPrivate vs f64) — the wrapper erases to `inner_ty` at the type level, so “construct the private value” is not a usable surface without additional API. The parameter-position witness above is the correct negative control.

---

## 7. Proposed correction (for founder decision — not implemented)

Ordered options; pick one direction before any `self-hosted/` work.

| ID | Proposal | Cost class | Risk |
|---|---|---|---|
| **P0** | Document status quo in TypeKind / concept materials: surgical wrappers = **Executable ceremony**, Lean modules = **Claim-ready algebra**, bridge = **open**. | Docs only (this PR). | None if claims stay honest. |
| **P1** | Madaros E201–E207 parity with lean_single (effect-id only). | Small checker path in Madaros modular check; dual-engine gate. | Still ceremony; stops false green on default engine. |
| **P2** | Closed intrinsic surface (`forget_contribution`, sedenion mul-by-A) required at ExactlyPrivate use-sites; syntactic/dataflow check. | Medium; stdlib + examples must call intrinsics. | No Lean discharge yet; better than empty body. |
| **P3** | Parameterise `A` + emit Lean obligation importing `Sounio.SurgicalInterventions` per site; CI `lake` consume (mirror Equiv-Theory emit). | Large; new emit family + CI. | Closest to claim-ready; G9 becomes real only if Audited uses this path. |
| **P4** | Keep wrappers as documentation keywords; move guarantees entirely to Lean + numerical examples. | Shrink type system surface. | Honest; loses type-system marketing. |

**This dispatch recommends P0 now (done by merging this doc), P1 as the next measured engineering slice if default Madaros remains the claim clock, and P3 only after P1 so obligations are not built on a non-firing gate.**

---

## 8. INDETERMINATE list

| Item | What would close it |
|---|---|
| Local `lake build SounioSurgicalInterventions` on cpu-ops | Install elan/lake on compute image or always cite CI job logs (current method). |
| Whether modular Madaros check shares the same `lower_exactly_private_type` body bit-for-bit with lean_single | Forensic diff of madaros check package vs `check.sio` lines 16728–16740 under a dedicated engine-parity dispatch. |
| Runtime residual of `zd_machine_unlearning` on current Madaros native | Re-run example under named engine; not required for the ceremony claim. |
| Interpretable product intent | Zero surface uses; confirm whether AMI is docs-only or a deferred keyword. |

---

## 9. Refutation criteria

This dispatch is wrong if any of the following is shown with named engine + command:

1. A Sounio programme with `ExactlyPrivate` whose **body** is rejected unless an annihilating intrinsic/`A` is present (beyond `with ZD`).
2. `madaros emit-lean-obligations` (or successor) emitting a goal that imports `unlearning_kernel_exact` for an `Audited`/`ExactlyPrivate` site.
3. Madaros default `souc check` on `witness_nozd.sio` returning non-zero with E201 (would refute the Madaros gap; would **not** refute ceremony-with-ZD).
4. Lean CI `lake build` failing or excluding `SounioSurgicalInterventions` from default targets (would collapse the positive control).

---

## 10. Commands to re-run

```bash
# Negative control (name the engine)
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
./bin/souc --version
./bin/souc check docs/audit/exactly_private_lean/witness_ceremony_with_zd.sio
./bin/souc check docs/audit/exactly_private_lean/witness_nozd.sio
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc check docs/audit/exactly_private_lean/witness_nozd.sio

# Positive control (CI or local lake)
cd formal/lean4 && lake build SounioSurgicalInterventions
# or re-pull: gh run view <id> --repo Sounio-lang/sounio  # Lean Proofs job
```

---

## 11. Bottom line

Lean already proves the **algebra** that `ExactlyPrivate` marketing cites — generically (4 annihilators per primitive) and concretely (`alice_kernel` for `e3+e10`). The Sounio type lowers to **inner type + optional ZD effect id**. The flagship unlearning example never places `ExactlyPrivate<` in type position. G9 does not emit Lean. Default Madaros does not even enforce E201 on the measured witness.

**The bridge is a founder decision, not a missing one-line fix.** This document is the forensic packet for that decision.
