<!-- docs:meta
topic_id: repo.docs.audit.exactly-private-ta-dispatch-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: grok-cli4
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.exactly-private-ta-dispatch-2026-08-19
-->

# ExactlyPrivate&lt;T, A&gt; — putting the ZD element in the type

**Date:** 2026-08-19  
**sha measured:** `29c2b22498` (`origin/main` tip at branch cut)  
**Host (souc check):** Slurm `cpu-ops` / `cpuops-t560-proxmox` (tarball `srun`)  
**Write set:** this document + `docs/audit/exactly_private_ta/**` + governance sync  
**Forbidden this round:** any edit under `self-hosted/` or `formal/`

**Prior receipts (not re-argued):**

| Receipt | Role |
|---|---|
| [`EXACTLY_PRIVATE_LEAN_BRIDGE_DISPATCH_2026-08-19.md`](./EXACTLY_PRIVATE_LEAN_BRIDGE_DISPATCH_2026-08-19.md) | Ceremony vs Lean; candidate **(a)** = kernel-built values of a declared element |
| [`ZD_ANNIHILATE_BUILTIN_DISPATCH_2026-08-19.md`](./ZD_ANNIHILATE_BUILTIN_DISPATCH_2026-08-19.md) (#2017) | Closed builtin `zd_annihilate`; product is an **operation**, not a constructor |
| [`ZD_EXACTNESS_FLOATING_POINT_BOUNDARY_2026-08-19.md`](./ZD_EXACTNESS_FLOATING_POINT_BOUNDARY_2026-08-19.md) (#2023) | Kowalski / IEEE-754: **formal exactness at compile time, not numerical exactness at run time** |

Algebra already measured (Lean): `ker(u)=4 ⊂ fiber(u)=12 ⊂ 84` primitives; `every_primitive_has_4_annihilators` is **generic** over all 84, not Alice-only.

Receipts: `docs/audit/exactly_private_ta/`

---

## Semantic lane declaration

```text
Semantic-Lane-ID: exactly-private-ta-20260819
Owner: grok-cli4
Concept-IDs: none
Intent-Preserved: type names a decidable proposition; IEEE residual stays out of the type claim
Transformation: none — design forensic + cost
Types-Changed: none this PR
Claims-Introduced:
  - multi-parameter types already parse (Robust, Knowledge ε); ExactlyPrivate is single-inner today
  - recommended A form: closed PrimSed locus (name or index) in ExactlyPrivateTypeInfo, not full dependent types
  - closed constructors ≠ zd_annihilate; annihilate is the forgetting op
  - named proposition is model-kernel membership + op recognition; float residual not in the type
  - surface break set for second param is small (order 6–8 files outside self-hosted)
Claims-Forbidden:
  - syntax landed
  - runtime f64 zero promised by the type
  - full dependent types required before multiparam exists
Write-Set: docs/audit/EXACTLY_PRIVATE_TA_DISPATCH_*, docs/audit/exactly_private_ta/**
Positive-Witness: Robust<i64, Stable, InDistribution> annotation check OK (Madaros); Knowledge<f64, epsilon < 0.1> check OK (Madaros)
Negative / contrast: ExactlyPrivate single-arg ceremony still OK; two-arg not a designed surface
Acceptance-Gate: re-run Slurm multiparam control
Authoritative-Only-If: engines named; no self-hosted/formal edits
```

---

## 0. Mandatory control first — multiparameter types **already exist**

### Finding (lead with this)

**The compiler already accepts types with two or more parameters.**  
`ExactlyPrivate<T, A>` does **not** require inventing multiparameter type syntax from zero. It requires extending the **ExactlyPrivate-specific** parser path that today only reads a single `inner`.

### Positive control (Slurm, Madaros v0.80.0)

| Witness | What it shows | Engine | rc |
|---|---|---|---:|
| `witness_robust_annot_only.sio` | `fn id(r: Robust<i64, Stable, InDistribution>)` typechecks | **Madaros** | **0** |
| `witness_knowledge_epsilon.sio` | `Knowledge<f64, epsilon < 0.1> = measure(...)` typechecks | **Madaros** | **0** |
| `witness_ep_one.sio` | `ExactlyPrivate<f64>` + `with ZD` still ceremony-OK | **Madaros** | **0** |

Full transcript: `docs/audit/exactly_private_ta/slurm_multiparam_control.txt` · host `cpuops-t560-proxmox`.

### Parser sites that already read multiple parameters

| Surface | Function | File:lines (approx.) | Mechanism |
|---|---|---|---|
| **`Robust<T, Level, Scope>`**, `Contest<…>`, many epistemic wrappers | `parse_generic_wrapper_type` → `parse_type_args_after_lt` | `self-hosted/parser/types.sio` **1426–1461** | After `<`, parse `TypeExpr` list separated by `,` into `type_args: TypeExprList` |
| **`Knowledge<T, epsilon < …, …>`** | `parse_knowledge_type` | `types.sio` **968+** | Specialised comma loop into `KnowledgeTypeInfo` (`epsilon`, provenance, …) |
| Named generics | `parse_type_args` | `types.sio` **403+** | Same comma-list pattern |

**`parse_type_args_after_lt` (excerpt of behaviour):** first type, then `while Comma { parse_type }`, then `>`. This is the multiparameter reader the control exercises via `Robust<i64, Stable, InDistribution>`.

### What ExactlyPrivate does **today** (not multiparam)

| Artifact | Content |
|---|---|
| `ExactlyPrivateTypeInfo` | **`inner_type` only** — `parser/ast.sio` ~510–513. **No `A` field.** |
| `parse_exactly_private_type` | `Lt` → **one** `parse_type` → `Gt`; stores `inner: Some(...)`, `type_args: None` — `types.sio` ~1179–1198 |
| Checker | `lower_exactly_private_type(te.inner, …)` — effect ZD only; returns `inner_ty` |

Contrast: **`ForgettableTypeInfo` already declares** `zd_locus: Option<Name>` with comment `// e.g. "e3e10"` (`ast.sio` ~503–506) — but **`parse_forgettable_type` never fills it** (same single-`inner` shape as ExactlyPrivate). The AST anticipated a locus; the parser did not.

### Engine notes (honest)

- **lean_single** does not run the full Contest/Robust frontend surface (witness fails on `contest` / `M1`). Multiparam **proof for this dispatch is Madaros** on Robust annotation + Knowledge ε.  
- `ExactlyPrivate<f64, f64>` is **not** a designed second parameter: Madaros reaches a type mismatch on assigning `f64` (annotation still looks like ExactlyPrivate); lean_single may accept the file as ceremony. **Do not read that as support for `A`.**  
- `prove_robust_basic` under this Madaros prebuilt failed annotation equality (`expected Robust` vs `found Robust<i64, level=…>`) — a separate engine-parity papercut; **annotation-only multiparam still rc=0**, which is the control the dispatch needs.

### Parser cost if founder approves `ExactlyPrivate<T, A>`

| Work | Scale |
|---|---|
| Reuse `parse_type_args_after_lt` **or** Knowledge-style second component | **Small** (~20–60 lines in `parse_exactly_private_type`) |
| Extend `ExactlyPrivateTypeInfo` with locus / index | **Small** (one field; wire through resolve/check) |
| New dependent-type engine / value-in-type generally | **Not required** for the recommended design |
| lean_single parity for the new field | **Must** be in the same implementation lane |

**Primary finding restated:** multiparameter types are live; ExactlyPrivate is a **one-arg special case** that must be upgraded, not a greenfield type system.

---

## 1. The form of `A` — evaluate against what exists

| Form | What it means | Supported today? | Verdict for ExactlyPrivate |
|---|---|---|---|
| **Type parameter** (second `TypeExpr` in `type_args`) | `ExactlyPrivate<T, AliceKer>` where `AliceKer` is a phantom/unit type name | **Yes** for Robust-style names via `parse_generic_wrapper_type` | **Viable** if a **closed catalogue** of locus types (or one `PrimSed<N>` family) is defined; heavy if 84 distinct names |
| **Value in type position** (true dependent) | `ExactlyPrivate<T, a>` with runtime `a` | **No** general dep types for this; `dependent.sio` is a separate track | **Reject for v1** — wrong cost class |
| **Integer index 0…83** | `ExactlyPrivate<T, 17>` mapping to `validPrims[17]` | **Partial:** Knowledge parses **float/int after epsilon keyword**, not bare index as second type arg for ExactlyPrivate | **Viable** with a **small specialised parse** (second component = int lit or `prim(i,j)`); best Lean alignment |
| **Ghost on effect** | `with ZD(A)` carrying payload | Effect system is **id flags** (ZD=18), not payloads | **Not available** without effect-payload design |
| **Name locus** (string/ident) | `ExactlyPrivate<T, e3e10>` or `zd_locus` | **AST foreshadowed** on Forgettable (`zd_locus: Option<Name>`); parser unwired | **Viable** — lowest surface friction; table maps name → PrimSed |

### Recommendation

**`A` is a compile-time PrimSed locus**, stored on `ExactlyPrivateTypeInfo`, **not** a runtime value and **not** a free dependent index.

Preferred surface syntax (pick one in implementation; both map to the same info field):

```text
ExactlyPrivate<T, prim(3,10)>     // two-support of A = e3+e10  (recommended clarity)
// or
ExactlyPrivate<T, e3e10>          // closed name table (matches Forgettable comment)
// or
ExactlyPrivate<T, 42>             // index into validPrims (compact; less readable)
```

**Not** `ExactlyPrivate<T, &[f64;16]>` — that re-opens float identity and is not decidable.

**Lean image of `A`:** an element of the finite `PrimSed` / ordered-pair set already used by `every_primitive_has_4_annihilators` and `alice_kernel` (special case `primA`).

---

## 2. Closed constructor set

`zd_annihilate` (#2017) is a **forgetting / left-product operation**. It does **not** introduce an `ExactlyPrivate` value; it **consumes** weights (and may be required on paths that **discharge** the privacy obligation).

### Legitimate producers of `ExactlyPrivate<T, A>`

| Constructor (proposed name) | Meaning | Lean image |
|---|---|---|
| **`ep_kernel_embed(A, c0..c3)`** or `ep_from_annihilator_coords(A, coords: [f64;4])` | Build a payload whose sedenion encoding is the linear combination of the **4 right-annihilators** of locus `A` | Point in `annihilatorsOf(A)` span (PrimSed model) |
| **`ep_kernel_basis(A, i)`** for `i ∈ 0..3` | Unit basis vector of `ker(A)` | One element of the 4-list |
| **`ep_zero(A)`** | Zero contribution in that kernel | Trivial kernel element |
| *(optional later)* tracking combinators that **preserve** kernel membership under declared ops | Closure of the set | Must be proved or axiomatised per op |

### What must be refused

| Illegal intro | Why |
|---|---|
| `let p: ExactlyPrivate<T, A> = arbitrary_t` | No proof of kernel membership |
| `as ExactlyPrivate<…>` / unchecked cast | Bypasses constructors |
| “Any `T` under `with ZD`” | Today’s ceremony |
| Inferring ExactlyPrivate from a bare `zd_annihilate` result | Product output is a weight, not a typed kernel certificate unless a **checked** post-condition is added |

### Role of `zd_annihilate` relative to the type

```text
// Conceptual — not implemented
fn forget(p: ExactlyPrivate<W, A>) -> W with ZD {
    zd_annihilate(locus_vector(A), p.payload)   // required shape of forgetting
}
```

- **Intro:** closed constructors only.  
- **Elim / forget:** must call `zd_annihilate` with the **same** `A` (or proven equivalent locus).  
- **stdlib today:** `forget_contribution` is **project-and-subtract** on a hard-coded Alice basis — neither constructor nor `zd_annihilate`; migration is a semantic choice (#2017).

---

## 3. Named proposition (SOUNIO-TYPE-INTERROGATION)

### One sentence

> **`ExactlyPrivate<T, A>` interrogates whether the value’s sedenion encoding is a linear combination of the four right-annihilators of the declared PrimSed locus `A` in the finite exact model, and whether forgetting applies the closed annihilating product at that same `A` — not whether IEEE-754 execution residual is zero.**

### Decidability

| Fragment | Decidable? | How |
|---|---|---|
| `A` is a valid primitive / two-support | **Yes** | Finite table (`validPrims`, 84) |
| Kernel dimension 4 for that `A` | **Yes** | `every_primitive_has_4_annihilators` |
| Constructor args are coords in that basis | **Yes** (at intro) | Syntax of closed constructors |
| Forgetting call uses `zd_annihilate` with matching `A` | **Yes** (syntactic/dataflow once builtin exists) | #2017 spine |
| Runtime `f64` vector equals model combination after ops | **No** | Kowalski / #2023 |

### Lean discharge

| Obligation | Theorem / module |
|---|---|
| Generic kernel size | `every_primitive_has_4_annihilators` / `_restated` (`SounioZeroDivisorBridge` / `SounioSurgicalInterventions`) |
| Concrete Alice case | `unlearning_kernel_exact`, `alice_kernel_is_4d`, `alice_kernel_fully_annihilated` |
| Locality (for Editable, not EP) | `editing_locality_kernel_bound` |
| Complement (for CapabilityGated) | `capability_removal_preserves_complement` |

The type’s proposition is **discharged by the generic annihilator theorem + constructor well-formedness + builtin call recognition**. It is **not** discharged by measuring `||A·x||` in f64.

---

## 4. Floating-point boundary in the type design

From #2023: formal exactness at compile time; numerical exactness **not** guaranteed at run time.

| Option | Meaning | Verdict |
|---|---|---|
| **(i) Ignore float** | Keep marketing “algebraically zero” at runtime | **Reject** — fails TYPE-INTERROGATION and Kowalski bound |
| **(ii) Carry residual bound on the type** | e.g. `ExactlyPrivate<T, A, residual < 1e-12>` | **Reject as primary** — confuses with DP/`Knowledge` ε; residual is **execution-history dependent**, not a stable decidable type predicate in the same sense as PrimSed membership |
| **(iii) Require exact arithmetic** | Only rational/PrimSed symbolic path | **Reject as sole path** — kills the f64 scientific surface the examples use; may exist later as a separate exact backend |
| **(iv) Split claims (recommended)** | Type = **model + op** proposition; residual = **operational** measurement / demo / optional assert | **Accept** |

### Recommendation: **(iv)**

- **Type promise (decidable):** kernel membership in the Lean/PrimSed model + required `zd_annihilate` at `A` on forget paths.  
- **Runtime honesty:** demos may print `||residual||`; gates may bound residuals **as tests**, not as the type’s named proposition.  
- **Wording:** never “the contribution is zero in memory”; always “zero in the verified model; programme verified to perform the annihilating operation; IEEE residual bounded operationally.”

Against **(ii)** specifically: putting ε on ExactlyPrivate would make it a cousin of `DiffPrivate`/`Knowledge` and invite the false reading that tightening ε restores Lean exactness. It does not (#2023).

---

## 5. Cost — who breaks, who needs the same treatment

### Versioned `ExactlyPrivate<…>` use (angle-bracket)

| Bucket | Count | Paths |
|---|---:|---|
| **Surface (stdlib + artifacts; excl. self-hosted, excl. this dispatch’s witnesses)** | **6** | `stdlib/privacy/exactly_private.sio`, `stdlib/regulatory/{gdpr,eu_aiact}.sio`, `stdlib/clinical/biomarker.sio`, `stdlib/epistemic/revocable.sio`, `artifacts/zd-ssm/model.sio` |
| compile-fail | 1 | `tests/compile-fail/exactly_private_requires_zd.sio` |
| self-hosted (parser/check/lexer) | 4 | implementation, not call sites |
| **Migration if second param becomes mandatory** | **~6–8 call-site files** | Add default locus **or** require explicit `A` (breaking but small) |

Most surface uses are **regulatory façade annotations**, not live kernel embeddings. Cost is **documentation + annotation update**, not a 400-file rewrite (#2017’s `sed_mul` census is a different problem).

**Compatibility strategy (founder choice):**

| Strategy | Breakage |
|---|---|
| **Mandatory `A`** | All 6 surface files + compile-fail must gain a locus |
| **Default Alice `A = e3+e10` when omitted** | Soft migration; still ceremony until constructors enforced |
| **`ExactlyPrivate<T>` deprecated alias** | Temporary dual surface |

### Sibling wrappers — same algebra, different projection

| Wrapper | Algebraic object | Second parameter should be | Same treatment? |
|---|---|---|---|
| **ExactlyPrivate** | **4D kernel** of `A` | PrimSed locus `A` | **This dispatch** |
| **Forgettable** | Same kernel story; AST already has `zd_locus` | Wire `zd_locus` (may **alias** EP or stay weaker) | **Yes, cheap** — field exists |
| **Editable** | **Fibre locality** (12-primitive fibre / xor label) | Fibre id **or** locus `A` whose fibre is edited | **Yes, but `A` means fibre context**, not “kernel only” — `Editable<T, A>` ≠ copy-paste of EP |
| **CapabilityGated** | **Complement** (79) of ker(`A`) | Same locus `A` (gate removes ker, preserves complement) | **Yes** |
| **Composable** | Orthogonal merge along ker/complement | Pair of loci or proof of disjointness | **Later** — needs two-sided story |
| **Audited / Revivable / Interpretable** | Witness / temporal / 168-basis | Different second params (not PrimSed kernel) | **Not the same `A`** |

Nested algebra reminder: `compose = unlearn ⊔ gate` as **disjoint union** in the Lean scaffold — Composable’s parameter story is **not** “another ExactlyPrivate”.

---

## 6. How it is written in Sounio (target shape)

Illustrative only — **not implemented**:

```sounio
// Locus A = e3+e10 as two-support (maps to Lean primA / alice_kernel)
type AliceEP = ExactlyPrivate<[f64; 16], prim(3, 10)>

fn contribute(coords: [f64; 4]) -> AliceEP with ZD {
    ep_from_annihilator_coords(prim(3, 10), coords)   // closed constructor
}

fn forget(p: AliceEP) -> [f64; 16] with ZD {
    zd_annihilate(locus_as_vec(prim(3, 10)), payload(p))  // builtin + matching A
}
```

**Erasure:** runtime representation can remain `[f64;16]` / `T` (as today’s lowering to `inner_ty`); `A` is a **ghost/index** for checking, like Robust’s level/scope metadata — unless founder demands residual tracking structs.

---

## 7. Implementation cost packet (approval only)

| Slice | Depends on | Cost class |
|---|---|---|
| **P-A** Parser + `ExactlyPrivateTypeInfo.locus` | multiparam already exists | Small |
| **P-B** Checker: store locus on TypeEntry; refuse bare cast | P-A | Small–medium |
| **P-C** Closed constructors + refuse arbitrary intro | P-B, #2017 builtin ideally | Medium |
| **P-D** Forget paths must `zd_annihilate` at same locus | #2017 landed | Medium |
| **P-E** lean_single parity + dual-engine gates | all above | Mandatory discipline |
| **P-F** Editable/CapabilityGated second param | algebra table | Separate lane after EP |

**Order:** #2017 builtin (B1) → **P-A/B** → constructors **P-C** → couple forget **P-D**. Type without constructors is still ceremony with a longer name.

---

## 8. Refutation criteria

1. Multiparameter `Robust` / `Knowledge` annotations fail to parse on Madaros (control dead).  
2. `ExactlyPrivateTypeInfo` already carries `A` and the parser fills it (would obsolete §0–§1).  
3. A published path gives decidable IEEE-754 exact ZD residual as a type predicate without model split (would reopen §4).  
4. Surface `ExactlyPrivate` call sites number in the hundreds (would change §5 cost).

---

## 9. Bottom line

- **Multiparameter types are not the blocker** — `parse_type_args_after_lt` and Knowledge’s comma components are live (Slurm Madaros controls).  
- **ExactlyPrivate is a one-parameter special case**; `ExactlyPrivateTypeInfo` has only `inner_type`; Forgettable’s `zd_locus` shows the intended direction, unwired.  
- **Write `A` as a compile-time PrimSed locus** (name / `prim(i,j)` / index), not a runtime sedenion and not a full dependent type.  
- **Constructors** embed into `ker(A)`; **`zd_annihilate` forgets**; arbitrary `T` must not convert.  
- **Named proposition** is model-kernel membership + matched annihilating op — **#2023 forbids** promising f64 zero in the type.  
- **Breakage** is small (~6 surface files). Editable/CapabilityGated need **related but not identical** second parameters (fibre / complement).

No `self-hosted/` or `formal/` lines were changed in this packet.
