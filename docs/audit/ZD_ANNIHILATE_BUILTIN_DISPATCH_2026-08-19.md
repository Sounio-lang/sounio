<!-- docs:meta
topic_id: repo.docs.audit.zd-annihilate-builtin-dispatch-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: grok-cli4
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.zd-annihilate-builtin-dispatch-2026-08-19
-->

# Closed builtin surface for ZD annihilation — forensic dispatch

**Date:** 2026-08-19  
**sha measured:** `335edd4b12` (`origin/main` tip at branch cut)  
**Founder ruling (same day):** annihilation by zero-divisor becomes a **closed builtin**, not a library call the checker must guess at.  
**Prior context (read first):** [`EXACTLY_PRIVATE_LEAN_BRIDGE_DISPATCH_2026-08-19.md`](./EXACTLY_PRIVATE_LEAN_BRIDGE_DISPATCH_2026-08-19.md)  
**Write set:** this document + `docs/audit/zd_annihilate_builtin/**` + governance registry sync  
**Forbidden this round:** any edit under `self-hosted/` or `formal/`

Receipts: `docs/audit/zd_annihilate_builtin/`

---

## Semantic lane declaration

```text
Semantic-Lane-ID: zd-annihilate-builtin-20260819
Owner: grok-cli4
Concept-IDs: none
Intent-Preserved: Lean ZD algebra remains claim-oracle; this lane only costs a closed builtin that makes candidate (b) checkable
Transformation: none — measurement + proposed signature/entry/migration
Types-Changed: none (ExactlyPrivate<T,A> deferred — §6)
Effects-Changed: none (proposal requires ZD at call site when implemented)
IR-Changed: none this round
Claims-Introduced:
  - call_expr_is_builtin_* pattern is live (measure positive control on Slurm, both engines)
  - recommended builtin is zd_annihilate(A, x) over vector/sedenion form, not 32-scalar sed_mul_out
  - recognising the call establishes "this site is the annihilating product", not "contribution is zero"
  - sed_mul call-sites cannot be auto-split into ZD-annihilation vs ordinary product (~458 hits)
  - do not promote whole sed_mul to builtin
Claims-Forbidden:
  - builtin implemented (docs only)
  - ExactlyPrivate non-ceremony closed by this PR alone
  - "all sed_mul are annihilations"
Assumptions: census via \bsed_mul(_out)?\s*\( on versioned *.sio
Write-Set: docs/audit/ZD_ANNIHILATE_BUILTIN_DISPATCH_*, docs/audit/zd_annihilate_builtin/**
Read-Set: self-hosted/check/check.sio, self-hosted/ir/lower.sio, stdlib/algebra/sedenion.sio,
          examples/zd_*, formal/lean4/SounioSurgicalInterventions.lean (read-only), prior ExactlyPrivate forensic
Positive-Witness: measure() builtin recognised — Madaros+lean_single rc=0 on Slurm cpuops-t560-proxmox
Negative-Witness: not_a_real_builtin_xyz — Madaros E137 / lean_single unknown identifier, rc=1
Acceptance-Gate: re-run tarball srun dual-engine control; re-count sed_mul census
Integration-Target: founder approves signature before any self-hosted implementation lane
Authoritative-Only-If: engines named per cell; no self-hosted/formal edits in the PR
```

---

## Why this dispatch exists

The ExactlyPrivate forensic scored three candidates for non-ceremony:

| Candidate | Prior verdict | Role of a closed builtin |
|---|---|---|
| **(a)** value built from annihilator kernel of declared `A` | Best static core | Still needs a **named op** that realises kernel action / product |
| **(b)** body contains `A ∘ x` tied to the theorem | Heuristic **unless** `A` and `∘` are builtins | **This ruling:** make `∘` a recognisable builtin |
| **(c)** emitted Lean obligation | Architecture exists for Equivalence Theory only | Builtin gives a **stable call-site** for a future emit family |

The founder ruling selects the closed-builtin path so (b) becomes verification, not grep-shaped hope, and so (a) has an operation to attach to.

---

## 1. Recommended builtin signature

### Name

**`zd_annihilate`**

Rationale: intent is surgical left-action by a ZD element (unlearn / gate / edit kernel), not general Cayley–Dickson product. Distinct from `sed_mul` so ordinary algebra stays a library.

### Preferred arity and types

```text
zd_annihilate(a: &[f64; 16], x: &[f64; 16]) -> [f64; 16]   with ZD
```

Equivalently, if the struct surface is preferred for typed code:

```text
zd_annihilate(a: Sedenion, x: Sedenion) -> Sedenion   with ZD
```

**Semantics (when implemented):** return the sedenion product `a * x` (left multiplication), identical in float behaviour to today’s library `sed_mul` / example `sed_mul_out`, but **only** this name is treated as the surgical annihilator by the checker.

**Effects:** require effect id **18 (ZD)** on the enclosing function (same gate family as ExactlyPrivate E201). Missing ZD → dedicated error (new code or reuse E201-class messaging).

### Rejected / demoted shapes

| Shape | Why not primary |
|---|---|
| **32-scalar `sed_mul_out(a0…a15, h0…h15)`** (today’s zd examples) | Checker sees 32 independent `f64`s. No value identity for `A`, no `PrimSed` link, no dataflow. Matches “library written by hand; nothing recognisable.” |
| **Make all `sed_mul` a builtin** | ~**458** call hits across ordinary SNN training, SSM demos, associators, GPU kernels. Most are **not** annihilation. Cost and false ceremony. |
| **`zd_annihilate` with only `x` (implicit global Alice `A`)** | Over-specialises to `e3+e10`; Lean already proves **generic** 4-annihilators (`every_primitive_has_4_annihilators`). Founder ruling: builtin need not be Alice-only. |

### Optional second form (for theorem linking)

If the implementation lane wants a **decidable** link to Lean `PrimSed` without float const-prop:

```text
zd_annihilate_prim(i: i64, j: i64, x: &[f64; 16]) -> [f64; 16]   with ZD
```

where `(i,j)` is the two-support of a primitive imaginary (`e_i ± e_j` encoding fixed by a table shared with `SounioZeroDivisorBridge`).  

- **Checker can:** reject pairs not in `validPrims` / not a ZD left factor (finite table).  
- **Lean image:** `isZeroPair` / annihilator lists for that primitive.  
- **Cost:** dual API or sugar `zd_annihilate(prim_sed(i,j), x)`.  

**Recommendation:** ship vector form first (matches stdlib `[f64;16]` and zd examples); add `zd_annihilate_prim` when ExactlyPrivate gains an `A` index (§6). Do not block the closed surface on PrimSed sugar.

### Why the checker can bind this to the theorem (and the scalar form cannot)

| Form | What the checker can see | Lean hook |
|---|---|---|
| `zd_annihilate(a, x)` | Callee identity + two args + ZD effect | Site is “left product by `a`”; if `a` later carries PrimSed / ghost `A`, discharge `∀v∈ker(A). A·v=0` style facts already proved generically |
| 32-scalar `sed_mul_out` | 32 floats, local fn name | **No** stable operator identity; theorem citation impossible without full symbolic reconstruction |
| Bare `sed_mul` library | Ordinary call | Same product, **not** marked surgical; drowning in non-ZD uses |

---

## 2. Where it enters (follow existing builtin pattern)

Do **not** invent a parallel mechanism. Mirror **`measure`** / epistemic builtins.

### Pattern inventory (live today)

| Stage | `measure` locus (read-only) | `zd_annihilate` analogue |
|---|---|---|
| **Recogniser** | `call_expr_is_builtin_measure` — `check.sio:14247` (`name_matches_str7` on `"measure"`) | `call_expr_is_builtin_zd_annihilate` (`name_matches_strN` on `"zd_annihilate"`) |
| **Bridge-by-value fan** | `call_expr_should_bridge_by_value` — `check.sio:7133` | Add one `if` next to measure / prove_robust cluster (`7114–7137`) |
| **In-place call spine** | `checker_check_expr_inplace` intercept — `check.sio:7727` | Same intercept list |
| **By-value check_call** | `check_measure_expr` — `check.sio:21590` / `23612` | `check_zd_annihilate_expr`: arity 2, arg types `[f64;16]` or `Sedenion`, require ZD effect id 18, result type = product type |
| **IR lower** | `lower_measure_call_ref` — `ir/lower.sio:14612` (struct Knowledge fields) | Either (i) **lower to ordinary call** of `stdlib/algebra/sedenion.sed_mul` / internal helper, or (ii) new `IrZdAnnihilate` if a distinct opcode is needed for FO/audit later |
| **Native** | measure builds Knowledge in IR then normal native | **Prefer delegate** to existing product implementation — no new microcode required for v1 |

### Implementation cost class (not scheduled)

| Piece | Rough size (order of magnitude) | Notes |
|---|---|---|
| Recogniser + three dispatch sites | ~40–80 lines in `check.sio` | Copy measure/admit_action shape |
| `check_zd_annihilate_expr` | ~30–60 lines | Types + ZD gate; optional PrimSed table later |
| Lowerer branch | ~20–80 lines | Thin if library-delegating |
| lean_single parity | **must** mirror Madaros | Prior forensic: Madaros already weaker on E201; do not ship builtin on one engine only |
| Tests | compile-fail missing ZD; run-pass product equals `sed_mul` | Dual-engine gate |
| **No** Mathlib / formal change required for v1 | — | Theorems already generic |

**Native emission:** v1 should **not** require a bespoke ELF sequence. Lowering to the same codepath as `sed_mul` (or inlining the same Cayley–Dickson body once) keeps the closed surface a **checker fact**, not a second algebra.

### What not to touch

- Lexer keywords: **not required** — builtins are bare idents matched in the checker (like `measure`, `prove_robust`).  
- Making `sed_mul` itself builtin: **out of scope** (§4).

---

## 3. What recognising the call establishes (and what it does not)

### Proposition that becomes establishable

> **At this call site, the programme applies the designated ZD left-product operator `zd_annihilate(a, x)`, under an explicit `with ZD` effect, producing `a * x`.**

Corollaries the checker may add later (not free with recognition alone):

1. **Operator identity** — dataflow can mark the result as `Image(zd_annihilate, a, x)` rather than opaque `[f64;16]`.  
2. **Effect honesty** — surgical product cannot hide in a non-ZD function (once Madaros/lean_single both enforce).  
3. **Hook for ExactlyPrivate** — a forgetting path can be required to contain ≥1 `zd_annihilate` whose `a` matches the declared unlearning element (when `A` exists on the type — §6).  
4. **Hook for Lean emit (c)** — a future `emit-lean-obligations` family can emit “product by primitive A” goals keyed off this callee only.

### Proposition that remains out of reach

| Claim | Why recognition is insufficient |
|---|---|
| “The user’s contribution is now zero” | Needs encoding of “contribution ⊆ ker(A)” (candidate **a**) + float/algebraic residual story. `forget_contribution` today **projects and subtracts** onto a hard-coded Alice basis — it does **not** call `sed_mul` at all. |
| “`a` is a zero-divisor” | True only if `a` is constrained (literal PrimSed, ghost type, or runtime check). Arbitrary `[f64;16]` is undecidable as ZD in the checker. |
| “`x` lies in the 4D right-annihilator of `a`” | Finite and decidable **on PrimSed coefficients**; not on opaque runtime weights without a kernel constructor API. |
| GDPR / “Right to be Forgotten” as law | Regulatory façade; algebra does not discharge legal identity of the data subject. |

**One-line honesty:**  
**Builtin recognition ≠ proof of annihilation of a contribution.**  
It proves **which operation** ran. Proof that the **operand** was kernel-supported is a separate closed constructor / type-parameter problem (ExactlyPrivate forensic candidate **a**).

---

## 4. Migration census

### Method

Versioned `*.sio` matched with `\bsed_mul(_out)?\s*\(`. Definitions (`fn sed_mul`) excluded from call counts where line-local.

### Aggregate (measured this worktree)

| Family | Call hits (approx.) | Files | Notes |
|---|---:|---:|---|
| sedenion demos / door_* / SSM | **318** | 34 | Ordinary product, orbits, EEG, associators |
| snn local copies | **35** | 8 | Local `fn sed_mul` redefinitions |
| **zd_surgical examples** | **30** | 9 | All carry hand-rolled `sed_mul_out` |
| other | 20 | 8 | mixed |
| algebra canonical | 18 | 3 | `stdlib/algebra/sedenion.sio` etc. |
| tests | 18 | 8 | |
| gpu / math / self-hosted | ~19 | ~6 | |
| **Total call hits** | **~458** | **~75 files** | |
| **`sed_mul_out` hits** | **23** | **9** | All under `examples/zd_*.sio` |

### Can we tell ZD annihilation from ordinary multiplication?

**Finding: not reliably from syntax alone.**

- **Ordinary product majority:** SNN `sed_mul(h,r)`, SSM recurrence, associator tests, `sed_mul(s, inv)`, GPU kernels — left factor is **not** a surgical ZD eraser.  
- **Surgical intent cluster:** `examples/zd_*.sio` (~9 files). Even there, `zd_model_editing_locality.sio` multiplies **deltas** by kernel/complement probes (measurement of locality), not only “erase contribution.”  
- **Alice unlearn story:** `zd_machine_unlearning.sio` uses `apply_a_to_w` → `sed_mul_out` for verification `A ∘ alice`; the stdlib GDPR helper `forget_contribution` uses **orthogonal projection subtract**, not product.  
- **No versioned call** today is spelled `zd_annihilate`.

**Migration implication:**

1. **Do not** rewrite the 458-site `sed_mul` corpus.  
2. **Do** introduce `zd_annihilate` as the **only** name ExactlyPrivate / surgical gates will accept.  
3. **Opt-in migrate** the zd_* examples and any stdlib surgical helper that truly means left-product-by-A (small, human-reviewed set — order **10 files**, not 75).  
4. Leave `sed_mul` as the general algebra API forever.

### stdlib note

`stdlib/privacy/exactly_private.sio::forget_contribution` is **not** a `sed_mul` call site. Migrating it to the builtin is a **semantic choice** (product vs project-and-subtract), not a rename. Founder should decide whether the blessed forgetting op is:

- **Product form:** `w' = zd_annihilate(A, w)` (example Method 3 narrative), or  
- **Projection form:** subtract ker components (current stdlib), possibly as a second builtin `zd_forget_kernel` later.

This dispatch only closes the **product/annihilator-action** surface named in the ruling.

---

## 5. Mandatory control — existing builtin is live

### Programme

`docs/audit/zd_annihilate_builtin/witness_measure_builtin.sio`:

```sounio
fn main() -> i32 {
    let k: Knowledge<f64> = measure(1.0, uncertainty: 0.1)
    0
}
```

### Checker site that catches it

1. `call_expr_is_builtin_measure` — `self-hosted/check/check.sio:14247`  
2. Dispatched from `check_call_expr` — `check.sio:21590` → `check_measure_expr` (`:23612`) returning `Knowledge<T>`  
3. In-place spine — `check.sio:7727`  
4. Lower — `self-hosted/ir/lower.sio:14612` `lower_measure_call_ref`

### Negative programme

`witness_unknown_fn.sio` calls `not_a_real_builtin_xyz(1.0)` — must **not** typecheck.

### Slurm dual-engine receipt

Host: **`cpuops-t560-proxmox`** (tarball via `srun` stdin; login `/workspace` not mounted on compute).  
Engine versions: **Madaros v0.80.0** default; **lean_single** via `SOUNIO_SOUC_ENGINE=lean_single`.  
Full transcript: `docs/audit/zd_annihilate_builtin/slurm_builtin_control.txt`

| Cell | Engine | rc | Observation |
|---|---|---:|---|
| `measure(...)` | **Madaros v0.80.0** | **0** | `check: OK` — builtin recognised without defining `measure` |
| unknown name | **Madaros v0.80.0** | **1** | `error[E137]` undeclared `not_a_real_builtin_xyz` |
| `measure(...)` | **lean_single** | **0** | compiles; epistemic meas channel notes present |
| unknown name | **lean_single** | **1** | `unknown identifier not_a_real_builtin_xyz` |

**Verdict:** the `call_expr_is_builtin_*` pattern is **live on both engines**. A `zd_annihilate` builtin can sit on the same spine without inventing infrastructure. (If this control had failed, the proposal would be blocked.)

---

## 6. Decision deferred: what the builtin demands of the type

**Out of this dispatch (as ordered):** `ExactlyPrivate<T, A>` parameterisation.

### What the builtin alone needs from types today

| Requirement | Level |
|---|---|
| Enclosing function declares **`with ZD`** | Effect ceremony (existing id 18) |
| Args are sedenion-shaped (`[f64;16]` or `Sedenion`) | Ordinary type check |
| Result is the same shape | Ordinary type check |
| Callee name is exactly `zd_annihilate` | Builtin recogniser |

No change to `ExactlyPrivate<T>` is **required** to land the builtin.

### What ExactlyPrivate still needs after the builtin exists

From the prior forensic minimum proposition, non-ceremony still needs:

1. **Declared `A`** on the type or ghost (`ExactlyPrivate<T, A>` or effect payload).  
2. **Kernel membership** of the forgotten contribution (closed constructors).  
3. **Link:** forgetting path must call `zd_annihilate(a, ·)` with `a` definitionally that `A`.  
4. Optional **Lean discharge** of the PrimSed facts (emit path).

**Builtin without type parameter:** enables (b) recognition only — programmes can still call `zd_annihilate` and ignore ExactlyPrivate, or use ExactlyPrivate with empty body (still ceremony until a gate requires the builtin).

**Founder next decision (explicit):**

| Option | Builtin role | Type role |
|---|---|---|
| **B1** | Land `zd_annihilate` + ZD gate only | Leave ExactlyPrivate as today (ceremony + docs) |
| **B2** | Land builtin | Add gate: ExactlyPrivate-lowering functions must call `zd_annihilate` at least once (still weak without `A`) |
| **B3** | Land builtin + PrimSed form | Introduce `ExactlyPrivate<T, A>` and require `zd_annihilate`/`_prim` with matching `A` |

**This forensic recommends B1 as the implementation slice immediately after approval, then B3 as the claim-ready end state.** B2 alone is a mild improvement and easy to game (dead call).

---

## 7. Proposed correction packet (for approval — not implemented)

1. **Add** checker builtin `zd_annihilate` on the `measure` spine (Madaros **and** lean_single).  
2. **Lower** by delegating to existing sedenion product (no new native algebra).  
3. **Require** `with ZD` at the call’s function.  
4. **Do not** builtin-ify `sed_mul`.  
5. **Migrate** only intentional surgical sites (zd examples + chosen stdlib helper) under review.  
6. **Tests:** dual-engine pass (product equals library) + refuse without ZD + refuse unknown arity.  
7. **Defer** `ExactlyPrivate<T, A>` to a separate semantic lane after B1 is green.

### Risks

| Risk | Mitigation |
|---|---|
| Engine split (Madaros misses effect gates) | Dual-engine gate from day one; prior E201 gap is a **blocker** for ExactlyPrivate coupling |
| Users wrap `sed_mul` and claim surgery | Only `zd_annihilate` counts for surgical types |
| Float residual ≠ Lean zero | Document residual demos separately; Lean stays PrimSed |
| Name collision if stdlib defines `fn zd_annihilate` | Builtin wins in checker (same as measure); forbid user redefinition or error |

### Cost summary for founder

| Item | Scale |
|---|---|
| Checker + lowerer | Small–medium (hundreds of lines, not thousands) if library-delegating |
| Migration | ~9 zd example files + optional stdlib; **not** 458 sed_mul sites |
| Formal/Lean | **Zero** for v1 |
| Claim-ready ExactlyPrivate | **Separate** lane (type parameter + kernel ctors) |

---

## 8. Refutation criteria

This dispatch is wrong if:

1. `measure` fails to typecheck without a user definition on Madaros or lean_single (pattern dead).  
2. A versioned automatic classifier cleanly separates all ZD-annihilation `sed_mul` sites from ordinary product with zero false positives (would change migration).  
3. Founder ruling is interpreted as “builtin-ify `sed_mul`” without a separate cost argument — this doc argues that reading is incorrect.  
4. Implementation lands only on one engine.

---

## 9. Commands to re-run

```bash
# Builtin control (login; or tarball srun as in scripts/dev/type_enum_denominator_slurm.sh pattern)
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
./bin/souc --version
./bin/souc check docs/audit/zd_annihilate_builtin/witness_measure_builtin.sio
./bin/souc check docs/audit/zd_annihilate_builtin/witness_unknown_fn.sio
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc check docs/audit/zd_annihilate_builtin/witness_measure_builtin.sio

# Census
rg -c --glob '*.sio' '\bsed_mul\s*\(' | awk -F: '{s+=$2} END{print s}'
rg -c --glob '*.sio' '\bsed_mul_out\s*\(' | awk -F: '{s+=$2} END{print s}'
```

---

## 10. Bottom line

The founder’s **closed builtin** ruling is the right cut: it turns candidate **(b)** into a checker fact without drowning ordinary `sed_mul` algebra.  

**Ship `zd_annihilate(a, x)` with ZD**, on the live `call_expr_is_builtin_*` spine proved by the `measure` control on Slurm. **Do not** promote general `sed_mul`. **Do not** claim contribution-zero from recognition alone. **`ExactlyPrivate<T, A>` remains the next founder decision**, not part of this packet.

No `self-hosted/` or `formal/` lines were changed in producing this document.
