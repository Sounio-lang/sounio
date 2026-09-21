<!-- docs:meta
topic_id: repo.docs.audit.surgical-calculus-disjoint-union-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: cursor-1 (Slurm lake build SounioSurgicalCalculus; Lean 4.30.0)
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.surgical-calculus-disjoint-union-2026-08-19
-->

# Surgical calculus: cardinality vs disjoint union (2026-08-19)

**Verdict: PROVADO** — COMPOSE is the disjoint union of UNLEARN and GATE
as *sets*. The docstring's ordered-concat reading, and the audit slogan
`AUDIT u = UNLEARN u ++ [u]`, are false. Both refutations are
`native_decide`d. No `sorry`. No Mathlib.

Existing cardinality theorems were not edited.

## Control: the file compiled as it stood

Mandatory first measurement, Slurm partition `cpu-ops` (host
`cpuops-t560-proxmox`), not the workspace pod. Sources staged to
`/orangefs/training/sounio/cursor-1-uniao` because the node cannot see
`/workspace` or the pod `/tmp`.

Command:

```text
export HOME=/tmp/elan-home-uniao
export ELAN_HOME=/orangefs/training/sounio/toolchains/elan
export PATH="$ELAN_HOME/bin:$PATH"
cd /orangefs/training/sounio/cursor-1-uniao/formal/lean4
lake build SounioSurgicalCalculus
```

Output (unchanged `SounioSurgicalCalculus.lean` from `origin/main`
`4775e84540`):

```text
✔ [2/6] Built SounioCayleyDickson (496ms)
✔ [3/6] Built SounioZeroDivisorBridge (862ms)
✔ [4/6] Built SounioSurgicalInterventions (572ms)
✔ [5/6] Built SounioSurgicalCalculus (1.3s)
Build completed successfully (6 jobs).
AS_IS_RC=0
```

Toolchain on the node: Lake 5.0.0 / **Lean 4.30.0**
(`elan` home `/orangefs/training/sounio/toolchains/elan`). The repo pin
`formal/lean4/lean-toolchain` is `leanprover/lean4:v4.33.0`. Restoring
that pin and re-running `lake build` failed with
`error during download` / `SSL peer certificate or SSH remote key was
not OK` (`PIN433_RC=1`). The source compiled; the pinned compiler is
not installed on this node and cannot be fetched.

## Why lists-as-sets, not ordered lists

`applyOp` returns `List PrimSed`. There is no Mathlib `Multiset` or
`Finset` under the `formal/` contract. The three relevant clauses are:

| op | definition |
|---|---|
| UNLEARN | `orderedZDPairs.filter (p.1 == u) \|>.map (·.2)` |
| GATE | `validPrims.filter (v => u ≠ v && ¬ isZeroPair u v)` |
| COMPOSE | `validPrims.filter (v => u ≠ v)` |
| AUDIT | `u ::` the same filter/map as UNLEARN |

COMPOSE and GATE walk `validPrims`. UNLEARN walks `orderedZDPairs`.
Those are different orders, so

```lean
applyOp .compose u == applyOp .unlearn u ++ applyOp .gate u
```

is a stronger claim than "disjoint union" and is **not** what the
definitions compute. The correct Mathlib-free encoding is membership
equality plus disjointness (`List.contains`), which is what a finite
disjoint union *is* when the carrier is `List PrimSed`.

Already-proved cardinalities (`4`, `79`, `83`) plus set-equality plus
disjointness also rule out hidden duplicates: a repeated UNLEARN
partner would shrink the unique set below 83 and fail `listSetEq`.

## Theorems added (none of the old statements changed)

In `formal/lean4/SounioSurgicalCalculus.lean` §3b:

| Theorem | Claim | Result |
|---|---|---|
| `compose_is_disjoint_union_of_unlearn_and_gate` | `disjoint(UNLEARN, GATE)` and `setEq(COMPOSE, UNLEARN ++ GATE)` for every valid `u` | **proved** (`native_decide`) |
| `compose_not_ordered_unlearn_append_gate` | ordered `COMPOSE = UNLEARN ++ GATE` is not identically true | **proved** (the identity is false) |
| `compose_ordered_concat_fails_everywhere` | the ordered identity fails for **every** valid `u` | **proved** |
| `audit_eq_cons_unlearn` | `AUDIT u = u :: UNLEARN u` | **proved** (definitional) |
| `audit_not_unlearn_snoc` | `AUDIT u = UNLEARN u ++ [u]` is not identically true | **proved** (the identity is false) |
| `audit_unlearn_snoc_fails_everywhere` | the snoc slogan fails for **every** valid `u` | **proved** |

Whole-file rebuild after the additions:

```text
✔ [5/6] Built SounioSurgicalCalculus (1.9s)
Build completed successfully (6 jobs).
BUILD_RC=0
```

## Counter-example (also the first `validPrims.find?`)

`u = { lo := 1, hi := 10, neg := false }` (and every other valid
primitive). `#eval` on Slurm:

```text
UNLEARN u = [
  { lo := 4, hi := 15, neg := true },
  { lo := 5, hi := 14, neg := false },
  { lo := 6, hi := 13, neg := true },
  { lo := 7, hi := 12, neg := false }]

AUDIT u           = u :: UNLEARN u          -- self first
UNLEARN u ++ [u]  = UNLEARN u ++ [u]        -- self last

COMPOSE u starts  { lo := 1, hi := 10, neg := true }, …
UNLEARN ++ GATE   starts with the four annihilators

listSetEq COMPOSE (UNLEARN ++ GATE) = true
listDisjoint UNLEARN GATE           = true
```

`u :: xs` equals `xs ++ [u]` only in degenerate cases. The UNLEARN
kernel has four elements and does not begin and end with `u`, so the
snoc slogan is false everywhere. COMPOSE's `validPrims` walk never
begins with that kernel, so ordered concatenation is false everywhere.

## What this does not claim

- It does not change `compose_decomposes_as_unlearn_plus_gate` or
  `audit_is_unlearn_plus_self` (still cardinality).
- It does not prove a Mathlib `Finset` identity.
- It does not run under Lean 4.33.0 on this node.
- `surgical_calculus_closure` still packages the weak (cardinality)
  facts. Strengthening that conjunction is a later edit.
