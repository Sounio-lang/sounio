<!-- docs:meta
topic_id: repo.docs.audit.epistemic-trust-map-2026-07-14
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.epistemic-trust-map-2026-07-14
-->

# Epistemic trust map — which stdlib/epistemic results a real program can trust today

**Date:** 2026-07-14
**Toolchain:** `./bin/souc` → Madaros v0.80.0 (default **native** engine)
**Scope:** core uncertainty primitives (GUM, uncertainty propagation, Knowledge<T>,
correlation/covariance, imprecise probability). Bounded, not exhaustive.
**Gate:** `scripts/epistemic_trust_gate.sh` → `EPISTEMIC_TRUST_GATE_OK`

## Why this exists

Sounio's differentiator is *trustworthy* uncertainty. But building one modest I/O
vertical surfaced five distinct imported-module codegen defects, one of which
(the `f64→i64` cast bitcast) makes the GUM coverage factor **silently wrong**.
That raised the real readiness question: *when a program imports an epistemic
module — the way real code uses the stdlib — which results can it trust, and which
are silently corrupted?* Nobody had mapped it. This is that map, for the default
native path (the lean_single engine is a separate, narrower story).

## The two axes

A result is usable under native import iff **both** hold:

1. **Importable** — the module compiles when `use`d by a program. Fails when the
   module transitively `use`s another stdlib module (multi-module native path —
   thin-link failure / segfault, see the dispatches) or hits a module-specific
   miscompile.
2. **Numerically correct** — the called path contains no `f64 → i64/i32` cast
   (which bitcasts under import — `MADAROS_IMPORTED_MODULE_F64_CAST_BITCAST`) and
   no `&local_array`→builtin (`DATA_IO_TRILHA_B_BUILTIN_BUFPTR_DISPATCH`).

## The map (empirically classified; all verified by the gate)

### ✅ Trustworthy under native import

| Module | Verified result | Notes |
|---|---|---:|
| `gum` | `gum_value`, `gum_std_u` (u_c), `gum_dof` (nu_eff) | value=98.3, u_c=0.290402, nu_eff=4.17 — first-principles exact |
| `correlation` | `covariance` of independents = 0 | self-contained; this is the **analytic** (shared-source) covariance, exact 0 for no shared variable — not a finite-sample estimator, so the exact-zero check is appropriate |
| `knightian` (p-box) | `pb_gap`, `pb_midpoint` | self-contained |
| `covariance` | `cov_new` and accessors | self-contained |
| `knowledge` (**free-function** `ep_*` API) | `ep_measured` / `ep_add` / `ep_mul` / `ep_merge` / `ep_gate` | **D3 partial 2026-07-19** — free-function surface imports under Madaros; see `EPISTEMIC_KNOWLEDGE_MADAROS_D3_2026-07-19` |
| `order_spread_exact` (`order_spread4`) | CPC N=4 exact spread ≈ `2.044226` (scaled µ-units `2044225`/`2044226`) | **stdlib leaf 2026-07-20** — algebra inlined via field-wise `OsOct` (no `algebra::` use). Gate: `scripts/madaros_order_spread_native_gate.sh` + Section A `ORDER_SPREAD_TRUST_OK`. `product4_exact` remains available (pulls free-function `knowledge`); method-form `Epistemic::measured` still Root-2. |

Heuristic: **self-contained modules (no stdlib `use` deps) that avoid `f64→i64`
casts and method-call sites import cleanly and return correct numbers.**
(`order_spread_exact` is the measured exception that may `use` free-function
`knowledge` while keeping the multiply path fully local.)

**Update 2026-07-20:** `algebra::octonion::oct_mul` imports cleanly under default
Madaros after splitting the 8-component exclusive-ref body into `oct_mul_lo` /
`oct_mul_hi` (each ≤ ~0x1200 spill frame). Full unrolled body needs ~0x23e0 and
SEGV'd (measured single-file and multi-module). Gate:
`scripts/madaros_algebra_octonion_import_gate.sh`.

### ⚠️ Importable but specific outputs CORRUPTED

| Module | Corrupted output | Root |
|---|---|---|
| `gum` | `gum_k95`, `gum_u95`, `gum_u99` (coverage factors / expanded uncertainty) | `dof_to_i64(nu_eff)` bitcasts → k95 = 1.960 for **all** dof; correct is `t95(nu_eff)` (e.g. 2.776 at nu=4). `u_c` and `value` are unaffected. **Note:** stdlib GUM-site workaround may land separately (prescription-chain lane). |

**Guidance:** report point estimate + combined standard uncertainty `u_c`; do
**not** rely on `U95`/`U99` from an imported `gum` until the cast bug is fixed.

### ❌ Not usable via native import (compile fails)

| Module / form | Failure | Consequence |
|---|---|---|
| `knowledge` **method-call** form (`Epistemic::measured`, `e.val()`) | SEGV in method-call lowering (Root 2) | use free `ep_*` API under Madaros; methods still OK under lean_single |
| `propagate` | blocked / fragile multi-module | propagation layer not yet native-trustworthy |
| `algebra::associator_field` (imported exclusive-ref path) | **runtime SEGV** after successful native compile (large exclusive-ref chain) | CPC N=4 uses `order_spread_exact`; `algebra::octonion::oct_mul` is green after lo/hi split (2026-07-20) |
| `uncertain_eq` | method / multi-module path | equality-under-uncertainty native-import-blocked |

Method-form and remaining modules are usable today only by **free-function rewrite**,
**inlining into `main()`**, or via the **lean_single** engine.

## Blast radius

The corruption/blockage is not peripheral. `Knowledge<T>` (the headline type) and
the entire `propagate` uncertainty-propagation layer are **native-import-unusable**;
`gum`'s coverage intervals are **silently wrong** for finite samples. What still
works under native import is the self-contained numeric core: GUM point+`u_c`,
correlation/covariance, and p-box dispersion. A real PBPK/GUM pipeline that imports
`knowledge`/`propagate` must run under lean_single (which the dissertation gate
already does) — it is **not** portable to the default native engine today.

Every failure here reduces to one of three already-filed compiler dispatches:

- `MADAROS_IMPORTED_MODULE_F64_CAST_BITCAST_2026-07-14` — silent numeric corruption.
- `DATA_IO_TRILHA_B_BUILTIN_BUFPTR_DISPATCH_2026-07-14` — `&buf`→builtin.
- multi-module native path (`MADAROS_MULTIMODULE_*`, and this doc's segfault
  witnesses) — importing any module with `use` deps.

Fixing the imported-module native path therefore unblocks I/O readers, cross-module
reuse, correct coverage factors, **and** the Knowledge<T>/propagation core at once.
That single infrastructure fix, not more application verticals, is what makes the
epistemic stdlib trustworthy under real use.

## Living boundary

`scripts/epistemic_trust_gate.sh` gates the trustworthy set (Section A) and carries
trip-wires (B/C) that print when a known-broken result starts working — so this map
is updated the moment a compiler fix lands, not by memory.

## AI disclosure

Classification and repros by AI agent (Claude) under human direction, on Madaros
v0.80.0. Substantive math claims (Student-t coverage factor; RSS combined
uncertainty) were confirmed by the mandatory math-review offload in the linked
dispatches; the elementary trust criteria (Cov of independents = 0, p-box
gap/midpoint, u_c) by a math-review offload logged in `.claude/llm_offload_log.md`.
No `stdlib/epistemic` or `self-hosted/` sources were modified. GAIDeT-ICMJE 2025.
