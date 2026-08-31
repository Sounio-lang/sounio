<!-- docs:meta
topic_id: repo.docs.audit.epistemic-trust-map-2026-07-14
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.epistemic-trust-map-2026-07-14
-->

# Epistemic trust map — which stdlib/epistemic results a real program can trust today

**Date:** 2026-07-14 (Wave10 k95 closeout 2026-07-21; **C1 imported ep-var preserve 2026-08-06**)
**Toolchain:** `./bin/souc` → Madaros v0.80.0 (default **native** engine)
**Scope:** core uncertainty primitives (GUM, uncertainty propagation, Knowledge<T>,
correlation/covariance, imprecise probability). Bounded, not exhaustive.
**Gate:** `scripts/epistemic_trust_gate.sh` → `EPISTEMIC_TRUST_GATE_OK`

## Why this exists

Sounio's differentiator is *trustworthy* uncertainty. Building one modest I/O
vertical originally surfaced five distinct imported-module codegen defects, one
of which (the `f64→i64` cast bitcast) made the GUM coverage factor **silently
wrong**. That raised the real readiness question: *when a program imports an
epistemic module — the way real code uses the stdlib — which results can it
trust, and which are silently corrupted?* This is that map, for the default
native path (the lean_single engine is a separate, narrower story).

## The two axes

A result is usable under native import iff **both** hold:

1. **Importable** — the module compiles when `use`d by a program. Fails when the
   module transitively `use`s another stdlib module (multi-module native path —
   thin-link failure / segfault, see the dispatches) or hits a module-specific
   miscompile.
2. **Numerically correct** — the called path must not hit residual mis-lowerings
   (historical: `f64 → i64/i32` param cast bitcast — **D1 CLOSED** #983/#1252;
   residual: some exclusive-ref / fn-ptr fragile paths; D2 `&local_array`→builtin
   largely fixed for I/O).

## The map (empirically classified; all verified by the gate)

### ✅ Trustworthy under native import

| Module | Verified result | Notes |
|---|---|---:|
| `gum` | `gum_value`, `gum_std_u` (u_c), `gum_dof` (nu_eff), **`gum_k95` / `gum_u95` / `gum_u99`** | value=98.3, u_c=0.290402 on Type-B-heavy budget; **finite-dof** Type-A-dominant (n=5, tiny Type-B) → ν_eff≈4, **k95≈2.776**, U95≈0.372 — Student-t, not normal 1.96. **Wave10 residual closeout** (mis-designed 1960 witness retired). Gate: Section A `GUM_TRUST_OK` + `witness_gum_k95` → `2776`. |
| `correlation` | `covariance` of independents = 0 | self-contained; this is the **analytic** (shared-source) covariance, exact 0 for no shared variable — not a finite-sample estimator, so the exact-zero check is appropriate |
| `knightian` (p-box) | `pb_gap`, `pb_midpoint` | self-contained |
| `covariance` | `cov_new` and accessors | self-contained |
| `knowledge` (**free-function** `ep_*` API) | `ep_measured` / `ep_add` / `ep_mul` / `ep_merge` / `ep_gate` | **D3 2026-07-19** — free-function surface imports under Madaros; see `EPISTEMIC_KNOWLEDGE_MADAROS_D3_2026-07-19` |
| `knowledge` (**method** form) | `Epistemic::measured` / `e.val()` / `e.add` / `e.mul` / `e.std` | **Wave9 residual closeout** — free vs method parity under multi-module Madaros. Gate: `scripts/madaros_knowledge_method_residual_gate.sh` + Section A `KNOWLEDGE_METHOD_PARITY_OK` / `KNOWLEDGE_METHOD_OK` |
| `order_spread_exact` (`order_spread4`) | CPC N=4 exact spread ≈ `2.044226` (scaled µ-units `2044225`/`2044226`) | **stdlib leaf 2026-07-20** — algebra inlined via field-wise `OsOct` (no `algebra::` use). Gate: `scripts/madaros_order_spread_native_gate.sh` + Section A `ORDER_SPREAD_TRUST_OK`. `product4_exact` remains available (pulls free-function `knowledge`); method-form `Epistemic::measured` is also green (Wave9). |
| `product_nonassoc` (structural variance) | Fano variance `0.25` / non-Fano `4.25` (κ=1, base σ²=0.25) | **stdlib leaf 2026-07-20** — algebra inlined via field-wise `PnOct` (no `algebra::` use). Gate: `scripts/madaros_product_nonassoc_native_gate.sh` + Section A `PRODUCT_NONASSOC_TRUST_OK`. Knowledge-free `product_nonassoc_augment` is the hard numeric witness; `product_nonassoc(Epistemic,…)` uses field-form `Epistemic` under Madaros (direct `ep_*` + leaf multi-import trips E035). Historic `epistemic::propagate::product_nonassoc` removed. |
| `propagate` (delta-method + MC) | product `6`/`0.25`; `exp_delta(1,σ²=0.01)` → `e` / `e²·0.01`; MC identity mean≈`1` var≈`0.01`; MC square E[X²]≈`4.01` var≈`0.16` | **2026-07-20 multi-module green** for free-function `Epistemic` + `exp_delta`/`product`/`ln`/`sin`/`cos_delta` and value-style LCG MC kernels (`monte_carlo_identity`, `monte_carlo_square`). Gate: `scripts/madaros_propagate_native_gate.sh` + Section A `PROPAGATE_TRUST_OK`. Caveats (pre-Wave6-C): literal `exp`/`cos` SEGV — **fixed**. ~~exclusive-ref xoshiro untrustworthy~~ **remeasured GREEN 2026-08-06** (`scripts/ci/madaros_xoshiro_imported_gate.sh`). Generic `monte_carlo(x,f,n)` fn-ptr **GREEN 2026-08-23** (`scripts/ci/madaros_propagate_monte_carlo_fnptr_gate.sh`). |
| `algebra::associator_field` | non-Fano ‖α‖²=`4`, g2=`2`, aug var=`4.25`; pentagon (e1,e2,e4,e1) var=`0.96` | **2026-07-20 multi-module green** after #1274 oct_mul lo/hi split + pub API surface (`assoc_field_*`, `pentagon_*`, `af_*`). Gate: `scripts/madaros_associator_field_native_gate.sh`. L0: `associator_field_octonion` + `associator_field_pentagon`. |
| `algebra::octonion` (`oct_mul`, `oct_associator`, …) | e1·e2→e3; non-Fano ‖[e1,e2,e4]‖²=`4` | **2026-07-20** lo/hi frame split. Gate: `scripts/madaros_algebra_octonion_import_gate.sh`. |
| `particle_physics::nonunitary_amp` + `sm_params::mass_bottom` | imported `Epistemic` **variance** bit-identical Madaros≡lean_single (scaled i64×1e18): `mass_bottom` → `900000000000000` (=0.0009×1e18); `h_bb_yukawa_amplitude_nu` amp → `1354` (~1.354e-15×1e18) | **C1 2026-08-06** — EXP `print_f64` `0.000000` was display rounding of ~1e-15, not Var corruption. Gate: `scripts/ci/madaros_imported_ep_var_preserve_gate.sh`. Audit: `docs/audit/MADAROS_IMPORTED_EP_VAR_PRESERVE_2026-08-06.md`. |

Heuristic: **self-contained modules (no stdlib `use` deps) import cleanly and
return correct numbers** under default Madaros. Method-call form on `Epistemic`
is TRUSTWORTHY (Wave9). Finite-dof GUM coverage (`k95`/`U95`/`U99`) is TRUSTWORTHY
(Wave10 — D1/#983/#1252 + stdlib `dof_to_i64` arithmetic-source + rounding).

**Update 2026-07-20 (oct_mul):** `algebra::octonion::oct_mul` imports cleanly under
default Madaros after splitting the 8-component exclusive-ref body into
`oct_mul_lo` / `oct_mul_hi` (each ≤ ~0x1200 spill frame). Full unrolled body
needs ~0x23e0 and SEGV'd (measured single-file and multi-module). Gate:
`scripts/madaros_algebra_octonion_import_gate.sh`.

**Update 2026-07-20 (associator_field):** with oct_mul green, the remaining
`associator_field` multi-module blocker was E175 privacy (Madaros hard-errors
on non-`pub` imports; lean_single only warned). Publicizing the L0 surface
makes `use algebra::associator_field` compile+run with correct sentinels.

**Update 2026-07-21 (Wave10 — gum k95):** D1 closed by #983 root-cause + #1252
joint D5+D1 land; stdlib `dof_to_i64` uses arithmetic-source + half-up round.
The Section B trip-wire that printed `CONFIRMED CORRUPT (k95=1960)` was **false
negative**: its budget was Type-B-dominant (ν_eff≈2.8e4), so k95=1.960 is the
correct normal approximation. Replaced with Type-A-dominant witness (`k95i=2776`)
and promoted into Section A. Finite-sample U95/U99 from imported `gum` are safe
to report under default Madaros.

### ⚠️ Importable but specific outputs CORRUPTED

| Module | Corrupted output | Root |
|---|---|---|
| — | *(none currently gated)* | Historical D1 gum k95 → **CLOSED Wave10**. |

**Guidance:** report point estimate + `u_c` + expanded `U95`/`U99` from imported
`gum` freely for finite-dof budgets. Prefer Type-A-dominant smoke tests when
validating Student-t coverage (Type-B-only or Type-B-dominant budgets correctly
converge to k≈1.96).

### ❌ Not usable via native import (compile fails / fragile)

| Module / form | Failure | Consequence |
|---|---|---|
| `propagate` **export names** `exp` / `cos` (call site) | **FIXED Wave6 C** — empty-stub builtins only (`instr_count==0`); user-bodied `exp`/`cos` keep IR | call `exp`/`cos` freely under multi-module Madaros; `exp_delta`/`cos_delta` remain aliases |
| `propagate::monte_carlo` (generic fn-ptr form) | **GREEN 2026-08-23** — DCE reachability + `IrCallIndirect` f64 return markers. Gate: `scripts/ci/madaros_propagate_monte_carlo_fnptr_gate.sh` | promoted; prefer value-style kernels when fn-ptr not needed |
| `uncertain_eq` | method / multi-module path | equality-under-uncertainty native-import-blocked (check current residual gates before quoting) |

Stdlib `Epistemic` method form is TRUSTWORTHY under multi-module Madaros (Wave9).
Language generic `Knowledge<T>` method form is a separate surface — do not
conflate with `epistemic::knowledge::Epistemic`. Remaining fragile modules still
need free-function rewrite, inlining into `main()`, or lean_single.

## Blast radius

What works under native import today includes the free-function **and method-form**
`Epistemic` numeric core, **full GUM** (value + `u_c` + finite-dof `k95`/`U95`/`U99`),
correlation/covariance, p-box dispersion, `order_spread4`, `product_nonassoc`,
and **`propagate` delta-method + value-style MC kernels + generic fn-ptr MC**. A real PBPK/GUM
pipeline can import `gum` + `knowledge` + `propagate` under default Madaros when
it uses those surfaces. ~~Residual trap: generic `monte_carlo` fn-ptr~~ **closed 2026-08-23**.
Language `Knowledge<T>` generics remain a separate surface.

Historical failures reduce to filed compiler dispatches (status as of Wave10):

- `MADAROS_IMPORTED_MODULE_F64_CAST_BITCAST_2026-07-14` — **CLOSED** (#983/#1252; gum k95 gated).
- `DATA_IO_TRILHA_B_BUILTIN_BUFPTR_DISPATCH_2026-07-14` — largely fixed for I/O readers/writers.
- multi-module native path (`MADAROS_MULTIMODULE_*`) — substantially improved;
  residual memory-wall / fragile exclusive-ref chains remain.

## Living boundary

`scripts/epistemic_trust_gate.sh` gates the trustworthy set (Section A), including
finite-dof gum k95 (`witness_gum_k95` → `2776`) and **C1 imported ep-var preserve**
(`madaros_imported_ep_var_preserve_gate.sh` — Madaros≡lean_single scaled Var).
Section B (k95 trip-wire) is **retired**. Update this map when a residual
Section C / fragile form graduates.

## AI disclosure

Classification and repros by AI agent under human direction, on Madaros
v0.80.0. Substantive math claims (Student-t coverage factor; RSS combined
uncertainty) were confirmed by the mandatory math-review offload in the linked
dispatches; the elementary trust criteria (Cov of independents = 0, p-box
gap/midpoint, u_c) by a math-review offload logged in `.claude/llm_offload_log.md`.
Wave10 closeout re-measured on `origin/main` (post-#1252): Type-A-dominant
`k95i=2776`, Type-B-dominant `k95=1.960` (correct). GAIDeT-ICMJE 2025.
