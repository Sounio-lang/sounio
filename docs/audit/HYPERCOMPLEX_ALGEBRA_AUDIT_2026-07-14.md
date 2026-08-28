<!-- docs:meta
topic_id: repo.docs.audit.hypercomplex-algebra-audit-2026-07-14
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.hypercomplex-algebra-audit-2026-07-14
-->

# Hypercomplex Algebra — Full-Repo Audit

**Date:** 2026-07-14
**Toolchain:** `./bin/souc` → Madaros v0.80.0 (default); generic-`F` on the fable5/lean_single engine
**Scope:** Everything the repo contains on complex / quaternion / octonion / sedenion /
Cayley–Dickson / Clifford / Jordan / Fano algebra and the SSM variants (O-SSM / H-SSM / S-SSM /
ZD-SSM) — compiler, stdlib, GPU, formal proofs, and runnable probes.
**Method:** Two parallel source+doc inventories cross-checked against `main` and the four open PRs.
Verdict statements below distinguish **language/compiler** state (the primary concern) from
**research/science** state (listed for completeness).

---

## Summary

Hypercomplex algebra is one of the largest areas of the repo: **~230 `.sio` code files** (excluding
artifacts), spanning four layers — self-hosted compiler → stdlib → GPU emitters → research probes.
It is a **first-class language feature**, not merely a library: there is a native `TyHyper` type, a
dedicated type-checker, IR-level algebra semantics, x86 lowering, a PTX emitter, and two custom
effects (`NonAssoc`, `NaturalityG2`) threaded through non-associative multiplication signatures.

The core is healthy. The binding constraints are **compiler-owned blockers** (the #919/#921 codegen
defects + Madaros M1–M3 generics); the **consolidation-debt problem** — the same product
re-implemented across ~11 sites with divergent Fano sign conventions — has since been **resolved**
(canonicalized onto Convention X, PRs #940 + #942, merged 2026-07-15; see the consolidation section).
On the science side, the headline honest result stands: **octonion structure is null on the brain**,
while the positive synthetic control works — and the single most important unbuilt experiment is
isolating the *associator* from the non-commutativity confounder.

Open PRs at audit time: **#907** (hypercomplex/SSM lane — clean home for the octonion commits),
**#904** (epistemic-GUM hardening, currently also carrying those commits), **#742** (GPU od256
div/sqrt), **#541** (octonion mass-δ, draft). Merged `feat/exact-*` family: #687/#689/#701/#704/#706.

---

## ✅ TEMOS (verified / present in `main`)

### Language / compiler layer (`self-hosted/`)

| Module | Role | State |
|---|---|---|
| `check/types.sio` | `TyHyper` (tag 22), `Hyper<Algebra,T>`, `algebra_kind` on every `TypeEntry`; `ty_clifford(p,q,elem)` | complete |
| `check/hyper.sio` (268L) | algebra-promotion for `+,-,*`; field access `.real/.imag`, `.w/.x/.y/.z`, `.e0..`; `.conjugate/.norm/.inverse` gated on division-algebra property; `is_division/associative/has_zero_divisors` | complete |
| `check/cayley_dickson.sio` (188L) | compile-time CD property model: legal reassociation, coercion, dimension, σ twist, Fano-triple, Artin applicability | complete |
| `ir/algebra.sio` (236L) | IR semantics: dimension, sedenion ZD-risk, Clifford grade-product, `ir_should_reassociate` | complete |
| `ir/ir.sio` | IR instructions carry `algebra_tag` (bits 3:0: QUAT=2, OCT=3, SED=4) | complete |
| `native/hyper_lower.sio` (476L) | x86 lowering: Clifford geometric product, Hamilton quaternion, conjugate, norm², AVX-512 Cl(1,3) fast path | complete |
| `gpu/kernels/hypercomplex.sio` (1320L) | PTX emitter for quat+oct (Fano tables, FourVec/EightVec/SixteenVec regalloc) | complete |
| `gpu/kernels/ossm_forward.sio` (1910L) | O-SSM **forward + backward/BPTT** in PTX (f64/f32), associator kernel, sigmoid gates | complete |
| `hypercomplex/octonion.sio` (597L), `quat_simd.sio` (412L) | standalone Oct+Sed + SIMD quaternion (4/8-lane) in the compiler tree | complete |

Effect system: `NonAssoc` and `NaturalityG2` are real effects appearing in `oct_mul`/`sed_mul`
signatures — the language **models non-associativity as a tracked effect**.

### stdlib, by algebra

- **Complex** — `stdlib/complex/lib.sio` (378L): full arithmetic + transcendentals + `EpistemicComplex`.
- **Quaternion** — most authoritative: `math/ga/quaternion.sio` (703L, geometric-algebra) +
  `self-hosted/hypercomplex/quat_simd.sio` (SIMD). Full NN stacks in `qnn/` and `nn/`.
- **Octonion** — most complete: `stdlib/algebra/octonion.sio` (533L) with **G2 structure**
  (3-form, SU(3)-branching, associator-deviation field). Full `onn/` NN stack incl.
  `g2_activation.sio`. Octonion-labeled graphs in `hypercomplex_graph/oct_graph.sio` (634L).
- **Sedenion** — `math/sedenion64.sio` (associator + 256-entry mul matrices) and `math/sedenion.sio`
  (595L, ZD detection + inverse). Full `snn/` stack. **Novel compiler use:**
  `compiler/ast/sedenion_ops.sio` + `sedenion_encoding.sio` encode **AST nodes as sedenions**
  (16-dim structural/semantic signatures).
- **Cayley–Dickson generic** — `stdlib/algebra/cayley_dickson_exact.sio`: **trait-generic exact CD**
  over an `ExactRing`, with the decidable `zd_exact` (no f64 tolerance). Plus `ladder.sio`
  (capability tiers) and `associator_field.sio`.
- **Related** — Clifford `Cl(p,q)` (`clifford.sio`), Jordan/Albert `J3(𝕆)` (`jordan.sio`), Fano
  plane + 168 count (`fano.sio`): all complete.

### SSM variants

`stdlib/ssm/lib.sio` (H-SSM step + S-SSM step + ZD-init/erase/gate/kernel/cokernel),
`ssm/fingerprint.sio` (7-class result), O-SSM on GPU (`ossm_forward.sio`). ~40 runnable probes
(EEG, per-patient seizure chb01/03/05/06/10/11, ablations, `conversational_ossm/`, `cognitive_ossm/`).

### Formal (Lean, 0 `sorry`)

`OctonionAlgebra.lean` (norm multiplicativity), `OctonionGraph.lean` (path-product norm
invariance), `SounioSedenionBipartite.lean`, the **168 theorem** + associator-norm dichotomy
‖[eᵢ,eⱼ,eₖ]‖ ∈ {0,2}, Der(𝕊)=g₂, PSL(2,7)↪G₂. (`GUM.lean` depends on 3 IEEE-754 axioms.)

---

## 🟡 EM VOO

- **PR #907** — the exclusive lane for the octonion commits (`oct_truth`, `oct_algebra`,
  `ossm_recover`, `ossm_separation`, rewritten `associativity_probe_benchmark`). Open. Once merged,
  PR #904 can drop the duplicated commits.
- **Phase-1/2 non-associative connectomics** — all Phase-2 code landed and synthetic-verified;
  real pilot **blocked** (see below).
- **Psychiatric-regime octonion arm** — hysteretic-associator witness
  (`clinical_hysteretic_associator_memory_witness.sio`) landed on
  `research/psychiatric-regime-contest-20260712`; passes its gate.

---

## 🔴 FALTA — compiler-owned blockers (language priority)

| ID | Blocker | Impact | Status |
|---|---|---|---|
| ~~**#651**~~ | ~~`[struct;N]` multiply-accumulate codegen corrupts~~ — **RE-DIAGNOSED 2026-07-14 (Madaros): misdiagnosed.** Not aggregate codegen; the exact CD product over ℚ **runs correctly** (256-iter sedenion mul, `RAT-ZD PROVED`). The d8 SIGSEGV was two conflated defects (below); the N=16 garbage was lean_single/fable5, plus a separate value-copy aliasing already fixed by `ff7afab69`. | Exact ZD over ℚ **no longer blocked** on Madaros. ℚ proof landed (`tests/run-pass/cd_exact_rational_concrete.sio`, PR #816); dispatch PR #923. | **resolved as filed**; split into #919/#921 |
| **#919** | native handle-table wraps at **2²⁰** allocations → `gc_empty_frame_reset` wipes live heap (liveness probe under-detects boxed value-locals). Scalar → wrong value; array-of-struct → SIGSEGV. Root cause of the d8 segfault; same family as PBPK (PR #555). | any Sounio program exceeding 2²⁰ heap-boxed value-struct allocs corrupts (training loops, sweeps). The 256-iter science target is **unaffected**. | open, B1; **dispatched to CODEX-2** (`docs/handoff/compiler_651_defects_codex_dispatch_2026-07-15.md`, PR #966); root fix: don't heap-box ≤16B value-struct returns |
| **#921** | multimodule thin-link (compact-IR ELF writer) fails rc=12 when `math::rational` is imported alongside a second module | forces single-module/inline (code duplication) for exact-arithmetic clients | open, E1; **dispatched to CODEX-2** (PR #966) |
| **#891** | Madaros v0.80.0 codegen: `print_int` garbled after f64; scalar-global-in-unit-fn does not persist to caller; `[i64;1]` SIGSEGV (use len ≥2); many f64-locals-across-calls corrupt the return addr | probes need manual source workarounds; Madaros is not the oracle for these paths | `BLK-20260714-madaros-print_int-f64`, owner codex-2 |
| — | Madaros support for **generic** exact CD (tracks M1–M3) | generic-`F` engine (`cd_exact_generic_i64.sio`) runs only on lean_single/fable5, not the default `souc` (Madaros); the concrete/monomorphized ℚ path works on Madaros | pending |

## 🟢 consolidation debt (language) — convention divergence RESOLVED 2026-07-15

- **~~`oct_mul` / `sed_mul` divergent Fano conventions~~ → CANONICALIZED onto Convention X
  (`cd_sigma` / XOR, `e1·e2=+e3`).** The divergence was verified real (octonion X vs Y differ 42/64
  basis products; `snn/base.sio` sedenion table 56/256 off), then unified:
  - **PR #940** (merged, `0c2493899`) — executable oracle
    `tests/run-pass/hypercomplex_convention_crosscheck.sio` (asserts the full 8×8+16×16 `cd_sigma`
    tables; guards future drift) + `stdlib/algebra/README.md` convention section + fixed a real **e7
    sign bug** in `ssm/lib.sio` (the flip made the octonion sub-product non-normed / non-alternative).
  - **PR #942** (merged, `dcd147e16`) — migrated the two Convention-Y octonion files
    (`hypercomplex_graph/oct_graph.sio`, `self-hosted/hypercomplex/octonion.sio`) and the divergent
    `snn/base.sio` sedenion sign table onto `cd_sigma`. Migrating the self-hosted `oct_mul` also
    reconciled its `oct_mul_cayley_dickson` (already X). X and Y are isomorphic (both 168
    non-associative triples), so invariants (norm, associator-norm, ZD) are unchanged — only
    component-level outputs of those files move (connectome/EEG artifacts should re-baseline).
- Still open: GA engine duplicated between `ga/quaternion.sio` and `ga/dual_quaternion.sio`;
  disabled stubs (`genomics/quaternion_protein.sio`, `genomics/octonion_grn.sio` — "pending parser
  support"; `hypercomplex_graph/mod.sio`; `algebra/sedenion_verdict.sio` f256 boundary tag).
- ~~`stdlib/algebra/README.md` is stale~~ → updated by PR #940 (now declares X canonical + a
  per-file conformance map + the exact layer).

## 🔵 FALTA — research/science gaps (outside this window's focus; listed for completeness)

- **Honest verdict:** octonion is **null on the brain** (ABIDE O-SSM vs H-SSM gap −0.52pp;
  0/200 ROIs survive Holm) — but the **positive synthetic control works** (+35.4pp recovery).
- **Central unresolved confounder:** the +35.4pp comes from **non-commutativity (Fano mixing)**, not
  the associator. The single most important unbuilt experiment is injecting a known associator≠0 and
  demonstrating recovery *as associator* — `docs/briefings/OSSM_ASSOCIATOR_RECOVERY_TEST.md`
  ("DESENHO. Nada corrido."). Non-associativity is materially exercised in exactly one file
  (`brain_ossm_classifier.sio`); cosmetic/absent in several `ossm_*` files despite prose.
- Phase-1 real pilot blocked: `frames.bin` absent at `/orangefs/...`; no numpy/scipy on cluster nodes.
- O-CSSM / Tapestry preprints are prose-only, not deposited; the homology functor F is only partially
  constructed. The conjectured associator spectral bound is **not found in code** (stale index line);
  the G₂ bridge is separately **falsified** on CC200 Laplacian eigenmodes.

---

## Recommendation (highest language leverage)

1. **~~Unblock #651~~ → resolved as filed (2026-07-14).** The exact ZD product over ℚ already runs on
   Madaros (concrete path; ℚ proof landed PR #816, dispatch PR #923). What remains are the two split
   defects: **#919** (2²⁰ handle-table wrap — the general-purpose root fix: don't heap-box ≤16B
   value-struct returns) and **#921** (multimodule thin-link with `math::rational`). Unbounded-ℚ over
   the *generic* engine still waits on Madaros generics (M1–M3), but `zd_exact` over ℚ is no longer
   the exact layer's blocker.
2. **~~Canonicalize `oct_mul` / `sed_mul`~~ → DONE (2026-07-15, PRs #940 + #942, merged).** Convention
   X (`cd_sigma` / XOR) is now the single convention; a run-pass oracle guards against drift, and the
   `ssm/lib.sio` e7 sign bug was fixed along the way.
3. **Madaros M1–M3** for exact CD — remove the generic-`F` dependency on lean_single/fable5. Now the
   highest-leverage remaining language item, alongside the #919/#921 compiler fixes.

On the science side, the highest-value single build is the **pre-registered associator-injection
recovery test** that isolates the associator from the non-commutativity confounder.

---

## Provenance

Built from two parallel repo inventories (source-by-algebra + docs/plans/blockers) cross-checked
against `main` and PRs #907/#904/#742/#541 on 2026-07-14. Every `docs/research/*` doc cited is tagged
`authority: historical`; the canonical current surface is the lane doc + briefings + the open PRs.

**Update 2026-07-14** — #651 was attacked and re-diagnosed on Madaros: not a `[struct;N]` aggregate
bug; the exact ZD product over ℚ runs correctly (proof landed PR #816). Split into #919 (2²⁰
handle-table wrap, root cause) and #921 (multimodule thin-link). Full forensic dispatch:
`docs/audit/HYPERCOMPLEX_651_ROOTCAUSE_2026-07-14.md` (PR #923).

**Update 2026-07-15** — the `oct_mul`/`sed_mul` convention divergence flagged as consolidation debt
was verified real (X vs Y differ 42/64) and **canonicalized onto Convention X** (`cd_sigma` / XOR):
PR #940 (oracle test + README convention map + `ssm/lib.sio` e7 sign-bug fix) and PR #942 (migrated
`oct_graph.sio`, `self-hosted/hypercomplex/octonion.sio`, `snn/base.sio`) — both merged. The
`stdlib/algebra/README.md` staleness is also cleared. Note: the checked-in `artifacts/self-hosted/
madaros` (Jul-11) is stale w.r.t. main — build fresh before concluding a codegen bug is live (the
"multiple `&[f64;N]` ref-param" and ssm-harness SIGSEGV symptoms were stale-binary artifacts).

**Update 2026-07-15 (dispatch)** — the two remaining compiler defects #919 (2²⁰ handle-table wrap)
and #921 (multimodule thin-link) were **dispatched to CODEX-2** as a forensic fix prompt:
`docs/handoff/compiler_651_defects_codex_dispatch_2026-07-15.md` (PR #966, merged), cross-linked on
both issues. Each carries root-cause mechanism (`file:line`), a checked-in repro, and acceptance
criteria. They are compiler-owned (`self-hosted/`, CLAUDE.md §8) — now in CODEX-2's queue.
