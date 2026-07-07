<!-- docs:meta
topic_id: repo.docs.research.eisa-precision-track-2026-07-05
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.eisa-precision-track-2026-07-05
-->

<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# EISA Precision Track — design note (2026-07-05)

Status: Phase 0 (blocker recheck) and Phase 1 (dd64 + witnesses + math-review)
complete on lane `lean_single`. Phases 2–4 scoped, not started. This note is
repo-internal; no public claim is made by it.

## Thesis

The semantic atom of the Sounio machine is `Knowledge`, not the bare number.
The precision track is therefore not a set of unrelated numeric features: it
is the first CPU instruction family of an **epistemic ISA (EISA)** — a
semantic contract between the language and the machine in which every
operation carries what it knows about its own numerical quality. The GPU
prototype already exists and is measured:
`self-hosted/gpu/kretikos_emit_epistemic_wmma.sio` emits dual
`mma.sync.aligned` paths (value fragments + uncertainty fragments) with
GUM/JCGM 100:2008 RSS quadrature — one *semantic* instruction lowered to a
bundle of real ones.

### EISA instruction table (semantic contract, not silicon opcodes)

| EISA op | Semantics | CPU lowering (bundle) | Status |
|---|---|---|---|
| `EADD k1, k2` | add propagating u by quadrature | `addsd` + u-quadrature | via `Knowledge` lowering (existing GUM path) |
| `EMUL k1, k2` | multiply with GUM relative-u rule | `mulsd` + u-rule | via `Knowledge` lowering |
| `EMMA` | epistemic GEMM tile | dual WMMA (PTX, measured) / CPU pair pending | GPU measured; CPU pending |
| `EOBS k, bound` | observation checking precision adequacy | ErrorDist chain vs u(x) gate | Phase 3 |
| `EHORIZON k, T` | "trustworthy until horizon T" (error growth rate) | Lyapunov-rate metadata check | future work, out of scope |

Operating principle: **bold in semantics, conservative in evidence** — every
phase closes with a measured witness and a receipt, in the S3/S4 style of
PR #622.

## Scope decisions

- **dd64 (double-word) first.** Parity with particle-physics practice: the
  qd library (Hida, Li & Bailey 2001) is what pySecDec/FIESTA use for
  Feynman loop integrals with catastrophic cancellation. dd64 is pure
  Sounio over f64 hardware ops, needs **no new ABI** (a 16-byte aggregate of
  two f64, identical under SysV, AAPCS64 and Darwin), and — unlike qd —
  composes with `Knowledge` GUM metadata.
- **IEEE binary128 later**, as a Madaros v2 S5 milestone, hard-gated on f64
  print/return/call witnesses passing on the default path
  (`docs/research/madaros-v2-sota-plus-plus-plan-2026-07-04.md`, §S5 rule).
- **binary256 rejected.** No known hardware, no ecosystem; if more than 113
  bits is ever needed, quad-double (~212 bits, same EFT technique) or
  arbitrary precision is the principled route.

## Phase 0 — blocker recheck (DONE, 2026-07-05)

`docs/audit/MADAROS_SEED_BEGIN_RECHECK_2026-07-05.md`:

- `BLK-MADAROS-SEED-BEGIN` **persists** post-PR #622 (the PR touches the
  `bin/madaros` HLIR/receipt lane, not `self-hosted/ir/lower.sio`).
- 6/6 `tests/stdlib/theorem/test_smt_*.sio`: segfault on default lane,
  `ALL PASS` on `SOUNIO_SOUC_ENGINE=lean_single`.
- **New finding:** the freshly written single-import `use math::dd64::*`
  tests reproduce the same segfault — the blocker is *not* smt-specific;
  the default lane is currently unusable for multi-module stdlib programs
  generally (failure point `lower_array: dep_begin 1` after `seed_done`).
- Consequence: all Phase 1 witnesses carry `validated_lane: lean_single`
  explicitly. No silent green.

## Phase 1 — dd64 (DONE, 2026-07-05)

Library: `stdlib/math/dd64.sio`. Algorithms and quoted error bounds
(u = 2^-53) follow Joldes, Muller & Popescu (2017), ACM TOMS 44(2)
["JMP2017"], Dekker (1971) and Knuth TAOCP §4.2.2:

| Function | Algorithm | Bound |
|---|---|---|
| `two_sum` | Knuth TwoSum, 6 flops | exact (EFT) |
| `quick_two_sum` | Dekker FastTwoSum | exact given precondition |
| `dd_split` | Veltkamp, splitter 2^27+1 | exact |
| `two_prod` | Dekker product, no FMA | exact (barring overflow) |
| `dd_add` | JMP2017 AccurateDWPlusDW | 3u²/(1-4u) |
| `dd_add_f64` | JMP2017 DWPlusFP | 2u² |
| `dd_mul` | JMP2017 DWTimesDW (no-FMA) | ~5u² |
| `dd_mul_f64` | JMP2017 DWTimesFP | 2u² |
| `dd_div` | qd-style long division, 2 corrections | ~10u² |
| `dd_sqrt` | f64 Newton seed + 1 dd correction | full dd accuracy |

No-FMA baseline is deliberate: the self-hosted x86 backend emits plain SSE2
scalar ops (`self-hosted/native/encode.sio`), so EFT exactness depends only
on IEEE round-to-nearest and evaluation order.

**Compiler contract:** EFTs are exact only without reassociation or FMA
contraction. The witnesses check this bit-exactly on the produced binary; if
a future optimiser pass breaks them, the pass is wrong, not the library.
This is the first *measured* interlock between the numeric library and the
optimiser — the seed of the Phase 3 gate.

Witnesses (all `ALL PASS` on `lean_single`; default lane blocked, above):

- `tests/stdlib/math/test_dd64_eft_exact.sio` — bit-exact EFT identities
  (tie-to-even residual of 1e16+1, (2^27+1)² split residual, 2^-60 capture).
- `tests/stdlib/math/test_dd64_cancellation.sio` —
  quartic (x-1)^4 Horner at x = 1+2^-20: f64 loses the value, dd64 < 1e-12
  relative; Rump (1988) f(77617, 33096): honest boundary — needs ~122 bits,
  dd64 (106 bits) is *not* expected to get the right value, but must beat
  f64 by ≥10 orders of magnitude (measured: ~1e21 → <1e5 absolute). This
  measured boundary is the standing motivation for qd128.
- `tests/stdlib/math/test_dd64_sum_e2e.sio` — Σ 1/k², 10⁶ terms vs analytic
  reference: err(dd) ≤ err(kahan) ≤ err(naive) with sanity bounds.
- `tests/stdlib/math/test_dd64_algebra.sio` — add/mul/sqrt round trips at
  2^-100 relative; comparison semantics resolving on the lo limb.

Mandatory math-review: xai/grok-4-1-fast-reasoning, PASS, "no mathematical
leaps or incorrect bounds found" (`.claude/llm_offload_log.md`, 2026-07-05).

## Phase 2 — S4 error-bounded rewrites (SCOPED, blocked on lane merge)

The S4 e-graph lane (preflight `s4_ready = true`) lives on
`origin/work/madaros-v2-sota-codex` (PR #622) and is **not on this branch**;
Phase 2 starts after that lane lands on the working branch.

Rules (proposals through the e-graph, never direct IR mutation), each
emitting a `madaros.v2.s4.err/0.1` receipt
(`rule_id`, `expr_hash_before/after`, `err_bound_before/after`,
`bound_citation`, `cost_delta_flops`, `decision`, `lane`):

- R1 `fadd → two_sum` — error-free, information-gaining.
- R2 `sum_naive → sum_kahan` — O(n·u) → O(2u + n·u²).
- R3 `sum_naive → dd_sum` — tighter budget, ~2× Kahan cost.
- R4 `horner ↔ estrin` — error-neutral, cost-relevant (shows one mechanism
  carries both cost and error).

Differential vs Herbie/Precimonious: inside the compiler, receipt-carrying,
composable with S5 — not an external advisory tool.

## Phase 3 — ErrorDist (NEXT on this branch)

Mirror of the epistemic latency model already in
`self-hosted/native/target_policy.sio` (`LatencyDist { mean, variance }`,
"independent variances add along a chain", upper-quantile decisions): an
`ErrorDist` per operation, chain accumulation, upper-quantile roundoff
compared against the `Knowledge` u(x):

- roundoff_q ≪ u(x): silent OK;
- roundoff_q ≳ u(x)/10: "marginal precision" diagnostic, suggesting R2/R3;
- roundoff_q ≥ u(x): opt-in compile error — a type system that rejects
  numerical nonsense.

The scheduler decides *when* to execute; ErrorDist decides *with how many
bits*. Same machinery, second use.

## Phase 4 — IEEE binary128 (CONDITIONAL, not started)

Hard precondition: f64 witnesses green on the default path (S5 rule).
Design summary (full detail in the approved plan):

- `abi_lower.sio`: `ABI_TYPE_F128`, size 16 align 16; SysV = stack + sret;
  AAPCS64 = Q0–Q7 with a separate 16-byte FP slot counter; Darwin arm64 =
  same internal ABI, plus an FFI fence — Apple's `long double` is binary64
  (`darwinpcs`), so passing Sounio f128 as C `long double` must be a checker
  error, never a silent truncation.
- `aarch64.sio`: `LDR/STR Qt` encoders (V=1 load/store group); arithmetic is
  always a `BL` to the runtime — AArch64 has no binary128 ALU.
- Runtime `stdlib/math/f128_soft.sio`: two u64 limbs, IEEE 1+15+112, no
  libquadmath dependency (absent/unstable on ARM).
- Witness suite `tests/selfhost/native_runtime/abi_f128_*` with ABI receipt
  hashes on the CI lanes already green in PR #622 (Linux x86-64,
  macOS arm64).

## Related work (honest positioning)

- qd (Hida–Li–Bailey 2001): the practice standard; library-level, no type
  system, no uncertainty metadata.
- JMP2017: the bounds we quote; theory, not a compiler.
- Herbie / Precimonious: rewrite/precision-tuning as *external* tools; no
  receipts, no in-compiler composition.
- GCC `__float128`/libquadmath: types without semantics — the compiler does
  not know what the number means.
- Sounio's position: the only stack where the same object carries value,
  GUM uncertainty, provenance — and, with this track, its own accumulated
  roundoff — from the type judgment down to the emitted instruction bundle.

## Out of scope (explicit)

binary256; GPU octuple; extended-precision u metadata (u stays f64);
`EHORIZON`/Lyapunov horizon typing; the DISC/solver track (separate plan).
