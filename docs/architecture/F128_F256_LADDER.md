<!-- docs:meta
topic_id: repo.docs.architecture.f128-f256-ladder
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.architecture.f128-f256-ladder
-->

# F128/F256 Implementation Ladder (V0-B → V0-E)

**Dispatch**: Wave-1 from fleet-orchestrator (claude-1), full context in `docs/internal/coordination/MADAROS_FOCUS_PLAN_2026-08-16.md` §WS-G.  
**Owner**: grok-cli3 (lane `ws-g-f128-spec`).  
**Claim**: `bin/sounio-coord claim --agent grok-cli3 --lane ws-g-f128-spec --intent 'WS-G f128/f256 ladder spec' --files docs/architecture/F128_F256_LADDER.md` (active).

This document lifts the current **V0-A boundary** (parser rejection of `f128`/`f256` source forms with E249 before any check/IR/SOIR/ABI/native lowering) into a staged, gate-defined ladder. No changes to `self-hosted/` are authorized in this phase. All progress is expressed through:

- New/updated test fixtures and compile-fail cases.
- Extension of the three existing scaffolds in `self-hosted/compiler/` (`f128_f256_format_descriptor_probe.sio`, `f128_f256_numeric_payload_probe.sio`, `f128_f256_numeric_wire_probe.sio`).
- New CI gates that exercise the ladder without claiming semantic arithmetic or epistemic surface until V0-E.
- Updates to `docs/EXACT_CORE.md`, `docs/compiler/KNOWN_LIMITATIONS.md`, and the epistemic trust map.

The ladder aligns with the single-semantic-clock rule (executable meaning for language and library claims is owned by Sounio under the default compiler, Madaros / `bin/souc`), precision preservation (`docs/internal/concepts/precision-preservation.md`), and the `Verdict` enum evolution in `stdlib/algebra/sedenion_verdict.sio`. The semantic-clock rule was drafted as ADR-008 on a branch that never landed, so there is no ADR of record for it — `docs/decisions/` stops at ADR-007.

## Current V0-A Boundary (as of Madaros v0.80.0)

From `docs/EXACT_CORE.md:55-57` and `self-hosted/parser/types.sio:27-45`:

```sounio
// Parser immediately rejects with E249
fn identity_f128(x: f128) -> f128 { x }  // compile-fail
let a: f128 = 1.0                         // compile-fail
let b = a + 1.0f128                       // compile-fail
```

- `MeasuredF256 { eps256 }` exists only as a future witness shape in `Verdict`; no arithmetic.
- Format descriptors, limb pools, and wire formats are already partially exercised by the three probes (binary128/binary256 IEEE-like, LSW-first limbs, roundtrip, negative cases, IR constant emission).
- Gates: `scripts/ci/madaros_f128_f256_format_identity_gate.sh`, `madaros_f128_f256_numeric_payload_gate.sh`, `madaros_f128_f256_numeric_wire_gate.sh` (included in IR/SOIR bridges).
- No literals, no user-visible operations, no stdlib surface, no `print_f128`/`Knowledge<f128>` interaction.
- All `tests/compile-fail/f128_f256_*.sio` and `tests/native-v2/f128_format_identity_*` remain authoritative negatives.

This boundary is **structural-only** and enforced **on Madaros** before type checking or IR lowering.

### Engine split (FINDING 2026-08-17; V0-B lift 2026-09-06)

| Engine | `f128`/`f256` type spellings + literals | Arithmetic / casts |
|---|---|---|
| **Madaros** (default `bin/souc`) | **V0-B green:** accepted through `check` (E249 lifted). | Still refused (E004 / E248 / mismatch). |
| **lean_single** (bootstrap seed) | Accepts type names; arithmetic behaviour is seed-owned (see #2387 / lean_single f128 work). | Not the Madaros V0-B contract. |

V0-A (parser E249 on all `f128`/`f256` source forms) was the Madaros-owned boundary before this stage. Compile-fail fixtures that still mention E249 under lean_single suite participation may carry `//@ known-failure: lean_single-only gap…` where needed — that documents an engine gap, **not** permission to treat f128 arithmetic as accepted under Madaros.

The authoritative V0-B contract remains `scripts/ci/madaros_f128_f256_ladder_gate.sh --stage v0b` under default Madaros.

## The V0-B..E Ladder

Each stage has a **gate definition** (CI command, positive/negative witnesses, success receipt, semantic-lane ID, acceptance criteria). Gates are additive; later stages subsume earlier ones. **V0-B green is judged on Madaros** (`madaros_f128_f256_ladder_gate.sh`); lean_single suite participation uses known-failure annotations until the seed gains E249 or is retired from this surface. Success receipts must be exact-string matched in gate scripts.

### V0-B: Literals Accepted End-to-End Through Check

**Goal**: Parser accepts `f128`/`f256` type spellings and decimal/hex/binary literals. Type checker accepts them as distinct from `f64` (no implicit conversion). No arithmetic, no casts, no runtime values yet. Negative witnesses for arithmetic remain.

**Gate**:
```bash
bash scripts/ci/madaros_f128_f256_ladder_gate.sh --stage v0b
```
(or integrated into `madaros_f128_f256_format_identity_gate.sh --stage v0b`).

**Positive witnesses** (new):
- `tests/run-pass/f128_v0b_literal_smoke.sio` — binds literals, passes to imported identity functions (see existing `tests/fixtures/f128_format_identity_leaf.sio`), `check` only.
- `tests/run-pass/f256_v0b_hex_literal.sio` — `0x1.0p+0f256`, array initializers.

**Negative witnesses** (updated/expanded):
- Existing `tests/compile-fail/f128_f256_arithmetic_unimplemented.sio`, `f128_f256_literal_unimplemented.sio` (now only arithmetic/cast cases fail).
- New: implicit conversion, generic arg misuse (already partially present).

**Success receipt**:
```
PASS f128_f256_v0b_literals check=green parser=E249_lifted typecheck=distinct_no_implicit literals=decimal+hex+binary negative_arithmetic=4
```

**Status (2026-09-06):** gate green on Madaros. Positive probes
`tests/run-pass/f128_v0b_literal_smoke.sio` /
`tests/run-pass/f256_v0b_literal_forms.sio` reach `check: OK`.

### V0-C: Wire Format / Limb Pools (Extend Existing Three Probes)

**Goal**: Fully exercise and extend the existing scaffold probes for 128/256-bit payloads. Limb pool management, wire serialization (byte-exact roundtrip, adler32 checksums, negative encoding), descriptor queries, IR constant emission for wide literals, SOIR/BSS layout. No arithmetic.

**Gate**:
```bash
bash scripts/ci/madaros_f128_f256_ladder_gate.sh --stage v0c
```
(extends `madaros_f128_f256_numeric_payload_gate.sh` + `numeric_wire_gate.sh`).

**Positive witnesses** (extensions of existing):
- `self-hosted/compiler/f128_f256_format_descriptor_probe.sio` (already green; extend with more formats).
- `self-hosted/compiler/f128_f256_numeric_payload_probe.sio` (extend limb counts, LSW-first validation, pool merging, 256-bit 4-limb cases).
- `self-hosted/compiler/f128_f256_numeric_wire_probe.sio` (extend to full 256-bit wire buffers, more negative cases, transactional reseal, IR→SOIR roundtrip).

**Negative witnesses**: malformed limb counts, wrong format_id, checksum mismatch, overflow.

**Success receipt** (example, expanded):
```
PASS f128_f256_v0c_wire limbs=8 order=lsw-first payloads=4 wire_bytes=272 roundtrip=exact decode_negative=24 encode_negative=4 checksum=adler32 ir_emit=green soir_bss=green
```

**Acceptance criteria**:
- Existing three probes pass with expanded coverage (at least 256-bit full limb support).
- Wide literals from V0-B emit as `IrWideNumericPayload` constants without fallback.
- Wire format is authoritative for ABI and future softfloat constants.
- No runtime arithmetic emitted.
- Semantic-Lane-ID: `WS-G-V0C-WIRE-LIMB-POOLS`.
- Ties into `docs/architecture/SOIR_REFERENCE.md` and native-v2 SRET/ABI.

### V0-D: Softfloat Arithmetic (Compiler-Owned Limb Routines)

**Goal**: Implement `add`/`sub`/`mul`/`div`/`cmp` (and `sqrt`, `fma` if natural) as compiler-owned routines operating on limb pools. Constant folding where possible. Rounded vs exact semantics defined. No user `+`/`-` surface yet (still compile-fail for source ops); used internally for constant evaluation and future `MeasuredF256`.

**Gate**:
```bash
bash scripts/ci/madaros_f128_f256_ladder_gate.sh --stage v0d
```

**Positive witnesses**:
- `tests/run-pass/f128_v0d_softfloat_const_fold.sio` — compile-time evaluation of wide constants.
- Extended numeric probes exercising limb routines (add/sub/mul/div/cmp on known values, including edge cases: subnormals, NaN, inf, zero-divisor proximity for epistemic use).
- New `self-hosted/compiler/f128_f256_softfloat_limb_probe.sio`.

**Negative witnesses**: overflow without rounding mode, incorrect rounding, provenance loss.

**Success receipt**:
```
PASS f128_f256_v0d_softfloat ops=add/sub/mul/div/cmp limb_routines=green const_fold=exact rounded=ieee754-2019 negative_cases=32 MeasuredF256_witness=structural
```

**Acceptance criteria**:
- Compiler owns the limb implementations (no libc dependency for these ops).
- `Verdict::MeasuredF256` can now be populated with real eps256 from softfloat.
- Ties directly to GUM propagation rules and `stdlib/epistemic/`.
- No stdlib surface or `print` yet.
- Semantic-Lane-ID: `WS-G-V0D-SOFTFLOAT-LIMB-ROUTINES`.
- Updates `docs/compiler/KNOWN_LIMITATIONS.md` (removes arithmetic unimplemented note) and epistemic trust map.

### V0-E: Stdlib Surface + Printing + GUM Interaction

**Goal**: Full user-visible surface. `stdlib/math/float128.sio` (or `wide_float.sio`), `print_f128`, `format`, `Knowledge<f128>`, `Knowledge<f256>`, refinement types, epistemic propagation (`GUM` variance, provenance), units integration if applicable. Full `souc run` support. Printing must be deterministic and match softfloat results.

**Staged delivery (2026-09-06):**

| Slice | Gate | Status bar |
|---|---|---|
| **V0-E.1** print + limb stdlib API | `bash scripts/ci/madaros_f128_f256_ladder_gate.sh --stage v0e` | Deterministic softfloat decimal/hex print; `stdlib/math/wide_float.sio`; Madaros `check` + seed-run hex wire smoke. Language `f128` `+` still E004; GUM/`Knowledge`/`MeasuredF256` deferred. |
| **V0-E.2** source ops + builtin print + GUM | (future; umbrella `f128_f256_full_ladder_gate.sh`) | Lift E004, Madaros-run `print_f128`, `Knowledge`/`GUM` k95, `MeasuredF256=executable`. |

**Gate** (V0-E.1):
```bash
bash scripts/ci/madaros_f128_f256_ladder_gate.sh --stage v0e
```

**Positive witnesses (V0-E.1)**:
- `tests/run-pass/f128_v0e_stdlib_smoke.sio` — limb constructors + hex wire print
- `scripts/dev/ws_g_v0e_wide_print_oracle.py` — decimal/hex vs softfloat (rump ≠ f64)
- `stdlib/math/wide_float.sio` — `F128Bits` / `F256Bits` API

**Success receipt (V0-E.1)**:
```
PASS f128_f256_v0e_surface print=deterministic stdlib=wide_float.sio hex_wire=softfloat_match madaros_check=green seed_smoke=green gum=deferred MeasuredF256=deferred
PASS madaros_f128_f256_ladder_gate stage=v0e
```

**Positive witnesses (V0-E.2 — not yet)**:
- `tests/run-pass/f128_v0e_stdlib_smoke.sio`, `f256_gum_interaction.sio`, `print_wide_precision.sio`.
- Integration with `sedenion_verdict.sio` and `Knowledge<T>` for high-precision measurements.
- CPC-style receipts using `f256` where `f64` was previously marginal.

**Success receipt (V0-E.2 full)**:
```
PASS f128_f256_v0e_full stdlib_surface=complete print=deterministic gum_interaction=k95_trust MeasuredF256=executable epistemic_trust_map=updated ladder_complete=V0-E
```

**Acceptance criteria (V0-E.1)**:
- Print oracle bit-stable and rump decimal ≠ f64 greenwash path.
- `stdlib/math/wide_float.sio` is the limb API source of truth; smoke hex wires match softfloat.
- Language source arithmetic / GUM / `MeasuredF256` remain explicitly deferred (V0-E.2).
- Semantic-Lane-ID: `WS-G-V0E-STDLIB-GUM-SURFACE`.

## Implementation Notes & Non-Goals

- **Order**: Must be strictly staged. V0-B before V0-C, etc. Each gate must pass independently.
- **No self-hosted edits in this doc**: Implementation of gates, probes, and softfloat routines belongs to subsequent dispatches (after this spec is reviewed/registered).
- **Auditability**: Every stage requires positive + negative witnesses, exact receipts, and updates to the claim oracle. No retrofitted tolerances.
- **Coordination**: Use `bin/sounio-coord` for any overlapping lanes (especially parser, IR, stdlib, epistemic). See `docs/internal/concepts/SEMANTIC_LANE_CONTRACT.md`.
- **Dependencies**: Builds on existing numeric payload/wire infrastructure. Softfloat must be limb-based (no external libm for core ops).
- **Next actions after this spec**:
  1. Register semantic lane in `docs/governance/topic-registry.v1.json`.
  2. Implement V0-B gate + probes (next Codex lane).
  3. Update `EXACT_CORE.md`, `KNOWN_LIMITATIONS.md`, and CI inclusion.
  4. LLM-offload review of the full ladder for epistemic/math claims.
  5. Handoff back to fleet-orchestrator.

**Status**: Spec complete (V0-A baseline + full staged ladder with gates). Ready for review.

*Last revised 2026-08-16. See `git log --oneline docs/architecture/F128_F256_LADDER.md` for updates. Measure before claiming (run the ladder gate).*
