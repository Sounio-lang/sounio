# K-AXI to PTX lowering inventory for PBPK-28 GPU work

Status: working inventory, verified 2026-10-30 on branch
`pbpk-28-gpu-parity`. Each claim cites the source inspected.

//! Scope: which Sounio constructs lower to GPU (HLIR to GPU-IR to
//! K-AXI to PTX) and which do not. Feeds dissertation GPU chapter and
//! the PBPK-28 parity gates (tasks 1-3 of the PBPK-28 GPU track).

## Pipeline (verified)

Sounio source -> HLIR (`self-hosted/gpu/hlir_to_gpu.sio`) ->
GPU-IR op array (`self-hosted/gpu/kaxi_backend.sio:gpu_lower_to_kaxi`,
general, 256-op cap) -> K-AXI asm text ->
PTX (`self-hosted/gpu/kaxi_to_ptx.sio:kaxi_transpile_to_ptx_unified`,
line-oriented, op-general).

The PBPK GPU witnesses bypass the front end: hand-written emitters
(`kretikos_emit_kaxi.sio`, `kretikos_kaxi_to_ptx.sio` — 233 lines,
witness-specific) plus C samplers (`scripts/gpu/kaxi_pbpk_2comp*.c`),
gated by `scripts/ci/kretikos_kaxi_phase_y_gate.sh`. Golden reference:
`tests/golden/kaxi_ptx/f32_gum/pbpk_2comp_gum_4step.ptx`
(sha256 `a76f7e5a...`, byte-identical).

## Lowers today

- K-AXI line ops: add/sub/mul/div/fma/min/max/neg/abs,
  lg2/ex2/sqrt/rsqrt/sin/cos/recip, setp/selp predicates,
  global and immediate loads/stores, getvar/get, ret, predicated
  branches, barrier (`kaxi_to_ptx.sio`, f64/f32/i32 + epistemic).
- HLIR: BINARY (incl. setp comparisons), LOAD, STORE,
  GET_ELEMENT_PTR, CONST (float/int/unit/undef), COPY, CAST,
  UNARY neg (f64/f32/i64/i32), CALL_DIRECT only for GPU intrinsics
  (tid/bid/ntid) and oct/sed batch-mul, BRANCH/COND_BRANCH
  terminators, RETURN.
- Types with lowering arms: i32/i64/f32/f64/u32/u64/bool/ptr.
- Epistemic shadow propagation for arithmetic (Sprint 28):
  per-value eps/valid/prov shadow registers, quadrature rules.
- Integer remainder: expanded as sub(a, mul(div(a,b),b))
  (`hlir_to_gpu.sio` HlirOpRem arm; O44 oracle in
  `self-hosted/test_gpu_oracle.sio` pins 3 ops in Div/Mul/Sub order).

## Explicitly refused (fail-closed, with reason codes)

`hlir_to_gpu.sio` refusal reasons (`hlir_gpu_refusal_reason`,
10 codes): bool const, float-lit-in-int-slot, exotic consts
(string/array/struct/null/fn/global), unary NOT (deref/ref path),
unary neg of unknown type, ALLOCA (`var` locals have no PTX
storage), PHI (control-flow join values, incl. if/match expression
values), non-intrinsic CALL (helper fns must inline), unknown ops,
SWITCH/match statements. Further refused op kinds (op-kind namer):
GET_FIELD_PTR, EXTRACT_VALUE, INSERT_VALUE, TUPLE, ARRAY, STRUCT
literals, effect ops, SELECT. Struct/array state must live in
globals/params and be reached via GET_ELEMENT_PTR (which lowers).

Consequences for a PBPK-28 ODE kernel body:
- `while` time-stepping loop: loop-carried compartment state needs
  PHI/ALLOCA, both refused. Either unroll fixed trip counts (CN
  stepper has fixed 13-organ + reduction structure, closed-form
  Schur complement, no Newton iteration) or implement PHI lowering.
- `if`-EXPRESSION value selects (volume floors, sink terms in the CN
  stepper) lower via PHI: use statement-`if`s or rely on `selp`.
- Helper `fn` calls (e.g. Euler/CN step helpers) must inline.

## Silent hazards (no refusal, wrong or degraded code)

- DONE (fixed on this branch): `HlirOpRem` emitted lone `Sub(a,b)`.
  Fix + O44 oracle landed; no golden pattern used Rem (no churn).
- `hlir_type_kind_to_gpu` has no Knowledge arm: Knowledge-typed
  values silently become `GpuI64`. Unreachable today (no caller);
  deferred to the GUM kernel task, which needs an f64 value channel
  (`hlir_lower_type` maps Knowledge to F32 by default — insufficient
  for the 1e-9 GUM parity gate).
- Unknown binops default to `GpuAdd`; unknown GPU-IR ops lower to
  NOP (`gpu_lower_to_kaxi`). Both must be covered by gate assertions.
- EXP/SIN branches of `gpu_lower_to_kaxi` skip the `seq` increment.
  `seq=` is write-only metadata (never parsed downstream, absent from
  goldens): cosmetic, dispositioned, revisit if the scheduler consumes it.
- `HlirOpRem` float path: none exists (integer only) — unchanged.

## Task mapping

- Task 1 (generalize K-AXI to PTX): reuse `gpu_lower_to_kaxi` +
  line transpiler; close PHI/ALLOCA/CALL gaps or unroll; document
  every construct here as it starts lowering (golden gate
  `kaxi_ptx_golden_gate.sh` pattern x mode inventory is the seed).
- Task 2a (GUM kernel): needs the Knowledge f64 channel + eps-shadow
  seeding from prior sigma; replicate CPU semantics exactly
  (central-difference Jacobian, diagonal combination, fail-closed
  perturbed solves, exact-0.0 column tripwire — see SESSION_NOTES).
- Task 2b (MC kernel): MRG32k3a done and bit-exact
  (`stdlib/random/mrg32k3a.sio` + oracle test); per-thread skip-ahead
  via A1p76 powers; identical polar/lognormal/fail-closed rules.
