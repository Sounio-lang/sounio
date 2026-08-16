<!-- docs:meta
topic_id: repo.docs.architecture.mli-design
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.architecture.mli-design
-->

# MLI Design — Machine-Level IR

**Status:** draft for approval (WS-D Phase 1). **No `self-hosted/` edits** until this
document is approved.

**Authority:** Founder objectives 2026-08-16 (“Madaros E2E operacional, SOIR, MIR,
MLI, HLIR EISA, f128 e f256 implantados e verificados”) and the binding design
principle in
[`docs/internal/coordination/MADAROS_FOCUS_PLAN_2026-08-16.md`](../internal/coordination/MADAROS_FOCUS_PLAN_2026-08-16.md)
§WS-D.

---

## 0. Executive summary

**MLI** (Machine-Level IR) is a **new** lowering layer between the portable
machine-oriented IR (**MIR**, WS-C route TBD) and the existing native emit
surface under `self-hosted/native/` (x86-64 ELF today; AArch64/RISC-V stubs).

Today, two Sounio-defining features are **erased or demoted to library/struct
calls** before codegen:

1. **`Knowledge<T>` / GUM uncertainty** — value + variance/epsilon + confidence
   (and, on GPU, validity + provenance lanes).
2. **Cayley–Dickson (CD) hypercomplex algebras** — dimension-tagged multiply /
   conjugate / associator already first-class in `self-hosted/ir/ir.sio`
   (`IrHyperMulQ/O/S`, `IrAssociator`, Door-β variance-of-associator) and
   **validated on GPU tensor cores** (`self-hosted/gpu/emit_ossm_oct_step_ptx.sio`,
   sedenion associator emitters; cross-backend epistemic SPIR-V 4-lane shadows).

**MLI is the last layer where those need not disappear.** This design evaluates
first-class epistemic and CD operand *kinds* alongside IEEE floats (`f32`…`f256`)
and chooses an explicit dual-track (R0 scalar / R1 epistemic-algebra) rather than
silently defaulting to “just another GPR IR.”

**Phase-1 exit (this doc approved):** layer contract, instruction/operand model,
register/stack discipline, verification story, staged ladder, named open
decisions blocked on WS-C/WS-G.

**Phase-2 exit (wave 3+, not this doc):** one function lowered MIR→MLI→x86
**bit-identical** to the current direct native path.

---

## 1. Naming disambiguation (read first)

| Name in repo | What it is today | Relation to MLI |
|---|---|---|
| **SIR / HIR / IR** (`self-hosted/ir/`) | SSA-ish compiler IR with hypercomplex and epistemic opcodes | **Above** MIR/MLI; rich but not machine-level |
| **`native::machine_ir`** (`self-hosted/native/machine_ir.sio`) | Native-v2 **legalize substrate** for x86-64 (MIR_* opcodes, GPR64 + stack slots + SSE2 float) | **Not** MLI. Today’s end-stage pseudo-ISA. MLI **feeds** this path or **replaces its pseudo layer** over time |
| **MIR** (WS-C) | Frontier ENIR→MIR pipeline on `origin/canon/madaros-v2-sota` (not fully on `main` yet) | **Input** to MLI (pending WS-C route) |
| **ENIR** | Epistemic numeric IR on the frontier branch; contract docs under `docs/internal/coordination/enir-*.md` | Source of MIR; verification pattern to mirror |
| **MLI** (this doc) | **New** Machine-Level IR | Between MIR and native emit |

If “MIR” is used without qualifier in this document, it means **WS-C portable MIR**
(the MLI *input*), not `native::machine_ir`.

---

## 2. Binding design principle — evaluation (required)

### 2.1 What mainstream backends do

LLVM / Cranelift / GCC treat:

- uncertainty as **user structs + library calls**,
- quaternions/octonions as **arrays of f32/f64** or SIMD vectors with **no algebra
  identity** in the type system.

Register allocation never asks “is this a GUM-coupled pair?” or “is this an
octonion (non-associative) ZMM pair?” Instruction selection never forbids illegal
reassociation of `Knowledge` arithmetic or Fano-line short-circuits.

### 2.2 What Sounio already has (evidence)

| Surface | Epistemic | CD algebra | Machine-ish? |
|---|---|---|---|
| Typecheck (`check/epistemic.sio`, `check/cayley_dickson.sio`) | first-class | first-class | no |
| IR opcodes (`IrMeasure`, `IrLiftKnowledge`, `IrHyperMul*`, `IrAssociator`, Door-β) | first-class | first-class | partial |
| GPU PTX/SPIR-V (`epistemic_spirv.sio` 4-lane shadows; `emit_ossm_oct_step_ptx.sio` WMMA L(A)·H) | first-class | first-class (layout/matrix) | **yes for GPU** |
| Native x86 path (`native/machine_ir.sio` + `hyper_lower.sio`) | **erased** to f64 / library | **demoted** to stack-slot loops or EVEX sequences late | scalar/SIMD without kind |

GPU already proves the research claim: **four-lane epistemic shadows** and
**tensor-core octonion left-multiply** are implementable when the IR *keeps* the
kind. CPU native is where erasure still wins by default.

### 2.3 Design options (must not skip)

#### Option A — Scalar-only MLI (silent default we reject as *sole* design)

MLI operands = `{i8…i64, u*, f32, f64, f128, f256, ptr, flags}`.  
Knowledge / Hyper lower to **field projections + calls** before MLI.

| Pros | Cons |
|---|---|
| Fast path to Phase-2 bit-identity | Throws away the only “beyond any existing language” opportunity called out in the plan |
| Matches LLVM mental model | Duplicates GPU epistemic work as a dead end on CPU |
| Smaller regalloc | GUM and non-associativity bugs reappear as “optimizer freedom” |

#### Option B — Full first-class epistemic + CD kinds from day one in emit

Every MLI op is kind-polymorphic; regalloc tracks multi-lane values; no scalar
subset until “later.”

| Pros | Cons |
|---|---|
| Maximum novelty | Blocks Phase-2 bit-identity for months |
| Unified CPU/GPU mental model | Couples MLI to unfinished WS-G (f128/f256) and WS-C MIR shape |
| | Verification surface explodes before any golden exists |

#### Option C — **Dual-track (recommended): kinds first-class in the type system; R0 emit first**

- **MLI type system and opcode space** include epistemic and CD kinds from the
  start (no “we’ll add tags later” lie).
- **R0 implementation** only *emits* scalar/float subset; epistemic/CD ops are
  **legal MLI** but lower via explicit **expand** pass to scalar MLI (or stay
  unexpanded and fail closed).
- **R1 implementation** enables native multi-lane regalloc + GUM op selection +
  algebra-aware schedules without rewriting the layer contract.

| Pros | Cons |
|---|---|
| Honours binding principle without silent default | Two stages of regalloc complexity |
| Phase-2 vertical slice unblocked (R0) | Risk of “R1 never ships” if not gated in CI as hard TODOs |
| GPU lessons transfer as **expand strategies**, not redesigns | Spec must freeze kind tags early |

### 2.4 Recommendation and tradeoff (explicit)

**Recommend Option C.**

- **Scope/schedule cost:** +1–2 weeks design/fixtures for kind tags and expand
  rules; R1 emit is multi-week *after* Phase-2 (wave 3+).
- **Novelty gain:** MLI becomes the first CPU machine IR in this lineage where
  **uncertainty-aware and algebra-aware instruction selection** are *expressible*
  without pretending they are structs.
- **Risk control:** Phase-2 success criterion remains bit-identity on a pure
  scalar function; R1 is separately gated.

**Not recommended:** Option A as the *architecture* (even if R0 code looks like
A). Documenting only A would violate the binding principle.

---

## 3. Layer contract

### 3.1 Pipeline position

```text
  source
    → lexer/parser/check
    → IR / HLIR  (self-hosted/ir, self-hosted/hlir)
    → [WS-B] SOIR  (serial / durable form)
    → [WS-C] ENIR → MIR   (portable machine IR; route TBD)
    → [WS-D] MLI          ← THIS LAYER
    → self-hosted/native/  (legalize / regalloc / encode / ELF)
    → x86-64 ELF (today)
```

Optional later consumers of the same MLI:

- GPU path: MLI → existing KAXI/GPU IR (share kind tags; different expand).
- Softfloat (WS-G): MLI `f128`/`f256` ops expand to routine calls or wide slots.

### 3.2 Input contract (depends on WS-C)

**Assumed MIR properties** (must be confirmed or adapted by WS-C route decision):

| Property | Requirement for MLI |
|---|---|
| SSA or near-SSA values | MLI may use virtual registers; φ-nodes resolved at MIR or early MLI |
| Explicit control flow | Blocks + terminators; no unstructured IR exceptions in R0 |
| Typed operands | MIR types map injectively into MLI kinds (see §4) |
| Side effects | Calls, stores, foreign ABI marked; pure arith free of hidden IO |
| Epistemic / Hyper | Either preserved as MIR types **or** already expanded with **explicit**
  “discharged epistemic” markers (never silent drop) |

**If WS-C chooses a scalar-only MIR:** MLI still defines R1 kinds; expand occurs
**at MIR→MLI** for epistemic/Hyper remaining in higher IR, or MLI accepts only
scalar and R1 is fed from IR→MLI side door. Prefer **one choke point**:
`mir_to_mli`.

### 3.3 Output contract (what `native/` consumes)

MLI after **target legalize** must be consumable by a thin emitter that looks like
today’s `native_v2` machine_ir consumers:

- finite opcode set mappable to `encode.sio` / `codegen.sio` emitters;
- operands resolved to **physical classes**: GPR, XMM/YMM/ZMM, k-mask, stack slot,
  immediate, reloc;
- ABI for calls already applied (arg moves, stack alignment);
- no unresolved virtual regs (post-regalloc) **or** a single documented
  “pre-regalloc MLI” form only used by the interpreter.

**Bit-identity Phase-2:** for a golden function, `encode(mli_legalize(mli))`
bytes equal `encode(current_direct_path)`.

### 3.4 What MLI is *not*

- Not a replacement for ENIR (epistemic *numeric* analysis IR).
- Not HLIR (high-level GPU/tensor scheduling).
- Not SOIR (on-disk interchange).
- Not “rename machine_ir.sio” without a kind model — that would fake completion.

---

## 4. Instruction and operand model

### 4.1 Operand kinds (first-class)

Kinds are **part of the type**, not comments.

```text
Kind =
  | Void
  | Int { bits: 8|16|32|64, signed: bool }
  | Ptr { aspace: flat }          // R0: single flat address space
  | Float { fmt: f32|f64|f128|f256 }
  | Flags                         // condition codes abstractly
  | VecF64 { lanes: 2|4|8 }       // XMM/YMM/ZMM geometric view (optional sugar)
  | Knowledge { base: Float|Int, lanes: KnowledgeLanes }
  | CD { dim: 1|2|4|8|16, coeff: Float }   // R,C,H,O,S over coeff format
  | Bundle { fields: [Kind; N] }  // rare; prefer Knowledge/CD tags
```

**KnowledgeLanes (R1, aligned with GPU):**

| Lane | Role | GPU analogue (`epistemic_spirv.sio`) |
|---|---|---|
| `val` | point estimate | `val_id` |
| `var` / `eps` | GUM variance or epsilon bound | `eps_id` |
| `conf` or `valid` | confidence 0–1000 **or** boolean validity | `valid_id` |
| `prov` | bit-packed provenance (optional in R0 expand) | `prov_id` |

Stdlib `Epistemic { val, variance, confidence }` is the **minimal CPU** shape;
full four-lane is the **GPU-compatible** shape. MLI should tag which shape is
active (`KnowledgeShape::Minimal3` vs `KnowledgeShape::Gpu4`) so expand rules
do not invent lanes.

**CD dimension:**

| `dim` | Algebra | Preferred machine packing (R1 hint) |
|---:|---|---|
| 1 | ℝ | scalar float |
| 2 | ℂ | 2-lane / xmm pair |
| 4 | ℍ | YMM (matches `IrHyperMulQ`) |
| 8 | 𝕆 | ZMM (matches `IrHyperMulO` / GPU L(A)·H tile) |
| 16 | 𝕊 | 2×ZMM (matches `IrHyperMulS`) |

Associativity / zero-divisor **policy** is not a kind field; it is an
**instruction attribute** or side table (see §4.3).

### 4.2 Operands

```text
Operand =
  | VReg { id, kind }           // virtual
  | Phys { class, idx, kind }   // post-alloc
  | Stack { slot, kind, offset_bytes }
  | Imm { bits, kind }          // kind restricts legal immediates
  | Reloc { symbol, addend }
  | BlockRef { id }
  | None
```

**Invariant:** every VReg/Phys/Stack carries a `kind`. Cross-kind `move` without
an explicit conversion op is a verify error.

### 4.3 Instruction classes

#### R0 (required for Phase-2)

| Class | Examples | Notes |
|---|---|---|
| Transfer | `mov`, `load`, `store`, `lea` | typed by kind |
| Integer arith | `add/sub/mul/div/rem`, shifts, logic | trapping policy flag |
| Float arith | `fadd/fsub/fmul/fdiv`, compares | IEEE fmt on kind |
| Convert | `i2f`, `f2i`, `fcast` (f32↔f64; f128/f256 when WS-G ready) | |
| Control | `jmp`, `br_cc`, `ret`, `call` | call ABI attrs |
| Compare | `cmp`, `fcmp` → Flags or bool vreg | |
| Stack | `alloca` / frame adjust (or precomputed frame) | |
| Pseudo | `copy`, `phi` (if not eliminated), `keep_alive` | |

#### R1 epistemic (first-class ops — design now, emit staged)

| Op | Semantics (sketch) | Expand (R0) |
|---|---|---|
| `k_measure` | raw → Knowledge | build struct / lanes |
| `k_add` / `k_sub` | val ±; var quadrature (GUM) | 3–4 float ops + sqrt |
| `k_mul` / `k_div` | first-order GUM; div near-zero guard | as GPU emitters |
| `k_fma` | optional | |
| `k_extract_val/var/conf` | lane project | field loads |
| `k_insert_*` | lane update | |
| `k_cast_base` | change base float width | |
| `k_lift` / `k_discharge` | to/from plain float (explicit) | |

**Illegal without attribute:** treating `k_add` as pure `fadd` on `val` and
discarding `var` (verify must reject).

#### R1 Cayley–Dickson (first-class ops)

| Op | Semantics | Expand / select |
|---|---|---|
| `cd_add` / `cd_sub` | componentwise | lane loop or SIMD |
| `cd_mul` | Hamilton / CD product by `dim` | Q: EVEX seq; O: Fano ZMM; S: 4×Fano; else libcall |
| `cd_conj` | conjugate | sign-flip imag lanes |
| `cd_norm2` | Σ xᵢ² | horizontal add |
| `cd_associator` | `[a,b,c]` for dim≥8 | matches `IrAssociator` |
| `cd_var_associator` | Door-β GUM correction | Fano short-circuit when exact |
| `cd_zd_probe` | zero-divisor proximity (policy) | exact i64 path remains **outside** MLI float path (Exact Core) |

**Attributes on `cd_mul` / associator:**

- `assoc_policy`: `strict` | `allow_reassoc_if_fano` | `force_expand`
- `zd_policy`: `ignore` | `flag` | `trap` (trap may be debug-only R0)

These attributes exist so optimizers **cannot silently reassociate** octonion
products (the historic “drift to mean” of IEEE thinking).

### 4.4 Module / function shape

```text
MliModule  { funcs, data, relocs, target_hint }
MliFunction { name, args: [Kind], ret: Kind, blocks, frame_info, attrs }
MliBlock    { id, params?, instrs, term }
```

**R0 constraint (Phase-2):** single return, no exception edges, SysV x86-64 only.

---

## 5. Register and stack discipline

### 5.1 Abstract register classes

| Class | Holds | x86-64 map (R0/R1) |
|---|---|---|
| `GPR` | Int, Ptr | rax…r15 |
| `FPR` | f32/f64 | xmm (scalar SSE2) |
| `V256` | f64×4 / partial CD | ymm |
| `V512` | f64×8 / O | zmm (needs EVEX path; already sketched in IR) |
| `KMASK` | predicates | k0–k7 |
| `FLAG` | conditions | eflags (transient) |
| `KNOW` | Knowledge multi-lane (R1) | **home**: consecutive stack slots or multi-phys bundle |
| `CD` | CD multi-lane (R1) | **home**: ymm/zmm pairs or stack blob |

### 5.2 Homing rules

1. **Scalar (R0):** virtual reg → physical class by kind; spill to stack slots
   sized by kind (`f64`=8, `f128`=16, `f256`=32 when live).
2. **Knowledge (R1):** preferred **SoA stack home** (val, var, conf[, prov]) for
   ABI simplicity; short-lived **FPR pairs** for val/var during GUM sequences
   (matches GPU SoA comment in `epistemic_spirv.sio`).
3. **CD dim=8:** prefer single ZMM home when EVEX available; else 8×f64 stack
   (today’s `hyper_lower.sio` fallback).
4. **CD dim=16:** two consecutive ZMM homes or 16×f64 stack.
5. **Calling convention:** R0 SysV: integers in rdi/rsi/…; floats in xmm0…;
   Knowledge/CD **passed by hidden pointer** unless a future ABI RFC freezes
   multi-reg returns (Madaros multi-module sret issues make “struct return”
   fragile today — fail closed to by-ref).

### 5.3 Stack frame

```text
[ inbound args ]
[ return addr ]
[ saved rbp ]
[ callee-save ]
[ MLI slots: scalar | Knowledge homes | CD homes ]  ← downward from rbp
[ red zone? only leaf pure R0, optional ]
```

**Frame metadata** for GC/stack maps stays out of R0 unless a function already
uses them on the direct path.

### 5.4 Interaction with existing `machine_ir.sio`

Proposed migration (post-approval, not in Phase 1 code):

1. Treat current `MachineInstr` as **post-legalize x86 pseudo** (keep MIR_* names
   or rename to `X86_*` later).
2. New modules: `self-hosted/mli/{ir,builder,verify,interp,lower_from_mir,legalize_x86}.sio`.
3. `legalize_x86`: MLI → today’s machine_ir ops (R0) or EVEX sequences (R1 CD).
4. Do **not** grow `machine_ir.sio` with Knowledge kinds ad hoc — that recreates
   erasure under a new name.

---

## 6. Verification story

Mirror the **ENIR pattern** described in the plan (`enir/interpreter.sio` +
`verify.sio`) and the existing IR habit in `self-hosted/ir/verify.sio`
(normalize + compare), even though `self-hosted/enir/` is **not on this branch’s
tree** (frontier / WS-C port).

### 6.1 Three layers of check

| Layer | What | Pass criterion |
|---|---|---|
| **V-struct** | Kind well-formedness, opcode arity, dominance, no raw cross-kind moves | pure static |
| **V-interp** | MLI interpreter executes a function on concrete inputs | matches reference oracle |
| **V-parity** | For R0 functions: shadow-exec MLI vs current native binary / direct path | bit-identical or value-identical per policy |

### 6.2 Interpreter (`mli_interp`)

- State: vreg file + stack memory + flags.
- Knowledge ops: implement **exact GUM rules** used by GPU emitters (document the
  formula table next to `epistemic_spirv.sio` so CPU/GPU cannot diverge silently).
- CD ops: implement **f64 component algebra** matching `stdlib/algebra` /
  `ir` conventions (Fano table, CD doubling); optional exact-i64 path remains a
  **separate** Exact Core concern (`docs/EXACT_CORE.md`) — do not mix “proved ZD”
  into float MLI.

### 6.3 Verify (`mli_verify`)

1. Structural verify (V-struct).
2. Optional **kind preservation** audit: every MIR epistemic/Hyper value either
   (a) remains Knowledge/CD through MLI, or (b) has an explicit `k_discharge` /
   `cd_expand` with source location.
3. Roundtrip: text or binary dump → parse → structural equality (like IR
   serialize/normalize).

### 6.4 Phase-2 golden

- Fixture: pure function `f(x: f64, y: f64) -> f64` (e.g. `x*y + 1.0`).
- Pipeline A: today’s native compile.
- Pipeline B: MIR→MLI→legalize→encode.
- **Pass:** identical `.text` bytes (or identical post-link function body under a
  fixed seed of nops). Prefer **bytes** to avoid “same math, different spills.”

### 6.5 R1 goldens (later)

- `k_add` matches stdlib `ep_add` on random inputs (value + variance + conf).
- `cd_mul` dim=8 matches `oct_mul` bit-for-bit on f64 components for a fixed set.
- Door-β Fano short-circuit: Fano triples emit cheap path (count ops or match
  mask).

---

## 7. Staged implementation ladder

| Stage | Deliverable | Gate |
|---|---|---|
| **S0** | This design approved | human sign-off |
| **S1** | `mli` module: kinds, builder, text dump, V-struct verify | unit tests on fixtures (no full compile) |
| **S2** | `mir_to_mli` for **scalar R0 only** (integers + f64) | interpreter tests |
| **S3** | `legalize_x86` → existing encode path | Phase-2 **bit-identical** vertical slice |
| **S4** | f32 + conversions; align with WS-G for f128/f256 **slots** (arith may libcall) | WS-G coordination |
| **S5** | Knowledge kinds + `k_*` ops + expand-to-R0 | parity vs `ep_*` free functions |
| **S6** | CD kinds dim≤8 + `cd_mul` select (EVEX or stack loop) | parity vs stdlib / IR hyper tests |
| **S7** | dim=16, associator, Door-β, zd_probe policy | GPU formula alignment checklist |
| **S8** | Optional: MLI→GPU kind-preserving bridge | reuses KAXI epistemic markers |

**Hard rule:** S3 does not require S5–S7, but S1 **must** define Knowledge/CD
kinds so S5 is not a breaking rewrite.

**Parallelism:**

- WS-C must freeze MIR type mapping before S2 lands on main.
- WS-G owns f128/f256 **arithmetic**; MLI only owns **kind + expand hooks**.
- WS-F EISA remains a separate conformance consumer; may later golden MLI dumps.

---

## 8. Worked examples

### 8.1 R0 scalar (Phase-2 target)

Source:

```sounio
fn add1(x: f64) -> f64 { x + 1.0 }
```

MLI sketch:

```text
func @add1(v0: f64) -> f64 {
  b0:
    v1: f64 = fadd v0, imm f64 1.0
    ret v1
}
```

Legalize → `machine_ir` float binop / movsd path → ELF.

### 8.2 R1 Knowledge (design now)

Source:

```sounio
fn add_k(a: Knowledge<f64>, b: Knowledge<f64>) -> Knowledge<f64> {
  // desugars to GUM add
  a + b
}
```

MLI sketch:

```text
func @add_k(a: Knowledge<f64, Min3>, b: Knowledge<f64, Min3>) -> Knowledge<f64, Min3> {
  b0:
    r: Knowledge<f64, Min3> = k_add a, b   // kind preserved
    ret r
}
```

R0 expand (legalize):

```text
// a.val, a.var, a.conf in stack homes
v = fadd a.val, b.val
var = fadd a.var, b.var          // or quadrature: sqrt(var_a+var_b)
conf = min(a.conf, b.conf)       // policy as today
```

**Without kind:** optimizers could CSE `a.val+b.val` with a plain float add
elsewhere and drop variance — the bug class MLI is designed to make *local* and
*checkable*.

### 8.3 R1 octonion (GPU-informed)

Source: `oct_mul(a, b)` or IR `IrHyperMulO`.

MLI:

```text
r: CD{dim=8, f64} = cd_mul a, b   assoc_policy=strict
```

Select:

- if target has EVEX ZMM: lower like current hyper EVEX sequence;
- else: 8-lane stack loop (`hyper_lower.sio` style);
- GPU sibling: expand to L(A)·H WMMA tile layout (not MLI’s job to emit PTX, but
  **kind+dim** must match so a future bridge does not repack blindly).

---

## 9. Risks and non-goals

### Risks

| Risk | Mitigation |
|---|---|
| WS-C MIR shape changes | Keep `mir_to_mli` adapter thin; freeze kind enum early |
| R1 never funded | CI checklist items for S5/S6 as explicit unpaid debt in MADAROS status |
| Kind explosion | No open-ended user algebras in MLI; CD dim ∈ {1,2,4,8,16} only |
| Exact ZD vs float MLI confusion | Exact Core stays separate; MLI float CD never claims `Proved` |
| Madaros multi-module / sret fragility | Knowledge/CD ABI by-ref in R0/R1 |

### Non-goals (Phase 1–3)

- Full optimizing mid-end on MLI (DCE/GVN can wait; MIR/IR already optimize).
- Replacing GPU IR.
- Implementing f256 arithmetic (WS-G).
- Porting ENIR itself (WS-C).

---

## 10. Open decisions (blockers)

| ID | Decision | Owner |
|---|---|---|
| O1 | Final MIR type → MLI kind mapping table | WS-C + WS-D |
| O2 | Knowledge shape default: Min3 vs Gpu4 on CPU | design review |
| O3 | Whether `native::machine_ir` renames or remains post-legalize only | native maintainers |
| O4 | f128/f256: softfloat libcall vs hardware when present | WS-G |
| O5 | Bit-identity vs value-identity if encoding noise unavoidable | Phase-2 gate owners |

---

## 11. Approval checklist (Phase-1 done)

- [x] Layer contract (MIR in → native out) stated
- [x] Instruction/operand model including **first-class** Knowledge + CD kinds
- [x] Binding principle **evaluated** with explicit R0/R1 tradeoff (Option C)
- [x] Register/stack discipline for multi-lane kinds
- [x] Verification story (struct + interp + parity), ENIR-pattern mirror
- [x] Staged ladder with Phase-2 bit-identity isolated from R1
- [x] Naming disambiguation vs existing `machine_ir.sio`
- [ ] Human approval (founder / orchestrator) — **pending**
- [ ] No `self-hosted/` MLI code until approval

---

## 12. References (repo-local)

- Plan: `docs/internal/coordination/MADAROS_FOCUS_PLAN_2026-08-16.md` (WS-D)
- Concepts: `docs/internal/concepts/epistemic-numeric-value.md`
- Exact vs measured: `docs/EXACT_CORE.md`
- ENIR contract lineage: `docs/internal/coordination/enir-semantic-interface-contract-2026-07-12.md`
- IR hyper/epistemic ops: `self-hosted/ir/ir.sio`
- Native substrate: `self-hosted/native/machine_ir.sio`, `hyper_lower.sio`, `codegen.sio`
- GPU epistemic 4-lane: `self-hosted/gpu/epistemic_spirv.sio`
- GPU octonion TC: `self-hosted/gpu/emit_ossm_oct_step_ptx.sio`
- Stdlib epistemic: `stdlib/epistemic/knowledge.sio` (`Epistemic { val, variance, confidence }`)
- IR verify pattern: `self-hosted/ir/verify.sio`

---

## 13. Document control

| Field | Value |
|---|---|
| Draft author | grok-cli2 (WS-D) |
| Claim | `bin/sounio-coord` lane `ws-d-mli-design` |
| Branch context | working tree may be research/*; doc is integration-agnostic |
| Next action | Reviewer approval → open S1 implementation dispatch (still no silent R1 drop) |
