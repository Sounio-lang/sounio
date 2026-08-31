<!-- docs:meta
topic_id: repo.docs.architecture.mli-design
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.architecture.mli-design
-->

# MLI Design — Machine-Level IR

**Status:** Option C **founder-approved** (2026-08-16). Preflight amendments D1–D4
from [`WS_C_D_PREFLIGHT_REVIEW_2026-08-16.md`](WS_C_D_PREFLIGHT_REVIEW_2026-08-16.md)
folded in (this revision). **No `self-hosted/` MLI code** until implementation
dispatch after this amended design is accepted.

**Authority:** Founder objectives 2026-08-16 (“Madaros E2E operacional, SOIR, MIR,
MLI, HLIR EISA, f128 e f256 implantados e verificados”); binding design principle
in [`MADAROS_FOCUS_PLAN_2026-08-16.md`](../internal/coordination/MADAROS_FOCUS_PLAN_2026-08-16.md)
§WS-D; Route B for WS-C (`MIR_PORT_PLAN.md`) is **settled** — do not re-litigate
Route B or Option C here.

**Adversarial preflight:** fable-1 reviewed Option C; it **survived**. Amendments
below close input-contract and kind-model gaps only.

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

**Phase-2 exit (wave 3+, not this doc):** one function lowered **IR→MLI→x86**
(see §3.2 feed choice) **bit-identical** to a **pinned** direct native path
(see §10 O5 — resolved). Not expressible from Route-B EMIR alone.

---

## 1. Naming disambiguation (read first)

| Name in repo | What it is today | Relation to MLI |
|---|---|---|
| **SIR / HIR / IR** (`self-hosted/ir/`) | SSA-ish compiler IR with hypercomplex and epistemic opcodes | **Above** MIR/MLI; rich but not machine-level |
| **`native::machine_ir`** (`self-hosted/native/machine_ir.sio`) | Native-v2 **legalize substrate** for x86-64 (MIR_* opcodes, GPR64 + stack slots + SSE2 float) | **Not** MLI. Today’s end-stage pseudo-ISA. MLI **feeds** this path or **replaces its pseudo layer** over time |
| **MIR / EMIR** (WS-C Route B) | Frontier `self-hosted/enir/` → EMIR (`enir/mir.sio`): **epistemic-bundle** machine IR | **Not** a general scalar machine IR; see §3.2 gap |
| **ENIR** | Epistemic numeric IR on `origin/canon/madaros-v2-sota`; contract docs under `docs/internal/coordination/enir-*.md` | Upstream of EMIR; verification pattern to mirror |
| **MLI** (this doc) | **New** Machine-Level IR | Between **IR and/or EMIR** and native emit |

**Disambiguation rule:** `native::machine_ir` constants use the historical
`MIR_*` prefix on main — **not** the same object as Route-B EMIR. See §5.4 /
preflight D4: renaming those constants to `X86_*` (or a header note) is a
**precondition of WS-C PR1**, not optional later cleanup.

If “MIR” is used without qualifier below, prefer **EMIR** when Route B is meant,
and **never** mean `native::machine_ir`.

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

**Option C is founder-approved (2026-08-16).** Preflight re-affirmed it; this
document does **not** re-open A vs B vs C.

- **Scope/schedule cost:** +1–2 weeks design/fixtures for kind tags and expand
  rules; R1 emit is multi-week *after* Phase-2 (wave 3+).
- **Novelty gain:** MLI becomes the first CPU machine IR in this lineage where
  **uncertainty-aware and algebra-aware instruction selection** are *expressible*
  without pretending they are structs.
- **Risk control:** Phase-2 success criterion remains bit-identity on a pure
  scalar function fed by the **IR→MLI side door** (§3.2); R1 is separately gated
  and is the natural home for **Route-B EMIR epistemic bundles**.

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
         │
         ├─[WS-C Route B] ENIR → EMIR (epistemic bundles only)
         │                      └─ emir_to_mli ──► MLI R1 Knowledge …
         │
         └─ ir_to_mli (S2–S3 primary) ──────────► MLI R0 scalar …
                                                    │
  [WS-D] MLI  ◄─────────────────────────────────────┘
    → legalize_x86 → self-hosted/native/ (encode / ELF)
    → x86-64 ELF (today)
```

Optional later consumers of the same MLI:

- GPU path: MLI → existing KAXI/GPU IR (share kind tags; different expand).
- Softfloat (WS-G): MLI IEEE `f128`/`f256` ops expand to routine calls or wide slots
  (**not** qd128 — see §4.1).

### 3.2 Input contract — Route B gap and chosen S2/S3 feed (preflight D1)

#### 3.2.1 What Route-B EMIR actually is (measured, not assumed)

WS-C **Route B** lands frontier `enir/mir.sio` (EMIR). It is **not** a general
scalar machine IR. Against that source (preflight fable-1, 2026-08-16):

| Fact | Consequence for MLI |
|---|---|
| **10 opcodes only:** `CONST ADD SUB MUL DIV SQRT OBSERVE LOAD STORE MOVE` | **No** integer ops, **no** `call`, **no** `ret`, **no** compare/branch |
| Module verify: `type_count == 1` and that type is an **epistemic bundle** `{value_kind: f64, error_kind: qd128, uncertainty_kind: gum1, status_tracked, provenance_tracked}` | **Every** EMIR value is a bundle, never a bare scalar |
| Post-E3D non-claims include general N-way SSA, alias, **ABI**, **MachineIR** | Exactly the pieces a naive `mir_to_mli` for R0 would need |

**Named gap:** ladder stages that assumed “MIR has functions, integers, control
flow, and scalar f64” **cannot be fed from Route-B EMIR as shipped**. In
particular the Phase-2 golden `add1(x: f64) -> f64` is **not expressible** in
EMIR (no function ABI, no `ret`).

This is **not** a re-litigation of Route B — EMIR is correctly scoped as an
**epistemic numeric** machine IR. The design error was MLI assuming a
*general* MIR that Route B does not claim to deliver.

#### 3.2.2 Chosen feed for S2/S3 (decision)

**Decision: re-anchor S2/S3 on the IR→MLI side door** (option (b) from
preflight). Do **not** block Phase-2 on a post-E3D EMIR generalisation tranche.

| Feed | What it carries | MLI track |
|---|---|---|
| **`ir_to_mli` (primary for S2–S3)** | `self-hosted/ir/` (and/or HLIR) scalar + full language surface | **R0** integers, f64, call/ret, branches |
| **`emir_to_mli` (primary for R1 epistemic)** | Route-B EMIR epistemic bundles | **R1 `Knowledge`** — maps naturally onto Gpu4-like shape (`val`/`error≈qd`/`uncertainty`/`status`/`prov`) |
| Optional later | Post-E3D EMIR generalisation (WS-C follow-on, **costed separately**) | Could eventually feed R0 if ABI+control land; **not** required for S2 |

**Why this is correct (not a gap only):**

1. Route-B EMIR’s single-type epistemic bundles are an **argument for** MLI’s
   first-class `Knowledge` kind (R1), not a failure of Option C.
2. EMIR `error_kind: qd128` is **double-double family**, not IEEE f128 — see
   §4.1 exclusion (D2).
3. Phase-2 bit-identity needs call/ret/f64 that **already exist** on the IR →
   native path; that is the golden surface to mimic (§6.4, O5).

**Choke points (two, not one):**

```text
  ir_to_mli   → R0 scalar MLI → legalize_x86 → encode   (S2–S3)
  emir_to_mli → R1 Knowledge MLI → expand or native k_*  (S5+, parallel)
```

A single `mir_to_mli` name is **retired** for the scalar path to avoid
implying EMIR can host `add1`.

#### 3.2.3 Optional WS-C follow-on (not on S2 critical path)

If product owners later want EMIR to host general functions:

| Item | Rough cost | Notes |
|---|---|---|
| Integer + compare + branch ops | multi-week | breaks `type_count==1` or adds second type |
| Function ABI + `call`/`ret` | multi-week | currently post-E3D non-claim |
| N-way SSA / MachineIR claims | large | full MIR generalisation |

**Do not** schedule this as a silent dependency of S2. If pursued, open a
**costed WS-C follow-on** with its own gates.

#### 3.2.4 Properties MLI itself still requires (from whatever feed)

| Property | Requirement |
|---|---|
| Explicit control (R0) | Blocks + terminators on **IR-fed** MLI |
| Typed operands | Map into MLI kinds (§4); EMIR→ always `Knowledge` shape |
| Side effects | Loads/stores/calls marked; pure arith free of hidden IO |
| No silent discharge | Epistemic lanes never dropped without `k_discharge` |


### 3.3 Output contract (what `native/` consumes)

MLI after **target legalize** must be consumable by a thin emitter that looks like
today’s `native_v2` machine_ir consumers:

- finite opcode set mappable to `encode.sio` / `codegen.sio` emitters;
- operands resolved to **physical classes**: GPR, XMM/YMM/ZMM, k-mask, stack slot,
  immediate, reloc;
- ABI for calls already applied (arg moves, stack alignment);
- no unresolved virtual regs (post-regalloc) **or** a single documented
  “pre-regalloc MLI” form only used by the interpreter.

**Bit-identity Phase-2 (pinned — O5):** for a golden function,
`encode(mli_legalize(mli))` bytes equal `encode(direct_path)` where
`direct_path` is the **session-built** engine pinned in §10 O5 — not “any
checked-in ELF”, and not an unspecified lean_single vs Madaros mix. See also
§6.4: S3 is **emitter mimicry** for one function.

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
  | Int { bits: 8|16|32|64, signed: bool }   // NOT 128 — see exclusions
  | Ptr { aspace: flat }          // R0: single flat address space
  | Float { fmt: f32|f64|f128|f256 }  // IEEE binary formats only
  | QD128                         // double-double error lane (NOT IEEE f128)
  | Bool                          // i1 logical; R0 branches consume this
  | VecF64 { lanes: 2|4|8 }       // XMM/YMM/ZMM geometric view (optional sugar)
  | Knowledge { base: Float|Int, lanes: KnowledgeLanes, shape: … }
  | CD { dim: 1|2|4|8|16, coeff: Float }   // R,C,H,O,S over coeff format
  | Bundle { fields: [Kind; N] }  // rare; prefer Knowledge/CD tags
```

**Exclusions and non-identities (preflight D2 — freeze in S1):**

| Topic | Rule |
|---|---|
| **qd128 ≠ IEEE f128** | Frontier EMIR `error_kind: qd128` is **double-double family** (`stdlib/math/qd128.sio`). MLI `Float{f128}` is **IEEE binary128** (WS-G). **Forbidden:** silent `f128 := qd128`. Use kind **`QD128`** (or Knowledge error lane typed `QD128`) for EMIR error components. Mapping either way without an explicit conversion op is a **semantic miscompile by construction**. |
| **No `Int{bits:128}` in R0 MLI** | Language `i128`/`u128` already reaches ELF on the **direct** wide-int path (2026-06). MLI R0 kinds intentionally list 8/16/32/64 only. **Until** an S4+ wide-int tranche: `ir_to_mli` must either (a) **reject** i128/u128 functions for the MLI path with a clear diagnostic, or (b) expand to multi-limb `i64` pairs under an explicit `WideInt` expand — never pretend coverage. Document which; default **(a) fail closed** for Phase-2 goldens (scalar i64/f64 only). |
| **Flags** | **Retired as a first-class long-lived kind.** See §4.3 R0 control: branches consume **`Bool` vregs**; legalize may fuse `fcmp+br` into x86 flag use. Transient eflags are an **x86 legalize detail**, not an MLI value that can be stored across arbitrary ops. |

**KnowledgeLanes (R1, aligned with GPU + EMIR):**

| Lane | Role | GPU analogue | EMIR Route-B analogue |
|---|---|---|---|
| `val` | point estimate | `val_id` | `value_kind: f64` |
| `var` / `eps` | GUM variance **or** epsilon | `eps_id` | uncertainty / GUM lane |
| `err` | **qd128** arithmetic error lane when present | — | `error_kind: qd128` |
| `conf` or `valid` | confidence 0–1000 **or** boolean validity | `valid_id` | `status_tracked` |
| `prov` | bit-packed provenance | `prov_id` | `provenance_tracked` |

Stdlib `Epistemic { val, variance, confidence }` is the **minimal CPU** shape
(`KnowledgeShape::Minimal3`). Full GPU four-lane is `Gpu4`. EMIR bundles map
to **`KnowledgeShape::EmirBundle`** (val + **QD128** err + gum uncertainty +
status + provenance) — this is a **positive** Route-B → R1 argument, not only
a feed gap.

MLI must tag which shape is active so expand rules do not invent or drop lanes.

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
| Integer arith | `add/sub/mul/div/rem`, shifts, logic | **i8–i64 only** (no i128) |
| Float arith | `fadd/fsub/fmul/fdiv` | IEEE fmt on kind; **not** qd128 |
| Convert | `i2f`, `f2i`, `fcast` (f32↔f64; IEEE f128/f256 when WS-G ready) | explicit `qd_from_*` if ever needed |
| Compare | `icmp`, `fcmp` → **`Bool` vreg** (not Flags) | |
| Control | `jmp`, `br` (cond: **Bool**), `ret`, `call` | call ABI attrs |
| Stack | `alloca` / frame adjust (or precomputed frame) | |
| Pseudo | `copy`, `keep_alive` | **no φ in MLI** — see §4.4 |

**Flags / eflags (D2):** R0 MLI **does not** model live `Flags` values. A
compare produces a `Bool`; `br` consumes that `Bool`. On x86, `legalize_x86`
may fuse `fcmp`+`br` into `ucomisd`+`jcc` and must treat eflags as
**clobbered by nearly every ALU op** — V-struct cannot catch a separated
cmp/br pair if Flags were first-class. Optional verify rule if a future
lowering ever reintroduces flag temps: **flags def must be adjacent to its
sole consumer** (no intervening ops). Default for S1–S3: **Bool only**.

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
MliBlock    { id, instrs, term }   // NO block params
```

**φ / block-params decision (D2 — pick one for S1):**

| Choice | Decision |
|---|---|
| Representation | **No block parameters and no φ nodes in MLI.** |
| Where SSA join is resolved | **Before** MLI: in IR / EMIR / `ir_to_mli` (copy insertion or already-straight-line). |
| Rationale | Frontier EMIR has no general N-way SSA; Phase-2 goldens are straight-line or simple CFG with explicit moves. Avoids builder churn at S5 if both forms were half-specified. |

If a future feed needs SSA joins inside MLI, that is a **versioned extension**
(`MliBlock.params`), not the S1 default.

**S1 storage layout (measured constraint, 2026-08-16 — binding for S2+):**
the shapes above are the *logical* model; the S1 implementation
(`self-hosted/mli/ir.sio`) stores instructions as a **struct-of-arrays pool**
on `MliFunction` (scalar columns + `[MliOperand; N]` columns; `MliInst` and
`MliBlock` are assembled views via `mli_inst_get`/`mli_inst_put` and
`mli_block_get`), and block metadata as scalar SoA columns. This is **not** an
arbitrary choice: AoS layout (`blk.instrs[i] = inst`) miscompiles under
current Madaros native-v2 — depth-3 aggregate element stores SIGSEGV and
scalar field writes into array-of-struct elements through `&!` misaddress
silently (#1678/#1749 residual family; same remedy the IR arena took in
#1717). Evidence and acceptance gate:
[`../audit/MADAROS_NESTED_AGGREGATE_ELEMENT_STORE_DISPATCH_2026-08-16.md`](../audit/MADAROS_NESTED_AGGREGATE_ELEMENT_STORE_DISPATCH_2026-08-16.md);
live witness: `self-hosted/mli/aggregate_store_diag.sio`. S2's `ir_to_mli`
must build through the pool accessors and must not "simplify" back to AoS
while that witness still prints `OBSERVED`.

**R0 constraint (Phase-2):** single return, no exception edges, SysV x86-64 only;
function must come from **`ir_to_mli`** (or hand-built MLI fixture), **not** raw EMIR.

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
| `FLAG` | **not a long-lived MLI class** | eflags only inside `legalize_x86` fusion |
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

### 5.4 Interaction with existing `machine_ir.sio` (preflight D4)

**Name collision is imminent:** on main, `native/machine_ir.sio` already owns
the `MIR_*` constant prefix (`MIR_OPERAND_GPR64`, `MIR_MAX_INSTRS`, …) while
WS-C Route B lands a different “MIR” (EMIR) under `enir/`. This plan cycle
already produced a swapped-wording incident. Leaving the collision optional
is insufficient.

**Precondition of WS-C PR1 (required, not optional-later):**

1. **Either** rename `MIR_*` symbols in `native/machine_ir.sio` (and call sites
   in `codegen.sio` / regalloc) to **`X86_*`** / `NATIVE_MIR_*`, **or**
2. At **minimum**, add a **naming-disambiguation note in the file header** of
   `native/machine_ir.sio` stating: “These `MIR_*` constants are the **x86
   native-v2 legalize substrate**, not Route-B EMIR / `enir/mir.sio`.”

Prefer (1) when a native-touching PR is open; (2) is the floor for PR1 merge.

**Migration (post-MLI approval, implementation phases):**

1. Treat current `MachineInstr` as **post-legalize x86 pseudo** only.
2. New modules: `self-hosted/mli/{ir,builder,verify,interp,ir_to_mli,emir_to_mli,legalize_x86}.sio`.
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
| **V-struct** | Kind well-formedness, opcode arity, dominance, no raw cross-kind moves; **Bool** (not live Flags) for branches | pure static |
| **V-interp** | MLI interpreter executes a function on concrete inputs | matches reference oracle |
| **V-parity** | For R0 functions: shadow-exec MLI vs current native binary / direct path | bit-identical or value-identical per policy |

### 6.2 Interpreter (`mli_interp`)

- State: vreg file + stack memory + **Bool** temps (no live Flags).
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

### 6.4 Phase-2 golden (preflight D3 — pin before S3)

**S3 is not a generic legaliser.** It is: *mimic the existing emitter for one
function* — register choices, constant materialisation, and scheduling included.

| Pin | Decision (O5 resolved) |
|---|---|
| **Golden engine** | **Madaros native-v2** via default `bin/souc` / `bin/madaros` after a **session-local rebuild** (`make build-madaros` or project equivalent under `souc-build-lock`). **Not** lean_single (different bytes). **Not** a checked-in ELF from `bin/` alone without rebuild receipt. |
| **Provenance receipt** | Record: git SHA, engine path, `souc --version`, build command, UTC timestamp of the binary used as golden. |
| **Fixture** | Pure scalar function expressible on IR→native today, e.g. `fn add1(x: f64) -> f64 { x + 1.0 }`. **Not** an EMIR program. |
| **Pipeline A** | Direct path: IR → current native legalize/encode (pinned engine). |
| **Pipeline B** | `ir_to_mli` → MLI V-struct → `legalize_x86` → same encode surface. |
| **Pass** | **Bit-identical** `.text` (or whole function body) for that fixture. |
| **`imm f64`** | No native x86 encoding for immediate f64 in arithmetic. Legalize **must** reproduce the **existing** emitter’s constant strategy (constant pool / `movsd` from rip-relative / `movabs`+`movq` — whichever the pinned path uses). Do not invent a cleaner constant path and call it “equivalent.” |

**Estimate S3 as emitter-mimic work**, not as proof that general legalisation is
cheap. Value-identity is a **fallback diagnostic** only if bit-identity fails;
it is not the pass criterion once O5 is pinned to bytes.

### 6.5 R1 goldens (later)

- `k_add` matches stdlib `ep_add` on random inputs (value + variance + conf).
- `cd_mul` dim=8 matches `oct_mul` bit-for-bit on f64 components for a fixed set.
- Door-β Fano short-circuit: Fano triples emit cheap path (count ops or match
  mask).

---

## 7. Staged implementation ladder

| Stage | Deliverable | Gate |
|---|---|---|
| **S0** | This design (as amended) accepted | human / orchestrator |
| **S1** | `mli` module: kinds (**incl. QD128, no i128, Bool not Flags**), builder (**no φ**), text dump, V-struct verify | unit tests on fixtures |
| **S2** | **`ir_to_mli`** for **scalar R0** (i64 + f64 + call/ret); **not** `emir_to_mli` | interpreter tests on IR-derived MLI |
| **S3** | `legalize_x86` **mimicking** pinned Madaros native-v2 emitter for **one** golden | Phase-2 **bit-identical** vs session-built golden (O5) |
| **S4** | f32 + conversions; IEEE f128/f256 **slots** (WS-G arith may libcall); still **no** qd≡f128 | WS-G coordination |
| **S5** | Knowledge kinds + `k_*` + expand-to-R0; start **`emir_to_mli`** for Route-B bundles → `KnowledgeShape::EmirBundle` | parity vs `ep_*` and EMIR oracle lanes |
| **S6** | CD kinds dim≤8 + `cd_mul` select (EVEX or stack loop) | parity vs stdlib / IR hyper tests |
| **S7** | dim=16, associator, Door-β, zd_probe policy | GPU formula alignment checklist |
| **S8** | Optional: MLI→GPU kind-preserving bridge | reuses KAXI epistemic markers |

**Hard rules:**

- S3 does **not** require S5–S7, but S1 **must** define Knowledge/CD/**QD128**
  kinds so S5 is not a breaking rewrite.
- **S2/S3 are not fed by Route-B EMIR.** EMIR feeds S5+ via `emir_to_mli`.
- Optional post-E3D EMIR generalisation (WS-C follow-on) is **out of band** for
  S2; see §3.2.3.

**Parallelism:**

- WS-C PR1: EMIR land + **D4 naming precondition** (§5.4).
- WS-G owns IEEE f128/f256 **arithmetic**; MLI owns kind + expand hooks + **QD128**
  as a distinct kind for EMIR error lanes.
- WS-F EISA remains a separate conformance consumer; may later golden MLI dumps.

---

## 8. Worked examples

### 8.1 R0 scalar (Phase-2 target — IR feed only)

Source (Sounio / IR — **not** EMIR):

```sounio
fn add1(x: f64) -> f64 { x + 1.0 }
```

MLI sketch:

```text
func @add1(v0: f64) -> f64 {
  b0:
    v1: f64 = fadd v0, imm f64 1.0   // imm is MLI-level; x86 has no f64 imm
    ret v1
}
```

Legalize must materialise `1.0` **exactly as the pinned Madaros native-v2
emitter does** (pool / rip-relative / etc.), then emit the same float binop
path. This fixture is **out of scope for `emir_to_mli`**.

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
| Route-B EMIR ≠ general MIR | S2/S3 use `ir_to_mli`; EMIR → R1 only (§3.2) |
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

| ID | Decision | Status / resolution |
|---|---|---|
| O1 | EMIR bundle → `KnowledgeShape::EmirBundle` (+ QD128 err lane); IR scalars → R0 kinds | **Draft mapping in §3.2 / §4.1**; freeze at S1 |
| O2 | Knowledge shape default on CPU: Min3 vs Gpu4 vs EmirBundle | **Open** — default Min3 for stdlib path; EmirBundle for `emir_to_mli` |
| O3 | `native::machine_ir` `MIR_*` → `X86_*` (or header note) | **Resolved as WS-C PR1 precondition** (§5.4 / D4) — not optional |
| O4 | IEEE f128/f256: softfloat libcall vs hardware | **Open** — WS-G; independent of QD128 |
| O5 | Phase-2 identity criterion + golden engine | **Resolved (D3):** **bit-identical** vs **session-built Madaros native-v2**; not lean_single; not unchecked `bin/` ELF. Value-identity is diagnostic only. |

---

## 11. Approval checklist (Phase-1 done)

- [x] Layer contract (MIR in → native out) stated
- [x] Instruction/operand model including **first-class** Knowledge + CD kinds
- [x] Binding principle **evaluated** with explicit R0/R1 tradeoff (Option C)
- [x] Register/stack discipline for multi-lane kinds
- [x] Verification story (struct + interp + parity), ENIR-pattern mirror
- [x] Staged ladder with Phase-2 bit-identity isolated from R1
- [x] Naming disambiguation vs existing `machine_ir.sio`
- [x] Option C founder-approved; preflight D1–D4 folded in
- [x] Route-B EMIR gap named; S2/S3 re-anchored on `ir_to_mli`
- [x] QD128 ≠ f128; i128 excluded; Bool not Flags; no φ in MLI
- [x] O5 pinned before S3; S3 scoped as emitter mimicry
- [x] D4 rename/note as WS-C PR1 precondition
- [ ] Implementation dispatch (S1) after amended design accepted
- [ ] No `self-hosted/` MLI code until S1 dispatch

---

## 12. References (repo-local)

- Plan: `docs/internal/coordination/MADAROS_FOCUS_PLAN_2026-08-16.md` (WS-D)
- Preflight: `docs/architecture/WS_C_D_PREFLIGHT_REVIEW_2026-08-16.md`
- WS-C route: `docs/architecture/MIR_PORT_PLAN.md` (Route B)
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
| Option C | Founder-approved 2026-08-16 (not re-litigated) |
| Preflight | fable-1 `WS_C_D_PREFLIGHT_REVIEW_2026-08-16.md`; amendments D1–D4 |
| Amend author | grok-cli2 lane `amend-mli-design` |
| Worktree | `/workspace/.wt/amend-mli` branch `amend/mli-design-20260816` |
| Next action | Land this amended doc; S1 may open; S2 = `ir_to_mli` not EMIR |

### 13.1 Amendment log

| Date | Change |
|---|---|
| 2026-08-16 | Initial WS-D draft (Option C recommended) |
| 2026-08-16 | D1: name Route-B EMIR gap; S2/S3 feed = `ir_to_mli`; EMIR → R1 Knowledge |
| 2026-08-16 | D2: QD128 kind; no Int128; Bool not Flags; no block-params/φ in MLI |
| 2026-08-16 | D3: O5 pin Madaros native-v2 session binary; S3 = emitter mimic |
| 2026-08-16 | D4: `X86_*` rename or header note = WS-C PR1 precondition |
| 2026-08-16 | S1 landed (`lane/cursor-2/mli-s1-20260816` @ `ab572a62d9`): §4.4 storage-layout note — SoA instruction pool as measured response to the nested-aggregate-element-store miscompile (cursor-2, lane mli-s1) |
