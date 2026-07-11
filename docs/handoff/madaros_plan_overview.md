<!-- docs:meta
topic_id: repo.docs.handoff.madaros-plan-overview
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.madaros-plan-overview
-->

# Madaros / Sounio — plan overview & navigation map

**Authored by:** Claude (EISA lane, `gpu/epistemic-tensor-core-next` @ `b515d2870`), 2026-07-11
**Type:** orientation map — synthesized from a 4-agent parallel doc sweep, cross-verified against primary sources.
**Purpose:** a single navigable map of the whole Madaros/Sounio plan — the founding thesis, the v2 master architecture, the *real* phase model, the advanced type-system and backend/hardware features (with **realized** status, not just ambition), and where to read the canonical originals.

> **Reading discipline (repo-wide):** Sounio's culture is anti-overclaim. Spec/design consistently runs **ahead** of what the active `use`-graph wires up. Treat spec sections as *design intent* and the checker imports + test fixtures + `*.gate-receipt` as *realized status*. Every claim below carries the repo's own status label where one exists.

## 1. The thesis — epistemic computing

`docs/MANIFESTO.md`: *"the compiler itself should know what is known, how well it is known, where it came from, and whether the next step is epistemically admissible."* A new design center ("Gradual Epistemic Compiler"), **not** feature-parity with Rust/Julia. The machine's semantic atom is **`Knowledge<T>` = value + uncertainty (GUM/JCGM-100) + confidence + provenance**, which cannot silently decay to bare `T`. Principles: all knowledge uncertain; provenance non-negotiable; uncertainty auto-propagates; confidence gates execution; standards-compliant (ISO 17025 / 21 CFR Part 11 / FAIR); **"primacy is earned by proof."**

## 2. The master architecture — Madaros v2 "SOTA+++"

`docs/research/madaros-v2-sota-plus-plus-plan-2026-07-04.md` — **marked `historical`, self-rated L0 (sketch), "not implemented."** It is the fullest ambition statement, **plan not state**:

> a **receipt-carrying, seven-stage (S0→S7) epistemic compiler** where typed scientific uncertainty, equality saturation (persistent e-graphs), and **E-KAN** (Epistemic Kolmogorov–Arnold Network compiler passes) are first-class compiler objects, and every source→binary stage is validated by bit-identical receipts + translation-validation + exact fallback.

| Stage | Job |
|---|---|
| S0 | current compiler as safety **oracle** (not the architecture) |
| S1 | canonical source/AST/module graph + `MadarosV2S1Receipt` |
| S2 | typed HIR/THIR with effects + epistemic declarations |
| S3 | HLIR SSA (ownership/effect normalization) |
| **S4** | **Eq/E-KAN optimization IR** ← novelty center (e-graph + receipted surrogates) |
| S5 | MIR, ABI, numeric tower (f64→f128, i128/i256, SRET, tensor regs) |
| S6 | target IRs + accelerator lowering (CPU, LLVM, WASM, GPU PTX, TensorIR) |
| S7 | self-hosted fixed point (stage N builds stage N+1, bit-identical receipts) |

Validation ladder **L0→L7** (doc is L0). Rule: *"E-KAN suggests. Receipts decide. Exact semantics win."*

## 3. The real phase model (where the project actually is)

Two vocabularies, one trajectory (*monolithic lean_single seed → modular Madaros → promoted default*):

| Horizon (source comments) | Stage contract (`docs/compiler/STAGE0_STAGE1_COMPILER_CONTRACT.md`) | Meaning |
|---|---|---|
| Horizon 1 | **Stage0 = `lean_single.sio`** | frozen bootstrap seed / fixed-point oracle. No new semantics may live only here. |
| Horizon 2 | (transition) | incremental modular wiring of `main.sio`. |
| **Horizon 3 ← current** | **Stage1 = Madaros** → **Stage2 (promoted default)** | the "serious" modular compiler; `bin/souc → Madaros` (receipt-gated). |

**Bootstrap:** C seed `bootstrap/stage0.c` → `boot0..boot4` → `gen1→gen2→gen3`, `md5(gen2)==md5(gen3)` = fixed point. **Proven only over `lean_single.sio`, NOT over Madaros/`main.sio`.** *"Madaros compiles Madaros"* is a separate, larger, not-done milestone. **"Foundry" = the Slurm/HPC validation cluster** (`docs/ops/foundry_slurm_handoff.md`), not a compiler stage. Receipt-gating: `artifacts/self-hosted/madaros.gate-receipt` pins the ELF sha256; else fall back to `bin/madaros-relocgate` → prebuilt → lean_single.

## 4. Advanced type system (7 features) — realized status

| Feature | Wired? | Realized maturity |
|---|---|---|
| **Epistemic (`Knowledge<T>`/GUM)** | yes | ★ flagship; static/typecheck **works** (`check/epistemic.sio` ~2331 ln: ε-quadrature, confidence lattice, provenance, `.value` gate). Gaps: native guard lowering (`backend_guard_count=0`), GUM v1 assumes uncorrelated inputs. |
| **Generics/monomorphization** | yes | **landed for 1–2 params** (turbofish + generic-struct-return; `cd_exact_generic_i64` GREEN 2026-07-07 WP-A5). Gaps: **3+ params**, trait-bound enforcement, trait objects. |
| **Effects** (`with …`) | yes (`effects.sio` only) | annotation checking **stable** (~22 effect IDs incl. exotic ZD/Witness/Temporal/Learn). Dormant: row inference (`effects_row.sio`), CPS handlers (`effects/*`) — not imported. |
| **Refinement** `{x:T\|p}` | yes (static) | static engine **works**; FM/SMT middle tier (`smt_qflia.sio`) unwired; spec's Z3 absent. |
| **Units** (dimensional) | yes | ★ most complete; 7 SI dims + rational scale. Gap: affine units (°C/°F). |
| **Traits** | yes | **works** (resolution, vtable, coherence, auto-derive). Gaps: bounds unenforced at call sites, no trait objects. |
| **Dependent** (Pi/Sigma) | imported, **never invoked** | **dormant/scaffolded** — only `pub struct`, zero callers, no docs/tests. Least mature. |

Theory spine: `docs/research/beta5_unified_type_theory_draft.md` (effects + refinement-by-SMT + GUM + algebra e-graph; rule `[K-Value]` gates `.value` on an `Epistemic` effect + SMT confidence witness).

## 5. Advanced backend / hardware (the frontier), ranked

| # | Feature | What | Status |
|---|---|---|---|
| ① | **EISA / Metron VM** ★ | self-contained epistemic ISA (`.eisax`), 3 unbypassable lanes/reg (value + EFT roundoff + GUM σ); surface lang **Metron** | **works, gated** through qd128; 3 bit-identical executors; flagship Rump-1988 cancellation receipt; falsifiable C1∧C2∧C3 "first". `lean_single`-only. |
| ② | **GPU epistemic tensor cores** | GUM through `mma.sync`/WMMA (`U²=VA·B²+A²·VB+VC`) | **proven on real Blackwell GB10** (reference kernel, ε_C=√7 exact, native SASS). Open: **compiler-generated** WMMA matmul **fails** to build. |
| ③ | **od256** (~424-bit oct-double) | 8-limb EFT "binary256 in software" for GPU | CPU + mpmath ref PASS; GPU kernels emitted **bit-exact ~430 bits** (branchless renorm). Not on real silicon yet. |
| ④ | **Exact algebraic core `<F>`** | sedenion zero-divisor census by exact ℤ equality | **168-census DELIVERED for F=i64** (cross-checked vs Python). Walled: **F=Rational/BigInt** (struct coeffs) miscompiles — codegen **#651** (`[struct;N]` aggregate-loop). |
| ⑤ | **K-AXI / Kretikos** | own GPU IR (29 opcodes) → PTX → CUBIN (no nvcc); mirrors a Verilog fabric | **works, audited** (0/318 violations; receipt-CUBINs on L4/GB10). Open: `Knowledge<Vec/Mat>` lane (~0%, by-value segfault). |
| ⑥ | **K-AXI FPGA fabric** | AXI4-Stream carrying `Knowledge<f64>` with GUM+provenance **in hardware** | **real synthesizable Verilog** (Yosys/Icarus in `artifacts/fpga/`), bound **proven in Lean**. Sim/synth only. |
| ⑦ | **native-v2** (LLVM-free CPU codegen) | 7 arch×OS matrix; native epistemic/hypercomplex lowering | **partial, x86_64-linux only** — the subject of the reframe (issue #789). |

Secondary targets: SPIR-V/Vulkan (mature; 4 IDs per `Knowledge<T>`) ≈ Metal/MSL (works) > Verilog (real) > **wasm (real code, stalled)**.

## 6. Formal verification + the foundational application

- **Lean 4 corpus** (~180 files, ~1807 theorems): the **168 = |PSL(2,7)| theorem is zero-sorry/zero-axiom** (`native_decide`), as are the type-checker & ELF-linker invariants. GUM/confidence **rest on ~10 axiomatized IEEE-754 primitives** (removal roadmap: `docs/research/lean_float_real_roadmap.md`). Not globally axiom-free (8 sorry, 12 axiom) — documented honestly by the repo's own auditor.
- **Dissertation** (`stdlib/darwin_pbpk/`): *"GUM-Native PBPK via Epistemic Gradual Compilation"*, PUC-SP, defense **2026-09-22**. **6/6 gates green**; Sounio↔Node parity <1% RMSE; headline **ω²(IIV) ⊥ u_c(GUM)**. The most mature deliverable.

## 7. Planned vs delivered — the honest ledger (incl. nulls)

- Compiler/algebra/formal core **over-delivered**; the empirical **neuroscience program was falsified**: ABIDE/O-SSM octonionic associator = **NULL/invalidated** (≈chance accuracy; 0 Holm-significant ROIs; an `oct_mul` sign bug flipped prior A/B/C "octonionic" results negative). Honestly down-scoped to "controlled instrumentation framework."
- **Engine split hides science behind a compiler artifact:** dissertation + EISA + exact-core are green **only under `lean_single`**; red under default Madaros is a **compiler defect, not a science failure** → this is exactly the native-v2 reframe track ([[native_v2_maturity_reframe]], #789).
- Live codegen defects walling the frontier (`docs/handoff/souc_v0800_defects.md`): **#637** (cross-module aggregate SIGSEGV), **#639** (data-enum `match` wrong arm), **#651** (`[struct;N]` aggregate-loop) — #651 walls exact-algebra with `F=Rational`.
- Tracking is layered (no single dashboard): claim-readiness (`docs/serious-language/*.tsv` + CI gate) · work-progress (`docs/architecture/compiler-maturity-blueprint.md` M4→M9) · machine receipts (`artifacts/omega/*.json`). Governance stamps say `2026-03-07` but the live frontier is July 2026 — "not in the ledger" ≠ "not real."

## 8. Canonical paths — the map (read the originals)

**Vision / master plan:** `docs/MANIFESTO.md` · `docs/research/madaros-v2-sota-plus-plus-plan-2026-07-04.md` · `docs/research/beta5_unified_type_theory_draft.md` · `README.md`
**Phase model / bootstrap:** `docs/compiler/STAGE0_STAGE1_COMPILER_CONTRACT.md` · `bootstrap/README.md` · `CLAUDE.md` (§Bootstrap chain) · `docs/MADAROS_STATUS.md` · `self-hosted/compiler/main.sio` (Horizon banner) · `docs/architecture/compiler-maturity-blueprint.md`
**Type system:** `docs/spec/LANGUAGE_SPECIFICATION.md` (§3.9 epistemic, §7 effects, §9 units, §10 refinement) · `docs/compiler/KNOWN_LIMITATIONS.md` · checker `self-hosted/check/{epistemic,effects,refinement,units,traits,specializer,check}.sio`
**Backend / hardware:** EISA `docs/research/eisa-stack-architecture-2026-07-05.md` + `stdlib/eisa/` · tensor cores `docs/design/EPISTEMIC_TENSOR_CORE_GUM_TURING.md` · od256 `docs/research/od256-oct-double-spec-2026-07-08.md` + `stdlib/math/od256.sio` · exact core `docs/EXACT_CORE.md` + `stdlib/algebra/cayley_dickson_exact*.sio` · K-AXI/Kretikos `docs/kretikos/UNIQUE_FEATURES.md` + `self-hosted/gpu/` · FPGA `stdlib/hardware/kaxi.sio` + `artifacts/fpga/` · native-v2 `docs/compiler/NATIVE_V2_SERIOUS_TRACK.md` + `self-hosted/native/`
**Formal / dissertation:** `formal/lean4/` + `scripts/ci/lean_proof_status_audit.py` · `docs/dissertation/QUALIFICATION_STATUS_2026-06-23.md`
**Status ledgers:** `docs/serious-language/readiness-ledger.md` + `public-claim-registry.v1.tsv` · `docs/architecture/truth-frontier.md` · `docs/handoff/souc_v0800_defects.md`

## 9. Relation to the current lane (2026-07-11)

- The **modular greenup (W1–W5) + main-merge** drove the Madaros modular typecheck to 0 — a step toward Stage2 / "Madaros compiles Madaros." See [[eisa-main-merge-landed]] / #767.
- The **native-v2 reframe** (#789, [[native_v2_maturity_reframe]]) is exactly the "engine split hides science" problem the plan names. **Resolved (2026-07-11):** the **generic-`<F>` `[F;N]` residual (Lane 3)** is a **front-end typecheck** monomorphizer-inference gap — `infer_type_params_from_fields` (`check.sio:22300`) binds a struct type param only from a *directly-`TyNamed`* field, not from a param inside an `[F;N]` array field → `__T` leak → E009, never reaching codegen. It is **distinct** from **#651** (a *codegen/runtime* `[struct;N]` aggregate-loop miscompile that its own filing declares "NOT generics"). Feature ④ (exact-algebra `<F>`) is walled by **both**: Lane 3 at typecheck, #651 at codegen for `F=Rational`.
