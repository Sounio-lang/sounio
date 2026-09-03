<!-- docs:meta
topic_id: repo.docs.compiler.compiler-plan-consolidated
authority: repo_only
audience: contributors
last_validated: 2026-03-07
validated_by: A4
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.compiler.compiler-plan-consolidated
-->

# The Sounio compiler plan, consolidated

**Assembled:** 2026-07-26, against `main@8d6eebc0d`.
**Why this document exists:** the compiler plan is real and coherent, but it is
scattered across five lanes, two long-lived branches, ~19 stacked draft PRs and a
plan-overview map that never reached `main`. Nobody could read it in one place.
This is that one place.

## What changed on 2026-07-26

This document was assembled the same day a batch of compiler fixes landed, so the
state below already reflects them. Recording them here because several are
load-bearing for the rest of the document.

**Three silent miscompiles were root-caused and fixed on `main`:**

| | defect | fix |
|---|---|---|
| #1454 / #1194 | `policy` and its family were reserved words the expression and pattern parsers rejected as identifiers | #1470 |
| #1474 | `[Aggregate; N]` repeat literals made all N elements alias one object | #1476 |
| #1475 | binding an aggregate local (`var b = a`) copied the *handle*, not the value — including plain structs with no array anywhere | #1480 |

**#1474 was the IR programme's real blocker.** `ir_empty_module()` is
`[ir_empty_function(); 2048]` and `ir_empty_function()` is `[ir_nop(); 4096]`,
so before the fix every `IrModule` was 2048 aliases of one function, each
holding 4096 aliases of one instruction. #885's *"the raw zeroed IrModule
reservation does not contain valid aggregate objects"* and #910's *"fixed-array
ident-copy alias blocker"* were descriptions of this.

**The size threshold that Lane A routes around does not exist.** Measured on
both engines with loop-filled, fully-checksummed programs: return by value,
parameter by value, tuple return and bare-vs-wrapped arrays all pass at 24 B,
1 KiB, 8 KiB, 64 KiB, **128 KiB**, 256 KiB, 1 MiB and **8 MiB**. The specific
artefact #887 calls impossible — a 128 KiB `[i8; 131072]` passed *and* returned
by value with a full checksum — passes on both. Several Lane A workarounds
(scalar-column storage, handle bridges, avoiding by-value returns) were built to
route around a wall that measurement does not find, while the actual defect
(aliasing) went undiagnosed. The whole stack is worth re-testing against a fixed
compiler before more is layered on.

**#1480 was corrupting dissertation-path science, not just compiler internals.**
`tests/run-pass/rapamycin_epistemic_pbpk.sio` reported `s_cl=0.2486 /
s_kp=0.2493 / s_fu=0.5021`, concluding fu_plasma dominates AUC uncertainty. The
alias was contaminating the per-parameter perturbation copies. Corrected:
`0.8503 / 0.000001 / 0.1497` — **CL_hepatic dominant**, independently confirmed
by lean_single at `0.850705 / 0.000002 / 0.149293`. Any variance decomposition
derived by perturbing copies of a struct before this date should be re-derived.

**Still open from the same batch:** silent truncation of the relocation table
above 65,536 entries (an ELF is written with unpatched relocation sites, rc=0);
nested aggregate *fields* still share storage after #1480, because Madaros lays
an array-typed field out as a handle where lean_single inlines it; and the
branch-aware linear merge (#1471) is halted with eight eliminated hypotheses.

## How to read this

Two rules, both learned the hard way in this repository.

1. **Spec runs ahead of wiring.** Treat design documents as *intent* and the
   checker imports, test fixtures and `*.gate-receipt` files as *realized status*.
   Every claim below carries a status label.
2. **Green in CI is not evidence for compiler guarantees.** `full-test-suite`
   runs **souc-stage2 (lean_single)**, the frozen bootstrap seed. Most guarantees
   in this document live in the **modular Madaros** compiler, which lean_single
   does not implement. A change to the Madaros checker or parser can be fully
   green in CI and still be broken. Verify on a Madaros **built from the source
   under test** — the prebuilt `bin/souc` / `bin/madaros` lag source and diverge.

Status labels used here: **SHIPPED** (on `main`, gated), **BRANCH** (exists,
verified, not on `main`), **BLOCKED** (implemented, cannot land), **PLANNED**
(designed, not implemented), **ASPIRATION** (goal, no design yet).

---

## 1. The thesis

From `docs/MANIFESTO.md`: *"the compiler itself should know what is known, how
well it is known, where it came from, and whether the next step is epistemically
admissible."*

The semantic atom is **`Knowledge<T>` = value + uncertainty (GUM / JCGM-100) +
confidence + provenance**, which cannot silently decay to a bare `T`. The design
centre is a *gradual epistemic compiler*, explicitly **not** feature parity with
Rust or Julia. Principles: all knowledge is uncertain; provenance is
non-negotiable; uncertainty propagates automatically; confidence gates execution;
*"primacy is earned by proof."*

## 2. Where the project actually is

| Horizon (source comments) | Stage contract | Meaning |
|---|---|---|
| Horizon 1 | **Stage0 = `lean_single.sio`** | frozen bootstrap seed / fixed-point oracle. No new semantics may live only here. |
| Horizon 2 | (transition) | incremental modular wiring of `main.sio`. |
| **Horizon 3 ← current** | **Stage1 = Madaros** → Stage2 (promoted default) | the modular compiler; `bin/souc` routes to Madaros, receipt-gated. |

`Madaros v0.80.0`, self-hosted, self-compiling. The full gate
(`scripts/ci/madaros_full_gate.sh`) covers CLI, multimodule visibility
diagnostics, source→native ELF, native-v2 ABI witnesses, the package manager and
imported dereferenced-f64-array lowering: **11 checks**.

## 3. The v2 master architecture — S0→S7

`docs/research/madaros-v2-sota-plus-plus-plan-2026-07-04.md`, marked `historical`,
self-rated **L0 (sketch), "not implemented."** It is the fullest ambition
statement and must be read as **PLANNED**, not as state:

> a receipt-carrying, seven-stage epistemic compiler where typed scientific
> uncertainty, equality saturation (persistent e-graphs) and E-KAN (Epistemic
> Kolmogorov–Arnold Network compiler passes) are first-class compiler objects,
> and every source→binary stage is validated by bit-identical receipts.

| Stage | Job |
|---|---|
| S0 | current compiler as safety **oracle** (not the architecture) |
| S1 | canonical source / AST / module graph + `MadarosV2S1Receipt` |
| S2 | typed HIR/THIR with effects + epistemic declarations |
| S3 | HLIR SSA (ownership / effect normalization) |
| **S4** | **Eq / E-KAN optimization IR** ← the novelty centre |
| S5 | MIR, ABI, numeric tower (f64→f128, i128/i256, SRET, tensor registers) |
| S6 | target IRs + accelerator lowering (CPU, LLVM, WASM, GPU PTX, TensorIR) |
| S7 | self-hosted fixed point (stage N builds N+1, bit-identical receipts) |

Validation ladder L0→L7. Governing rule: *"E-KAN suggests. Receipts decide.
Exact semantics win."*

Lanes 4–8 below are the concrete work that feeds these stages.

---

## 4. Lane A — the IR architecture (S1/S3/S5)

The largest body of unlanded compiler work in the repository. Five layers, all
built **shadow / off-default**, with the legacy pipeline retained as the
**differential oracle**:

| Layer | What it is | PRs | Status |
|---|---|---|---|
| **SOIR v1–v5** | the IR serialization wire format; fail-closed capacity handling; a bounded codec core extracted from a 4,852-line serializer | #870, #883, #893 | BLOCKED |
| **IrModuleArena v2** | one arena owns all function/instruction slots; generational handles `(slot, generation)` reject ABA, stale, cross-arena and cross-module use; scalar-column setters avoid whole-`IrFunction` copy-and-republish | #895, #962 | BLOCKED |
| **Place IR v0** | explicit places: root identity + address space, ordered `Field`/`Index`/`Deref` projections, mutability, value category, type/layout provenance — replacing ad-hoc raw-field identity | #894, #973, #979 | BLOCKED |
| **DefinitionRegistry** | generational IDs for modules, nominal types, fields and TypeExpr bindings. Paths *select* declarations; paths, hashes, declaration order, field ordinals and storage slots are **never** semantic identity | #998 | BLOCKED |
| **TargetDataLayout registry** | compiler-minted `TargetDataLayoutId`; closed profiles for x86_64-linux, x86_64-darwin, aarch64-linux | #992 | BLOCKED |

Supporting closures: heap module bridge (#881), bounded heap-graph
materialization (#885), scalar-result writer receipts (#887), legacy/Arena
identity differential (#899), Place-Arena-SOIR D1 characterization (#910),
`pub(crate)`/`pub(super)` recovery in boot4 tokens (#946), f64-bitcast DefId
provenance through SOIR merge (#947), raw-field identity closure (#956), native
codegen capacity (#960).

### Why none of it has landed

The engineering discipline here is the best in the repository — exact bases
pinned by SHA, source-fresh build receipts, fail-closed at every boundary,
explicit "NOT DONE" boundary lists. The blockage is structural, not sloppiness.

**(a) Ten-deep stacks.** Two chains, each PR based on the previous one:

```
#870 → #881 → #883 → #885 → #887 → #893 → #894/#895 → #899 → #910
#946 → #947 → #956 → #960 → #961 → #962 → #965 → #973 → #979 → #992
```

Nothing merges while the bottom is blocked, and every rebase of the bottom
invalidates every pinned base above it.

**(b) A bootstrapping deadlock — but not the one the drafts describe.** The
blockers were stated as codegen limits of the current compiler: native-v2 cannot
execute the wide-`IrModule` roundtrip witness; 128 KiB arrays cannot be returned
or passed by value; fixed-array ident-copy aliasing; code-capacity overflow.

Measured on 2026-07-26, **the size limits are not real** (see "What changed"
above — by-value aggregates pass at every size to 8 MiB on both engines). The
`IrModule` roundtrip failure and the "ident-copy alias" were the same single
defect, #1474, now fixed. What remains genuinely true is the *shape* of the
deadlock: architecture work was gated on compiler defects, and those defects
were mis-described because nobody reduced them to a minimal repro. Three agents
dispatched at three separately-catalogued blockers converged on the same two
lines of `ir/lower.sio`.

**(c) The oracle costs double.** Keeping the legacy pipeline as the differential
oracle is correct, but it means every layer must be built twice and proven
identical before it can be selected — and the differential runner itself
(#899, #910) reports `promotion ready: false`.

### What to do about it

- Land the **leaves that stand alone**, not the stack. #946 (visibility) and
  #947 (DefId provenance) are compiler fixes with their own witnesses; they do
  not depend on Place IR or the Arena.
- Fix the **codegen blockers first**, as ordinary Madaros bugs on `main`, with
  minimal repros. They are the real critical path; the IR work is downstream of
  them.
- Re-cut the rest as **shallow, independently verifiable PRs** against a settled
  `main`. A 10-deep stack of drafts cannot converge.

## 5. Lane B — EISA, the epistemic ISA (S2/S5)

> **Location:** source and specs are **not on `main`**. They live on branch
> `gpu/epistemic-tensor-core-next` (`stdlib/eisa/*.sio`, specs
> `docs/research/eisa-*-2026-07-0{5,6}.md`). `main` carries only the compiled
> golden corpus `artifacts/eisa/*.eisax.elf` and its receipts. **Status: BRANCH.**

A CPU-side instruction set whose registers are **triples, not scalars**
(`stdlib/eisa/core.sio`, type `EReg`):

| lane | meaning |
|---|---|
| `val` | IEEE-754 binary64 — the stored value |
| `err` | signed roundoff correction, `true ≈ val + err`, carried as a **`Dd64` double-double** (~106-bit, exact via error-free transforms) |
| `u` | GUM / JCGM-100:2008 **standard uncertainty** σ of the true value |

The distinction is the whole point: `err` is how far the stored value is from
true (knowable, correctable); `u` is how far true is from the physical quantity
(irreducible).

- **v0 core opcodes:** `econst`, `eload`, `eadd`, `esub`, `emul`, `ediv`,
  `esqrt`, `egate`, `estore` (0–8). Opcodes 9–15 are rejected. `egate` always
  gates a register against **its own `u` lane** → `ok | marginal | reject`.
- **v1a** adds fuel / branch / control-flow (opcodes 9–13).
- **Pipeline:** textual `.eisa` → assembler (`stdlib/eisa/asm.sio`) → dual-plane
  `.eisax` container → **Metron VM** (`stdlib/eisa/evm.sio`). A Sounio→`.eisax`
  surface compiler exists (`stdlib/eisa/backend.sio`).
- **x86-64 AOT conformance bridge** (`stdlib/eisa/bridge_x86.sio`) is checked
  byte-for-byte against the VM by `scripts/ci/eisa_bridge_conformance_gate.sh`.
- **Receipts** are the evidence surface:
  `eisa-receipt: v=1 prog=<hash> gate=<k> reg=<eN> val=… roundoff=… u=… poisoned=0 frail=0 stop=fuel`.
  `poisoned` = NaN/Inf contamination, `frail` = catastrophic-cancellation flag —
  these are the knowledge-carrying status bits.
- **Golden corpus with negative controls** (on `main`, `artifacts/eisa/`):
  `golden-{add,mul,sqrt,poison}`, `e5-cancellation`, `v1-{loop,fuel,highreg,…}`.
  The gate **must reject** `golden-mul-tampered` and `golden-poison`.

## 6. Lane C — the precision ladder (S5)

Project ethos: **extended precision is a software construct.** `dd64` and `qd128`
already are exactly that. "No hardware exists" is therefore not a blocker — the
same argument as building 32-bit arithmetic on an 8-bit CPU.

```
f64 (hardware)  →  dd64 ~106-bit  →  qd128 ~212-bit  →  od256 ~424-bit
                   EISA v0/v1 err     EISA v2 err       oct-double, 8 limbs
```

| type | status | notes |
|---|---|---|
| `dd64` double-double | **BRANCH** | 16-byte hi+lo pair, ABI-free under SysV / AAPCS64 / Darwin |
| `qd128` quad-double | **BRANCH** | Priest renormalization, ~212-bit |
| `od256` oct-double | **verified** | 8 limbs, ~424-bit. `two_sum`/`two_prod`/`add`/`mul` confirmed **bit-exact vs CPU reference on real NVIDIA L4 and DGX Spark GB10**, up to 4096 cases; 432/428 bits vs mpmath. Known defect: the emitted GPU `add` kernel uses 2 VecSum passes and drops to ~215 bits for partial-overlap magnitude gaps — 5 passes required; this blocks GPU div/sqrt |
| `f128` (IEEE binary128) | **PLANNED** | a real **S5 milestone**, hard-gated on f64 print/return/call witnesses (`docs/research/eisa-precision-track-2026-07-05.md`). The literal `f128` tokens on `main` today are 128-byte buffers / 128-bit SIMD, not the type |
| `f256` (IEEE binary256) | **out of scope** | no hardware, no ecosystem — correctly rejected *as an IEEE hardware type* |
| octuple via EFT | **ASPIRATION → partly real** | this is the live goal, and `od256` above is its first working rung. Do not read "f256 rejected" as "octuple rejected"; the software route is alive and measured |

Adjacent, on `main`: the Cayley–Dickson tower — `octonion` (R⁸, normed and
alternative), `sedenion` (R¹⁶, power-associative with zero divisors, **not**
alternative). A single sign error in `e2*e5` once silently broke alternativity,
so always re-verify `‖a·b‖ = ‖a‖‖b‖` and `(a·a)·b = a·(a·b)` to ~1e-15 after
touching a multiplication table.

## 7. Lane D — GPU (S6)

K-AXI IR / mini-ISA → NVIDIA PTX → CUBIN (`ptxas`), with the Kretikos wrapper, a
ptxas-acceptance gate and a golden PTX corpus. K-AXI — not the frozen
`souc --backend gpu` binary — is the real GPU engine. Epistemic MMA and the
scorer are proven on GB10. Open: real matmul, native CUBIN, f64.

## 8. Lane E — the type system (S2)

**`Knowledge<T>` is a compiler construct, not a stdlib struct** — threaded
through `self-hosted/check/types.sio`, `check/epistemic.sio`,
`native/lower_ir.sio`. Refinement-typed, carrying mean + covariance +
confidence; `.value` projection is gated by the `Epistemic` effect **and** an
SMT-discharged confidence refinement, so a function requiring `ε >= 0.82`
rejects under-confident data at compile time. GUM Jacobian / delta-method
propagation. Keep IIV (biological variability, ω²) and GUM uncertainty (u_c)
distinct — conflating them is a bug this repository has already made once.

**Ownership: linear and affine.** Current measured state (2026-07-26):

| case | expected | before | after the fix in flight |
|---|---|---|---|
| linear, straight-line replay | E039 | E039 ✓ | E039 ✓ |
| linear, never consumed | E040 | E040 ✓ | E040 ✓ |
| linear, first use via `let` + by-value param | accept | **E039** ✗ | accept ✓ (#1464, merged in #1447) |
| linear, consumed in **both** branches | accept | **E039** ✗ | accept ✓ (#1471) |
| linear, consumed in **one** branch only | reject | **accepted** ✗ | E040 ✓ (#1471) |
| **affine, straight-line double use** | reject | **accepted** ✗ | **still accepted** |

`affine` carries **no enforcement at all** today — not across control flow, not
even in straight-line code. That is the real content of PR #1290 ("enforce
affine ownership across control flow"), whose title describes the second half of
a problem whose first half is that there is nothing to extend yet. Recorded with
three-line repros in #1471.

Five `tests/compile-fail/linear_*.sio` fixtures are still accepted
(`linear_capture_closure`, `linear_early_return`, `linear_field_unconsumed`,
`linear_loop_consume`, `linear_parameter_unused`) — remaining holes, unclaimed.

## 9. The two failure modes to design against

Everything blocked in this document is blocked by one of these.

**Blocked-by-bootstrap.** New compiler architecture is limited by the codegen of
the compiler that must build it. Consequence: *codegen defects on `main` are the
critical path for architecture work,* and should be prioritised as such rather
than worked around inside the new lane.

**Blocked-by-stack.** Deep chains of pinned drafts cannot converge, because the
base moves. Consequence: prefer shallow PRs against a settled `main`, each with
its own repro that fails before and passes after. A blocker catalogued with an
ID, an owner and a severity is not a blocker resolved — #1194 was classified with
full rigour on 2026-07-19 and stayed open for a week; six three-line variants
root-caused it in minutes.

Add a third, cross-cutting: **the CI measures the wrong compiler.** Until
`full-test-suite` exercises Madaros, agents working on the modular compiler get
misleading green feedback. Tests for Madaros-only semantics must carry
`//@ requires: madaros` and run on the madaros gate.

## 10. Suggested order

1. **Codegen blockers on `main`** — 128 KiB by-value aggregates, fixed-array
   ident-copy aliasing, code capacity tiers. Unblocks Lane A wholesale.
2. **Standalone leaves of Lane A** — #946, #947 re-cut against current `main`.
3. **Ownership** — finish linear (#1471), then give `affine` an implementation at
   all; then revisit #1290.
4. **EISA and the precision ladder onto `main`** — they are verified on a branch
   and invisible from `main`, which is how a working `od256` ends up looking like
   an aspiration.
5. **Then** Place IR / Arena promotion, with a differential runner that can
   actually report `promotion ready: true`.

---

## Provenance

- Lane A reconstructed by reading PRs #870–#998 and the `codex/ir-*`,
  `codex/soir-*`, `codex/place-*` branches on 2026-07-26.
- §1–§3 condense `docs/handoff/madaros_plan_overview.md`, which exists only on
  `gpu/epistemic-tensor-core-next` and had never reached `main`.
- Lane B/C figures come from the EISA specs on that same branch plus the
  `artifacts/eisa/` receipts on `main`, and from the `od256` GPU gate runs.
- Lane E figures were measured on 2026-07-26 against Madaros built from source at
  `main@8d6eebc0d`, not from a prebuilt binary.
