<!-- docs:meta
topic_id: repo.docs.audit.hlir-disconnect-cost-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: grok-cli3
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.hlir-disconnect-cost-2026-08-19
-->

# HLIR disconnect cost — measurement, not reconnection

Date: 2026-08-19
Lane: `hlir-disconnect-cost-20260819`
Worktree: `/workspace/.wt/grok-cli3`
Instrument: prebuilt `artifacts/self-hosted/madaros` (Madaros v0.80.0, 99964767 B, 2026-08-17 17:01)
Slurm: job **10326** on `cpuops-t560-proxmox` at 2026-08-19T10:19:29Z
This is not a reconnection. No variant was added. No `self-hosted/hlir/` file was edited.

Companion: [`HLIR_DISCONNECT_COST_2026-08-19.tsv`](HLIR_DISCONNECT_COST_2026-08-19.tsv)

---

## Semantic-Lane declaration

```text
Semantic-Lane-ID: hlir-disconnect-cost-20260819
Owner: grok-cli3
Concept-IDs: none
Intent-Preserved: compile success != runtime parity; effect annotation != physical mechanism; a type that exists only downstream of the checker is not a sayable type
Transformation: none — measurement of a disconnected layer
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: package-level `souc check self-hosted/hlir/mod.sio` is rc=0 on this instrument; HlirTypeKind has 42 unique names and 2 duplicate identifiers; there is no TypeKind→HlirTypeKind function
Claims-Forbidden: "HLIR is in the live pipeline"; "HlirTypeOctonion is a sayable type"; "the layer is unrecoverable because three files fail isolated check"; "supports first-class Octonion" from the existence of HlirTypeOctonion
Assumptions: Madaros is the claim-oracle; the import instrument is `^use <pkg>::` in tracked `self-hosted/**/*.sio` outside `self-hosted/<pkg>/`
Write-Set: docs/audit/HLIR_DISCONNECT_COST_2026-08-19.md; docs/audit/HLIR_DISCONNECT_COST_2026-08-19.tsv
Read-Set: self-hosted/hlir/*; self-hosted/check/types.sio; self-hosted/check/layout.sio; self-hosted/parser/ast.sio; self-hosted/compiler/main.sio; self-hosted/gpu/hlir_to_gpu.sio; self-hosted/llvm/type_convert.sio
Positive-Witness: none claimed (no reconnection)
Negative-Witness: none claimed
Acceptance-Gate: the five questions below are answered with per-file rcs, named importers, named duplicate variants, a named conversion function or its absence, and a counted (not written) variant list
Integration-Target: none
Authoritative-Only-If: a later Madaros rebuild changes isolated-check behaviour of these five files
```

---

## Refutation criteria (written before the run)

1. The layer is **not recoverable as source** if any of the five files fails `souc check` with a parse error (`parse_failed=true`) or with a diagnostic that is not a missing sibling name. Isolated E137 on a name defined in a sister file of the same package is incomplete wiring, not an ill-formed file.
2. The layer is **not in the live pipeline** if its only `use hlir::` importers outside `self-hosted/hlir/` sit on the `--gpu-target` side door and the default `check`/`run` path never calls `hlir_lower_module`.
3. The `#1949` "44 lines / 2 duplicate names" claim is **false** if a mechanical count of `HlirTypeKind` variant identifiers is not 44, or if the two colliding names are not findable.
4. A TypeKind→HlirTypeKind conversion **exists** only if a function takes `TypeKind` (or `TypeEntry`) and returns `HlirType` / `HlirTypeKind`. Mapping `TypeExpr` (parser AST) is a different function.
5. The Octonion-annotation cost is **zero new variants** only if parser `TypeExprKind`, checker `TypeKind`, and layout `LayTypeKind` already have an Octonion constructor, or a documented partial that `hlir_type_from_ast` already consumes for the bare name `Octonion`.

If a number disagreed with the dispatch's positive control (parser=93, ir=48), the ruler is wrong and the number is not reported as fact.

---

## Instrument validation

Import syntax is `use <dir>::`. The ruler that reproduces the named controls is:

- tracked `*.sio` (`git ls-files`)
- line matching `^use <pkg>::`
- file under `self-hosted/`
- file **outside** `self-hosted/<pkg>/`

| package | this tree | dispatch control |
|---|---:|---:|
| parser | **93** | 93 |
| check | 49 | 49 |
| ir | **48** | 48 |
| native | 20 | 19 |
| wasm | 17 | 15 |
| hlir | **2** | 2 |
| gpu | 2 | 1 |
| enir | 0 | 0 |
| mli | 0 | 0 |
| llvm | 0 | 0 |
| vm | 0 | 0 |
| effects | 0 | 0 |

parser=93 and ir=48 match. native/wasm/gpu differ by 1–2 files on this tree; that is reported, not rounded to the dispatch. A `use|import|from|mod` grep was not used.

`souc check` positive control on the same Slurm job: `tests/effects/archaeology/io_pass.sio` → rc=0, `check: OK`. The binary runs on the compute node. A second calibration — isolated `self-hosted/parser/ast.sio` — is rc=1 with `AST closure incomplete … parse_failed=false`, no `error[E…]`. Isolated self-hosted files are not automatically green. That is the ruler, not a HLIR defect.

Slurm note: `/workspace` is not mounted on compute nodes. Payload (Madaros ELF + the five HLIR files + the two controls) was shipped as a stdin tarball. Job 10326, `cpu-ops` / `cpuops-t560-proxmox`. No `build_modular_madaros.sh`. No pod check.

---

## Q1 — does each of the five files pass `souc check` today?

Per file. No aggregate.

| file | bytes | isolated `madaros check` rc | diagnostic | what the log actually says |
|---|---:|---:|---|---|
| `self-hosted/hlir/mod.sio` | 1334 | **0** | NONE | `about to check 5 modules` … `verdict=0` `check: OK` |
| `self-hosted/hlir/ir.sio` | 49095 | **0** | NONE | `check: OK` |
| `self-hosted/hlir/builder.sio` | 15333 | **1** | E015, E137 | undeclared `hlir_empty_name`, `HLIR_INVALID_VALUE`, `hlir_module_add_global`, `hlir_module_add_typedef` (all defined in `ir.sio`) |
| `self-hosted/hlir/lower.sio` | 107486 | **1** | E015, E137 | undeclared `hlir_name_from_str` and other `ir.sio` / builder names; pulls `opt_strategy` via `use hlir::opt_strategy::*` (2 modules) |
| `self-hosted/hlir/opt_strategy.sio` | 15699 | **1** | E015, E137 | undeclared `HLIR_INVALID_VALUE`, `HLIR_UNIT_VALUE`, `hlir_function_add_block` |

Package-level: `mod.sio` is `pub use ir::*` / `builder::*` / `lower::*` / `opt_strategy::*` and checks as **five modules, verdict=0**. The three isolated failures are missing sibling names, not parse failures. Under the pre-run criterion this is recoverable as a package and not recoverable as five independent translation units.

`mod.sio` also emits `stack frame too large` warnings on by-value `HlirModule` / `HlirFunction` (reported sizes in the 10¹²–10¹⁵ byte range). That is a reconnection cost if anyone wires this into the live compiler. It is not a check failure.

`mod.sio` comments mention `lower_effects.sio`. That file does not exist.

---

## Q2 — who are the two importers, and is the use real?

Under the validated ruler, files outside `self-hosted/hlir/` containing `^use hlir::`:

| file | import | what it calls | live default pipeline? |
|---|---|---|---|
| `self-hosted/compiler/main.sio:81–85` | `use hlir::ir::*` `use hlir::builder::*` `use hlir::lower::*` | `hlir_lower_module` at `:28198`, inside `run_gpu_compile_pipeline` | **No.** Comment at `:28152`: only when `--gpu-target` is set. Default `check`/`run` is parser → check → ir → native. |
| `self-hosted/gpu/hlir_to_gpu.sio:20` | `use hlir::ir::*` | consumes `HlirModule` / `HlirType` to emit GpuKernelIr / PTX / Metal / SPIR-V | **No** for default `souc`. Real GPU lowering, not a `test_*.sio` self-test. Reached only from the same GPU side door. |

Self-import (not counted): `self-hosted/hlir/lower.sio:16` `use hlir::opt_strategy::*`.

Call site the import ruler does **not** count: `self-hosted/main.sio:502` calls `hlir_lower_module` with **no** `use hlir::`. It is not an importer under the instrument that produced "2".

Neither importer is a self-test. Both are the GPU bypass. HLIR is not on the oleoduto that survived (`parser → check → ir → native`).

---

## Q3 — `HlirTypeKind` variants and the two duplicate names

Source: `self-hosted/hlir/ir.sio:106–158`.

Mechanical count of variant identifiers: **44 lines, 42 unique names, 2 duplicates**.

The duplicates, same spelling, same comment, second copy under a second heading:

```
150:    HlirTypeContest,         // Contest<inner> with disagreement metadata
151:    HlirTypeRobust,          // Robust<inner> with disagreement metadata
…
156:    HlirTypeContest,         // Contest<inner> with disagreement metadata
157:    HlirTypeRobust,          // Robust<inner> with disagreement metadata
```

`#1949` is confirmed on those two names. This is the class no gate sees — two identifiers with the same name in one enum, analogue of the `#1695` `IR_NAME_POOL_LEN` collision.

The 42 unique names, in source order:

Void Bool I8 I16 I32 I64 I128 U8 U16 U32 U64 U128 F32 F64 Ptr Array Struct Tuple Function
Vec2 Vec3 Vec4 Mat2 Mat3 Mat4 Quat Octonion Sedenion
QuatLinear QuatConv2d QuatRnnState QuatGate
Vec2d Vec3d Vec4d Dual
Knowledge Contest Robust Intervention Counterfactual Validated

Hypercomplex that exist **only here** (and in the dead LLVM convert): `HlirTypeQuat`, `HlirTypeOctonion` (`:135`), `HlirTypeSedenion`, the four `HlirTypeQuat*`, Dual, the six Vec/Mat.

---

## Q4 — shortest link from checker to HLIR

**There is no `TypeKind` → `HlirTypeKind` function.** A search of `self-hosted/**/*.sio` for a function that takes checker `TypeKind` / `TypeEntry` and returns `HlirType` / `HlirTypeKind` is empty.

What exists instead:

| function | from | to | file:line | on the live path? |
|---|---|---|---|---|
| `hlir_type_from_ast` | parser `TypeExpr` | `HlirType` | `self-hosted/hlir/lower.sio:1990` | only if `hlir_lower_module` runs (GPU side door) |
| `convert_hlir_type` | `HlirType` / `HlirTypeKind` | LLVM type | `self-hosted/llvm/type_convert.sio:73` | llvm has **0** outside importers; `type_convert.sio` has no `use` line at all |

`hlir_type_from_ast` is the shortest existing link, and it **skips the checker**. `run_gpu_compile_pipeline` even says so (`compiler/main.sio:28184`): `hlir_lower_module` consumes the AST, not the type-checked `TypeEntry`s.

Unknown `TypeNamed` falls through to `hlir_type_i64()` (`lower.sio:2060–2061`). Bare `Octonion` would become i64. `Hyper<Octonion, f64>` is already special-cased at `:2023–2027` and returns `hlir_type_octonion()`.

That is the finding the dispatch named: the type exists downstream of the site where it can be said.

---

## Q5 — cost to write an `Octonion` annotation that arrives as `HlirTypeOctonion`

Counted, not written. Three layers named in the dispatch, plus the conversion that is not an enum.

### Bare annotation `let x: Octonion = …`

| layer | enum | Octonion constructor today | partial already there | new variants needed | exact insertion line (not written) |
|---|---|---|---|---:|---|
| parser | `TypeExprKind` `self-hosted/parser/ast.sio:815–871` | **none** (`rg Octonion` in `ast.sio` is empty) | `TypeNamed` at `:816` already parses any identifier, including `Octonion` | **0** if `TypeNamed` is kept; **1** (`TypeOctonion`) if a dedicated variant is wanted — after `:816` or after last variant `TypeAxiom` at `:870` | |
| checker | `TypeKind` `self-hosted/check/types.sio:16–133` | **none** (no `TyOctonion`) | `TyHyper` at `:60` is `Hyper<Algebra, T>`, a different constructor; algebra tag 3 = Octonion lives in `check.sio` / `hyper.sio`, not in the enum | **1** (`TyOctonion`) — append after `TyF256` at `:132` (file says append to preserve discriminants) | |
| layout | `LayTypeKind` `self-hosted/check/layout.sio:45–63` | **none** (no `TkOctonion`) | none. Closest are `TkStruct` / `TkArray` / `TkOpaque` | **1** (`TkOctonion`) after `TkOpaque` at `:62` | |
| HLIR | `HlirTypeKind` | **already** `HlirTypeOctonion` at `ir.sio:135` | `hlir_type_octonion()` at `:413` | **0** | — |
| AST→HLIR | function, not enum | maps `TypeNamed "Hyper"` + arg `"Octonion"` only | fallback for bare `"Octonion"` is `hlir_type_i64()` at `lower.sio:2061` | **0 variants**; **1 branch** on `tname == "Octonion"` next to `:2023` | |
| TypeKind→HlirTypeKind | **absent** | — | — | **1 function**, does not exist anywhere | would live next to `hlir_type_from_ast` in `lower.sio` or in a new check→hlir file |

**New enum variants for the three named layers: 2** (checker + layout) if parser stays on `TypeNamed`, **3** if parser also gets `TypeOctonion`.

Plus one missing function (TypeKind→HLIR) if the type is required to survive the checker rather than jump from the AST. Plus one string-compare branch in `hlir_type_from_ast` even on the existing AST jump, because bare `Octonion` is not handled.

### Annotation that already almost works: `Hyper<Octonion, f64>`

- parser: `TypeNamed` + type args — **0** new variants
- checker: `TyHyper` + algebra tag — **0** new variants
- layout: still no `TkOctonion` — **1** if layout must distinguish it; **0** if it is a struct
- `hlir_type_from_ast`: **already** returns `hlir_type_octonion()` at `:2023–2027`

That path still only fires inside `hlir_lower_module`, which default `souc` never calls.

### Downstream that must not be mistaken for a sayable type

`self-hosted/llvm/type_convert.sio:282` already lowers `HlirTypeOctonion` to `<8 x float>`. llvm has 0 outside importers. Existence of that arm is not evidence that a user can write `Octonion`.

---

## Cost summary (before anyone promises a date)

| item | count |
|---|---:|
| HLIR files | 5 |
| package check via `mod.sio` | rc=0 |
| isolated file check green | 2 / 5 (`mod`, `ir`) |
| isolated file check red (sibling E137) | 3 / 5 (`builder`, `lower`, `opt_strategy`) |
| outside importers (`use hlir::`) | 2, both GPU side door |
| default-pipeline importers | 0 |
| `HlirTypeKind` unique / duplicate | 42 / 2 (`Contest`, `Robust`) |
| `TypeKind` → `HlirTypeKind` functions | 0 |
| new enum variants for bare `Octonion` on the three named layers | 2 (or 3 with dedicated parser variant) |
| missing conversion functions for a checker-honest path | 1 |
| `lower_effects.sio` (named in `mod.sio`) | 0 (file absent) |

Reconnection is a separate dispatch. This document does not start it.

---

## Semantic outcome

```text
Semantic-Outcome: HLIR source is package-checkable and disconnected from the live pipeline; Octonion is a HLIR-only kind
Concept-Status-Before: "HLIR has two importers and the hypercomplex algebra lives only there"
Concept-Status-After: confirmed; importers are the GPU side door; no TypeKind→HlirTypeKind; 2 (or 3) new variants to make bare Octonion sayable
Distinctions-Added: isolated-check rc != package-check rc; TypeExpr→HlirType != TypeKind→HlirTypeKind; HlirTypeOctonion != sayable Octonion
Distinctions-Preserved: compile success != runtime parity; Madaros is the claim-oracle
Distinctions-Erased: none
Evidence-Run: Slurm job 10326, cpuops-t560-proxmox, 2026-08-19T10:19:29Z, artifacts/self-hosted/madaros v0.80.0
Fallback-Path: none claimed
Legacy-Kept: all five HLIR files untouched
Conflicting-Lanes: grok-cli4 holds ENIR/MIR cost; grok-cli5 holds effects-layer cost and the docs registry. This lane did not write those files.
Next-Semantic-Interface: a reconnection dispatch would have to decide whether Octonion is TyHyper, a new TyOctonion, or AST-only, before anyone writes a variant
```
