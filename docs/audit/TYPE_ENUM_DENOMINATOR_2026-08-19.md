<!-- docs:meta
topic_id: repo.docs.audit.type-enum-denominator-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: grok-cli4
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.type-enum-denominator-2026-08-19
-->

# Type-enum denominator — the archaeology ruler under the ladder

**Date:** 2026-08-19  
**sha:** `1dc0df549dbf` (`origin/main` at measure)  
**Engine surface:** source archaeology (enums in tree), not Madaros run  
**Measure host:** Slurm `cpu-ops` / `cpuops-t560-proxmox` (not the login pod)  
**Receipt:** `docs/audit/type_enum_denominator/MEASUREMENT_RECEIPT.txt`  
**Script:** `scripts/dev/type_enum_denominator_measure.py`  
**Launch:** tarball→stdin→`srun` (workspace invisible on compute; see ops note)

---

## Semantic lane declaration (mandatory)

```text
Semantic-Lane-ID: typekind-denominator-20260819
Owner: grok-cli4
Concept-IDs: none (measurement of existing type-kind vocabularies; no concept rewrite)
Intent-Preserved: ladder positions remain derived from fixture pairs (#1943/#1945);
  this lane only names the denominator those fixtures sit under
Transformation: none to language meaning — observational census + defended count
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced:
  - there are exactly 8 included type-kind enums in the active Madaros tree (measured)
  - sum of unique variants across those 8 = 238; union of normalized stems = 143
  - tests/typekind index covers 29 kinds, 9 with real fixtures (not ghosts)
  - HlirTypeOctonion has no checker TypeKind twin — layer gap, not a dual occurrence
Claims-Forbidden:
  - "17/99 is the archaeology fraction" (99 is one enum, not the denominator)
  - identity of types by source-order discriminant or raw integer code equality
  - casting lean_single EXPR_TY / ETY_KIND / stdlib TYPE_* codes onto TypeKind ordinals
  - treating ghost_*.sio attempts as fixtures
Assumptions:
  - primary sources are self-hosted/ (Madaros) + listed stdlib compiler surfaces
  - bootstrap/ and *_test.sio / *_lean.sio copies are shadows, not extra enums
Write-Set:
  docs/audit/TYPE_ENUM_DENOMINATOR_2026-08-19.md
  docs/audit/TYPE_ENUM_DENOMINATOR_2026-08-19.tsv
  docs/audit/type_enum_denominator/**
  scripts/dev/type_enum_denominator_measure.py
  scripts/dev/type_enum_denominator_slurm.sh
Read-Set:
  self-hosted/{parser,check,hlir,gpu,compiler,lsp}/**/*.sio
  stdlib/compiler/types/type.sio
  stdlib/compiler/transform/type_annotation.sio
  tests/typekind/**/index.tsv
Positive-Witness: MEASUREMENT_RECEIPT.txt n_included_enums=8 r1_match=True
Negative-Witness: R4_DUP_VARIANTS HlirTypeContest/Robust listed twice in source
Acceptance-Gate: re-run scripts/dev/type_enum_denominator_slurm.sh (or measure.py on a
  tree the compute node can read); receipt sha + counts must match
Integration-Target: archaeology fleet — replace "99" as sole denominator in reports
Authoritative-Only-If: receipt regenerated from listed script on stated sha
```

---

## Refutation criteria (written before the run)

| ID | Criterion | Result |
|---|---|---|
| **R1** | Included type-kind enum count must equal the claimed 8, or the report states N≠8 with path:line for every candidate | **PASS** — N=8 |
| **R2** | No same-type verdict may use source-order index or raw integer code equality (two numberings exist) | **PASS** — identity by normalized stem + bridge evidence only |
| **R3** | Stem match alone ≠ same type; without a bridge → homonym | **PASS** — then manual bridge correction for known false negatives |
| **R4** | Impossible/duplicate source rows → RULER_SUSPECT, do not invent | **FLAG** — `HlirTypeContest` and `HlirTypeRobust` appear **twice** in `HlirTypeKind` body; unique count used |
| **R5** | Fixture = non-`-` path + file exists + not `*.ghost_*` | **PASS** |
| **R6** | bootstrap / `*_test` / `*_lean` are shadows | **PASS** — listed in `candidates.tsv` |

If a cell looked impossible, the ruler was suspected first (R4 duplicate variants). That is the same reflex that produced **Reserved**.

---

## 1) The eight type-kind enums

Measured N = **8**. Not 7, not 9. Inclusion is the curated pipeline set below; every other discovery is shadow or adjacent.

| # | enum | path:line | unique variants | role |
|---|---|---|---:|---|
| 1 | **TypeExprKind** | `self-hosted/parser/ast.sio:815` | 54 | parser AST type-expression kind |
| 2 | **TypeKind** | `self-hosted/check/types.sio:16` | **99** | checker semantic kind (the old sole denominator) |
| 3 | **LayTypeKind** | `self-hosted/check/layout.sio:45` | 17 | layout / ABI kind |
| 4 | **OwnTypeKind** | `self-hosted/check/ownership.sio:87` | 4 | ownership class (Linear/Affine/Copy/Drop) |
| 5 | **HlirTypeKind** | `self-hosted/hlir/ir.sio:106` | 42* | HLIR value type kind (*44 raw lines; 2 duplicate names) |
| 6 | **HlirTypeDefKind** | `self-hosted/hlir/ir.sio:1329` | 2 | HLIR type-definition kind (Struct/Enum) |
| 7 | **GpuType** | `self-hosted/gpu/kernel_ir.sio:46` | 12 | GPU kernel IR type |
| 8 | **TypeKind** *(legacy thin)* | `self-hosted/compiler/parser.sio:96` | 8 | lean-path thin AST TypeKind — **name collision** with #2 |

\* Source lists `HlirTypeContest` and `HlirTypeRobust` twice (lines in the epistemic block). Unique names = 42. Ruler flag, not silent.

### Discovered but not in the eight

| bucket | name | path:line | n | why out |
|---|---|---|---:|---|
| candidate / adjacent | LsphTypeCategory | `self-hosted/lsp/hover.sio:81` | 11 | LSP presentation, not compile pipeline |
| candidate / adjacent | HirType | `stdlib/compiler/transform/type_annotation.sio:29` | 11 | stdlib sketch, not Madaros driver |
| shadow | TypeExprKind | `self-hosted/bootstrap/bootstrap_v0.sio:2775` | 45 | bootstrap copy |
| shadow | TypeKind | `self-hosted/compiler/parser_test.sio:93` | 6 | test double |
| shadow | GpuType | `self-hosted/gpu/kernel_ir_wmma_lean.sio:136` | 12 | lean duplicate |
| **not an enum** | TYPE_* const table | `stdlib/compiler/types/type.sio` | 45 | integer kind codes — **second numbering** |
| **not an enum** | lean_single ETY/EXPR_TY | `self-hosted/compiler/lean_single.sio` | 25 documented | seed integer table — **must not cast onto TypeKind ordinals** |

If someone counted Lsph+HirType and dropped HlirTypeDefKind or legacy TypeKind, they would also get 8 — a different 8. This report pins the list by path:line. Disagree with evidence, not with a round number.

### Two numberings (why ordinals never identify)

1. **Enum discriminants** in Madaros Sounio enums (`TypeKind::TyI64` as source order).  
2. **Integer kind tables** in the seed and stdlib (`ETY_KIND` 0=unknown,1=i64,…; `TYPE_I64()→6`; lean `EXPR_TY` / `type_name_kind` 1=int,2=float,…).

Comment in `lean_single.sio` even documents a **third** map inside `ety_from_tnk` (tnk 1=int → ety 1=i64, tnk 3=string → ety 4=string). Comparing raw codes across these rulers without a named bridge is how silent mismatches land. **R2 forbids it.**

---

## 2) Coverage vs `tests/typekind/**/index.tsv`

Indices loaded: `tests/typekind/index.tsv` (F+H) + `tests/typekind/c/index.tsv` (family C).  
**29** kind rows total; **9** with a real `pass.sio`/`refuse.sio` pair (ghosts do not count).

| enum | variants | in index | with fixture |
|---|---:|---:|---:|
| TypeExprKind | 54 | 2 | 2 |
| TypeKind (checker) | 99 | 29 | 9 |
| LayTypeKind | 17 | 4 | 4 |
| OwnTypeKind | 4 | 0 | 0 |
| HlirTypeKind | 42 | 6 | 6 |
| HlirTypeDefKind | 2 | 0 | 0 |
| GpuType | 12 | 3 | 3 |
| TypeKind (legacy parser) | 8 | 1 | 1 |
| **Σ rows (not union)** | **238** | — | — |

Machine table: `docs/audit/TYPE_ENUM_DENOMINATOR_2026-08-19.tsv`  
Per-variant: `docs/audit/type_enum_denominator/variant_index_coverage.tsv`

**17 of 99 was never the fraction.** Even restricting to checker TypeKind alone, the index now has 29/99 rows and 9/99 fixtures. Against the multi-enum tree the fraction is smaller still.

---

## 3) Cross-layer: same type or homonym?

Method: normalize stems (strip `Ty`/`Type`/`HlirType`/`Tk`/`Gpu`/…; alias Void↔Unit, Ptr↔RawPtr, …).  
**Never** use `source_index`.  
Multi-enum stems → search bridge files (`check.sio`, `types.sio`, `type_convert.sio`, `lower.sio`, …).  
No bridge → provisional homonym; then manual bridge pass for false negatives (primitives split across files).

| after correction | n |
|---|---:|
| multi-enum stems | 57 |
| **same_type** (stem + bridge) | **57** |
| **homonym** (stem, no bridge) | **0** |

Full table: `docs/audit/type_enum_denominator/cross_layer.tsv`

### The Octonion question

> Um Octonion no HLIR e um Octonion no checker são um tipo ou dois?

**Measured answer: there is no Octonion variant in the checker `TypeKind` enum.**

- HLIR: `HlirTypeOctonion` at `self-hosted/hlir/ir.sio:135`, used in `type_convert.sio` (→ `<8 x float>`).  
- Checker `TypeKind`: **no** `TyOctonion` (rg clean on `types.sio` / `ast.sio`).  
- Parser `TypeExprKind`: **no** `TypeOctonion`.

So this is **not** a same-vs-homonym pair across two enums. It is a **layer gap**: a type that exists in HLIR (and LLVM lower) without a checker/parser kind name. Under the founder rule “every type in every layer”, Octonion is a witness of **incomplete integration**, not of dual identity.

Same HLIR-only family (no checker TypeKind twin):  
`Octonion`, `Sedenion`, `Quat`, `QuatLinear`, `QuatConv2d`, `QuatRnnState`, `QuatGate`, `Dual`, `Vec2/3/4`, `Vec2d/3d/4d`, `Mat2/3/4`, `U16` (HLIR has U16; checker TypeKind has U8/U32/U64/U128 but the stem map may still pair some widths — see variant table).

### Same-type examples (bridge-backed)

| stem | layers (names) | bridge evidence (abbrev.) |
|---|---|---|
| i64 | TyI64, TkI64, HlirTypeI64, GpuI64 | check.sio + type_convert.sio |
| knowledge | TypeKnowledge, TyKnowledge, HlirTypeKnowledge | check.sio / epistemic.sio |
| array | TypeArray, TyArray, TkArray, HlirTypeArray, legacy Array | check.sio + lower.sio |
| contest | TypeContest, TyContest, HlirTypeContest | check.sio |
| unit/void | TypeUnit, TyUnit, TkUnit, HlirTypeVoid, legacy Unit | check.sio + lower.sio |

### Name collision: two enums named `TypeKind`

| | checker | legacy compiler parser |
|---|---|---|
| path | `check/types.sio:16` | `compiler/parser.sio:96` |
| n | 99 | 8 |
| variants | TyI64, TyKnowledge, … | Named, Tuple, Array, Function, Generic, Unit, Ref, RefMut |

These are **not** the same enum. Stem overlap on `array`/`unit`/`ref` is real surface structure shared with TypeExprKind; the legacy enum is a thin AST skeleton, not a second copy of the 99. Reports must qualify `TypeKind` with path.

---

## Defended denominator

| candidate denominator | value | when it is honest |
|---|---:|---|
| checker TypeKind variants | **99** | claims about **checker** ladder rows only |
| sum of unique variants across 8 enums | **238** | upper bound on “named slots”; double-counts same_type stems |
| **union of normalized stems across 8 enums** | **143** | **defended default** for “how many type concepts the tree names somewhere” |
| stems present in all pipeline layers | ≪143 | the *integrated* count the founder rule aims at — **not yet measured as a green count**; Octonion shows the gap |
| index rows / fixtures | 29 / 9 | archaeology progress, not the type system size |

**Recommendation for fleet reports:**

1. Stop writing `k/99` as if 99 were the tree.  
2. Write coverage as  
   `fixtures=9 | index_rows=29 | checker_TypeKind=99 | union_stems_8_enums=143 | sum_slots=238`.  
3. For “every type in every layer”, the open work is the **integrated** stem set (intersection across parser/checker/layout/HLIR), not growing the F+H index alone.  
4. Never cast seed/stdlib integer codes onto enum ordinals (R2).

---

## What this does to archaeology next

- A new fixture still derives position from the two-program gate.  
- The **denominator** for “how much of the type system is fixture-covered” is no longer a handwritten 99.  
- HLIR-only kinds (Octonion, …) need either checker/parser kinds or an explicit “Reserved at surface / Executable at HLIR” split — inventing a checker Claim-ready without a `TyOctonion` would be a label again.

---

## Reproduce

```bash
# On a host that can srun (payload is streamed; /workspace need not be on compute):
bash scripts/dev/type_enum_denominator_slurm.sh
# Or, if the tree is local to the measure process:
SOUNIO_ROOT=$PWD python3 scripts/dev/type_enum_denominator_measure.py
```

Orangefs receipt (compute-visible):  
`/orangefs/training/sounio/type_enum_denominator/20260819T092457Z/`
