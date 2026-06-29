# Madaros large-module wall — CORRECTED diagnosis (supersedes the 128KB framing)

Date: 2026-06-29
Branch: `research/solver-ts3-parallel`, tip `83513addb`
Supersedes: `MADAROS_128KB_CODE_PATCH_WALL_2026-06-29.md` (its root-cause is **wrong**)
Status: **BOTH RESOLVED** — wall (a) `f5d4466cf`; println (b) `af5610d84`. The DIMACS corpus
now compiles AND runs to the correct answer end-to-end with the unmodified `println(result)`
harness (small/mid600/bi_1000=1.35MB → 0=UNSAT; a SAT instance → 1).

## RESOLUTION of (b) — imported-call return types not preseeded cross-module (`af5610d84`)

The `println(smt_solve-result)` crash was NOT smt_solve-specific and NOT memory-corruption (the
session's mid-investigation guess, disproved by an instrumented build). Real cause: the
multi-module externs preseed registers cross-module STRUCT layouts but NOT function return
types, so an IMPORTED callee's return type is empty in module.functions at body lowering →
`let r = imported_i32_call()` isn't tagged scalar_kind=1 → println(r) routes to string `print`
→ derefs the int as char* → SIGSEGV. Local calls work (own signatures are preseeded); test_smt
passes only because it prints via string-compare. Fix: a module-global side table
(LOWER_EXTERN_FN_RET_*: name-hash→1=int/2=f64) populated from sibling ASTs in the externs loop,
consulted as a fallback in expr_result_is_int_ref. Trivial 2-file repro: `use mymod::*;
println(ret5())` for `pub fn ret5()->i32`.

## RESOLUTION of (a) — the wall is the 4096 `flat_reloc` table cap (commit `f5d4466cf`)

A diagnostic build (instrumented `nc_add_flat_reloc`, since reverted) settled it: the unpatched
rel32 are **all `e8` calls** = relocations, and `NativeCompiler` held the reloc table as four
by-value `[i64;4096]` fields — `nc_add_flat_reloc` **silently dropped every reloc past index
4096**. Dropped CALL relocs keep `rel32=0` → jump-to-garbage → SIGSEGV. Reloc counts:
bi_100=**4000** (<4096 → 0 dropped → runs correct), bi_200=**6603**, bi_600=17015 (thousands
dropped). It is a COUNT cap, not an offset cap (~26 relocs/clause; `.text`≈256KB is merely where
the 4096th reloc happened to land). The "256KB" was a coincidence of reloc density.

**Fix:** moved `flat_reloc_{offsets,kind_codes,is_functions,target_indices}` to module globals
`NV2_RELOC_*[131072]` (codegen_x86_linux.sio), mirroring the `NATIVE_ELF_BUF` pattern. Lifts the
cap to 131072 (covers a 4096-clause corpus ≈106K relocs), removes 128KB of by-value aggregate
from `NativeCompiler` (binary 103.60→103.37MB), and kills the O(n²) copy-out-modify-copy-back the
in-struct array forced per reloc. **Verified:** test_smt 6/6 + run-pass green; known-UNSAT ladder
bi_100..**bi_1000 (1.35MB .text)** all print correct `RESULT_UNSAT` (was SIGSEGV >256KB);
mid600_unsat 0 unpatched rel32. The OpNot fix (`83513addb`) was orthogonal latent hardening; this
reloc-table fix is what actually unblocked large corpora.

---
(original investigation below, kept for the record)

## TL;DR

The prior audit said the 600-clause DIMACS corpus SIGSEGVs because of **128KB by-value
*builtin* truncation** (sqrt/print_f64/etc. emitted past offset 131072 into the by-value
`CodeBuffer`). That is **disproven**. The corpus crash is the sum of **two unrelated bugs**,
neither of which is builtin truncation:

- **(a) a ~256KB rel32-patch wall** — call/jump relocations whose patch site is at code
  offset ≥ ~262144 are left **unpatched** (`rel32 = 0`) → control transfer to garbage → SIGSEGV.
- **(b) `println(smt_solve-result)`** — passing `smt_solve`'s i32 return *directly* to
  `println`/`print_int` SIGSEGVs at **all sizes** (size-independent, separate codegen bug).

## What was verified and LANDED (commit `83513addb`, test_smt 6/6)

1. **Debt#4 4MB ELF buffer** (cherry-pick `481f3e9de`): `.text` mirror, ELF file buffer, and
   all mirror read/patch/sync sites widened 256KB→4MB. Lifts the rc=13 ELF cap. Real.
2. **OpNot mirror-direct** (`codegen_x86_linux.sio:6666`): the live `core_ir_into` path
   lowered boolean NOT via a by-value `(*nc).code = emit_not_bool_rax + sync_from_narrow`
   capped at the 131072 by-value buffer. Now `nc_emit_test_rax_rax/sete_al/movzx_rax_al`
   (mirror-direct). **Latent hardening** — no test this session reached a >128KB function
   using NOT; it changed no observed corpus behavior. Correct, non-regressing. Keep it, but
   it is NOT the corpus fix.

## How the 128KB-builtin framing was disproven (empirical chain)

- `sqrtsd` bytes (`F2 0F 51 C0`) are **absent from every compiled ELF**, including small
  passing ones → the `sqrt` builtin is dead-code-eliminated, never emitted. Not the culprit.
- The 6 passing `test_smt` ELFs are **139–151KB (already >131072)** and run fine. So programs
  with `.text > 128KB` are not categorically broken.
- The live function-compile path is `compile_ir_function_v2_core_ir_into` (mirror-direct
  `nc_emit_*`). Swept it + its call graph: no by-value `native_v2_*` helper calls, mirror-direct
  prologue (`begin_function_from_ir_into`), `_into` handle-resolve. OpNot was its **only**
  residual hybrid — and even fixing it changed nothing for the corpus.

## (a) The real wall: ~256KB rel32-patch cutoff — EVIDENCE

Known-UNSAT instances (`x0 ∧ ¬x0` appended; string-compare output `RESULT_UNSAT`, NOT
`println(result)` — see (b)). Run on `bin/madaros-opnotfix`:

| clauses | .text size | result |
|--------:|-----------:|--------|
| 100 | 0x3c214 (248KB) | `RESULT_UNSAT` ✓ correct |
| 200 | 0x5ad71 (363KB) | SIGSEGV / no output |
| 300 | 0x798ce (498KB) | SIGSEGV |
| 600 | 0xd5ae5 (874KB) | SIGSEGV |

Unpatched `rel32` scan (`E8|E9` followed by `00 00 00 00`) in `.text`:

| ELF | unpatched total | beyond offset 262144 | first beyond |
|-----|----------------:|---------------------:|-------------:|
| bi_100 (works) | 0 | 0 | — |
| bi_200 | 183 | 120 | 0x42e34 |
| bi_300 | 286 | 235 | 0x40222 (262690 — just past 262144) |

So: **call/jump patches at code offset ≥ ~262144 are silently not applied.** `.text` is fully
present and dense (bytes emitted via `nc_emit_byte`, mirror widened to 4MB) — this is a
*patch-application* gap, not byte truncation.

### Where it is NOT (already widened to 4194304)
`native_v2_text_byte_at` (1342), `native_v2_text_patch_u8` (1349), `nc_patch_u32_le` (1652,
gates on `code.len`), `apply_relocations_into` (4653), `native_v2_reloc_is_call_patch` (4638),
`native_v2_reloc_is_rip_disp_patch` (4643) — all read via the 4MB mirror / gate on `code.len`.
Label-patch recording (`codegen:1921`) and the global label arrays look uncapped here.

### Next-session pinpoint (needs ONE diagnostic build)
By inspection every primitive is 4MB-clean, yet patches past 262144 don't land. Instrument:
in `apply_relocations_into` and the label-forward-mirror patch, log `patch_offset`,
`NATIVE_V2_TEXT_LEN`, and whether `is_call_patch`/`is_rel32_branch` returned true, **gated on
`patch_offset >= 262144`**. Prime hypotheses to confirm/kill:
  - `NATIVE_V2_TEXT_LEN` stuck/clamped at 262144 at apply time (would make `byte_at` return 0 →
    applicability checks fail → silent skip). Check every assignment to `NATIVE_V2_TEXT_LEN`.
  - the `flat_reloc_*[4096]` table or a 256-entry label array overflowing for call-dense
    modules (first-unpatched tracks offset≈262144, which argues for an *offset* cap over a
    *count* cap — but verify the reloc count for bi_200 vs bi_100).

## (b) `println(smt_solve-result)` — separate, size-independent

Minimal repro (tiny, <128KB) still SIGSEGVs:
```
let result = smt_solve(&!ctx)
println("BEFORE")   // prints
println(result)     // SIGSEGV — never reaches next line
```
Works: `println(result + 0)`, `println(result as i64)`, `println(99)` after solve,
`if result==0 {println("UNSAT")}` (string-compare), and generic `let r=f(); println(r)` for a
trivial `f()->i32`. So it is **specific to `smt_solve`'s return path**, not "direct i32 temp".
`minA` faults (139) while near-identical `copy_through` exits 0 truncated → memory corruption;
black-box variant probing misleads. Pinpoint by `objdump`-diffing the print-arg setup of the
crashing vs working case, not by more variants. `b5849a4be` (route println non-literal i32 to
print_int) did **not** fully fix this.

## Why this matters / scope
The requested "128KB CodeBuffer refactor" rested on a wrong root cause. The by-value
`CodeBuffer`/builtin path is a real latent limit but is **not** what blocks the SAT corpus.
The corpus RUN gate needs (a) AND (b), each its own rebuild+verify cycle. Debt#4-alone is a
regression for >256KB programs — do not merge to main until (a) lands.

## Repro assets (scratchpad, this session)
`gen_corpus.py` (force_unsat + string-compare output), `bi_{100..600}.sio`, `small_unsat.sio`,
`mid600_unsat.sio`; binaries `bin/madaros-debt4` (Debt#4 only) and `bin/madaros-opnotfix`
(Debt#4 + OpNot) for A/B.
