# M1.2 Step 3 Blocker — Global Identifier Resolution is Structural

**Date:** 2026-04-30
**Task:** Wire global identifier resolution into the Native-v2 driver
**Verdict:** STOP. Root cause is structural, not a missing wire. Fix exceeds the
~150-line / "mirror lean_single" budget.

## What was investigated

Reproduced from existing inventory `/tmp/sounio-parity-inventory.L2TcOI`
(run-pass corpus, 393 files). Sampled failures matching the cluster:

| File | First failure |
|---|---|
| `tests/run-pass/door5_epistemic_attention.sio` | `kind=123 text=RNG_A` |
| `tests/run-pass/epistemic_ode_14comp.sio` | `kind=123 text=EState` |
| `tests/run-pass/test_gp.sio` | `kind=123 text=e308` |
| `tests/run-pass/mcmc_integration.sio` | `kind=123 text=e30` |
| `tests/run-pass/octonion_basic_demo.sio` | `kind=123 text=Octonion` |

`door5_epistemic_attention.sio:47` confirms: `var RNG_A: i64 = 7777` — a plain
top-level `var` declaration. `Octonion`, `EState` are top-level types. `e308`
is a different cluster (lexer bug — see below).

## Structural finding: hardcoded vs. dynamic name tables

The N-v2 driver resolves top-level identifiers through two **hardcoded**
`if token_text_eq(...)` tables:

- `self-hosted/compiler/native_compile_driver.sio:733` —
  `fn driver_global_id_tok(tok) -> i64` returns a fixed integer for
  ~80 specific names (all of them globals the driver itself uses, e.g.
  `V2_LOCAL_TOK_IDX`, `STRUCT_NAME_TOKS`, `DRV_CODE_BYTES`). Anything
  else returns `-1`.
- `self-hosted/compiler/native_compile_driver.sio:817` —
  `fn driver_const_value_tok(tok) -> i64` returns the literal int value
  for ~50 hardcoded constant names (mostly `TK_*` token kinds and
  `V2_BUILTIN_*`). Returns `-999999999` sentinel otherwise.

Both are consulted from `parse_primary_ir` at `:1801` and `:1807`. There is
**no fall-through path** that walks the user program's AST to discover
declarations. There is no `register_global`, `register_const`,
`register_enum_variant` runtime that does anything beyond the
struct/enum tables (`STRUCT_NAME_TOKS`, `ENUM_VARIANTS`) which are
already populated by `collect_structs`/`collect_enums`.

By contrast, lean_single uses dynamic, hash-keyed tables populated at
parse time:

- `self-hosted/compiler/lean_single.sio:1193` — `fn gl_find(ns, ne) -> i64`
  walks `GL[i*4+0]` looking for a name hash.
- `self-hosted/compiler/lean_single.sio:1203` — `fn gl_add(ns, ne, esiz, alen)`
  registers a global, allocating BSS via `bss_alloc_aligned`.
- `self-hosted/compiler/lean_single.sio:1247` — `CONST_NS[]/CONST_NE[]/CONST_VAL[]`
  with `CONST_COUNT` — populated as `pub const NAME = VALUE` declarations
  are parsed.

These are not *also* present in N-v2 in some unwired form. They are absent.

## Why this is not a 150-line fix

To mirror lean_single in N-v2 requires, at minimum:

1. **New parallel tables** — `DRV_USER_GL_NAME_TOKS[]`, `DRV_USER_GL_BSS_OFF[]`,
   `DRV_USER_GL_INIT[]`, `DRV_USER_GL_COUNT`; same shape for user consts.
   ~50 lines of declarations (must respect the driver's "no global
   initialization beyond `0`" constraint — see `feedback_native_compiler_limits`).
2. **A pre-pass over the program tokens** — walk top-level `var`/`let`/
   `const` / type / enum-variant declarations and register them. Roughly
   mirrors `collect_structs` (`:1117`) and `collect_enums` (`:1218`), but
   needs to handle initializer expressions for `var X: T = ...`. ~80–120 lines.
3. **Resolver fall-through** — `parse_primary_ir` at `:1800-1840` must,
   after the hardcoded tables miss, consult the dynamic table. Then
   route to a *new* IR path that emits BSS load/store with relocation.
   Today `ufn_record_global_load(dst, gid, idx)` takes a dense `gid`
   integer that maps directly to BSS offsets the driver already
   allocated for itself. User-program globals don't have allocated BSS
   slots. ~50 lines for the resolver, plus codegen wiring.
4. **BSS allocation for user globals** — the driver currently emits its
   own BSS layout from the hardcoded set. User globals need disjoint
   BSS slots, each with a relocation through `DRV_RELOC_*`. ~40 lines.
5. **Initialization codegen** — `var RNG_A: i64 = 7777` requires emitting
   a `mov [rip+slot], imm64` (or arch-equivalent) at program entry,
   before `main` runs. lean_single does this in `emit_global_inits_x86`
   (`:6261`) and `emit_global_inits_a64` (`:6343`). N-v2 has no analog. ~80 lines.

Total realistic scope: **300–400 lines**, plus a non-trivial risk to the
N-v2 self-compile gate, because the driver's own global set is currently
the **only** global set, and adding a second-class user-global path
changes BSS layout assumptions. This is a multi-session structural refactor,
not the "wire what lean_single does" the task envisioned.

## Distinct sub-cluster: scientific-notation float lexer bug

`e30` (8), `e200` (5), `e308` (4) are **not** unresolved globals. They
are the lexer splitting `1.0e308` into a `TK_FLOAT(1.0)` followed by a
`TK_IDENT(e308)`. Example: `tests/run-pass/test_gp.sio:34` —
`if x > 709.0 { return 1.0e308 }`. The driver does not lex; tokens come
from the shared lexer (`self-hosted/compiler/lexer.sio`). Fix belongs
in the lexer's number scanner, not the driver. Estimated 20–30 lines.
**Worth separating from M1.2 step 3.** Closing this alone would drop
~17 punch-list cases.

## Recommendation for the convergence plan

Split M1.2 step 3 into three independent slices, sized for separate
sessions:

- **3a (lexer):** scientific-notation float parse — ~30 lines, low risk,
  drops ~17 cases. Do this first.
- **3b (top-level enum variants + types):** these are *already* handled
  via `collect_enums`/`collect_structs` for the bare-name path; double-check
  whether `Oct`, `Octonion`, `EState`, `CovMat`, `Span`, `Box` are failing
  because the type isn't being collected (e.g. `pub struct` keyword,
  generic parameters, attribute decorators) versus an unhandled use site.
  Likely a smaller-than-expected fix — possibly under 150 lines.
- **3c (user globals/consts):** the structural one above. Plan as a
  multi-day sprint with its own design doc covering BSS layout, init
  codegen, and self-compile invariant.

## Files / call sites for next session

- Driver hardcoded tables: `self-hosted/compiler/native_compile_driver.sio:733`,
  `:817`
- Driver primary resolver: `:1800-1840` (dispatch); `:2510` (assignment);
  `:2670` (calls)
- Driver collect_structs / collect_enums (the existing dynamic precedent):
  `:1117`, `:1218`
- Lean_single dynamic tables: `:1193 gl_find`, `:1203 gl_add`,
  `:1247 CONST_*`, `:6261 emit_global_inits_x86`, `:6343 emit_global_inits_a64`
- Inventory snapshot used: `/tmp/sounio-parity-inventory.L2TcOI` (run-pass,
  393 files, 324 nv2_compile failures)

## Working tree note

The agent harness placed cwd at
`/workspace/sounio/.claude/worktrees/agent-a0c84a8e91addb3da` (branch
`worktree-agent-a0c84a8e91addb3da`, commit `a8aecce5`). The
inventory + the M1.2 punch list + the most recent M1.2 commit
(`4430b7b6 docs(m1.2): Track A → N-v2 parity inventory baseline +
punch-list`) live on `/workspace/sounio` (branch
`claude/s-ssm-zero-divisor-gating-KbKQe`). This blocker note is committed
on the latter, where the work belongs.
