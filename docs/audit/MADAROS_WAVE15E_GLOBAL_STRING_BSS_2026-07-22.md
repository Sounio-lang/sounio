<!-- docs:meta
topic_id: repo.docs.audit.madaros-wave15e-global-string-bss-2026-07-22
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-wave15e-global-string-bss-2026-07-22
-->

# Madaros Wave15e — module-level string global BSS init (SEGV)

**Date:** 2026-07-22  
**Role:** Wave15 Agent E (ousadia residual hunter)  
**Branch:** `fix/madaros-wave15e-global-string-bss`  
**Tip measured:** `origin/main` @ `3e7ed9f52` (post-Wave13e #1405)  
**Engine:** current-source Madaros (stock prebuilt lacks this fix)

## Claim boundary

> Under **current-source Madaros**, a module-level string literal global:
>
> ```sounio
> let PREFIX: string = "hi"
> fn main() -> i64 with IO { println(PREFIX); 0 }
> ```
>
> compiles, places `"hi"` in rodata, stores a RIP-relative pointer into BSS at
> entry, and prints `hi` (rc=0). Pre-fix: compile succeeded, `"hi"` **absent**
> from the ELF, BSS word stayed 0 → `println` SEGV (rc=139).

### Owns

- Single-module `let` / `var` string globals initialised with a string literal
- Getter that returns that global
- Concat of global string with a literal (`PREFIX ++ "!"`)

### Does **not** own (claims_not_made)

- W15-A multi-stmt paramful global **element-list** fold
- W15-B `print_f64` negatives (already green on tip for `-2.0`)
- W15-C large multi-mod / #901 layout
- W15-D science dual expansion
- Merged W13/W14 surfaces (into-acc, bare Ident, method chains, bare float arith, #913, #921, cd_exact, DCE)
- Multi-module **import** of a string global that participates in `++` in the dep
  (measured residual: thin-link `rc=12` when dep has `pub let PREFIX: string = "hi"`
  plus `PREFIX ++ s`; separate class)
- Data-carrying enum payload match (wrong-code residual: payload always 0)
- Non-literal string global inits (calls, concat at global scope)

## Root cause

Global init recording (`items_record_global_int_init` → `GLOBAL_VAR_INIT_*`) only
folds **i64 words / f64 bits**. `ExprStringLit` was not recorded, so:

1. BSS slot stayed zero-filled (`returns_float` never set → no entry init).
2. The literal never entered rodata (`"hi"` not present in the ELF at all).
3. `println(PREFIX)` loaded a null pointer from BSS → SEGV.

Runtime store already worked (`PREFIX = "hi"; println(PREFIX)`) because that path
emits `IrLoadString` + store.

## Fix

| Layer | Change |
|---|---|
| `parser/ast.sio` | `GLOBAL_STR_INIT_*` side table + record/lookup; reset with int table |
| `parser/items.sio` | `ExprStringLit` → `ast_record_global_str_init` |
| `ir/ir.sio` | `BSS_INIT_STRING_MAGIC = -800001` |
| `ir/lower.sio` | `lower_apply_global_var_init`: magic + normalized payload in `return_struct_name` |
| `native/codegen_x86_linux.sio` | `emit_global_var_inits_into`: LEA rodata + store to BSS when magic |

## Gate

```bash
bash scripts/ci/madaros_global_string_init_gate.sh
# → MADAROS_GLOBAL_STRING_INIT_GATE_OK
```

Fixture: `tests/run-pass/global_string_lit_init.sio`  
Expects four lines: `hi` / `yo` / `hi` / `hi!` and rodata contains both payloads.

## Measured

| Program | Stock prebuilt (pre-fix) | Current-source Madaros |
|---|---|---|
| `let S: string = "hi"; println(S)` | compile OK, run SEGV 139; no `"hi"` in ELF | `hi\n`, rc=0; `"hi"` in rodata |
| `var S: string = "x"; S = "hi"; println(S)` | already OK (runtime store) | unchanged OK |
| `let N: i64 = 7; print_int(N)` | OK (control) | OK |

## Related residual (not this PR)

Multi-mod:

```sounio
// lib.sio
pub let PREFIX: string = "hi"
pub fn wrap(s: string) -> string { PREFIX ++ s }
// main
use lib::{wrap}
println(wrap("!"))
```

→ thin-link `rc=12` on stock tip. Distinct from the single-module SEGV class.
