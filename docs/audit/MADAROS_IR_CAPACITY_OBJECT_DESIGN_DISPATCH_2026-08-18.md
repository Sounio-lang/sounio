<!-- docs:meta
topic_id: repo.docs.audit.madaros-ir-capacity-object-design-dispatch-2026-08-18
authority: repo_only
audience: users
last_validated: 2026-08-18
validated_by: empryo-1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-ir-capacity-object-design-dispatch-2026-08-18
-->

# Dispatch — IR capacities should be ONE object, not dozens of coupled literals

**Filed:** 2026-08-18 · **Status:** OPEN (design dispatch, not yet implemented) · **Lane:** empryo-1

## The problem, measured

The `IR_MAX_FUNCS` raise (#1897, 8192 → 16384) needed **eight files** touched
in lockstep because the capacity is expressed as dozens of independent
literals that must agree:

| Site | Literal | Role |
|---|---|---|
| `ir/ir.sio:35` | `IR_MAX_FUNCS = 16384` | the constant itself |
| `ir/ir.sio:3161` | `[IrFunction; 16384]` | `IrModule.functions` |
| `ir/ir.sio:1131-1141` | `[i64; 16640]` ×5 + `ir_region_table_capacity()` | region table (MUST exceed `IR_MAX_FUNCS`) |
| `ir/normalize.sio` | `[IrFunction; 16384]` ×2 | sort + normalize scratch |
| `ir/lower.sio:682` | `elem_kinds: [i64; 16384]` | indexed by the functions slot |
| `native/codegen_x86_linux.sio` | `fn_offsets: [i64; 16384]` ×4 | backend offsets |
| `native/elf.sio`, `elf_bulk.sio`, `reloc.sio`, `frame.sio` | `fn_offsets` / `name_offsets` | ELF emission |

Miss one and the failure is either a loud type error (best case) or a silent
drop (the historical case the `ir_capacity` fixture exists to catch: past the
literal, a function's symbol and offset were simply DROPPED, exit 0, no
error). The same shape repeats for every other capacity:

- `IR_MAX_INSTRS = 16384` is coupled to `DCE_MAX_INSTRS = 8192`
  (`ir/dce.sio:31`) — a truncated liveness analysis is a WRONG analysis, not a
  weaker one, so these must move together or not at all.
- `SPEC_DCE_MAX = 8192` / `SPEC_DCE_SLOTS = 16384` (`check/specializer.sio`)
  and the `[i64; 16384]` mark arrays.
- `HLIR_MAX_LOCALS = 8192` (`hlir/ir.sio`).

## The design

These capacities should be **one object**, not scattered literals:

```
struct IrCapacities {
    max_functions: i64,
    region_slots: i64,          // MUST be > max_functions (+ margin)
    max_instructions: i64,
    dce_liveness_slots: i64,    // MUST be >= max_instructions
    backend_offset_slots: i64,  // MUST be >= max_functions
    ...
}
```

with a single constructor that enforces the invariants
(`region_slots > max_functions`, `dce_liveness_slots >= max_instructions`)
and a single place to raise a ceiling. The arrays that are currently
fixed-size inline (`[IrFunction; N]`, `fn_offsets: [i64; N]`) would be sized
from that object.

## Why this also fixes the value-semantics cost

Duplicating arrays inline is what makes the by-value semantics expensive:
`IrModule.functions: [IrFunction; 16384]` is what makes `IrModule` a ~23 MiB
struct, and every by-value pass of it invites a deep-copy reading even where
the lowering happens to handle-pass it. Moving the backing storage to
**dynamic allocation** (one heap/arena buffer per capacity, sized from the
object) collapses `IrModule` to a struct of handles + counts, makes by-value
passes cheap and unambiguous, and removes the entire class of "raise the
constant, forget an array" failures.

## Scope and stop criteria

This is a **design dispatch, not an implementation**. The implementation is a
large, multi-file refactor that must:

1. Land behind the `irfunction_instr_capacity_coherence_gate.sh` and the
   `ir_capacity` fixture (raised past the old ceiling).
2. NOT raise `IR_MAX_INSTRS` without raising `DCE_MAX_INSTRS` and every
   per-instruction context that stops at its own cap — that is the
   silent-miscompile trap named in #1897's safe-stop point.
3. Keep the recorded fixed-point rung at `check` until gen2 is independently
   unblocked.

## Related

- #1884 — the capacity raise is not self-contained (region-table coupling)
- #1897 — the coordinated `IR_MAX_FUNCS` raise, proven and regression-free
- `scripts/ci/irfunction_instr_capacity_coherence_gate.sh` — the coherence gate
- `tests/multimodule/ir_capacity/` — the boundary witness
