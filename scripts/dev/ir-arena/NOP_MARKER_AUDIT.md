# Audit: who may delete an `IrNop`, and who must not

Prompted by #1669 — `ocp_mfi_compact_nops` deleted every `IrNop`, including the
one carrying `IR_FLOAT_REG_MARKER_FLAG`, so **every `f64` returned from a call
came back as `i64` MAX** under `-O`. Five lines reproduced it, on `origin/main`
as well as on this branch.

The rule the tree relies on, written down here because it was not written
anywhere:

> **An `IrNop` whose `imm_flags` is `IR_FLOAT_REG_MARKER_FLAG` (64064) is not
> dead.** It is how lowering tells codegen that the register in its `dst` holds
> an `f64`. Deleting it, or overwriting it with a plain `ir_nop()`, silently
> retypes that register as an integer.

## Every `IrOpcode::IrNop` comparison in `self-hosted/`, classified

93 mentions, 26 files. Only two of them delete.

### Deletes a nop — must check the marker

| site | status |
|---|---|
| `ir/opt_cleanup.sio` `ocp_mfi_compact_nops` | **was the bug**, fixed |
| `ir/opt_cleanup.sio` `ocp_compact_nops` (by-value, ~9010) | same shape, fixed. Self-test/probe spine only today, so it was not reachable from `-O` — but a self-test exercising the wrong behaviour is a trap |

### Consumes the marker — these are the reason it must survive

| site | what it does |
|---|---|
| `native/codegen_x86_linux.sio:7406` | *"Always honor an explicit float-marker IrNop"* — marks the register float |
| `native/machine_ir.sio:1013` | `is_float_slot[dst] = 1` |
| `native/machine_ir.sio:1158` | `is_float_slot[dst] = 1` |
| `ir/opt_cleanup.sio:664` | `ocp_rebracket_has_float_marker_before` tests `imm_flags & IR_FLOAT_REG_MARKER_FLAG` |

Three independent consumers, in two subsystems. That is what made the deletion so
damaging and so quiet.

### Skips over nops without removing them — safe

`ir/tailcall.sio` (331, 834, 861 — scanning forward for a return),
`ir/opt_cleanup.sio:9214` (`jump_to_return` looking past nops for the next real
instruction), `native/machine_ir.sio:653` (support check), and
`native/codegen_x86_linux.sio:7802` (emits no code for a nop, its meaning having
been consumed at 7406).

### Cannot reach a nop — safe by construction

`ocp_mfi_dce_once` overwrites dead instructions with `ir_nop()`, which *would*
erase a marker — but it is guarded by `ocp_has_dst(op)`, and `ocp_has_dst` has no
`IrNop` arm, so it falls to `_ => false`. A marker nop is never a candidate.

**This is load-bearing and accidental.** Adding `IrOpcode::IrNop => true` to
`ocp_has_dst` — which looks harmless, since a marker nop does have a `dst` —
would reintroduce #1669 through a completely different door.

### Neither reads nor writes — safe

Opcode-equality helpers (`ir/normalize.sio`, `ir/inline.sio`), opcode
classification and mapping (`ir/loop_opt.sio`, `ir/auto_vectorize.sio`,
`wasm/lower.sio`, `compiler/module_frontend.sio:1828`, `ir/serialize.sio`,
`emit/text.sio`), printers (`ir/disasm.sio`, `native/machine_ir.sio:381`,
`native/codegen_x86_linux.sio:10711`), the VM and peephole translations, and the
self-test assertions in `compiler/main.sio` (23 of the 93 mentions) and
`ir/{optimize,const_prop,dce}.sio`.

## What would make this structural

The marker is a flag on an opcode that everything else is entitled to treat as
dead. That is a poor place to keep load-bearing information, and the audit above
is a snapshot, not a guard. Two options, in increasing order of goodness:

1. **A distinct opcode.** `IrFloatMarker` instead of `IrNop + flag`. Then "a nop
   is dead" stays true, and every `match` on opcodes has to acknowledge the new
   arm — the type checker does the auditing.
2. **Carry float-ness on the register**, not on an instruction. The information
   is a property of the value; the marker instruction exists only because there
   is nowhere else to put it. `IrFunction` already carries `returns_float`.

Until either lands, a grep gate asserting that no new `!= IrOpcode::IrNop`
survivor test appears without a marker check is the cheap version.
