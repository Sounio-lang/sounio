---
name: feedback_irinstr_size_bug
description: Linter repeatedly introduces [IrInstr; 16] type error in new test functions — always fix to [IrInstr; 128]
type: feedback
---

Whenever the linter adds new optimizer test functions to `self-hosted/compiler/main.sio`, it uses the wrong array size `[IrInstr; 16]` instead of `[IrInstr; 128]`. It also sometimes uses `opt_cleanup_module(func)` instead of `opt_cleanup_function(func)` (passing IrFunction to a function that takes IrModule).

**Why:** `cp_empty_instrs()` returns `[IrInstr; 128]`. The linter template is wrong. This recurs on every new sprint batch.

**How to apply:** After any sprint that adds new test functions, run:
```bash
sed -i 's/var instrs: \[IrInstr; 16\] = cp_empty_instrs()/var instrs: [IrInstr; 128] = cp_empty_instrs()/g; s/let result = opt_cleanup_module(func)/let result = opt_cleanup_function(func)/g' self-hosted/compiler/main.sio
./artifacts/omega/souc-bin/souc-linux-x86_64-jit check self-hosted/compiler/main.sio
```
This is a one-shot fix. Apply it whenever typecheck gates fail with "Type mismatch: expected [IrInstr; 16], found [IrInstr; 128]".
