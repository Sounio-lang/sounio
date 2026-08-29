<!-- docs:meta
topic_id: repo.docs.audit.g1-wip.source-to-elf-bridge-prototype-2026-06-02
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.g1-wip.source-to-elf-bridge-prototype-2026-06-02
-->

# source→ELF bridge prototype — `--native-v2-compile` (2026-06-02)

Goal: connect real `.sio` source to the native-v2 back-half I hardened to 9 IR witnesses
(`compile_native_v2_preview_to_file`), i.e. the genuine source→ELF path the synthetic
emit witnesses only modeled. Prototyped a new mode that wires the **working lowering**
to that backend, bypassing the crashing old `compile_to_obj`.

## What was built
A ~30-line CLI mode `--native-v2-compile <src> -o <out>` (patch:
`native_v2_compile_bridge.patch`, applies on the g1 tip):

```
load_multimodule_ir(src)  →  (if ok)  compile_native_v2_preview_to_file(&module, spec, out)
```

Prototyped on a **detached** snapshot of `g1/qualify-bare-patterns` tip `102350faa`
(binary `9979340d`, built with `bin/souc e35ef063`) — the live FIX#2 worktree was NOT
touched. (It cannot be built on `modular/native-v2-e2e-gate`: that binary crashes in
**lowering** — `--probe-load-ir` rc=139 even on `fn main(){}` — because it lacks the g1
bare-pattern qualification fix, which repairs a bin/souc miscompile that the lowering path
also trips. The g1 frontend is a hard prerequisite for any source→ELF.)

## Result: architecture sound, two precisely-localized gaps (NOT a green E2E)

| Program | `--probe-load-ir` (`_fn_count_traced`) | `--native-v2-compile` (full `load_multimodule_ir`) | Where it breaks |
|---|---|---|---|
| `fn main(){}` (empty) | rc=0, functions=1 | **rc=139 crash** *before* backend | **full `load_multimodule_ir`** |
| `fn main(){ let x=1 }` | rc=0, functions=1 | **rc=139 crash** *before* backend | **full `load_multimodule_ir`** |
| `fn main()->i64{ return 13 }` | (n/a) | rc=0, "lowering failed" | **check-during-lowering** |
| `…{ let a=40; let b=2; return a+b }` | (n/a) | rc=0, "lowering failed" | **check-during-lowering** |

**Both gaps are UPSTREAM of the native-v2 backend.** A print marker placed immediately
after `load_multimodule_ir` returns ("load OK fns=… → calling backend") **never prints**
for empty/letx → the crash is *inside* `load_multimodule_ir`, before
`compile_native_v2_preview_to_file` is ever called. The backend was not reached and is not
implicated. (An earlier draft guessed "backend impedance mismatch" — **falsified** by the
marker build; the back-half stays unblamed and, so far, unexercised on real source.)

*Marker method validated against the documented "stdout buffering hides success" gotcha
([[project_modular_native_backend_alive]]): on this same binary/redirect, prints survive a
subsequent SIGSEGV — e.g. `--check` runs that exit rc=139 still leave "Type checking
module 0" in the log. So an executed marker would have appeared; its absence means the
line was never reached, not that it was buffered away.*

### Gap 1 — value-returning fns: blocked by the spurious-return bug (the census's #1)
`ret13`/`arith` fail with **"Type check failed during IR lowering for module 0"** carrying
the exact **E008 "expected `()` / found i64"** spurious return-type error. `load_multimodule_ir`
runs check internally, so the same `current_return_type → TyUnit` bridge-state bug that
emits 105/105 spurious E008 across the corpus (see
`MODULAR_CORPUS_CRASH_CENSUS_2026-06-01.md`) **also blocks real compilation of any
value-returning function.** Fixing the bridge-state propagation (census lever #1) unblocks
this — it is not a separate bug.

### Gap 2 — void fns: the FULL `load_multimodule_ir` crashes (not the backend)
`empty`/`letx` pass the count-only probe variant (`load_multimodule_ir_fn_count_traced` →
functions=1) but the **full `load_multimodule_ir`** (which assembles and returns the
complete `MultiModuleIrResult`/`IrModule`) crashes (rc=139) before returning. The delta
between the two variants is the full-module materialization/return path — a plausible
instance of the bin/souc large-struct value-move/SRET miscompile family (cf.
`project_g1_codegen_largestruct_mut`), surfacing here on IrModule return. This is a
**lowering / front-half-adjacent** crash, NOT a back-half issue.

## Takeaway
The bridge **mode is wired** (`load_multimodule_ir` → `compile_native_v2_preview_to_file`),
but the native-v2 back-half **cannot yet be reached on real source** — both blockers are
upstream:
1. **Bridge-state propagation** (census lever #1) → unblocks value-returning fns. G1 lane
   (front-half, check.sio:1146 / return at 2489).
2. **Full `load_multimodule_ir` IrModule-return crash** on even trivial void fns → likely
   the large-struct/SRET miscompile family. Lowering/front-half territory.

So source→ELF remains gated on the front-half/lowering, exactly as `--check`/`--emit-obj`
already implied — the prototype's value is **naming the two specific upstream blockers and
proving the backend is not among them.** The 9-witness back-half stays the regression net
to extend once real modules actually reach it.

Prototype + patch are durable here; the mode is not committed to any branch (lives on the
detached snapshot). Route the patch to the g1 lane when wiring the real bridge.
