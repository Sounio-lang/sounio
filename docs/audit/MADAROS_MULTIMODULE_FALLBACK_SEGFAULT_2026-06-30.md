<!-- docs:meta
topic_id: repo.docs.audit.madaros-multimodule-fallback-segfault-2026-06-30
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-multimodule-fallback-segfault-2026-06-30
-->

# Madaros multi-module native fallback: segfault in `lower_array: seed_begin`

**Date:** 2026-06-30
**Scope:** `bin/souc` (Madaros v0.80.0) native compilation of multi-module `.sio`
programs (any `use module::*` import), default `native-v2` engine.
**Status:** root-caused to two distinct, sequential bugs. First bug fixed by
rebuild (stale prebuilt binary). Second bug open, minimal repro below.

## Context

`stdlib/clinical/vancomycin_pbpk.sio` (`use epistemic::knightian::*`) and
`docs/audit/MADAROS_F64_NATIVE_V2_BUGREPORT.md` (this repo, 2026-06-30, commit
`9626d88a1`) independently reported that Madaros native-v2 fails on f64-heavy
and multi-module code, with `lean_single` (`SOUNIO_SOUC_ENGINE=lean_single` /
`bin/souc-lean-single-x86_64`) as the correct-output engine.

## Bug 1 (FIXED — stale prebuilt binary, not a source bug)

`bin/souc-native` (checked-in prebuilt) predates commit
`fix(madaros): fall back when compact imported IR emit fails`
(`self-hosted/compiler/module_native_driver.sio`, 2026-06-30T18:04:19Z) by
~8 hours. The prebuilt binary therefore never reaches the fallback path
(`module_frontend_compile_imported_to_file`) when the compact "imported
simple IR" table (`module_native_simple_driver.sio` — a bootstrap-stage
pattern-matcher over ~20 fixed function "kinds", not a general code
generator) fails to emit a function shape it doesn't recognize (which
includes essentially all real f64 arithmetic).

**Verified fix:** rebuilding via
`bash scripts/ci/build_modular_madaros.sh <out>` (seeded from
`bin/souc-lean-single-x86_64`, per the script's own seed-resolution order)
produces a binary that *does* reach the fallback — confirmed by the
`module_native_driver: compact IR load failed; falling back to full IR path`
diagnostic, which the stale binary never printed.

## Bug 2 (OPEN — segfault in the fallback's array/box lowering seed step)

The fallback path itself (`module_frontend_lower_programs_array_direct_box`,
`self-hosted/compiler/module_frontend.sio:4192`) segfaults on the **very
first** module it lowers, inside
`module_frontend_lower_program_items_box_traced_with_externs` — after the
`"lower_array: seed_begin\n"` trace print (line 4204) and before
`"lower_array: seed_done\n"` (line 4207). No further trace output is
produced; no panic/error message, just SIGSEGV.

### Minimal repro (2 files, deterministic)

```sounio
// lib.sio
pub fn add_one(x: f64) -> f64 { x + 1.0 }
```

```sounio
// main.sio
use lib::*
pub fn main() -> i32 with IO {
    println(add_one(4.0))
    0
}
```

```
$ ./madaros main.sio -o out.elf
...
imported_compile: lower_begin
lower_array: seed_begin
Segmentation fault
```

Reproduced with a freshly-rebuilt binary (today's `main`, post-Bug-1-fix), so
this is not a stale-binary artifact. The crash is upstream of any f64-specific
logic — even the trivial single-function `lib.sio` triggers it — so this
looks like a general defect in the imported/multi-module box-lowering seed
path, not specifically an f64 bug (though it blocks all multi-module f64
code, including `stdlib/clinical/vancomycin_pbpk.sio`).

### Suggested next step

Bisect inside `module_frontend_lower_program_items_box_traced_with_externs`
(`self-hosted/compiler/module_frontend.sio`) with additional trace prints, or
attach a debugger capable of reading the hand-emitted ELF (no DWARF is
emitted by the self-hosted backend, so `gdb` alone will not resolve symbols —
a print-statement bisection is the practical path used for Bug 1's sibling
report). Not attempted here due to unbounded scope within this session; flagged
for a dedicated follow-up.

## Practical resolution (unblocks the clinical digital-twin work now)

Until Bug 2 is fixed, multi-module and/or f64-heavy `.sio` programs —
including all of `stdlib/clinical/`, `stdlib/darwin_pbpk/`, and any program
importing `epistemic::knightian` — **must** be built with the `lean_single`
engine, not the default Madaros native-v2 path:

```bash
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run stdlib/clinical/vancomycin_pbpk.sio
# or, directly:
./bin/souc-lean-single-x86_64 stdlib/clinical/vancomycin_pbpk.sio /tmp/out.elf && /tmp/out.elf
```

This is not a workaround of last resort: `souc-lean-single-x86_64` is the
verified bootstrap seed compiler (it is literally what `scripts/ci/
build_modular_madaros.sh` uses to build Madaros itself), and both
`stdlib/clinical/vancomycin_pbpk.sio` and
`examples/dissertation_vancomycin_demo.sio` produce hand-verified-correct
numeric output under it (see `docs/audit/` PBPK session notes and this
report's companion verification below).

## Verification of the twin under `lean_single` (for the record)

`stdlib/clinical/vancomycin_pbpk.sio :: main()`, three scenarios:

| Scenario | Cmin band (mg/L) | Decision | Hand-check |
|---|---|---|---|
| Pre-TDM (0 samples), 78.5 kg, CrCl 65, 1000 mg q12h | [9.052178, 24.298861] | `PRE_REFUSE` | matches closed-form Fréchet corners (Vc_lo,CL_hi)/(Vc_hi,CL_lo) to 6 d.p. |
| Post-TDM (3 samples), same patient | [12.820636, 17.358234] | `POST_PRESCRIBE` | matches, band ⊂ [10,20] |
| Contract violation (weight 15 kg) | — | `CONTRACT_BLOCK` | correct refinement-gate refusal |
