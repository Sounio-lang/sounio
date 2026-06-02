# Struct-return crasher — fix ATTEMPT (does not reproduce in isolation) — 2026-06-02

Goal: fix the dominant body-check crasher (131/170), which faults at 0x4c2805b
reading a 16-byte TypeEntry from rdx=−1, on the modular repro
`fn f(x:i64)->i64{x} fn main()->i64{let y=f(5) 0}` → mc_fixed --check SIGSEGV.

## Method (same repro-driven approach that fixed the nested-store bug)

Built progressively faithful BOOTSTRAP-level repros of the hypothesised mechanism
(`(*c).fn_sigs.get(id)` returns FnSig by value → `fn_param_list_get(sig.params,idx)`
reads a param TypeEntry), all compiled with ds_fixed2 (the SAME fixed bootstrap that
produced the crashing mc_fixed). Sources in repro/sret_norepro_attempts/.

| repro | models | result on ds_fixed2 |
|-------|--------|---------------------|
| sret_repro | small struct w/ Box, plain fn return-by-value | r=77 OK |
| A | large struct (Name-sized), plain fn, local self | r=77 OK |
| B | large struct, method, local self | r=77 OK |
| D | small struct, method, self via `(*c).tbl` | r=77 OK |
| E | scalar return `self.entries[i].n` via deref-self | n=5 OK |
| F | materialize `(*c).tbl` then method | r=77 OK |
| chain | FULL chain: get()→struct by value, pass `.params` Box to a recursive
          list-get returning a struct, read `.ty` | r=99 OK |

**Every repro passes.** (An earlier apparent reproduction was a red herring: it used
the UNFIXED bin/souc, where the *setup* `(*c).tbl.entries[i]=S{…}` is the nested-store
bug this branch already fixes — not the return bug.)

## Conclusion

The crasher does NOT reproduce at bootstrap-repro scale. Minimal and faithful models
of the hypothesised get()/fn_param_list_get struct-return mechanism all compile
correctly under ds_fixed2 — the exact compiler that emits the crashing mc_fixed. So
either the hypothesised mechanism is not the true cause, or (more likely) the crash
is a SCALE/context-dependent codegen fault (register pressure, huge frames, the 17k-
line check function, 64K-entry tables) that only manifests in the full check.sio —
NOT a cleanly-isolable semantic codegen bug. This matches the repo's prior verdict
(project_modular_B_repro_verdict: the modular-checker crash is "layout-sensitive,
non-monotonic, non-bisectable, intractable without gdb").

**Therefore the repro-driven fix method that worked for the nested-store bug does NOT
apply here.** A real fix needs gdb-level debugging of the full mc_fixed binary
(map 0x4c2805b to a check.sio function via instrumented build / careful disassembly),
or surgical source-bisection with 2:36 modular rebuilds per step — a substantially
larger undertaking than the nested-store fix. NOT attempted further here; flagged as
its own lane.

The nested-store codegen fix (this branch) remains correct and complete on its own
terms; this attempt does not change it.
