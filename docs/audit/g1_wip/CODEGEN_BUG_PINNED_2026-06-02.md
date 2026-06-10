<!-- docs:meta
topic_id: repo.docs.audit.g1-wip.codegen-bug-pinned-2026-06-02
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.g1-wip.codegen-bug-pinned-2026-06-02
-->

# The bin/souc codegen bug — PINNED (2026-06-02)

The layout-sensitive return-address-smash that blocks effect/ident validation (and is the
[[project_modular_span_sensitive_crash]] whitespace-sensitive crash family) is **pinned**, with
a verified root cause, a build-independent 1-second reproducer, and an exact source fix site.
Found via a 5-strategy adversarial gdb workflow (Strategy-1's "SRET rep-movsq overrun" theory was
REFUTED live by a write-watchpoint firing zero times; strategies 2/3/4/5 converged).

## Root cause: unsound "all-paths-return" check skips the fall-through epilogue

`lean_single.sio:24549` (top-level fn) and `:9307` (closure twin) gate the trailing
implicit-return epilogue on a **byte-peek**:

```sio
if CL > 0 && CD[(CL - 1) as usize] != 0xc3 as i8 { ...emit (xor eax,eax); add rsp; pop rbp; ret }
```

`0xc3` = `ret`. The intent is "if the function already ends in `ret` (all paths return), don't
emit a second epilogue." But it is **unsound**: a function whose LAST statement is
`if cond { …; return }` **with no else** ends with the THEN-branch's inline `return` (`0xc3`) —
yet the FALSE path falls through and needs the epilogue. The byte-peek sees `0xc3`, wrongly
concludes all-paths-return, and emits nothing. The if-false `je` (emitted at `:19430-19431`,
patched at `:19542` `patch32(je_off, CL)`) then targets `CL` = one byte past the then-`ret` =
the **prologue of the next emitted function**. Entered by `je` (not `call`), that function's own
`ret` pops uninitialised stack: `0` → `rip=0` (mc_W2), or a stale `-1` lookup-miss sentinel
snapshotted into an arg slot and dereferenced (mc_T). Layout-sensitive because it lands in
whatever function is physically adjacent.

## Build-independent reproducer (`repro/fallthrough_crash_min.sio`)

```sio
fn other() -> i64 { 12345 }
fn h(c: bool) with IO, Mut, Panic, Div {
    if c { print(1) return }          // LAST statement: if-then-return, no else, no tail
}
fn main() with IO, Mut, Panic, Div { h(false) print(999) }
```

`bin/souc` compiles it in ~1s; running `h(false)` → **rc=139** (the false path falls off the end).
Adding a tail expression after the `if` (so the if is not the last statement) → correct. This is
why the boundary contracts crashed exactly as instrumented: `unit_boundary`/`ontology_boundary`
END with `if cond { report; return }` (→ crash); `knowledge_boundary` ends with `if cond { report }`
(no return → last byte ≠ 0xc3 → fine). It also explains why my earlier struct-shaped reproducer
hunt missed it — the trigger is control-flow, not struct size.

## The fix (lean_single, then re-bootstrap with FULL validation)

The byte-peek must be replaced by a real "does the last statement diverge on all paths" signal.
Two options:
- **Minimal/low-risk:** in the if-statement **no-else** branch (`:19542`, right after
  `patch32(je_off, CL)`), emit a 1-byte `nop` (`em(0x90)`). Then `CD[CL-1] != 0xc3` after any
  if-no-else, so the guard always emits the fall-through epilogue. Correct: if-no-else can always
  fall through; if-WITH-else (all arms return) is untouched and still correctly skips. The nop is
  on the fall-through path (harmless) and costs one byte per if-no-else.
- **Proper:** a `LAST_STMT_DIVERGES` flag set true only by bare `return`/`panic` and
  if-else/match where every arm diverges, false for if-without-else; gate `:24549`/`:9307` on it.

Validate on a SCRATCH `gen1` (bin/souc compiles patched lean_single → gen1.elf; gen1 compiles the
reproducer → must not crash; gen1 compiles the 847 corpus + tests/run-pass → no regressions).
ONLY then re-bootstrap gen2/gen3 (fixed point) and replace bin/souc. **Never overwrite bin/souc***
until the full suite passes — that gate is exactly what the `/workspace/sounio`
`Epistemic::measured val=0` re-bootstrap skipped.
