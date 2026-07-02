<!-- docs:meta
topic_id: repo.docs.audit.madaros-ir-max-instrs-1024-truncation-2026-06-28
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-ir-max-instrs-1024-truncation-2026-06-28
-->

# Madaros forensic dispatch — `IR_MAX_INSTRS = 1024` SILENTLY truncates long function bodies

Date: 2026-06-28
Branch: `recover/solver-gpu-arc` (madaros rebuilt `e274d1e8`, `test_smt` 6/6 green)
Class: **SILENT MISCOMPILE** (no `rc`, no diagnostic, binary writes + runs + exits 0)
Status: root-caused + reproduced; loud-or-grow fix proposed, NOT yet applied (this is a §8 precursor)

> This is the SEPARATE bug flagged in the "Related finding" of
> `docs/audit/MADAROS_RC13_ELF_256K_CAP_2026-06-28.md`. It is **orthogonal** to the
> `rc=13` 256 KiB ELF cap: that one is a LOUD capacity ceiling; this one is a silent
> per-function instruction-count truncation that produces a binary that lies.

## Symptom

A single Sounio function whose lowered body exceeds **1024 IR instructions** is
**silently truncated at exactly 1024 instructions**. There is no error, no non-zero
return code, no stderr. The compiler emits a complete, runnable ELF whose truncated
function simply *stops* at the 1024th instruction — every instruction past the cap
(including the function's tail logic) is dropped on the floor.

For an epistemic language this is the worst possible failure class: the toolchain
asserts success and hands back a binary that does **less than the source says**, with
nothing to distinguish it from a correct compile.

## Reproduction (deterministic, offline, on this branch)

Take a `use theorem::smt::*` harness whose `main()` is **unrolled / straight-line**
(no helper functions) so that all clause-construction lands in one function body.
Generate three variants with 800, 1500, and 2500 inline clause blocks:

```
800-block main  -> ELF 167944 bytes,  sha256 = A
1500-block main -> ELF 167944 bytes,  sha256 = B   (A != B)
2500-block main -> ELF 167944 bytes,  sha256 = C   (B != C)
```

All three ELFs are **byte-distinct** (different embedded clause *data*) yet **identical
in size** (167944 B) — the dead giveaway that the *code* amount is pinned, not the data.
Tracing confirms the pin:

```bash
SOUNIO_NV2_IR_TRACE=1 ./bin/souc compile <harness>.sio -o /tmp/x.elf 2>&1 | grep 'name=main'
# => name=main instr_count=1024     (for 800, 1500 AND 2500 blocks — invariant)
```

`instr_count` saturates at exactly `1024 = IR_MAX_INSTRS` regardless of source size.
Running the truncated binary:

```bash
/tmp/x.elf ; echo "rc=$?"
# => (no output)   rc=0
```

It exits 0 and prints nothing — it never reaches the tail `smt_solve` / `println`
that live past instruction 1024. **Success is reported; the work was discarded.**

## Root cause

`IR_MAX_INSTRS` is a hard, fixed-size per-function ceiling, and the emit path that
hits it drops instructions instead of failing loud.

- `self-hosted/ir/ir.sio:13` — `pub let IR_MAX_INSTRS: i64 = 1024`
- `self-hosted/ir/ir.sio:711` — `pub instrs: [IrInstr; 1024]` (the fixed `IrFunction` slot array)
- `self-hosted/ir/ir.sio:3478` — `instrs: [ir_nop(); 1024]` (its zero-init in `ir_empty_function`)

The central lowering emit helper enforces the cap by **silently refusing to append**:

`self-hosted/ir/lower.sio:3711` `fn emit(self, instr: IrInstr) -> Lowerer`:

```
let instr_count = (*current_func_box).instr_count
if instr_count < IR_MAX_INSTRS {
    (*current_func_box).instrs[instr_count as usize] = instr   // append
    (*current_func_box).instr_count = instr_count + 1
    ...
} else {
    if lo.probe_mode {                                         // <-- diagnostic ONLY in probe_mode
        print("lower_probe: emit_overflow instr_count=") ...
    }
    lo.report_error()                                          // sets had_error, drops the instr
}
```

What makes it silent has two parts — one proven, one still open:

1. **The overflow diagnostic is gated behind `probe_mode`** (`lower.sio:3731`), which is
   off in a normal compile — so nothing is ever printed. (PROVEN.)
2. **`report_error()` (`lower.sio:3249`) only sets `lo.error_count += 1` /
   `lo.had_error = true`**; it does not abort. Whether that `had_error` reaches an
   output-rejecting gate on the repro's compile path is **NOT pinned by this dispatch**
   (see the forensic note below).

The same fixed-`1024` bound is reasserted at every other appender — `lower.sio:1768`,
`:3725`, `:4957`, `opt_cleanup.sio:289/295`, `tailcall.sio:732/742/763/826/853/885` —
all of which no-op past the cap.

### Forensic note — the `had_error` propagation gap is UNPINNED (verify before fixing)

There is a genuine, unresolved contradiction the implementer must settle on a **fresh
madaros build** before deciding where to add the loud reject:

- `had_error` gating *does* exist on multiple lowering paths — `module_loader.sio:1115`
  and `:1331-1337` gate per-module on `lower_result.had_error`, and the single-module
  path `main.sio:978` gates too. `IrLowerResult.had_error` is propagated from the
  lowerer (e.g. `lower.sio:8895`).
- Yet the repro **still produced a silent `rc=0`, runnable, truncated binary.**

So one of these must hold, and it is not yet established which: (a) the truncation does
**not** set `had_error` on the repro's actual compile path (e.g. the `theorem::smt main`
is lowered via a route — imported/streamed body, `module_loader.sio:1111`
`lower_program_to_ir_trace` — whose `had_error` is not threaded into the gates above),
or (b) that path is simply **not one of the gated ones**. Caveat: the working tree shows
`self-hosted/ir/ir.sio` and `self-hosted/ir/lower.sio` as **modified (`M`)**, but the
empirical symptom came from madaros **e274d1e8** — the `had_error` logic being read here
may differ from the binary that produced the symptom. **Pin (a) vs (b) against a freshly
rebuilt madaros before choosing the fix site.** The cap facts (`ir.sio:13/711/3478`) and
the truncating drop (`lower.sio:3725`) are stable and build-independent; only the
"why didn't `had_error` halt the build" mechanism is open.

## Why the DIMACS corpus generator dodges this (and why that's not a fix)

`scripts/research/generate_sounio_dimacs_harness.py` splits clauses into 96-clause
`add_block_*` helper functions, so **no single function body** approaches 1024 IR
instructions. The truncation is a *per-function* cap, so structurally-fanned-out code
never trips it. That is luck of code shape, not a guarantee: any hand-written or
differently-generated harness with a fat straight-line body silently miscompiles.

## Proposed fix — **loud-or-grow, NEVER silent truncation**

Pick exactly one of the two; both eliminate silent truncation. Do **not** ship a
diagnostic that only fires in `probe_mode`.

### Option A — LOUD reject (minimal, ship-now)

Make overflow an unconditional, always-printed hard error and propagate it so the
binary is **not written**:

1. `lower.sio:3730-3738` — emit the `emit_overflow` diagnostic **unconditionally**
   (drop the `if lo.probe_mode` guard around it), naming the function and the cap, e.g.
   `error: function '<name>' exceeds IR_MAX_INSTRS=1024 (needs >N instructions); refusing to emit truncated code`.
2. First settle the forensic note above (which path drops the signal), then make
   `had_error` fatal on **that** path — i.e. wherever the repro's `main` is lowered, the
   truncation's `had_error` must thread into an output-rejecting gate (mirroring the
   existing gates at `module_loader.sio:1115/1331-1337` and `main.sio:978`) so the build
   **aborts with a non-zero rc instead of emitting**. A LOUD `rc` (like the `rc=13` ELF
   cap) is acceptable; a silent truncated binary is not. Do not guess the file — the
   "never consults `had_error`" assumption is explicitly disproven; confirm empirically.

### Option B — GROW the cap with a loud overflow guard (capacity + safety)

Raise `IR_MAX_INSTRS` (e.g. to `16384`, matching `HLIR_MAX_INSTRS` at
`self-hosted/hlir/ir.sio:18`) **and keep Option A's loud reject as the backstop** so the
new, larger cap can still never truncate silently:

1. `ir.sio:13` `IR_MAX_INSTRS` → larger value.
2. `ir.sio:711` `instrs: [IrInstr; 1024]` and `ir.sio:3478` `[ir_nop(); 1024]` → same
   value (the slot array and its zero-init MUST move together, or `ir_empty_function`
   under-fills the slot).
3. Audit every consumer that hardcodes `1024`/`IR_MAX_INSTRS` as a bound
   (`codegen_x86_linux.sio:6436/6482/9619`, `opt_cleanup.sio:289/295`, `tailcall.sio`
   sites) — they must read `IR_MAX_INSTRS`, not a stale literal, or a wider IR will be
   re-truncated downstream.
4. Retain the **unconditional loud reject** from Option A for any body that still
   exceeds the grown cap. Growing without the loud backstop just moves the silent cliff.

> Mandatory invariant either way: **a function body larger than the cap MUST cause a
> visible diagnostic and a non-zero rc — never a written-and-runnable truncated binary.**

## Verification gate (after the fix, build-locked)

1. Rebuild madaros: `bash scripts/ci/build_modular_madaros.sh artifacts/self-hosted/madaros`.
2. No-regression: `./scripts/run_sio_test_suite.sh test_smt --verbose --jobs 1` → **6/6**.
3. Repro is now LOUD (Option A) or COMPILES CORRECTLY (Option B):
   - Option A: the 1500-block straight-line `main` harness fails to compile with a
     printed `IR_MAX_INSTRS` overflow error and a non-zero rc — no ELF written.
   - Option B: the 1500-block harness compiles AND `instr_count` (under
     `SOUNIO_NV2_IR_TRACE=1`) tracks source size past 1024, AND the binary runs to its
     real tail logic (prints its `smt_solve` / `println` result), not silent `rc=0`.
4. Add a permanent gate: a `tests/` harness whose single function body exceeds
   `IR_MAX_INSTRS`, asserting either a loud overflow rc (A) or a correct full-body run
   (B) — so silent truncation can never silently return.

## Cross-references

- Sibling LOUD dispatch: `docs/audit/MADAROS_RC13_ELF_256K_CAP_2026-06-28.md`
  (256 KiB ELF staging buffer — separate, loud, capacity ceiling).
- This finding was first noted as the "Related finding" tail of that dispatch and is
  promoted here to its own forensic record.
