<!-- docs:meta
topic_id: repo.docs.audit.madaros-hardening-plan-2026-08-12
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-hardening-plan-2026-08-12
-->

# Madaros hardening plan — one root cause, five lanes (2026-08-12)

**Status:** PROPOSED. No compiler-owned file is changed by this document.
**Measured on:** `claude/sounio-status-report-am0myn` at `845f8bef`, using the
checked prebuilt `bin/madaros-linux-x86_64` (`Madaros v0.80.0`).

---

## Status update 2026-08-13

**Lane A:** landed on `main` via #1725 (honest exit status + `MADAROS_STACK_KB` + measure script).

**Lane B — instruction storage (B3-instrs):** DONE in source. `IrFunction.region: IrInstrRegion`
and `IR_MAX_INSTRS = 16384` are on `main`. A current-source Madaros rebuild compiles and
runs `tests/run-pass/knowledge_octonion_structure.sio` (14 389 IR ops) to `PASS`. The
checked-in `bin/madaros-linux-x86_64` can lag (still reports 4096 until rebuilt/shipped).

**Lane B residual (stack floor):** with current-source Madaros, minimal hello still needs
**64 MiB** stack (fails at 8/16/32). `IrModule.functions: [IrFunction; 8192]` remains
inline (~11 MiB) and codegen still has multi-10 MiB frames.

**B-nc peel (branch `fix/lane-b-nc-codebuffer-peel-20260813`, rebuilt):**
- `CodeBuffer.bytes` shrunk 262144 → 64 (vestigial; payload is `NC_BIG_CODE`)
- `trampoline_offset` takes `&NativeCompiler` (was by-value full NC)
- by-value `emit_entry_trampoline` is a thin wrapper over `_into`; `wide_driver` calls into

**Measured on current-source peel binary** (`artifacts/self-hosted/madaros-peel3`):

| Metric | pre-peel | post-peel |
|---|---:|---:|
| ELF frames ≥1 MiB | 926 | 613 |
| max `sub rsp` frame | 128.07 MiB | **54.32 MiB** |
| hello stack floor | 64 MiB | **64 MiB** (unchanged) |
| helper (+add3) floor | 128 MiB | **128 MiB** (unchanged) |

The vestigial CodeBuffer tax is gone from the binary, but the hello path still
hits a ≥32 MiB live frame (largest remaining is ~54 MiB). Next peel:
functions table as arena handles (B3-functions) and residual by-value
`NativeCompiler` on `emit_builtin_*` / `persist_builtin_emit_into`, plus
`MachineModule`/`MachineFunction` arena (still ~1.3 MiB per function of
inline MIR).

Re-measure: `MADAROS_RAW_BIN=artifacts/self-hosted/madaros bash scripts/dev/measure_madaros_stack_floor.sh`


## TL;DR

The shipped Madaros compiler **cannot compile a one-function program under the
default 8 MB Linux stack**. It needs somewhere between 32 MB and 64 MB for
`fn main() with IO { print("E\n") }`, and between 64 MB and 128 MB once a
three-argument helper is added. `bin/madaros:67` hides this with
`ulimit -s unlimited`, so the defect is invisible through the launcher and
fatal through any other entry point.

That is not a tuning problem. It is the visible edge of one architectural
decision — **the IR is a stack-resident value type of gigabyte scale** — and
most of the open compiler backlog is downstream of it. The plan below fixes the
root, then closes the symptom classes in a dependency order that makes each
phase measurable on its own.

---

## 1. The measurement

Every number here is re-runnable. Re-derive the stack floor with:

```bash
bash scripts/dev/measure_madaros_stack_floor.sh --reps 3
```

Observed (3/3 reps per cell, deterministic in both directions):

| Program | 8 MB | 16 MB | 32 MB | 64 MB | 128 MB | 256 MB |
|---|---|---|---|---|---|---|
| `fn main() with IO { print("E\n") }` | SIGSEGV | SIGSEGV | SIGSEGV | ok | ok | ok |
| the same plus `fn add3(a,b,c) -> i64` | SIGSEGV | SIGSEGV | SIGSEGV | SIGSEGV | ok | ok |

Supporting measurements, same tip, same binary:

| Quantity | Measured | Command |
|---|---:|---|
| Peak RSS to compile the 4-line program | **756 MB** | `python3 -c` around `resource.getrusage(RUSAGE_CHILDREN)` |
| Default launcher virtual-memory guard | **32 GB** | `grep -n MADAROS_VMEM_LIMIT_KB bin/madaros` |
| `IrModule.functions` | `[IrFunction; 8192]` inline | `self-hosted/ir/ir.sio:2166` |
| `IrFunction.instrs` | `[IrInstr; 4096]` inline | `self-hosted/ir/ir.sio:751` |
| `IrModule` passed by value / returned | 59 / 74 sites | `grep -rn ": IrModule\b" self-hosted --include=*.sio` |
| `IrFunction` passed by value / returned | 164 / 110 sites | `grep -rn ": IrFunction\b" self-hosted --include=*.sio` |
| `madaros --self-test` | **rc=139**, dies at T24 | `./bin/madaros --self-test; echo $?` |
| …of the 24 that run | **22 OK, 6 FAIL** | same |
| suite size | disputed: 1162 (#1680) vs 1156 (`epistemic_egraph_rewrite_gate.sh:11`) — the summary line that would settle it is past the crash | — |
| `instr_arena.sio` (the intended fix, #1649) | 403 lines, **referenced by no other file** | `grep -rn instr_arena self-hosted --include=*.sio -l` |
| `arena_v2_shadow.sio` | 307 lines, likewise unwired | same |
| CI gate scripts in `scripts/ci/` | 510 | `ls scripts/ci/*.sh \| wc -l` |
| Gates asserting on the `compiler/main.sio` suite result | **0** | `grep -rn self-test scripts/ci/*.sh Makefile .github/workflows/*.yml` |

Two further facts that matter for the plan, both read from source rather than
measured:

- `bin/madaros:521` `_compile_source_to_artifact` **discards the compiler's
  exit status entirely**. It decides success by `[[ ! -s "$out" ]]` — file
  non-empty. A compiler that segfaults after writing a partial ELF is reported
  as a successful build.
- `bin/madaros:67` `ulimit -s unlimited` is what makes the launcher path work.
  Remove it and `bin/souc run hello.sio` stops working.

---

## 2. The diagnosis

`IrModule` embeds `[IrFunction; 8192]`, each embedding `[IrInstr; 4096]`
inline. The compiler then passes `IrFunction` by value at 164 sites and returns
it at 110, and does the same for `IrModule` at 59/74 — because the optimiser
depends on those copies being deep (`var result = func; result.instrs[i] = …;
return (false, func)` as rollback). Every one of those boundaries is a
multi-megabyte stack copy.

That single decision explains, without further hypotheses:

| Open issue | Symptom | Why it is the same root |
|---|---|---|
| #1680 | `--self-test` SIGSEGV at T24 of ~1160 | T23 keeps **one** `IrModule` live and passes; T24 is the first test needing three |
| #1649 | `IR_MAX_INSTRS = 4096` too low; a 136-line program needs 14 389 | the limit cannot be raised — it multiplies through 8192 functions |
| #1646 | three more silent capacity walls (driver labels, `return_jumps`, float typing) | same fixed-array-with-no-overflow-check pattern |
| #1686 | self-compile blocker: seed aliases a loop counter with `IR_MAX_FUNCS` | the constant is load-bearing for layout, not just a bound |
| #1693 | layout-sensitive segfault; adding dead code fixes it | stack-frame arithmetic near the cliff |
| #1692 | a one-line file with a module-level global segfaults under `-O` | an extra pass, an extra live module |
| #1570, #1586 | codegen cliff / silent output truncation in a large `main` | the wall reached mid-function, with no diagnostic |
| #1658 | native runtime exhausts temporaries on long trajectories | same class in the emitted code |
| D3 residuals | multi-module "memory wall" | two modules live at once |

The compiler is not fragile in nine independent ways. It is fragile in one way,
observed from nine directions.

There is a second, compounding fault that is *not* the same root, and it is why
the first one survived this long: **the instruments lie**. The `compiler/main.sio`
suite exits 139 whatever happens, so its own `if passed == total { 0 } else { 1 }`
is unreachable and its `N/M passed` summary never prints. One gate does invoke it
— `scripts/ci/epistemic_egraph_rewrite_gate.sh:98`, as
`timeout 60 "$MADAROS" --self-test || true` — but it tolerates the crash by
construction and greps for three named tests, recording `NOT_RUN` when they are
not reached. No gate asserts on the summary line (#1680). The launcher throws away
the compiler's exit code. 14 tests assert a marker neither engine emits (#1706).
Compile-fail tests accept a SIGSEGV as a pass (#444). Typecheck errors in
imported modules are non-fatal and still emit code (#1494). Unknown SOIR opcodes
deserialise to `IrNop` (#878). Every one of those converts a loud failure into a
green run.

And a third, which sets the ceiling on how good Madaros can get: **Madaros is
built by a compiler with known silent miscompiles**. `lean_single` is still the
bootstrap seed, and it carries #1678 (`&(*boxed).array[i]` is a wrong address —
"it miscompiles the compiler"), #1655 (storing a struct into a global array is a
no-op), #1574 (literal-initialised global arrays always read element 0), #1644,
#1610. Meanwhile #725 records that the `stage0 → boot4` C chain no longer
reproduces the fixed point at all: the seed is a frozen binary.

---

## 3. The plan

Five lanes. Lane A is a precondition for honest measurement of everything else.
Lane B is the root fix. C, D and E depend on B in the order shown.

### Lane A — stop the instruments from lying (days, not weeks)

Nothing here touches codegen. It is what makes the rest of the plan measurable.

- **A1.** `_compile_source_to_artifact` and every sibling in `bin/madaros`
  propagate the raw compiler's exit status. A SIGSEGV that wrote bytes is a
  failure. *Acceptance:* a deliberately-crashing compile reports non-zero
  through `bin/souc`.
- **A2.** Replace the blanket `ulimit -s unlimited` with an explicit, named,
  measured reservation, and emit a diagnostic — not a signal — when the
  compiler exhausts it. *Acceptance:* a program that overruns the reservation
  prints `error: compiler stack exhausted while lowering <fn>` and exits 1.
  Keep the reservation until B lands; A2 makes it visible, B makes it
  unnecessary.
- **A3.** Wire `compiler/main.sio --self-test` into CI, asserting on the
  `N/M passed` summary line, never on the exit code. *Acceptance:* the gate
  fails today (it must — 6 of 24 are red), and its expected N/M is committed as
  a baseline that can only move up.
- **A4.** Every fixed-size capacity in the IR and native backend gets an
  overflow check that produces a diagnostic. No silent truncation anywhere.
  Sweep from #1646 and #1649. *Acceptance:* a generated program that exceeds
  each limit produces a named error, and a test asserts each one.
- **A5.** Vacuous-test detector: a test whose expected marker no engine can
  emit fails the harness (#1706), and a compile-fail test must assert an error
  pattern rather than accepting any non-zero death (#444).

### Lane B — the IR arena (the root fix)

This is the bold part, and it is already half-designed in-tree: `instr_arena.sio`
(403 lines: variable-size regions, generation-checked handles, sealing) and
`arena_v2_shadow.sio` (307 lines) exist and are wired to nothing. The previous
attempt (`probe/ir-soa-phase0`, #1649) moved the self-test crash *earlier*, to
T08 — which is the expected outcome of flipping 274 by-value boundaries at once.

Sequence it so that it cannot fail silently:

- **B1. Shadow.** Run the arena alongside the inline arrays. Every pass writes
  both; a checker compares them after each pass and aborts loudly on
  divergence. Slow and memory-hungry on purpose. *Acceptance:* the full test
  corpus compiles with shadow enabled and reports zero divergences.
- **B2. Seal the boundaries.** Sounio has no copy constructor, so each of the
  274 by-value sites must be classified by hand as *deep copy needed* or
  *handle share is correct*. Sealing (write through a sealed handle is refused)
  turns a missed classification into a loud refusal instead of a cross-function
  miscompile. *Acceptance:* every by-value site is annotated, and the count in
  §1 is reproduced by a gate so new ones cannot appear unclassified.
- **B3. Flip.** `IrFunction.instrs` and `IrModule.functions` become handles.
  Delete the inline arrays. *Acceptance:* the §1 table is re-measured and both
  programs compile at **8 MB**, 10/10; peak RSS for the 4-line program is under
  50 MB, down from 756 MB.
- **B4. Raise the walls.** With storage proportional to use, `IR_MAX_INSTRS`
  stops being a memory-multiplying constant. Close #1649 with the measured
  14 389-instruction program, and #1686 with the aliasing fix it blocks.

### Lane C — seed independence: Madaros builds Madaros

Depends on B (the self-compile blockers are memory and aliasing).

- **C1.** Swap the build seed from `lean_single` to Madaros.
- **C2.** Re-establish the fixed point over `main.sio`: gen2 == gen3
  bit-identical. Note the current fixed point is over `lean_single.sio` only —
  Madaros itself has never been fixed-point verified, and CLAUDE.md is already
  explicit about not claiming otherwise.
- **C3.** Repair the `stage0 → boot4` C chain (#725) so the seed is derivable
  from source rather than being a frozen binary. Until then the toolchain has
  no reproducible origin.

### Lane D — kill silent wrongness

Partly parallel with B; the differential harness is most valuable *during* B.

- **D1.** Differential harness: every `tests/run-pass` program compiled and run
  under `{-O0, -O} × {lean_single, Madaros}`, outputs compared. This is #1667's
  acceptance criterion 3, generalised — and it is the only instrument that
  would have caught `add3(1,2,3) == 3` at rc=0.
- **D2.** Land PR #1705 (propagation across basic-block boundaries) against
  #1667, naming the responsible pass as that issue requires.
- **D3.** A randomised program generator feeding D1. Wrong-answer-at-rc-0 is
  the defect class this repository keeps rediscovering by hand.
- **D4.** Make silence impossible at the boundaries already known: fatal
  typecheck errors in imported modules (#1494), unknown SOIR opcode is an error
  not `IrNop` (#878), fully-qualified call without `use` is an error not a
  no-op (#1568).

### Lane E — retire the divergence

Depends on C and D.

- **E1.** 63 `tests/run-pass` files are pinned to `lean_single`. Each pin is a
  Madaros defect; each gets an issue. *Acceptance:* the pin count is a gated
  number that can only decrease.
- **E2.** When it reaches zero, `lean_single` leaves the user-facing surface
  entirely and survives only as the historical bootstrap record.

---

## 4. What "working perfectly" means

Proposed acceptance matrix. These are the numbers that decide the claim; none
of them is currently green.

| # | Criterion | Today | Target |
|---|---|---|---|
| 1 | Compiles a one-function program at the default 8 MB stack | SIGSEGV | ok, 10/10 |
| 2 | Peak RSS, 4-line program | 756 MB | < 50 MB |
| 3 | `--self-test` executes and reports | 24 reached, rc=139 | every registered test runs or is counted as skipped; rc ∈ {0,1} |
| 4 | Fixed capacity reached | silent truncation | named diagnostic, rc≠0 |
| 5 | `-O` vs `-O0` over the corpus | not compared | identical, gated |
| 6 | Compiler exit status through `bin/souc` | discarded | propagated |
| 7 | Madaros builds Madaros, gen2 == gen3 | not attempted | verified |
| 8 | Seed derivable from `stage0` C chain | frozen binary (#725) | reproducible |
| 9 | Tests pinned to `lean_single` | 63 | 0 |

Criterion 1 is the one to lead with. A compiler that cannot compile
hello-world without its wrapper script raising the stack limit is not a
compiler the world can use, however good the science on top of it is.

---

## 5. Sequencing

```
A (days)  ───────────────────────────────────────────────►  gates honest
             │
             ├── B1 shadow ── B2 seal ── B3 flip ── B4 walls  (the root)
             │                              │
             │                              ├── C1 ── C2 ── C3   (seed)
             │                              │
             └── D1 ── D2/D3/D4 ────────────┴── E1 ── E2         (divergence)
```

A must land first: without A1 and A3 there is no instrument that can tell
whether B worked. D1 should land early and run continuously *through* B — a
differential harness is exactly what makes a 274-boundary refactor survivable.

Concurrency discipline from CLAUDE.md applies throughout: full self-compiles go
through `scripts/dev/souc-build-lock.sh`, one worktree per agent, ≤2 agents
doing compiler work on the pod at once.

---

## 6. What this plan deliberately does not do

- It does not touch `stdlib/`, the science lanes, or the dissertation path. The
  epistemic trust gate stays as the guard on what stdlib results survive native
  import.
- It does not add a new backend, target, or language feature. Every lane
  removes a failure mode; none adds surface.
- It does not propose collapsing the 510 CI gate scripts, though the fact that
  510 gates coexist with a compiler that segfaults on hello-world is worth its
  own dispatch. Gate count is not gate coverage.

## 7. The main risk

Lane B is a 274-boundary refactor in a language with no copy constructor, on a
compiler currently built by a *different* compiler with known silent
miscompiles (#1678 miscompiles the compiler itself). If B is attempted before
A3 and D1 are green, a miscompile introduced during B is indistinguishable from
a miscompile inherited from the seed. That ordering is not a preference; it is
the difference between a refactor that converges and one that cannot be
debugged.

---

## Appendix — reproduce the headline in 30 seconds

```bash
printf 'fn main() with IO { print("E\\n") }\n' > /tmp/hello.sio

# through the launcher, which raises the stack limit for you
./bin/souc run /tmp/hello.sio                      # E

# the same compiler, invoked the way any other tool would invoke it
./bin/madaros-linux-x86_64 /tmp/hello.sio -o /tmp/hello.elf
echo "rc=$?"                                       # rc=139
```
