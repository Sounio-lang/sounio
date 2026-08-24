<!-- docs:meta
topic_id: repo.docs.audit.handle-green-census-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: grok-cli3
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.handle-green-census-2026-08-19
-->

# Handle-expression greens — census, 2026-08-19

**Question:** how many tests and examples in the corpus contain a `handle`
*expression*, and what does the suite say about them today? In particular:
how many `tests/run-pass/` files use `handle`, pass under Madaros, and
therefore green a construct that #1993 showed is erased?

**Answer:** **zero.** There is no live algebraic-effect `handle` expression
in any versioned `.sio` file on the SHA measured. The perverse cell is
empty. The greens that *look* like handler tests either bind a variable
named `handle`, sit inside a comment, or were rewritten to `println`.

This receipt does not repeat #1993. #1993 established the execution
witness (`handle<IO>` type-checks on Madaros, emits an ELF, exits 0, and
runs nothing). This receipt asks how far that hole reaches into the
suite.

Measurement only. `self-hosted/` was not edited. No harness annotation
was changed.

```text
Semantic-Lane-ID: handle-green-census-20260819
Owner: grok-cli3
Concept-IDs: none created
Intent-Preserved: a green must not be a silent no-op of an erased construct
Transformation: none — census
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced:
  - 0 versioned .sio files contain a live handle expression on 92fade0be1
  - the perverse cell (run-pass + Madaros pass + handle expression) is 0
Claims-Forbidden:
  - this is not a claim that handlers work
  - this is not a claim that the suite has no handler-shaped greens by name
  - "14 of 17" style coverage numbers are not used here
Assumptions: comment-stripping matches Madaros `/* */` and `//`
Write-Set:
  docs/audit/HANDLE_GREEN_CENSUS_2026-08-19.md
  docs/audit/HANDLE_GREEN_CENSUS_2026-08-19.tsv
Read-Set: git ls-files '*.sio' at 92fade0be1
Positive-Witness: examples/effects.sio:46 and :54 (handle expressions,
  excluded because they sit in a block comment)
Negative-Witness: lower_handle_buf_arg_index; let handle = …
Acceptance-Gate: both controls shown; engines named; Slurm used for runs
Integration-Target: docs (audit)
Authoritative-Only-If: n/a — observational
```

## Instrument

| Field | Value |
|---|---|
| SHA | `92fade0be1` (census). `origin/main` later moved to `482051161c` (`#1987` `#1984` `#1985`); `.sio` delta is empty |
| Word scan | `git ls-files '*.sio'` then `\bhandle\b` |
| Classifier | comments and strings blanked; remaining token classified |
| Handle expression | `handle<…>`, `handle {`, `handle IDENT (` / `{` / `with` |
| Launch | `scripts/dev/slurm_srun_minimal.sh`, partition `cpu-ops`, node `cpuops-t560-proxmox` |
| `workspace_visible` | no |
| Staging | `/orangefs/training/handle-green-census-20260819` (stdin tar; login pod cannot read `/orangefs`) |
| Madaros | `bin/souc` default → `Madaros v0.80.0`; ELF sha256 `437bdd8f96a2…` (bit-identical to the committed `bin/madaros-linux-x86_64`) |
| lean_single | `SOUNIO_SOUC_ENGINE=lean_single` → `bin/souc-lean-single-x86_64` sha256 `337d5a86f44e…` |
| Stack | `ulimit -s 1048576`, `MADAROS_STACK_KB=524288` |
| Compile form | `souc compile <src> -o <elf>`; ELF magic `7f454c46` |
| #1993 fixtures | `tests/audit/handle_*.sio` are **not** on this SHA |

Companion TSV: `docs/audit/HANDLE_GREEN_CENSUS_2026-08-19.tsv`.

## 1. Census

94 versioned `.sio` files contain the word `handle`. 900 hits.

| class | hits | files | what it is |
|---|---:|---:|---|
| comment | 441 | 84 | `//` or `/* */` |
| ident | 371 | 13 | binding, parameter, or assignment (`let handle =`, `fn f(handle:`) |
| field | 59 | 5 | `.handle` / `handle.await` |
| string | 23 | 9 | inside `"…"` |
| other | 6 | 2 | `next.handles[i] = handle` — runtime handle **table**, `self-hosted/native/gc.sio` and its archive copy |
| **handle_expr** | **0** | **0** | algebraic-effect construct |

No row is `INDETERMINATE`. The classifier produced an empty expression
set, and the review of every `other` hit confirmed they are the GC table.

### Positive control — a handle expression, not counted

`examples/effects.sio` wraps its original body in a block comment
(`/*` at line 1, `*/` at line 66). Inside that comment sit two genuine
handle expressions:

```
46    handle divide(a, b) with {
54    let result = handle {
```

Those two lines are the shape this census exists to find. They are
**comments**. The live program below the closer is a stub that prints
`example: effects` and a checksum. It contains no `handle` token.

#1993's `handle<IO> { … }` programs are the same shape and are not on
this SHA.

### Negative control — the name traps

`self-hosted/ir/lower.sio:4810`:

```
fn lower_handle_buf_arg_index(n: Name) -> i64 with Div {
```

`\bhandle\b` does not match inside `lower_handle_buf_arg_index`
(underscore is a word character). The identifier is absent from the
900-hit table.

`tests/run-pass/closure_linear.sio:14`:

```
let handle = open_file(42)
```

Classified **ident**. So is `tests/run-pass/linear_correct_consume.sio:13`
(`let handle = FileHandle { fd: 42 }`) and
`tests/run-pass/async_spawn.sio:11` (`let handle = spawn {`).
`examples/showcase/linear_file_server.sio` uses `handle` as a field and
a parameter of a linear `FileHandle`. None of these is an effect
handler.

## 2. Suite state — files that could have been the perverse cell

No `tests/` file contains a handle expression, so the suite section for
the census set is empty. The files the dispatch named, plus every
`tests/run-pass/` file with a *live* `handle` token, were still run on
Slurm under **both** engines, so a reader can see what the suite
actually does with them.

| file | harness | live `handle` | **Madaros** | **lean_single** |
|---|---|---|---|---|
| `tests/run-pass/closure_linear.sio` | `//@ run-pass` **`//@ ignore`** | ident (`let handle =`) | check 0, ELF yes, run **0** (stdout `0`) | check **1** `linear value not consumed` |
| `tests/run-pass/linear_correct_consume.sio` | `//@ run-pass` | ident | check 0, ELF yes, run **0** | check 0, ELF yes, run **0** |
| `tests/run-pass/async_spawn.sio` | `//@ run-pass` | ident + field (`.await`) | check **1** `E012` no field `await` on `i32` | check 0, run **0**, `spawn_basic: PASS` |
| `tests/run-pass/handler_discharge.sio` | `//@ run-pass` `//@ expect-stdout: handler: PASS` `//@ native-pass` | **none** (comment only) | check 0, run **0**, stdout `handler: PASS` | check 0, run **0**, stdout `handler: PASS` |

`closure_linear.sio` is **ignored** by the harness (`scripts/dev/run_sio_test_suite.sh`
skips `//@ ignore`). A direct Madaros run still exits 0. That green is
about a linear file descriptor, not about effect handlers.

`handler_discharge.sio` is the cousin, not the cell. Line 4 says it was
"Rewritten to pass lean_single (`handle<IO>` not supported; prints
directly)". The body is `println("handler: PASS")`. Both engines print
that string and exit 0. The suite records a handler test as green
because the construct was deleted.

## 3. The perverse cell

**Definition:** `tests/run-pass/` + contains a handle expression +
passes under Madaros.

**Count: 0.**

There is no run-pass file whose algebraic-effect half is being silently
erased, because there is no run-pass file that still writes that half.
The hole #1993 measured is real. It is not currently laundered through
a run-pass green.

What *is* laundered is a **name**: `handler_discharge` is green on both
engines after the construct was replaced by a print. Reclassifying that
file is a founder decision. This receipt does not touch the annotation.

## 4. Examples and documentation claims

| file | live `handle` | **Madaros** | **lean_single** |
|---|---|---|---|
| `examples/effects.sio` | none (expressions are in `/* */`) | check 0, run 0, `example: effects` / `36` | check **1** — seed reports `E200` `` `handle` `` at the **commented** lines 46 and 54 (it does not skip the `/* */` body the way Madaros does) |
| `examples/effects/basic_handler_continuation.sio` | none | check 0, run 0, `Hello from handler!` | check 0, run 0, same print |
| `examples/effects/comprehensive_effects.sio` | none | check 0, run 0, banner `All tests completed successfully!` | check 0, run 0, same banner |
| `examples/showcase/effect_test_harness.sio` | none | check **1**, parse failure | check 0, compile 0, run **139** (SEGV) after printing the harness header |
| `examples/showcase/linear_file_server.sio` | ident / field (linear `FileHandle`) | check 0, run 0 | check 0, run 0 |

`basic_handler_continuation.sio` and `comprehensive_effects.sio` carry
`//@ run-pass` and comments that say they test effect handlers and
continuations. Their bodies are `println`. They contain no `handle`
token. Their greens are not the perverse cell. They are printlns
wearing a handler's name.

### Where documentation still points

| document | claim | against this measurement |
|---|---|---|
| `docs/spec/S07_EFFECT_HANDLERS.md:78–82` | Do not describe Sounio as having algebraic effect handlers. Do not cite `examples/effects.sio` or `examples/effects/*.sio` as evidence that handlers run. | **Agrees.** Those files do not contain a live handle expression. |
| `examples/showcase/README.md:65–68` | `effect_test_harness.sio` "Shows how algebraic effects decouple business logic from I/O." Features: "algebraic effects, captured log…" | **Contradicted.** The file's own header says it *simulates* the pattern with structs. Madaros cannot parse it. lean_single SEGV 139. |
| `docs/architecture/EFFECT_HANDLERS_IMPLEMENTATION.md:305–325` | `comprehensive_effects.sio` is an end-to-end handler test, "All passing with JIT interpreter", invoked via `cargo run --bin souc --features jit`. | **Stale.** No `handle` token. The success banner is printed by `println`. `cargo` / JIT is not the current clock. |

## What this is not

- Not a patch to `self-hosted/`.
- Not a reclassification of `handler_discharge` or any `//@ ignore`.
- Not a repeat of #1993's `handle<IO>` / `handle<NotARealEffect>` matrix.
- Not a claim that the construct is Reserved. lean_single's `E200` on
  the word `handle` remains ignorance, as #1993 recorded.

## Commands

```text
# census (login worktree, SHA 92fade0be1)
python3 /tmp/handle_expr_census.py /workspace/.wt/handle-census
# 94 files, 900 hits, handle_expr=0

# Slurm (cpu-ops / cpuops-t560-proxmox)
# stdin tar -> /orangefs/training/handle-green-census-20260819
# Madaros ELF reused from /orangefs/training/effects-cost-20260819/bin
#   (sha256 matches committed bin/madaros-linux-x86_64)
bash scripts/dev/slurm_srun_minimal.sh --time=00:25:00 -- '…'
```
