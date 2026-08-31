<!-- docs:meta
topic_id: repo.docs.audit.handle-execution-witness-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: cursor-2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.handle-execution-witness-2026-08-19
-->

# Handle execution witness

**Question:** a `handle` expression type-checks on Madaros and never appears in IR / native / ENIR. What happens when that program is compiled and run?

**Verdict (Madaros, canonical engine): QUEDA SILENCIOSA — the whole `handle` expression is erased.**  
Check is OK. An ELF is produced. The process exits 0. Stdout is empty. The handler does not run. The body does not run either. There is no diagnostic. This is worse than “the handler is ignored and the body runs”: the body is deleted too.

**Verdict (lean_single): RECUSA as an unknown identifier, not as `handle`.**  
`error[E200]: undefined identifier \`handle\``. The seed does not treat `handle` as a keyword. That is not a typed Reserved state.

The measurement that said `ExprHandle` is absent from `self-hosted/ir/`, `self-hosted/native/`, and `self-hosted/enir/` is **not** contradicted. Nothing in this run shows a path from `ExprHandle` to code.

## Instrument

| Field | Value |
|---|---|
| SHA of sources / staged binaries | `9fdaa5772b` (`origin/main` at branch creation) |
| Submitter | workspace pod; `/workspace` is **not** visible on the node |
| Launch | `scripts/dev/slurm_srun_minimal.sh` — `srun`, partition `cpu-ops`, `--chdir=/tmp`, `--export=NONE` |
| Node | `cpuops-t560-proxmox` |
| `workspace_visible` | no |
| `orangefs_visible` | yes |
| Stamp | `2026-08-19T16:07:39Z` |
| Madaros | `bin/souc` default → `Madaros v0.80.0` (`bin/madaros-linux-x86_64`) |
| lean_single | `SOUNIO_SOUC_ENGINE=lean_single` → `bin/souc-lean-single-x86_64` (`Usage: mini_native <source.sio> <output>`) |
| Stack | `ulimit -s 1048576`, `MADAROS_STACK_KB=524288` |
| Compile form | `souc compile <src> -o <elf>` (never the bare `-o` swallow) |
| ELF predicate | file exists, size > 0, magic `7f454c46` |
| First Slurm attempt | **void** — Madaros launcher `bin/madaros` was not staged (`env: …/bin/madaros: No such file`); lean_single control ELF lacked `+x` (rc=126). Not a language result. |
| Second attempt | launcher + `chmod +x` on produced ELFs. This table is that run. |
| Local TSV | [`HANDLE_EXECUTION_WITNESS_2026-08-19.tsv`](HANDLE_EXECUTION_WITNESS_2026-08-19.tsv) — reconstructed from the Slurm stdout of the second attempt. The node wrote `/orangefs/training/sounio/handle-exec-witness/HANDLE_EXECUTION_WITNESS_20260819T160739Z.tsv` (header only; rows were echoed to stdout). The pod cannot read `/orangefs`. |

Programs (also in-tree):

- [`tests/audit/handle_control.sio`](../../tests/audit/handle_control.sio) — no `handle`; prints `BODY_MARK`
- [`tests/audit/handle_io.sio`](../../tests/audit/handle_io.sio) — `handle<IO> { println("BODY_MARK") } with { println("HANDLER_MARK") }`
- [`tests/audit/handle_unknown_effect.sio`](../../tests/audit/handle_unknown_effect.sio) — same shape with `handle<NotARealEffect>`

## Positive control

The control program **compiled and ran on both engines**. `println` works. A later `handle` cell that prints nothing is therefore a fact about `handle`, not a broken printer.

| Engine | phase | rc | ELF | stdout |
|---|---|---:|---|---|
| **madaros** | check | 0 | — | `check: OK` |
| **madaros** | compile | 0 | yes `7f454c46` | `Compilation successful!` |
| **madaros** | run | 0 | yes | `BODY_MARK` |
| **lean_single** | check | 0 | — | (lean_single check log, no error) |
| **lean_single** | compile | 0 | yes `7f454c46` | ELF 36636 bytes |
| **lean_single** | run | 0 | yes | `BODY_MARK` |

## Matrix

Stdout/stderr below are the **run** phase unless the cell never produced an ELF.

| Engine | program | check rc | compile rc | ELF | run rc | run stdout | class |
|---|---|---:|---:|---|---:|---|---|
| **madaros** | `handle_control` | 0 | 0 | yes | 0 | `BODY_MARK` | control |
| **madaros** | `handle_io` | 0 | 0 | yes | 0 | *(empty)* | **QUEDA SILENCIOSA** |
| **madaros** | `handle_unknown_effect` | 0 | 0 | yes | 0 | *(empty)* | **QUEDA SILENCIOSA** (same as `IO`) |
| **lean_single** | `handle_control` | 0 | 0 | yes | 0 | `BODY_MARK` | control |
| **lean_single** | `handle_io` | 1 | 1 | no | — | — | **RECUSA** `E200` `` `handle` ``, `` `IO` `` |
| **lean_single** | `handle_unknown_effect` | 1 | 1 | no | — | — | **RECUSA** `E200` `` `handle` ``, `` `NotARealEffect` `` |

lean_single diagnostics (verbatim fragments):

```
error[E200]: undefined identifier `handle` at <main>:5
error[E200]: undefined identifier `IO` at <main>:5
typecheck: failed
```

```
error[E200]: undefined identifier `handle` at <main>:5
error[E200]: undefined identifier `NotARealEffect` at <main>:5
typecheck: failed
```

Madaros compile of `handle_io` / `handle_unknown_effect` reported `Merged IR: 1 functions`. The control reported `Merged IR: 3 functions`. The body that prints `BODY_MARK` is not in the IR that was emitted.

## Negative control — invented effect

On **Madaros**, `handle<NotARealEffect>` is indistinguishable from `handle<IO>`: check OK, ELF produced, run rc=0, empty stdout. `effect_name_to_id` returns `-1` for an unknown name; `check_handle_expr` only installs an effect when `eff_id >= 0`. The checker does not refuse an invented effect name.

On **lean_single**, both `handle<IO>` and `handle<NotARealEffect>` fail at `handle` itself. The seed never reaches an effect-name distinction.

## Which of the four outcomes

Asked to name one:

| Engine | Outcome |
|---|---|
| **Madaros** | **QUEDA SILENCIOSA.** Compiles, runs, rc=0, no diagnostic. The handler does not execute. The body is also erased, so this is not “handler ignored, body kept”. A program that looks like algebraic-effect handling is a no-op. |
| **lean_single** | **RECUSA** (`E200` undefined identifier). Not a `handle`-aware Reserved diagnostic. The construct is not in the seed. |

Not **FUNCIONA**: `HANDLER_MARK` never appears.  
Not **CRASH**: Madaros run rc=0; lean_single fails at typecheck with a named error.

`ExprHandle` does not reach code on the path measured here. No IR / native / ENIR arm is required to explain the empty ELF: the front-end accepts the node and lowering never sees it.

## What this file is not

- Not a patch to `self-hosted/`.
- Not a claim that lean_single is the language. Madaros is the clock; lean_single is the seed.
- Not a CI gate. Wiring a fail-closed check for this drop is a later decision.
