<!-- docs:meta
topic_id: repo.docs.research.self-falsifying-compiler-spec-2026-07-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.self-falsifying-compiler-spec-2026-07-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Self-Falsifying Compiler — Compile-Time Execution of Scientific Claims

- **Status**: implemented (Approach A, opt-in)
- **Date**: 2026-07-25
- **Branch**: research/particle-exp123-20260725
- **Depends on**: `docs/research/ast_native_claims_spec_2026-07-25.md` (AST-native
  claims, rung 1), `docs/research/falsification_ledger_spec_2026-07-25.md`
  (comment-scanned ledger)

## 1. Idea

A scientific claim embedded in a `.sio` source file should not be inert
documentation. When the compiler is invoked with `--verify-claims`, it becomes a
**self-falsifying compiler**: after the program passes type-check and *before*
any code generation runs, the compiler executes each claim's `gate` script in a
sandboxed subprocess. If a gate fails, the claim is treated as falsified and the
compilation aborts — the compiler refuses to emit code whose scientific
premises no longer hold.

Without the flag, claims remain exactly what rung 1 defined them to be:
compile-time metadata that never reaches resolve/check/codegen. The feature is
strictly opt-in.

## 2. Pipeline placement

```
parse → resolve → type-check → [CLAIM GATE EXECUTION] → lower → codegen
                                     ^
                              only when --verify-claims
```

Concretely, in Madaros (`self-hosted/compiler/main.sio`):

- `build` / `run` / `--native-v2-compile` lane: the hook fires inside
  `run_native_v2_compile_mode`, after the source/visibility preflights and
  before `module_frontend_compile_imported_to_file` (the fused front-half +
  backend call).
- default lane (`souc input.sio`): the hook fires in `main()` immediately after
  `parse_options`, before `compiler_try_default_native_v2_single` / `compile`.
- `check` / `--check` lane: the hook fires after the visibility preflight, in
  place of codegen (check mode emits no code).

The executor (`self-hosted/compiler/claim_executor.sio`,
`claim_executor_verify`) itself re-runs `preflight_multimodule_frontend`
(parse/resolve/type-check) on the source before executing any gate, so a
program that does not type-check never has its claims executed. It then
re-parses the main source file with `load_module_file`, which leaves the AST
claim registry holding exactly the claims of that file.

## 3. Claim source of truth

Claims come from the **parser claim registry** (`self-hosted/parser/ast.sio`),
populated by `parse_claim_item` during parsing. Rung 1 stored only
`ast_name_hash` digests, which cannot be inverted to recover gate paths. This
rung therefore extends the registry **additively**:

- `CLAIM_NAME_TEXT` / `CLAIM_NAME_TEXT_LENS` — claim name text (64 x 128 bytes)
- `CLAIM_FIELD_NAME_TEXT` / `CLAIM_FIELD_NAME_TEXT_LENS` — field name text
  (1024 x 128 bytes)
- `CLAIM_FIELD_VALUE_START` / `CLAIM_FIELD_VALUE_END` — field value **source
  byte-offset spans** (1024 entries)

plus read accessors:

- `ast_claim_slot_name(slot) -> Name`
- `ast_claim_slot_field_count(slot) -> i64`
- `ast_claim_slot_field(slot, field_idx) -> ClaimField` (name + kind; the value
  Name is always empty)
- `ast_claim_slot_value_start(slot, field_idx) -> i64`
- `ast_claim_slot_value_end(slot, field_idx) -> i64`

**Why values are spans, not Names:** Names nested inside `ClaimDecl` do not
survive the parse → registry path intact, and `current_name()` results for
StringLit tokens do not survive the trip into a recorded field either (struct
store / struct-return limitations — rung 1 deferred text recovery for this
reason; empirically, recorded StringLit values arrived truncated to 8 bytes
or empty). `parse_claim_item` therefore records, for each field, the token
span `[pt_read_start, pt_read_end)` covering the value (for paths the span
covers the whole `A::B::C` path; for string literals it includes the quotes),
via `ast_claim_note_field_span(field_idx, start, end)` — scalar i64s, immune
to the struct limitations. Claim and field names go through
`ast_claim_note_name_text` / `ast_claim_note_field_name` (these provably
survive). All note functions are called while the claim is being parsed,
before `ast_record_claim` increments `CLAIM_COUNT`, so the pending slot is
`CLAIM_COUNT`. The executor reads the compiled source file and slices the
value text out at the recorded spans.

The hash arrays and the existing `ast_claim_present` /
`ast_claim_field_count` / `ast_claim_field_kind` API are unchanged;
`ast_record_claim` keeps its original hash-only behavior, and the hash path
remains the only registry consumer besides the executor. The comment-scanned
ledger (`scripts/research/falsification_ledger_contract.py`) and the
`claim_ast` preprocessor remain the repo-wide extraction paths; the registry
covers the claims of the file being compiled.

## 4. Execution semantics

For each claim slot `0 .. ast_claim_count()`:

1. Locate the `verdict` field. If its value text contains `archived`
   (case-insensitive on the first letter, covering both `"archived"` string
   values and `Verdict::Archived` path values), the claim is **skipped**
   (`CLAIM_SKIP <name> (archived)`).
2. Locate the `gate` field. Claims without a gate are skipped
   (`CLAIM_SKIP <name> (no-gate)`); they are metadata placeholders.
3. Otherwise the gate is executed:
   - String values keep their quotes in the registry; the executor strips one
     surrounding pair of `"` before use.
   - The gate path is executed as `bash <gate-path>` via raw
     `fork`/`execve`/`wait4` syscalls — never in-process, never through a
     shell string, so no code runs inside the compiler process and there is no
     command-injection surface.
   - A wall-clock timeout (default 30 s) is enforced by polling
     `wait4(WNOHANG)` with 50 ms `nanosleep` intervals; on expiry the child is
     `kill(SIGKILL)`ed and reaped. A timeout counts as failure
     (`CLAIM_TIMEOUT`).
   - Gate exit code 0 → `CLAIM_PASS <name>`; anything else →
     `CLAIM_FAIL <name> rc=<decoded exit>`.

Summary markers on stdout:

- `VERIFY_CLAIMS_NOOP` — the source contains no claims (executor is a no-op).
- `VERIFY_CLAIMS_OK pass=P skip=S` — every executed gate passed.
- `VERIFY_CLAIMS_FALSIFIED fail=F pass=P skip=S` — at least one gate failed or
  timed out; the compiler aborts with a non-zero exit before codegen.
- `VERIFY_CLAIMS_TYPECHECK_FAIL` — the source did not pass the frontend; no
  gate was executed.

Return value of `claim_executor_verify`: `0` on no-op/all-pass, `>0` = number
of failed gates, `-1` on type-check failure. Any non-zero value aborts the
compilation.

## 5. Sandbox properties

- **Subprocess isolation**: gates run in a forked child; a crashing or
  malicious gate cannot corrupt compiler state.
- **No shell interpolation**: `execve("/bin/bash", ["bash", gate_path])` with a
  fixed argv; the gate path is data, not a command line.
- **Timeout**: bounded wall-clock per gate; a hung gate cannot wedge a build.
- **Bounded work**: at most 64 claims x 16 fields (registry capacity), one
  subprocess at a time.
- **No claims → no-op**: empty registry short-circuits before any fork.

## 6. Non-goals / limitations (this rung)

- Only claims in the **main source file** are executed; claims in imported
  modules are not (the registry is reset per module parse; aggregating across
  the module closure is a future rung).
- Gate paths are interpreted relative to the compiler's current working
  directory (repo root in CI).
- Claims do not feed the falsification ledger JSONL; the Python ledger remains
  the ledger writer. Unifying verdicts (gate failure → ledger `refuted`) is a
  future rung.
- The GPU lane (`--gpu-target`) does not run claim verification.

## 7. Files

| Artifact | Path |
|---|---|
| Claim executor | `self-hosted/compiler/claim_executor.sio` |
| Registry text extension | `self-hosted/parser/ast.sio` (additive) |
| Flag + hooks | `self-hosted/compiler/main.sio` (`--verify-claims`) |
| Test (mixed pass/fail/archived) | `tests/run-pass/self_falsifying_compiler_test.sio` |
| Fixture gates + sources | `scripts/ci/fixtures/self_falsifying_*` |
| CI gate | `scripts/ci/self_falsifying_compiler_gate.sh` |

## 8. Testing

`bash scripts/ci/self_falsifying_compiler_gate.sh` checks:

- **F1_SURFACE** — executor module, flag, and hook exist in source.
- **F2_INERT_DEFAULT** — the run-pass test compiles and runs *without* the
  flag; claims stay inert.
- **F3_NO_CLAIMS_NOOP** — flag on a claim-free source prints
  `VERIFY_CLAIMS_NOOP`, exit 0.
- **F4_PASS_ONLY** — flag on a source whose claims all pass prints
  `VERIFY_CLAIMS_OK`, exit 0.
- **F5_FAIL_BLOCKS** — flag on the mixed test source prints `CLAIM_PASS` for
  the passing claim, `CLAIM_SKIP` for the archived claim, `CLAIM_FAIL` for the
  failing claim, `VERIFY_CLAIMS_FALSIFIED`, and exits non-zero (no codegen,
  no ELF).
- **F6_TIMEOUT** (optional, `SFC_TEST_TIMEOUT=1`) — a sleeping gate is killed
  and reported as failure.
- **F7_DEFAULT_LANE_BLOCKS** — the default lane (no mode keyword) also aborts
  with non-zero exit and emits no ELF when a claim is falsified.

F4/F5/F6 require a claim-aware Madaros built from current source; the gate
builds `artifacts/self-hosted/madaros-self-falsifying` via
`scripts/ci/build_modular_madaros.sh` unless `SFC_SKIP_BUILD=1` is set and the
artifact already exists (same pattern as `claim_native_gate.sh`).
