<!-- docs:meta
topic_id: repo.docs.spec.2026-10.19-syntax-execution-receipt
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.spec.2026-10.19-syntax-execution-receipt
-->

# 19. Syntax Execution Receipt

## Result and claim boundary

This receipt executes one deliberately partial positive witness for every
`SYN-001`–`SYN-016` rule in [chapter 01](01-lexical-syntax-and-modules.md)
under both current-source Madaros and the canonical `lean_single` route. The
canonical matrix contains 32 rows: 30 are `WITNESS_PASS` and two are recorded
front-end divergences. Both divergences have dedicated follow-up tasks.

`WITNESS_PASS` is intentionally narrower than “rule confirmed.” Each row says
only that the fixture named in that row satisfied its byte-level or check-only
contract. The corpus does not exhaust every construct listed by a rule, does
not promote a rule's standing, and does not resolve `SYN-016`'s open evaluation-
order decision.

Raw material and attestation:
[`docs/audit/S01_SYN_PROBES_2026-11-10/`](../../audit/S01_SYN_PROBES_2026-11-10/README.md).
The canonical run is `20261110-0d619ed0-canonical`, Slurm job `22691`, on a
clean tree at `0d619ed0e8f23f630de01fd0af356e29cd70c687`.

## Engine identity

| engine | executable evidence |
|---|---|
| current-source Madaros | Built immediately before the matrix by `scripts/ci/build_modular_madaros.sh`; SHA-256 `6bb517501e2ea416d62f7c9ba423be3731784061c4d735543ed531821a18fd27`; `bin/madaros info` attests the exact raw path. |
| `lean_single` | `bin/souc-lean-single-x86_64`; SHA-256 `5a1ab6f9650d35d1efe368a61175ef0e016d7113a90f00fd8406cb1538ffe292`; `canonical_compiler_gate.sh` passed and an execution trace attests the wrapper route. |

The manifest also records the source files, build script, public wrappers,
resolver scripts, timeout binary, kernel/architecture, hostname, compiler build
log, fixture hashes, golden-output hashes, and empty staged/unstaged patch
hashes. Compiler invocation and native execution use scrubbed environments and
separate return codes. Complete stdout is compared byte for byte; a marker
substring is not sufficient.

## Rule-by-rule matrix

| rule | Madaros | `lean_single` | measured witness and limit |
|---|---|---|---|
| `SYN-001` | `WITNESS_PASS` | `WITNESS_PASS` | Multiplication/addition, addition/shift, and left-associative subtraction only. |
| `SYN-002` | `WITNESS_PASS` | `WITNESS_PASS` | Prefix `!`/`~`, `!=`, and `^` with exact five-line stdout. |
| `SYN-003` | `WITNESS_PASS` | `WITNESS_PASS` | Parenthesized newline continuation and its existing unparenthesized control. |
| `SYN-004` | `WITNESS_PASS` | `WITNESS_PASS` | Call, field, and index postfix composition; no universal label/reordering claim. |
| `SYN-005` | `WITNESS_PASS` | `WITNESS_PASS` | Typed function parameters, return type, and one effect list; check only. |
| `SYN-006` | `WITNESS_PASS` | `WITNESS_PASS` | A public method inside `impl`; modifiers and generic boundaries remain unmeasured. |
| `SYN-007` | `WITNESS_PASS` | `WITNESS_PASS` | Raw mutable/const pointer type recognition without unrelated initialization semantics. |
| `SYN-008` | `WITNESS_PASS` | `WITNESS_PASS` | Existing restricted compound refinement predicates; check only. |
| `SYN-009` | `WITNESS_PASS` | `WITNESS_PASS` | Qualified enum and wildcard patterns with exact output. |
| `SYN-010` | `FAIL_FRONT_END` | `WITNESS_PASS` | Madaros reports E137 for nested bindings `y` and `z`; see §3. |
| `SYN-011` | `WITNESS_PASS` | `WITNESS_PASS` | Double semicolon after a statement; leading-empty-statement behavior remains unmeasured. |
| `SYN-012` | `WITNESS_PASS` | `WITNESS_PASS` | Parenthesized hexadecimal return control only; the direct starter gap is not certified. |
| `SYN-013` | `WITNESS_PASS` | `FAIL_FRONT_END` | Madaros runs all eight `if let` controls; the seed refuses the literal-pattern arm; see §3. |
| `SYN-014` | `WITNESS_PASS` | `WITNESS_PASS` | Arity-two closure literals, capture, higher-order passing, and exact output. |
| `SYN-015` | `WITNESS_PASS` | `WITNESS_PASS` | Nonnegative integer payload syntax is accepted; payload retention is not proved. |
| `SYN-016` | `WITNESS_PASS` | `WITNESS_PASS` | Narrow FO product-rule call-once witness only; universal sequencing remains open. |

## Tracked divergences

### `SYN-010`: recursive tuple subpatterns

`tests/run-pass/tuple_destructure_let.sio` contains both flat and nested tuple
destructuring. Current-source Madaros refuses the file with E137 for the inner
bindings `y` and `z`; `lean_single` compiles it and prints the complete golden
line. Chapter 01 already observes that the direct-binding desugaring skips
recursive subpatterns, while the fixture's `run-pass` classification claims the
opposite. This public-contract choice is tracked as **SIO-629**; the receipt does
not silently select either semantics.

### `SYN-013`: literal `if let` pattern on the seed

Current-source Madaros compiles and runs all eight controls in
`tests/run-pass/if_let_desugar_forms.sio`. The canonical seed refuses
`if let 4 = n` at `=` with `parse error: expected expression`. The bounded seed
parity defect is tracked as **SIO-630**.

## Reproduction and independent controls

The run directory contains `manifest.tsv`, `results.tsv`, raw front-end/native
stdout and stderr, the Madaros build log, engine-route attestations, and the
source status/diff receipts. Native ELFs are omitted from Git; `results.tsv`
retains each artifact SHA-256. Re-running with the same run identifier is
refused, and each exact artifact path is removed before compilation, preventing
a stale output from satisfying a later row.

The harness received two independent read-only reviews. The first found stale-
artifact, partial-stdout, dirty-provenance, and inherited-environment hazards;
all four were corrected before the canonical run. The second review confirmed
those gates and the engine-route attestations. Preliminary infrastructure and
harness failures are not included in this receipt's matrix.

## What this receipt does not claim

- Exhaustive positive/negative coverage for every sentence in `SYN-001`–`016`.
- Equivalence between the two engines beyond the named witnesses.
- That a check-only success proves code generation or runtime behavior.
- That `SYN-012`'s direct-return starter gap is fixed or approved.
- That `SYN-015` retains distinct payload values merely because it accepts the syntax.
- Any universal operand sequencing or logical short-circuit contract for `SYN-016`.
