<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-sio-184-diagnostics.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-sio-184-diagnostics.readme
-->

# Optional E2E full diagnostics and materializer exit status

> **Status**: Beta | **Last validated**: unvalidated | **Source**: operational fake-compiler controls only

The E2E runner optionally retains complete combined command output while preserving its existing six-line result excerpt. This fixes the evidence loss observed when a genuine compiler error follows a long boundary/provenance banner. It does not fix or reclassify any scientific test failure.

```
bash scripts/stdlib/run_stdlib_e2e.sh --artifacts-dir /absolute/existing-parent/fresh-capture --json-out /absolute/results.json
```

The artifact directory must be new, absolute, normalized, have an existing writable parent without symlink components, and lie outside both the repository and configured stdlib input trees. Existing or dangling targets, overlapping paths and symlink ancestors are refused before test commands. Test selection, annotation parsing, stdout assertions, default commands, schema-v1 results, check deadlines and the native wrapper's 30-second run timeout remain unchanged. Help includes the new option.

Each actually executed check/run command retains `tests/stdlib/<full-relative-case>.sio/<check-or-run>/combined.output`, `start.json` and `result.json`. Commands, actual combined exit codes, UTC timestamps, source identities and SHA256 of complete raw output are recorded. Bash command substitution still drops NUL and trailing newlines when constructing the existing result excerpt; the regular raw file preserves those bytes. Bash's NUL warning references a different source line in the capture branch, which is the only normalized stderr difference in the comparison control.

Run is the existing wrapper command combining compilation and native execution. Its recorded status does not identify the inner failing phase; a returned139 is signal-compatible status, not proof of a directly observed signal. Wrapper and caller hashes identify files, not a universal execve trace or an imported-source dependency closure. The source census covers `.sio` test files; callers include runner, resolver, emitter, diagnostics helper and the selected wrapper path. Their identities are compared before/after. A requested metadata operation failing exits65 explicitly and prevents publication of a fresh schema-v1 result. A genuine command exit65 with a completed matching receipt remains a normal test failure. Interruptions can leave partial capture files; absence of `complete.json` is not a completed suite receipt.

## Operational controls

The current capture selftest runs28 controls and the launcher selftest runs17 controls, all passing. In addition to the original26 capture cases below, the terminal follow-up checks legitimate zero-phase completion and a later selected case mutating an earlier completed raw log: finalization must return65 with no complete.json or fresh suite result. Current capture receipts are in [terminal-capture01](terminal-capture01/README.md). The opt-in route has not yet been exercised with an actual Sounio compiler.

### Original26-control capture receipt

At parent239682aa004a386bd44e22ce96eaad063bb99f03, `node scripts/stdlib/test_stdlib_e2e_diagnostics.mjs` ran26 controls using a generated Bash fake compiler in an isolated temporary fixture. They compare default/capture stdout, exit status and complete schema-v1 rows/totals, excluding UTC and the Bash NUL-warning line number. They cover a true error after12 banner lines, NUL/final-newline retention, statuses0/1/65/124/139, missing stdout, skip/no-annotation, filter and check-only fallback, path refusals before compiler invocation, a missing completed receipt, an unwritable parent, lost raw output, raw-log symlink substitution, intermediate-directory symlink refusal before outside creation, changed raw bytes and changed source. The original26 receipt status is0. The SELFTEST files in this directory preserve that historical round. No real Sounio compiler or native science code runs in these controls.

### Launcher17-control receipt

`node scripts/stdlib/test_wrapper_materialization_status.mjs` runs17 controls for an adjacent launcher defect. Both original launchers used `if ! bash materializer; then _mat_rc=$?`, capturing the success status of `!` after materialization failed. Exact reconstructed before variants return0 for fake materializer78/125 without an override; the fixed normal-if/else variants return the actual78/125. Explicit executable `/bin/true` substitutes for an ELF override, retaining successful version routes and strict committed-only/invalid-override refusals. It is not a compiler. The actual32 invocation rows, source variant/tool hashes and controlled Python-deny receipts are retained. Source variants are verified unchanged after the controls. Real Python is never invoked; three executable Bash deny shims have separate125 probes, and the runtime deny log is empty.

The launcher patch changes only status capture, preserving materialization attempts, raw precedence, valid override behavior and strict checks. The whole E2E corpus and current-main compiler/seed/fixed point remain unqualified by this operational patch. No new Slurm job, timeout increase, skip, annotation change or compiler build is claimed.

Base: `b22d3484c95b826955b3e232310b41a6018be751`. `SOURCE_SHA256.txt` binds this patch's operational sources; `FILES.sha256` covers this text/metadata custody. Historical real627-case receipts remain distinct from these synthetic fixtures.

The original committed runner was independently replayed with the same fake fixtures against the patched default branch: stdout, status and schema-v1 rows match (UTC and the Bash NUL-warning source line excluded). The first metadata comparison attempt used a PATH without Node and failed before producing JSON; its attempt01 logs remain in the temporary handoff. A corrected explicit Node PATH produced the retained PASS receipt. This is operational harness comparison, not a scientific execution.
