<!-- docs:meta
topic_id: repo.docs.audit.pr2737-encoded-aggregate-escape-dispatch-2026-10-02
authority: repo_only
audience: users
last_validated: 2026-10-02
validated_by: codex
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.pr2737-encoded-aggregate-escape-dispatch-2026-10-02
-->

# PR #2737: Encoded References In Aggregate Results

Blocker-ID: BLK-20261002-pr2737-encoded-aggregate-escape
Status: locally-validated
Severity: B1
Class: compiler-semantics
Evidence-Level: E3
Owner: codex
Lane: pr2737-aggregate-escape
Worktree: pr2737-region-reclaim-real
Branch: codex/pr2737-region-reclaim
Source baseline: b7905db7e8467ca19e44e297a2c354c20baa40b2

## Source Mechanism

`rgn_copy_cost` excludes integer result slots because they can encode references,
but accepts floating slots and arrays. `rgn_emit_young_copy` copies their bits
without checking whether they name a young handle or raw arena address. A
returned wrapper can therefore retain bits naming an object which the region
reset reclaims. A declared representation is not proof of non-reference.

## Acceptance Gate

Compile and execute `madaros_region_reclaim_encoded_aggregate.sio` with the
current-source baseline, the opt-out control, and the corrected compiler.
The witness covers a floating field, a floating array element and a nested
wrapper. Keep all raw ELFs and receipts in the isolated upstream lab.

Baseline current-source compiler SHA256:
`f3661cebf54530c1ca01afe1d4291c414c88bdbba5c6b49dd807b2306129fca9`

Repro: `madaros tests/run-pass/madaros_region_reclaim_encoded_aggregate.sio -o witness.elf`
Observed: default compile succeeds, execution exits 3; opt-out compile succeeds,
execution exits 0 and prints the expected marker.
Expected: the default execution must preserve every recovered field.
Evidence: `/workspace/upstream-lab/pr2737-thinlink-default-20261002/receipts/aggregate-escape/`

The fix scans raw scalar words recursively before the escape decision. Array
scans use a runtime loop, and zero/unassigned aggregate slots are skipped.
`__rgn_barrier(word, 0)` treats the caller as older storage and prevents resets
which could invalidate an encoded reference. Copy eligibility remains unchanged.
The epilogue capacity estimate now budgets for the additional scan.

## Rebuilt Execution

CI run 37040225568 produced the compiler from 49b73d5ba. SHA256:
`46ca23bd4cd5d351e5ae2dc4e1fd4e01a573b91c48cf1caa792135993df3618a`.
With the opt-out unset, the new aggregate witness compiled and executed with
exit 0 and `REGION_RECLAIM_ENCODED_AGGREGATE_OK`. The ordinary aggregate result,
encoded scalar and ordinary escape witnesses likewise compiled and executed
with exit 0 and their expected markers. Compiler stack matched `bin/madaros`;
the generated programs ran with the original 8192 KiB stack.

For these single-module witnesses, invoke the raw compiler as
`<ELF> tests/run-pass/<witness>.sio -o <output>`; `--native-compile` selects an
imported-module route which explicitly refuses single-module streaming input.
Raw ELFs, compile logs and output are retained in
`/workspace/upstream-lab/pr2737-thinlink-default-20261002/receipts/current-source-stack/`.
The four default-mode integration gates passed against this same rebuilt ELF
after applying the launcher's stack contract in the gate harness. CI run
37040225568 itself is not green: its unpatched visibility harness segfaulted
under the default compiler stack. The subsequent exact-tip CI must pass too.

This evidence covers unchanged encoded words in returned aggregates. It does
not establish safety for reversible transformations of handle bits; provenance
at casts remains a separate open review issue. Do not clear the entire escape
contract based on these witnesses alone.

Fallback-Path: none in the production fix; opt-out used only as a control.
Legacy-Kept: yes; aggregate relocation stays enabled for ordinary results.
LLM-Offload: local Sounio scaffold failed verification; the witness was corrected
manually and must be compiled and executed before accepting it.
Secondary review: darwin-llm/coder-next reviewed the diff. Its null-dereference
recommendation was rejected: zero aggregate handles must be skipped, not read;
Option/Box shapes are excluded by eligibility. Its speculation about the
classifier was checked against the existing native implementation. This model
output is advisory, not compiler or execution evidence.
Next-Action: verify exact-tip CI with the repaired harness, then separately
reduce and address reversible handle transformations and other open review issues.
