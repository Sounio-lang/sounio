<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-inference-scale-2026-10-24.nested-pilot-6ffc.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-inference-scale-2026-10-24.nested-pilot-6ffc.readme
-->

# Completed nested B31 pilot on source6ffc

Slurm17026 completed with scheduler stateCOMPLETED/exit0:0. The real nested-traced
pilot produced one READY batch of16 synthetic datasets at n16/alpha0.25/B31.
The production harness recorded LAST_INVOCATION_STATUS0 and COMPLETE; its C++
batch and cell summaries report16 available, 0 unavailable and 0 partial traces.
This is a completed pilot and availability observation, not B199 qualification,
finite-sample calibration, a general performance result or authorization to launch
a round. No round or performance pair was submitted by the monitoring agent.

The source is the prepared central tree
`combined_final_cache04/source`, HEAD6ffc1a5bd8eaebeef2d15ccced830bfbc832a648,
with guarded compiler d74fbebe0ff442e8c77bec024d6d92325b83a98d3f1487caa1db59ef171a41e6.
The genuine functional receipt is bound to successful fullgate17020. Its SHA is
c415632ec6a64d57060e1ec2c62b16443d626c1e7cd5ad9645b2be07abfabfe3.
The rejected historical17009 receipt and earlier unsubmitted preparations remain
separate; none was rebound to this run.

## Actual pilot records

`pilot17026.tar.gz` retains actual per-dataset stdout/stderr/status/resources,
CAL_RESULT/CAL_P/CAL_WORK/CAL_TRACE output, C++ summaries, generated dataset/seed
metadata, source/receipt/compiler/tool/ELF identities, READY/completion records,
protocol and submission receipts. `generated-driver.sio.txt` is the actual small
batch driver for replay; no compiler ELF, native ELF or full source tree is copied.

The C++ summaries agree byte-for-byte with SHA
35b8d1d503f6395b85f747dbdfb3b90d62104f192d42d67d8867fecf35315d7f.
Actual result counters include 1024 fit attempts, 7936 count draws, 95338 uniforms,
80277 proposals and 45351 profile evaluations. The callsite trace records 1101488
full-PMF rows. Counters are copied from the original executed C++ summary and
native records; none is reconstructed from B, RNG state or another counter.
Conditional rejection diagnostics remain in the raw summary with its caveats;
they are not interpreted as evidence of calibration here.

The scheduler records 00:01:50 elapsed and batch MaxRSS 1419220K. This is the complete
pilot job including compilation and all invocations, not an isolated kernel or
speedup comparison. The harness wall counter is 110 seconds. Root's later budget
plan conservatively charges the full authorized 900-second pilot allocation, not
an automatically inferred remainder from that counter. No round ledger row is
created by this custody.

## Separate operational monitoring scopes

`monitor-wrongcwd.tar.gz` preserves an auxiliary verification attempt that used
the wrong cwd for relative canonical manifest paths. The resulting file-open
failures are operator-monitor errors, not source drift, failed datasets or a
failed pilot. No missing operator exit-status capture is synthesized.

`monitor-retry-pass.tar.gz` preserves the corrected actual monitor STATUS0.
It verifies the original full batch manifest, the C++ tool/native ELF identities,
campaign inputs and the complete current source catalog. The current catalog is
byte-identical to the source catalog approved in17020. Receipt/compiler identities
also match. The full catalog is stored only once in the pilot archive; the retry
archive records its hash rather than duplicating it.

The original batch FILES manifest and PREPARED_FILES manifest are provenance
records, not standalone custody validators: the executable is intentionally omitted
and the generated driver is renamed with a .txt suffix, while root appended the
pilot authorization after the prepared ledger-header snapshot. The actual worker
batch was verified intact before copying. Use the new archive custody manifests
to verify exactly the retained files.

## Verification and boundary

Run `sha256sum -c MANIFEST.sha256` and `sha256sum -c ARCHIVES.sha256` here. Extract
each archive into a separate temporary directory and check its inner MANIFEST.
Originals remain under
`/scratch/sounio-stdlib-inference-scale-20261024/nested_qualification_final03`.
No source or allocation ledger was modified by collection.

The separate versioned submit-phase-hardcap operator packet is prepared only,
with an exact minute-based Slurm cap and no change to the original pilot launcher.
It has not submitted a round. Root review of the pilot and the separately scheduled
three performance pairs precedes any later phase; a pilot does not imply their
results, a full round, or scientific acceptance.
