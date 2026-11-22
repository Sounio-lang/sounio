<!-- docs:meta
topic_id: repo.docs.plans.hc3-count-tsv-implementation-2026-10-19
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.hc3-count-tsv-implementation-2026-10-19
-->

# HC3 and incremental count TSV — implementation checkpoint, 2026-10-19

## Scope and acceptance boundary

This adds two scientific library implementations after the published sprint commit
`af017b565479136231728ecd75b444e221e9aa3d`, based on remote main `15a1983f`.
Both Sounio modules passed native Madaros type checking in Forge/Slurm.
This is an **experimental implementation checkpoint**: numerical concordance,
executable consumers, corpus inclusion and runtime resource measurements remain
pending. The earlier 101/101 acceptance does not cover these new modules.
No automated tests were added or executed in this checkpoint.

## OLS HC3 covariance

`stdlib/stats/regression/robust_covariance.sio` exposes
`multiple_linear_hc3(design,response,n,p,rcond)` for the existing physical 256x8
layout. It factors once with the existing equilibrated, pivoted QR kernel.
Two triangular solves provide leverage and coefficient influences. Production
forms neither normal equations nor their inverse. Influence norms and a normalized
Gram accumulation produce covariance and standard errors in original column order.

Covariance status is independent from the retained OLS fit. The module reports
failed fit, insufficient residual degrees of freedom, computed leverage >=1 and
arithmetic unavailability separately. It does not clip leverage or infer t/p/CI.
It does not provide cluster covariance or assess scientific assumptions.

The rounded leverage policy does not certify every mathematically unit-leverage
row. The documentation also distinguishes reported zero residuals from exact
mathematical fits, and discloses cancellation/underflow limits. An independent
review corrected the documentation's factor order to `S^-1 P v`; the code already
used original-column scales correctly.

Source SHA256: `b788b4cb384e2d9f82098a63264fe913b74c28df4a57dc79f2e70bbba0cf7fb7`.

## Counts, incremental QC and bounded normalization input

`stdlib/genomics/io/count_tsv.sio` parses nonnegative i64 TSV counts for 1..8
samples. Chunks hold up to 256 byte values in i64 cells. It handles LF, split CRLF,
EOF without a newline, decimal overflow before multiplication/addition, and
absolute error positions. Ready rows pause until explicitly acknowledged.
Errors are sticky and invalid rows are not emitted.

QC updates totals and zero counts transactionally; overflow does not leave partial
numerical updates. A checked in-place adapter assembles the existing 256x8 count
layout and rejects excess capacity. QC may continue beyond 256 features while
its integer counters remain representable; the normalization adapter may not.
The parser receives caller-supplied bytes and does not implement filesystem I/O.
Chunk-local median-ratio normalization is not claimed to equal global normalization.
`genomics/io/mod.sio` exports the new module.

Source SHA256: `ddb0e0b9216f6b5c6535d812cc85caa656ae9b69ffb69f1a0928d2b95fb1ca74`.

## Independent reference and swarm

`tests/reference/linear_hc3_reference.cpp` is an independent C++23/quad reference
using two-pass modified Gram-Schmidt on the original design, rather than the
production Householder factorization. It prepares 24 cases and explicitly
separates quad residuals, residuals based on rounded fitted values, and analytic
perfect-fit certificates. It has passed g++ C++23 syntax checking on t560;
it has **not** been executed, so its expected outputs are not accepted evidence.

Source SHA256: `4e792584d182ddcecdeb9ed5d3003ee2673911abfc14e7c669d9585a7d63ae8e`.

Three subagents owned HC3, count TSV and the independent reference respectively.
The count-TSV owner also reviewed HC3 without editing it; root reviewed the parser
and coordinated the export and Forge packet. Darwin was consulted before new
Sounio code; invalid candidates were corrected, not accepted as correctness proof.

## Native receipts

Run `stats-viz-hc3-counts-typecheck-20261019-v1`, Slurm15621, successfully checked
both Sounio modules but the combined dispatcher failed when it attempted g++:
the sanitized job image has no C++ compiler. This is preserved as an infrastructure
failure, not reported as a clean job.

The corrected v2 packet runs only the two Sounio checks and declares
`purpose=scientific-library-typecheck-only`, `corpus_acceptance=false`, and
`numerical_acceptance=false`. C++ syntax was checked separately on t560 with:

```sh
g++ -std=c++23 -Wall -Wextra -Wpedantic -fsyntax-only \
    tests/reference/linear_hc3_reference.cpp
```

That command exited 0. No numerical C++ executable was run.

## Remaining scientific work

Independent numerical concordance, executable public consumers, parser chunk
boundary controls, native runtime evidence and corpus integration remain open.
The planned scientific figure consumer is counts -> QC -> Canvas/SVG; no figure
has been produced for this checkpoint. Cluster covariance requires a separate
score-aggregation contract. No new performance or SOTA superiority is claimed.

Final typecheck receipt: v2, **Slurm15622**, status pass, all three orchestration
phases pass, no failed layer. The log contains TYPECHECK_OK for each module and
TYPECHECK_READY. This is two module checks, not three scientific tests.

## Superseding native acceptance

The numerical/consumer work left pending above is now covered by Slurm15625,
105/105 with zero failures/skips/errors, independent C++23 concordance and a
native counts-to-QC Canvas/SVG consumer. This file retains the historical
implementation checkpoint; see HC3_COUNT_TSV_ACCEPTANCE_2026_10_19.md for the
current result, controls and remaining limits.
