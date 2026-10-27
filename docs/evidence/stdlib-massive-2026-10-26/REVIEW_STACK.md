# Review stack and acceptance status

The campaign is split into these draft PRs, in dependency order:

| PR | Review scope | Base |
|---|---|---|
| [3239](https://github.com/Sounio-lang/sounio/pull/3239) | Inventory, independent receipts, raw diagnostics, Node boundary, C++23 hyper oracle | main |
| [3240](https://github.com/Sounio-lang/sounio/pull/3240) | Owned storage, binary reading, native platform, sync/time/profile migrations | 3239 |
| [3241](https://github.com/Sounio-lang/sounio/pull/3241) | Scientific consolidation, original refusal fixtures and migrated examples | 3240 |
| [3242](https://github.com/Sounio-lang/sounio/pull/3242) | VCF core/GT validation, protocol guards, migrations and receipts | 3241 |

The first publication put the changelog only in the last review unit. Impact correctly refused the other units. Each unit now adds its own ADR018 fragment(s), and the owned branches were restacked onto their owned updated parents. The restack compared the final tree against the pre-restack tree and found only changelog additions; no tested compiler, library, fixture or gate bytes changed in that operation. Exact leased pushes protect the prior published heads.

## Observed validation

| Job | Packet | Observed result |
|---|---|---|
| 17771 | source compiler | Fresh raw Madaros build passed; compiler implementation was unchanged from source base 5977c066 |
| 17800 | early packet 08 | Complete selection: 320 pass, 74 fail, 194 skip, 588 total; no selection errors |
| 17854 | integrated packet 14 | Complete selection: 344 pass, 61 fail, 230 skip, 635 total; no selection errors; science child results retained |
| 17864 | packet 15 | Focused native/reference/metadata gate passed, 0:0 |
| 17874 | packet 17 | Expanded focused gate including VCF and protocol mismatch passed, 0:0 |
| 17877 | repaired packet 18 | Previously failed DP54, interval, phonon and Wilcoxon child gates plus expanded native gate all passed, 0:0 |

Packet 14 predates the passing time/profiler/effect/golden/protocol/VCF changes and the restored original negative fixtures/example migrations. Its original four scientific child failures are kept in [the diagnostic receipt](integrated-17854.json). The four repaired children are rerun in [the repair receipt](repair-17877.json); the remaining fourteen scientific child gates passed packet 14. These separate executions do not constitute a complete final-tip deep-science replay.

The two complete selection totals have different test sets and are diagnostic records, not a like-for-like pass-rate comparison. The 61 packet-14 failures remain listed individually rather than being converted to skips. Later focused repairs do not justify inventing an adjusted full-suite total.

## Still open

- Complete final-tip full/deep-science replay, full slow lane and compiler corpus.
- The preserved original SNN fixture fails native emission with rc=12; its dynamic-slice implementation is not replaced by a smaller passing control.
- Legacy fixed containers, method/dynamic-slice paths, several domain tests and timeouts remain in the full diagnostic failure list.
- Lifetime/fixed-point/transport and Lean gates, where applicable, have no new completion receipt.
- Exact-head GitHub CI must be read independently after the leased stack update. Initial Impact failure was the missing per-unit changelog; initial Contracts/Decision failures cascaded from skipped shards. CodeQL or a focused Slurm pass does not stand in for those gates.

All PRs remain draft for human review. No merge, auto-merge, registry publication or production activation was performed.
