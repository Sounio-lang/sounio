CI (ADR-027 phase 1): `cancel-in-progress` now applies only to pull requests, so a push to `main` always gets a complete verdict (38 of 53 main runs were cancelled in the 2026-10-29..11-01 window). The five sub-minute gates (epistemic shape ratchet, language gap ratchet, effect-name closed list, EISA origin, knowledge-annotation silent drop) run as steps of one `Fast Checks` job inside `ci.yml` and count in `CI Decision`; their standalone workflows are removed. `Native Self-Host (macOS arm64)` runs in the merge group, on `main` and on dispatch, not on pull requests; the evaluator and its self-test pin both directions.

Build current-source Madaros once for receipt-verified Witness, Wave0, f64 and Deep Science consumers. Start Deep Science after the producer and move unchanged SCO controls to a parallel required job. Preserve independent bootstrap and modified-source refusal builds; require producer and SCO outcomes in CI Decision.

Forward the verified compiler through the full-gate dereference subgate, reuse it for the SCO runtime gate with source/ELF receipts, and run mandatory deep-science core/integration partitions in parallel. Preserve all scientific bodies and the private modified-source SCO refusal build.

Check the workflow-reference ratchet before expensive jobs and make impact-selected jobs wait for Fast Checks. Name the SCO orchestration selftest directly in the workflow.

Prepare independent C++ references concurrently, and size the science worker pool by CPUs and the existing four-GiB-per-worker memory budget instead of an additional fixed two-worker ceiling. Preserve explicit overrides and propagate child failures or missing receipts to the parent.
