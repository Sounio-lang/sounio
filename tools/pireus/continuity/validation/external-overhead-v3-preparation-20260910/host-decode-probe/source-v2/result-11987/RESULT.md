# Job 11987: inference completed; host probe window rejected

Slurm recorded COMPLETED / 0:0 from 2026-09-10 05:03:32 to 05:24:14 UTC. The pinned driver returned 0 without retry. The terminal collection preserves 235 files, with no missing artifacts, under SHA-256 b03202231b1bb4f890a8a29c5c0ee202df2bba6408c307b69519fda5c38fe109.

Both ranks completed all eight frozen requests, totaling 943 tokens per rank. Each response file matches its completion-receipt digest, and corresponding response files are byte-identical between ranks. Runtime binding passed separately for both ranks (1009 / 983 lifecycle records). This does not qualify full diagnostic custody.

The host probe inspection rejected the collected stream with probe prefix order. Only 23 rank-0 and 21 rank-1 probe records parse as complete JSON; 16 additional lines containing probe schema text are malformed. The stream has missing BEGIN/END entries within the first fifteen decode calls. Preserve original bytes. Interleaving of large records in aggregate stdout is a transport hypothesis; do not repair fragments into accepted evidence.

The guardians returned 0. Sampled minima were 35943071744 bytes on rank 0 and 35669508096 bytes on rank 1, above the unchanged 33 GiB early-stop (35433480192 bytes) and 32 GiB floor. Sampled minima are not atomic memory reservations or evidence that every instantaneous value was observed.

Run python3 analyze.py to check the pinned file inventory, response digests, rank equality, token counts and guardian exits, and enumerate malformed probe lines. The supplemental runtime audit and frozen inspector outputs are preserved separately. Inference completion is true; probe-window qualification, full diagnostic custody, timing eligibility and pilot acceptance remain false.

## Next boundary

Preserve 11987 as a completed instrumented eight-request run with rejected probe transport. Prepare a separately identified transport revision using an individual journal per rank, validate long records and incomplete writes without a model allocation, then freeze and qualify that revision before any new model attempt. Do not mutate this packet, silently recover missing records, retry 11987, promote it to the original 32-request pilot, or claim a causal memory explanation.
