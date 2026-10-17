# Per-rank host probe journal preparation

Job 11987 completed inference but its aggregated stdout contained malformed large probe lines. This revision preserves all probe fields, native units, observation times, request/decode scope and model computation from the pinned parent runtime. It changes only the transport and embeds its writer helper.

Each rank exclusively creates /scratch/pireus/receipts/host-decode-JOB-RANK.jsonl on its first probe. Existing files and symlinks are refused. Writes handle short returns and interruptions; failures close the journal and prevent further writes. stdout contains only a short informational digest receipt. Individual journal bytes must be collected and checked against an externally pinned digest and lifecycle identity. This revision adds no device synchronization or fsync durability claim. Files are unbuffered but do not constitute power-loss durability.

Nine new controls cover concurrent large records, pre-existing paths, short/interrupted writes, zero-progress failure with partial-line rejection, identity/order/limits, corrupt input, partial prefixes, the generated hook, and reversible source preparation. tests.log also includes the existing host-decode controls. Tests are synthetic CPU controls, not cluster or model qualification.

This is a prepared runtime, not a runnable experimental packet: the journal collector, lifecycle-bound inspection adapter, prospective protocol/freeze and source acceptance must be integrated before hardware qualification. Existing source-v2 packets, job 11987 and their rejection remain unchanged. Do not feed small stdout receipts to the old probe inspector as if they were the raw observations. No model was submitted by this preparation; timing and pilot acceptance remain false.
