# V3 screening stopped after baseline 11983

The fresh baseline completed checkpoint initialization and emitted the first token for request 0 on both ranks, then rank 1 crossed the unchanged 33 GiB guardian threshold. No complete request response was saved. Slurm reports FAILED and the launcher returned 75. The partial collection is retained and qualification refuses it as incomplete.

The observed arm was not submitted. No retry is authorized by this frozen pair protocol. This failure occurred without the external v3 observer, so this pair provides no observer-overhead estimate. CPU job 11982 remains qualified within its own scope; loaded-model job 11977 remains a separate negative result.
