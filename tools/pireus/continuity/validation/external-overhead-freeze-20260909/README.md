# Loaded-model external observer screening

The full runtime is frozen in the private remote directory recorded in
validation.json. Reconstruct it with freeze_external_overhead.py create.
Public execution-freeze.json pins both runtime maps, immutable inputs, the
exact runtime source and prospective acceptance criteria.

Run baseline then observed on the same eight without-feedback requests.
Both runs must complete, with rank and cross-arm byte parity. Each rank must
lose at most 256 MiB of minimum host-available memory and have a decode-span
ratio <=1.10. These are engineering screening thresholds, not general
statistical overhead claims. Fixed order can confound cache state; retain that
limitation. Do not infer equal residency or arbitrary 32-request acceptance.

The observed arm requires continuous external coverage, all required metrics,
no new OOM events, and complete lifecycle journals. Missing evidence cannot
pass. Guardian 33 GiB, protected floor 32 GiB, cache 6144 and workload are
unchanged. One attempt per arm; stop after any failed arm; no automatic retry.

The exact source CI run is 34402455439 at commit 054b7d672d. It is pending in
the archived observation. Custody checks passed separately. A readiness
receipt must verify the entire runtime and all three required exact-source
checks before fresh live preflight or allocation. No model job has started.

A new PR push cancels prior CI under this repository's concurrency rule.
Preserve the remote source head until its existing run finishes; local remote
workspace commits can retain this preparation meanwhile.
