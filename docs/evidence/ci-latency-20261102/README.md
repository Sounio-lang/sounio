<!-- docs:meta
topic_id: repo.docs.evidence.ci-latency-20261102.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.ci-latency-20261102.readme
-->

# CI latency baseline, 2026-11-02

GitHub Actions metadata for PR #3413, head run 41433054775 and merge-group
run 41438509619. This is a baseline, not candidate performance evidence.

`runs.tsv` projects workflow ID, event, created, started, updated, conclusion.
Jobs project job name, started, completed, conclusion. Steps project job name,
step name, started, completed, conclusion. Projections use jq without arithmetic.

Compile `timings.cpp` with a C++23 compiler and run:

```sh
c++ -std=c++23 -O2 timings.cpp -o timings
./timings runs runs.tsv
./timings jobs head-jobs.tsv
./timings steps head-steps.tsv
./timings jobs queue-jobs.tsv
./timings steps queue-steps.tsv
```

The committed outputs came from Xcode C++23 on macOS. Slurm job 21299 failed
with exit 4 because a skipped job had an inverted timestamp; the corrected
analyzer explicitly excludes skipped rows, while non-skipped inverted timestamps
remain an error. No Sounio build or runtime qualification took place on the Mac.

Workflow start/update intervals include workflow orchestration and are different
from per-job wall spans. Sum of job/step durations includes parallel work and
must not be called end-to-end latency. No statistical distribution or speedup is
inferred from these two runs.
