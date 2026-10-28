# Durable custody receipt — logical ownership

Slurm19346 completed0 in26s. Its stdout marker is LOGICAL_OWNERSHIP_CUSTODY_EXPORTED; stderr is empty. The export job created the archive, checked gzip integrity and SHA256, and retained the prior job metadata.

Archive:
`/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/services/logical-ownership-export01/logical-ownership.tar.gz`

SHA256:
`dc14a95092e06f6a62ec58846d947fc4e78a912e8369720b858e2c7897a773e8`

The archive includes logical-ownership01/02/03 staging, baseline and candidate source/checker ELF files, all focused and regression outputs, initial failed namespace/manifest diagnostics, the scratch-only cleanup bridge and mutants, validators, job scripts and the immutable ac39a59339a906790fbd6235c1bb977733eb7a15 patch. Production logical source was adopted by root as908f595d1f; this follow-up changes custody documentation only.

This receipt preserves checker-component evidence. Runtime ADR014 lowering, full BFS compilation, public WorkerFn acceptance and the scientific pipeline are separate gates.
