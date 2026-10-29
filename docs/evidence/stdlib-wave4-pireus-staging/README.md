# Five Pireus failures caused by incomplete staging

Slurm19611 completed0/41:45. All five unchanged original programs passed with the exact retained ELFs produced by baseline19392. No compiler invocation, source patch, expected-hash adjustment, fixture substitution, assertion change or deadline increase was used in this experiment.

The baseline source and all four worker copies had omitted docs and tools. Every one of the22 explicit parent files already matched the existing frozen SHA-256 constants in the tracked repository. This differs from the separately corrected material-engine sidecar, where five current documents had drifted.

The control copied the baseline's12588 files byte-for-byte into two fresh working directories. One remained incomplete. The other added only the5800 tracked docs/tools files from the baseline source commitadd24582e1e109e84536242b77ad19f70da96cd0. STAGING_DELTA.json verifies equality of every common file and restricts all additions to those two directories. PARENTS.json binds the22 declared parent paths and hashes. The raw11332 and native tools also matched their original manifests.

Each program executed the same retained ELF first in the incomplete directory, reproducing the original failure, then in the complete directory. The old baseline itself and its original result were preserved.

| Original program | Incomplete input status | Complete input status | Original native deadline | Observed complete native wall time | Native peak RSS, KiB |
| --- | --- | --- | --- | --- | --- |
| multiprobe_block_certification | 3 | 0 | 1200s | 3:47.05 | 1922304 |
| operator_autogenesis | 1 | 0 | 30s | 0:01.85 | 499200 |
| operator_discovery_engine | 1 | 0 | 3600s | 41:39.23 | 52224 |
| operator_novelty_frontier | 1 | 0 | 3600s | 37:35.46 | 1538304 |
| operator_seed_kernel | 1 | 0 | 30s | 0:00.23 | 28484 |

The original output markers reported executable OK for multiprobe and zero structural failures for each operator case. INPUTS, TOOLS, ELFS and WITNESSES final manifests were complete and all verified. This was one execution of each configuration; the timing observations are not a speedup comparison against the early missing-input refusals. Slurm allocated8CPUs/32GiB for the five concurrent cases; the table reports individual native measurements, not the allocation or compiler RSS.

This establishes a staging cause for these five historical failures. It does not close full-stdlib acceptance, the18 other native timeouts, cross-engine checks or production loop-region qualification. The retained ELFs use raw11332 and the original opt-in loop-region mode1. Those compiler/lifetime limits remain separate.

## Durable custody

Slurm19619 ran afterany:19611, so the collector would also preserve a failure or timeout rather than discard it. It completed0 and retained exact statuses plus final guards. Slurm19621 completed0 and copied the archive with SHA-256 and length readback.

`/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/services/pireus-staging-export09/pireus-input-closure.tar.gz`

SHA-256: `89fe070bc18d27a19cb7b48c1d441d2a0de3c8b410b3cc323f3b18abdd2485c1`

Size:225110525 bytes. Custody STATUS0, COPY_STATUS0, READBACK.verify0.

The archive contains both complete staged trees, all five original retained ELFs and copied ELFs, original source/ELF/raw/native-tool manifests, the original730-case baseline result, all stdout/status/resource receipts, input audit/delta manifests, raw11332 and the native platform DSO. The queued collector records its own custody status separately from the producer's native verdict.
