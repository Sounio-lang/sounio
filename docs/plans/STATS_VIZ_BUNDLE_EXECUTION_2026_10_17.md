<!-- docs:meta
topic_id: repo.docs.plans.stats-viz-bundle-execution-2026-10-17
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.stats-viz-bundle-execution-2026-10-17
-->

# Compile-once statistics/viz bundle — 2026-10-17

## Interface

scripts/dev/build_stats_viz_bundle.sh builds a packaged run-pass entry into a
new directory outside the runtime package. It requires a Slurm allocation.
Example inside a prepared runtime package on the worker:

    bash scripts/dev/build_stats_viz_bundle.sh \
      tests/run-pass/viz_statistical_interval_forest.sio /path/to/new-bundle

Subsequent executions on the worker:

    /path/to/new-bundle/run

The launcher verifies program.sha256, then executes program.elf directly.
It forwards program arguments and preserves the caller's working directory.
It does not call a compiler, scan the source tree, or automatically rebuild.

The bundle contains the ELF, checksum, compile log, complete input manifests,
source entry, compiler checksum, Slurm job ID, launcher and readiness marker.
Input manifests are compared before and after compilation; a changing package
cannot produce a ready bundle. Builds pin committed Madaros and the packaged
stdlib. These are explicit artifacts, not an automatic cache for mutable source.

## Package compatibility

The package builder now normalizes the legacy suite symlink into a regular
file. Job 14770 exposed this compatibility issue before compilation and exited 2.
The retry uses the same hard-linked source archive and performs exactly that
normalization before constructing its input manifest; its batch script records
the operation. No source archive bytes were duplicated for retries.

Job 14771 compiled the ELF but correctly refused readiness because the generated
bin/.madaros-linux-x86_64.verified stamp changed. Compiler and source hashes
otherwise matched. The builder now excludes only that derived verification stamp,
while retaining compiler bytes, wrappers, stdlib and entry sources in the receipt.
Job 14776 applies the corrected builder as a checksum-recorded overlay to the
same source archive. The effective inputs are captured before and after build.

## Validation protocol

Accepted job 14776, root:
 /var/lib/cephfs-slurm/training/sounio-foundry/runs/stats-viz-bundle-20261017-r3

- Build the forest interval fixture once, recording cold build RSS and time.
- Remove the package's bin directory from its canonical path.
- Invoke the ready bundle three times from /tmp.
- Require fixture success and compare complete stdout byte for byte.
- Alter the ELF, require launcher exit 78 and no program stdout, then restore it.
- Verify the restored ELF checksum.
- Record ELF identity, source identity, measurements and bundle disk size.

The fixture exercises Student confidence bounds, explicit interval metadata,
input rejection, Canvas pixel assertions and SVG geometry. This validates this
entry and bundle protocol, not all possible native program dependencies.

## Accepted execution and measurements

Slurm 14776: COMPLETED, exit 0:0, elapsed 64 seconds, BUNDLE_REUSE_PASS.
One cold build plus three warm executions. C++23 summary.cpp reads metrics.tsv:
- Cold build: 3036684 KiB peak RSS, 63.30 seconds.
- Warm execution medians: 3168 KiB peak RSS, 0.01 seconds.
- Bundle disk allocation: 1254 KiB, including ELF and provenance.

The cold phase includes materialization, input hashing, compilation and receipt
generation. It is not directly comparable to the earlier pure-compile timings.
These measurements describe one short fixture on a shared worker. Reuse avoids
compilation by construction; no general throughput ratio is claimed.

All three stdout hashes match the earlier visual fixture's complete stdout:
45ca4f64396199179ef8007813ea877033250ed41a2fe8919ad8e4f75a677f89

The bin directory was absent at the canonical source path during all warm runs.
An altered ELF returned 78 before program output; the original ELF was restored
and its checksum checked successfully afterward.

Retained native ELF SHA256:
52afed4d602861d6312d8c832464b789d95516aa3a8d90142afd0d468e1b3e6e

Compiler SHA256:
125c25703ac4062fc6befb7acf77e945abb13405f0adb5ab09fbc5a8d4b5b29f

Input manifest SHA256:
5c6e59b71fd8a43038fbec8c65f17e0c5c992e710f45684be20bad8e360c9540

## Source boundary

No .sio source changed during this stage. The prior four-test acceptance remains
job 14757; this new job validates bundle construction/reuse and the forest entry.
Worktree baseline f5b2c585; remote main rechecked at 6861acf. No rebase, commit,
push, PR, or merge. The active worktree remains remote and isolated.
