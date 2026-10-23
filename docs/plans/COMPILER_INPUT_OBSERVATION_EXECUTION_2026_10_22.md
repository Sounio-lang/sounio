<!-- docs:meta
topic_id: repo.docs.plans.compiler-input-observation-execution-2026-10-22
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.compiler-input-observation-execution-2026-10-22
-->

# Instrumented compiler pathname observation: execution receipt, 2026-10-22

## Separate source identities

Branch `sprint/compiler-input-observation-main-f02db-20261022` retains main
`f02db32df2f77bde26982b45d8c12c460549c3b2` and all prior science-sprint increments.
The main advance from7d67 changed documentation only.
Collector source commit: `3860980c8d16f513b9ba3e779c397d9afc0b4fd3`.

Compiler source is the existing immutable Foundry snapshot from run
nb2-balanced-current-source-20261022, frozen `eeeab3025c68bebfdbde284116e41176b9976770`.
The snapshot was validated and reused through the existing Forge --reuse-snapshot
mechanism; it was not copied to a new source.tar.gz. Collector/dispatch overlays
are separately hashed. The canonical builder/cache scripts in the snapshot remain
unchanged. The run-local runner changes exactly one corpus-smoke command to invoke
the observational overlay; original/patched runners and hashes are retained.
Generic corpus-smoke naming remains transport only; no tests/corpus acceptance.
This is not a claim that the compiler was built from the new collector Git tree.

## Observation and storage design

The wrapper records selected pathname/process/cwd syscalls and followed descendants,
including failed lookups, plus broad before/after source inventories. Strings in
ordinary arguments are intentionally truncated with -s0; pathname decoding is
separate. Version/hash/namespace/cwd and requested argv hash identify the context.
The existing build-environment fingerprint is recorded without environment dump.

It separates traced numeric status from collector outcome and retains raw/archive
hashes with decompression readback. A successful trace command is insufficient
if collection/inventory/compression/receipt creation fails. Nonempty tracer
diagnostics or endpoint mismatch prevent a captured marker. Signal semantics are
not re-raised by the shell wrapper. Raw paths may be sensitive and remain private.

All heavy output uses the cpu-ops host's /scratch, outside the shared quota:
raw/compressed traces, compiler, package provisioning and native example artifacts.
Durable compressed copies go to t560 ext4 custody. Shared Foundry storage receives
only small orchestration metadata/overlay files, with no new source archive.
No quota change or historical artifact removal is performed.

The worker initially lacked strace. The job downloads Ubuntu packages into isolated
scratch lists/cache and extracts them without system installation. A bound dynamic
loader launcher supplies extracted libunwind only to strace; no global loader
configuration or LD_LIBRARY_PATH is changed. Tracer package/bin/launcher/loader
identities are retained. The package update log contains a denied global apt cache
cleanup-hook attempt from the existing apt configuration; the job ran as unprivileged
sounio and no system package installation was performed.

## Execution

Slurm **15940 COMPLETED 0:0**, elapsed **00:07:18**, TotalCPU **07:04.184**,
batch peak RSS **2911784K**. This includes provisioning, instrumented build,
collection/compression and native demonstration; it is not an uninstrumented
compiler benchmark. Tracer package: strace6.8-0ubuntu2; extracted dependency:
libunwind8 1.6.2-3build1.1. The actual tracer reports version6.8.

Observer RESULT records traced_command_status0, collector_exit_status0 and
collector_status=captured_endpoints_equal, while cache_authority and
complete_input_closure remain false. Tracer stderr was empty. Both content-hash
and metadata endpoint inventories match by identity. Raw trace/archive bytes
were read back and compared before custody publication.

| Storage artifact | Recorded bytes/policy |
|---|---|
| Raw trace | 15888844 bytes |
| Lossless gzip trace | 383405 bytes |
| New shared run directory | 110568 apparent bytes |
| New source archive | Not created; validated existing snapshot reused |
| Large outputs | Private scratch and t560 ext4 custody |

The shared-volume observation remained62914560bytes available under the existing
3TiB quota. This is a point observation, not an attributed cluster-wide storage
change. No raw trace/compiler/package blob is committed to Git.

The fresh observed compiler hash is `0c896942018ff4be9a86d69f30e3a8aadac955f32451c445a07fbf502ae37fc3`.
The existing NB2 example typechecked, compiled and executed with exit0 afterward.
Its native-process GNU time observation: elapsed/user/system0.00/0.00/0.00s,
peak RSS1052KiB, exit0. Zero displayed centiseconds is reporting resolution,
not zero execution time. No isolated-kernel or comparative runtime claim follows.

Native ELF: `de6826dbe5bf74037a9e66c8ca92051af4d0da4ce40c6ac374890f09e9809686`.
Native stdout: `a189379a7d1a7e0e21d1b6e6cf90b0e98130cd040d645df49e01133628912788`.
These match the previous NB2 balanced run's ELF/stdout, establishing artifact
identity only. No new figure geometry or scientific arithmetic was introduced.

Trace raw SHA: `3c13a600762a47732f9e4b5f7cf4b670e8accc3dc72f6746d4f18a3ca2d43f22`.
Trace gzip SHA: `10352fe781b5251489fe032b1c223d6b6f7cdae64438c030fadfef3e1d4b86ff`.
[Compact evidence](../evidence/compiler-input-observe-2026-10-22/README.md),
[result](../evidence/compiler-input-observe-2026-10-22/observation/RESULT.txt),
[runner delta](../evidence/compiler-input-observe-2026-10-22/RUNNER.patch),
[relative hashes](../evidence/compiler-input-observe-2026-10-22/FILES.sha256).
Private raw/inventory/package/compiler custody:
/home/devsounio/sounio-artifact-archive/compiler-input-observe-20261022.

## Meaning and limitations

This is selected syscall observation, not a complete content-input closure.
Opened files can belong to hashing/provisioning/build orchestration as well as the
compiler. In particular the unchanged broad cache-key recipe itself opens the
entire source tree to hash it. A future analyzer must attribute calls to process
images/roles rather than interpret every open as a compiler import. Failed lookups
and search precedence remain relevant. Inherited descriptors, memory, stdin,
external symlink targets, namespace changes and decoding limits remain concerns.
Equal endpoint inventories do not rule out transient modification/restoration.
Ptrace changes timing and scheduling; this is not an uninstrumented benchmark.

The broad cache key/guards remain authoritative. Neither observation readiness nor
matching compiler bytes authorize narrowing the key or changing no-cache policy.
No tests were added or run. Native example success does not establish independent
numerical concordance, global accuracy, full corpus, CI or merge. The larger
statistics/scientific-rendering/math/genomics/physics/resource sprint is preserved.

Generated temporary seed content is not preserved by the unchanged builder after
its normal cleanup. Its observed execution path is not a retained content hash.
A future complete manifest must bind generated intermediates before cleanup.
The declared tracked compiler paths are unchanged between snapshot Git sourceeeeab302
and collector Git source3860980 (bin, self-hosted, stdlib Sounio and scripts except
the new observer/doc). This scoped comparison is not a complete-input-closure proof.
