# Observational Madaros pathname collector

Run `bash scripts/dev/observe_madaros_inputs.sh ROOT NEW_OUTPUT TRACER -- COMMAND...`
as the non-root build user on the Slurm worker. ROOT must contain bin, scripts,
self-hosted and stdlib. OUTPUT must be a new directory outside ROOT. Provision an
attested Linux strace explicitly; host availability does not imply worker availability.
No tool installation or build command is implicit in the collector.

## What it records

- Selected pathname/process/cwd syscalls for the command and followed descendants.
- Failed lookups, unfinished/resumed events, signals and process exits in raw trace.
- Tracer version/hash, collector hash, requested argv hash, identity/namespaces/cwd.
- The existing compiler cache environment fingerprint, as a hash only.
- Broad regular-file hashes and NUL-delimited metadata, including literal symlink
  targets, before and after the command. Symlink targets are not silently followed.
- Separate child stdout/stderr and tracer diagnostics, numeric traced exit and
  independent collector outcome. Compression includes raw/archive identities and
  decompression readback; raw files are retained in private scratch.

`-s 0` intentionally truncates ordinary argument strings, not pathname decoding.
It is not unlimited-string capture or recoverable argv identity. `-f -q -yy -xx`
and file/process/cwd filters retain failures and descriptor context without
filtering to successful or already-known paths. A shell shim redirects child
stdout/stderr separately. Its requested argv is bound by a NUL-delimited hash.
No full environment is dumped. Paths/fd annotations may still be sensitive;
raw traces and command streams belong in private custody, not automatic Git upload.

## Exit and readiness

A nonzero traced status is preserved numerically. This may be a command failure
or tracer startup failure; it does not prove the child was reached. Signals are
not re-raised by this shell wrapper. After a successful traced command, collector
failures use70. Final receipt-write failure cannot produce successful exit.
`OBSERVATION_CAPTURED_CACHE_AUTHORITY_FALSE` is printed only after successful final
receipt creation, matching endpoint inventories, verified compression and empty
tracer diagnostics. Failed/incomplete artifacts remain available for diagnosis.

## Limits and cache policy

This is an instrumented syscall observation, not a complete compiler-input or
content-consumption manifest. An open path does not establish bytes consumed.
Endpoint equality is not proof against transient modification/restoration.
Memory, inherited descriptors, network/stdin, external symlink targets, namespace
changes, undecoded syscalls and lookup precedence remain completeness concerns.
The existing environment fingerprint is not a complete environment closure.
Ptrace changes timing/scheduling; resource receipts are not uninstrumented benchmarks.

The collector neither creates nor narrows a cache key. It does not modify the
canonical builder or cache script, remove guards, authorize reuse, or change
no-cache policy. The existing broad key remains authority. Future observational
analysis must account for negative lookup candidates before any narrower cache
proposal. See the [manifest candidate](../../docs/plans/COMPILER_INPUT_MANIFEST_CANDIDATE_2026_10_22.md).

If TRACER is a launcher, its file hash binds the launcher only; separately retain
the native tracer, loader and library/package identities. The first observed run
uses such a bound launcher to avoid changing global loader settings.

## Observed execution checkpoint

Slurm15940 completed0:0 with traced status0, captured_endpoints_equal, empty tracer
diagnostics, lossless archive readback and existing scientific example execution.
The compiler used the prior validated immutable source snapshot; the collector
was a separately hashed overlay. No cache validity or complete-input claim follows.
[Execution receipt](../../docs/plans/COMPILER_INPUT_OBSERVATION_EXECUTION_2026_10_22.md).
