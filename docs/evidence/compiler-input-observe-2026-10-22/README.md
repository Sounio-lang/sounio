<!-- docs:meta
topic_id: repo.docs.evidence.compiler-input-observe-2026-10-22.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.compiler-input-observe-2026-10-22.readme
-->

# Instrumented compiler pathname observation

Slurm15940 completed0:0. Collector source3860980c8d16f513b9ba3e779c397d9afc0b4fd3 is a separately hashed overlay. Compiler source uses the existing immutable snapshot from nb2-balanced-current-source-20261022, frozen eeeab3025c68bebfdbde284116e41176b9976770. The snapshot was validated/reused without a new archive. The only run-local runner change invokes the overlay instead of the generic corpus-smoke command. No tests/corpus acceptance are claimed.

The fresh compiler build ran under strace6.8. Traced status0, collector captured_endpoints_equal, empty tracer diagnostics, verified lossless trace compression, and matching broad source content/metadata inventories are separate recorded facts. They do not establish complete content-input closure or authorize any cache key/reuse change.

Raw traces, command streams, full inventories, package bundles and compiler binaries remain in private t560 ext4 custody: /home/devsounio/sounio-artifact-archive/compiler-input-observe-20261022. They are not embedded here. Trace raw/archive hashes and sizes are metadata; they do not expose the underlying private pathname records. Provisioning/trace/inventory decompression identities were checked before publication.

The scientific example typechecked, compiled and executed after the observed build. Its ELF/stdout identities equal the previous source's artifacts; that is byte identity, not independent numerical concordance. Native stdout here contains synthetic report/figure output only, losslessly compressed. One native-process resource observation is recorded separately from whole instrumented-job accounting.

Shared run size was110568 apparent bytes. No new source.tar.gz was created; large artifacts use scratch/ext4. Existing quota remains3TiB; no historical removal was performed. Ptrace changes timing, so no uninstrumented speedup is claimed. Broad cache policy remains unchanged.

Scoped tracked-source comparison and original declared bootstrap/recipe identities are recorded separately. The normal builder cleans up its temporary generated seed without retaining a content hash; observed execution paths do not close that provenance gap. Future complete manifests must bind generated intermediates.
