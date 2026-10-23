<!-- docs:meta
topic_id: repo.docs.evidence.nb2-residual-bars-2026-10-23.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.nb2-residual-bars-2026-10-23.readme
-->

# Actual NB2 signed residual bars — 2026-10-23

Main basis94866e463d. Initial consumer/compiler build source99082d7bcc; corrected
consumer source13c825f7a7. The corrected consumer uses the original fresh compiler,
whose Git inputs bin/scripts/self-hosted/stdlib did not change in the correction.
No compiler rebuild is claimed for the retry.

Attempt1 Slurm16017: fresh compiler build0, nine typechecks0, then real native
consumer exit1 at the horizontal-axis budget. Attempt2 Slurm16019: two typechecks
and two real native programs exit0 after changing only the example's horizontal
axis interval count. The shared axis guard was preserved. No assertions, fixtures,
unit-test suite, numerical oracle or pixel comparison were added/run here.

`native/` contains actual report/HTML/SVG byte extracts, compressed original stdout,
resource receipts and program identities. Both SVGs parsed with xmllint; full-aspect
PNG rendering and root/independent visual inspection found no blocking issue for
the corrected residual bars. Vector originals remain here; PNG previews are local
export artifacts. The figure remains descriptive with supplied NB2 parameters.

Manifest endpoint checks cover listed inputs, not all files or immutability.
The wrapper adds bin/madaros-linux-x86_64 and bin/.madaros-linux-x86_64.verified.
The materialized prebuilt ELF matches its bound gzip; it is distinct from the
explicit fresh MADAROS_RAW_BIN used for compilation. After-inventory and hashes
retain this distinction. No narrower cache key or cache reuse policy is activated.

Source, temp/cache, logs and outputs use /scratch ext4 on the pinned cpuops worker.
Training quota was0 free before and after; no job outputs or source packet were
written there by this recipe. The Slurm scheduler still uses a separate Ceph PVC,
so this is not an all-Ceph independence claim. No quota change or historical removal.

Private compressed custody:
/home/devsounio/sounio-artifact-archive/nb2-residual-bars-20261023
It retains the complete original source archive, corrected consumer, control
recipes, compressed executable readbacks and both attempt receipts. Raw execution
files remain in scratch; uncompressed runtime ELFs were not duplicated into custody.
This is host-local custody, not an independent physical backup.

Full before/after source inventories remain in private custody. Public inventory
identities, selected critical-input hashes and the exact additions patch avoid
duplicating the full inventories in Git. The complete source archive is bound
by its snapshot hash and frozen Git revision.
