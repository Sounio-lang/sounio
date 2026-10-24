<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-inference-scale-2026-10-24.benchmark-drivers.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-inference-scale-2026-10-24.benchmark-drivers.readme
-->

# Exact benchmark-driver custody

This folder preserves source bytes and transformation provenance. It contains no
ELF, compiler binary, numerical recomputation or new performance calculation.
`.sio.txt` files are exact copies of the named Sounio sources, with the extension
changed only to avoid treating evidence snapshots as executable corpus entries.
SOURCE_MAP.tsv records original scratch paths and uncompressed source SHA256.
FILES.sha256 verifies all files in this custody folder; compressed manifests are
lossless copies of the original files recorded in SOURCE_MAP.

## Variants and execution scope

- **B199-original/before-driver.sio.txt** is the preserved original full02
  qualification driver for n256, alpha0, B199 and the original independent4096-count
  literal/seed schedule. Prior frozen qualification executions are separate
  evidence; this source copy does not assert a new benchmark execution.
- **B199-original/after-driver.sio.txt** is the source-only nested-entry candidate
  produced by recipe/transform.mjs. Its after-B199 benchmark is **prepared, not
  executed or accepted by this custody operation**. No three-pair B199 performance
  result is inferred from either B31 pilot.
- **pilot02-B31** preserves the actual before/after driver inputs to the first
  one-pair B31 pilot. Both use the original dataset/case selector and seed schedule.
  The smaller pilot changes explicitly declared budgets only: B199 to B31 and
  max_profile_attempts400 to64. Execution/performance receipts live separately
  under inference/pilot-B31; these are source inputs, not a new run.
- **pilot03-B31** preserves the actual before/after inputs to the subsequent
  one-pair B31 pilot. Its candidate library snapshot adds replica-null warm start
  while keeping the observed-null search cold. Driver-byte identity between
  pilots, where present, does not imply equal linked library snapshots: RUNS.tsv
  and the original input manifests bind those snapshots separately.

Each pilot directory includes its original submission/job, compiler identity,
RUNS manifest, process-level status and compressed source-input manifests. These
files identify the actual executions without copying binaries or entire trees.
A status0 receipt is not a statement of calibrated inferential validity, global
MLE certification, generic speedup or achievement of the B199 performance target.
Uniform/proposal equality cannot be inferred from drivers that emit only final
RNG/hits/B. Mathematical library revisions and compiler identities must also match
before describing a future replay as source-equivalent.

## Reproduce the source transformation

The recipe performs text edits, hashes and an exact inverse check; it does not
sample, fit or calculate a numerical reference. On a writable scratch surface:

```sh
cp B199-original/before-driver.sio.txt /scratch/path/original.sio
node recipe/transform.mjs /scratch/path/original.sio /scratch/path/nested.sio
cmp /scratch/path/nested.sio B199-original/after-driver.sio.txt
```

The generated JSON contains replay-local file paths, so compare its source and
transformer digests rather than requiring path-specific JSON byte identity.
The archived recipe/TRANSFORM.json is the original execution record. Its SHA256
identities describe the original and generated Sounio bytes.

B31 copies can be independently recovered from either B199 variant by replacing
only the two declared budget literals, then checking exact bytes against the
corresponding pilot input. Runtime nesting still validates the explicit embedding;
text transformation is not a substitute for that check. Compile/native execution
and any numerical or performance verification remain Slurm-only under ADR009.
