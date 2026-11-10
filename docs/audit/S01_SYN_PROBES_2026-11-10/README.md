<!-- docs:meta
topic_id: repo.docs.audit.s01-syn-probes-2026-11-10.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.s01-syn-probes-2026-11-10.readme
-->

# Syntax probes — raw material, 2026-11-10

This directory contains the executable evidence for `SYN-001`–`SYN-016` in
chapter 01. The probes are measurements, not a new language conformance suite:
they do not change a rule's standing, and `SYN-016` remains an open decision.

`cases.tsv` binds exactly one positive, deliberately partial witness to every
rule. It is not an exhaustive cross-product of every construct mentioned by a
rule. Where the chapter's original evidence fixture did not isolate the parser
branch, this directory supplies a small probe. `run_probes.sh` executes every
case under a current-source Madaros ELF and the actual `lean_single` public
route. It:

- scrubs inherited compiler-routing variables with `env -i`;
- invokes Madaros with `--science-boundary off` because these are syntax-only
  witness probes with no science manifest or claim contract; the manifest records
  this exclusion, while `lean_single` remains on its unmodified public route;
- records the full SHA-256 of both compiler ELFs, source revision, fixture,
  routing scripts, build log, and clean/dirty patch state;
- keeps front-end and native execution exit statuses separate;
- applies the execute bit to a nonempty native output before the separately
  recorded execution phase (the raw `lean_single` compile ABI does not);
- preserves raw stdout and stderr instead of condensing shell variables; and
- compares complete stdout bytes against independent golden files;
- refuses to reuse an existing run directory or native artifact; and
- fails if a rule is absent, a case is duplicated, an engine route cannot be
  attested, or a positive witness does not compile/run as required.

Run it on Linux after building Madaros from the same source revision:

```bash
mkdir -p .build/sio42
bash scripts/ci/build_modular_madaros.sh .build/sio42/madaros-current \
  >.build/sio42/madaros-build.log 2>&1
MADAROS_RAW_BIN="$PWD/.build/sio42/madaros-current" \
MADAROS_BUILD_LOG="$PWD/.build/sio42/madaros-build.log" \
  bash docs/audit/S01_SYN_PROBES_2026-11-10/run_probes.sh
```

The generated `runs/<run-id>/results.tsv`, `manifest.tsv`, and `logs/` are the
primary evidence. `WITNESS_PASS` means only that the named partial witness
satisfied its declared observation; it does not mean the entire `SYN-NNN` rule
was confirmed. A nonzero harness status does not erase the receipt; it means
one or more positive witnesses failed and must be represented by a tracked task
before the work card can close.

Canonical source-bound run: [`runs/20261110-27aa1274-canonical/`](runs/20261110-27aa1274-canonical/)
on Slurm job `22715`. Native artifacts are intentionally not committed; their
SHA-256 values remain in `results.tsv`, while the manifest, build attestation,
raw front-end/native logs, and exact expected stdout files are retained.
