<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-inference-scale-2026-10-24.compiler-effectbridge.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-inference-scale-2026-10-24.compiler-effectbridge.readme
-->

# Current-source compiler build, guards and fresh selfcheck

Source commit92c9e22b71997c413b5c03cfa0ceeab89f148f7b includes merged main8e2109063d
and the narrow effect/borrow repairs. The input manifest, source commit, build
scripts and resource receipts are preserved here. Full source and ELF remain at
`/scratch/sounio-stdlib-inference-scale-20261024/compiler_effectbridge01`.

The generated compiler is
`d74fbebe0ff442e8c77bec024d6d92325b83a98d3f1487caa1db59ef171a41e6`.
The private-function identity source is258a87a1daa002b2d5f9805afcd761dae387f533671f4b8f5780f83150a26ee6;
the checker source is2420eda8ecc3ab7001048fef56f3305958ecadf12b5f223f87d99e2ccd502937.
A read-only recursive comparison of the compiler `self-hosted` tree against the
CENTRAL snapshot257fb0d0a25abef4633bcbe7ac8163f1b2a9983d found no byte differences;
this is not a claim that every scientific stdlib input in those snapshots matches.

## Separate execution states

- **16983:** compiler build itself exited0, elapsed6:54.67 and maximum RSS736944KiB.
  The enclosing job subsequently failed when its guard harness attempted chmod
  BEFORE compiling the first guard ELF. Its `artifacts/STATUS=FAILED1` and original
  stderr remain preserved. This is an operational guard-harness failure, not a
  successful whole-job result. No expensive rebuild was performed afterward.
- **16985:** dependent selfcheck could not start after16983 failed. Its pending
  state was preserved, and only that owned job was cancelled before execution.
- **16992:** separate guards-only retry used the existing hash-bound compiler.
  Both positive native guards passed, and all seven negative guards rejected
  with expected exit1 and the specified E175 checks. Source/compiler hashes before
  and after passed. Its acceptance marker is `guard_retry01/STATUS=COMPLETE`.
- **16993:** a separate16GiB selfcheck allocation, dependent on the passing guards,
  ran the NEW compiler on `check self-hosted/compiler/main.sio`. Acceptance requires
  selfcheck_retry01/STATUS0, checkOK and input/compiler hash verification before
  and after. This is selfcheck evidence, not a gen2/gen3 byte-identical fixed point.

The earlier baseline, failed hypotheses and integrated old-compiler selfcheck
16973/16976/16979/16981 are maintained separately by the companion
compiler-selfcheck-scopes custody, not duplicated or reclassified here. The source
repairs propagate real effects/borrow ownership; no checker rule was relaxed.

This folder contains compact copied text receipts, stdout/stderr, resources,
script and source hashes. No ELF, cache or full source archive is included.
SOURCE_MAP.tsv records original paths, ORIGINAL_FILES.sha256 their original byte
hashes. Large text is losslessly gzip-compressed, so map paths precede optional
`.gz` extension. FILES.sha256 is the authoritative final copied-file manifest.
No numerical recomputation or additional native execution occurred during copying.
