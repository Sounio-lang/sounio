<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-inference-scale-2026-10-24.compiler-selfcheck-scopes.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-inference-scale-2026-10-24.compiler-selfcheck-scopes.readme
-->

# Compiler self-check scope controls

`selfcheck-scopes.tar.gz` is the compact evidence package for the closure-borrow
and private-function effect repairs. It contains diagnostic logs and identities,
not a compiler executable, compiler rebuild receipt or full source archive.
The final compiler and subsequent integrated science gate have separate custody.

All checks use source-bound native compilers under Slurm:

- **16973 baseline:** old8c and new92 both reject the same source690dc with one
  closure E038 and four private-function E035 errors. The closure failure is not
  introduced by the nested-record owner fix.
- **16976 rejected hypothesis:** explicit effect-array literals leave the same
  E038. This unsuccessful candidate is preserved and was removed from production.
- **16979 isolated bridge:** the closure E038 disappears; exactly the four PFI
  effect errors remain. Exit1 here is a scoped control, not an integrated pass.
- **16981 integrated:** closure bridge2420 plus PFI258a passes full
  `check self-hosted/compiler/main.sio`, with exit0, verdict0 and `check: OK`.
  Inputs and compiler hashes are verified. The measured complete self-check took
  3:26.69 with maximum RSS5366112KiB; this is one observation, not a benchmark.

The original caller wrapped `&!c` inside a pointer cast. Existing call-scoped
borrow release did not recognize that cast-wrapped argument, leaving the root
borrow active before `&!c.env`. A small borrowed bridge now accepts `&!Checker`
directly and casts only within the call; the pointer does not escape. No checker
criterion, effect inference or type-safety rule was weakened. See the archived
`DIAGNOSIS.txt`, patches and raw diagnostics for the precise source evidence.

Integrated source identities:

- `check.sio`: `2420eda8ecc3ab7001048fef56f3305958ecadf12b5f223f87d99e2ccd502937`.
- `private_fn_identity.sio`: `258a87a1daa002b2d5f9805afcd761dae387f533671f4b8f5780f83150a26ee6`.
- Compiler used: `92d2108ef595c5382d582aa363df6e95dab907a554f039d5f8f008073e535c21`.

The first integrated submission reply timed out. The accepted job16981 was
recovered by its unique queue name; no duplicate submission was made. Its raw
submission failure and recovery receipt are retained.

Check `selfcheck-scopes.tar.gz.sha256` before extracting. The archive contains
its own independently verified `MANIFEST.sha256`; this directory's manifest also
covers the archive, sidecar and README. A successful source self-check is distinct
from rebuilding the compiler and executing the final scientific suite.
