<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-sio-177.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-sio-177.readme
-->

# SIO-177 first increment: decision and structural contract

Base: remote main a2a44df195bb8caf3fb07cbc9517275b29325e94. ADR019 records the
maintainer-approved D34 package. `epistemic::interval_contract` is an explicit
structural implementation with private endpoints and status; it is not the
`Interval<F>` compiler primitive and exposes no arithmetic operators.

The constructor rejects NaN endpoints and infinite singleton intervals,
canonicalizes empty/zero/decorations, and keeps invalid input distinct from a
valid empty set. Membership contains real finite values. Strict comparison returns
Never/Possible/Certain/refused (0/1/2/3), never an implicit Bool. Hull combines
already evaluated values only; it is not effect-checked `hull_branches`.

## Reproduction and executable evidence

```sh
MADAROS_RAW_BIN=/absolute/identified/madaros.elf \
INTERVAL_CONTRACT_OUT=/absolute/fresh/result \
  bash scripts/stdlib/interval_contract_gate.sh
```

Final Slurm17107 completed0:0, UID1000, stack524288KiB. Frozen source07 and result07:
`/scratch/sounio-stdlib-inference-scale-20261024/SIO177`.
Retained compiler SHA256:
`d74fbebe0ff442e8c77bec024d6d92325b83a98d3f1487caa1db59ef171a41e6`.
This is not a fresh build of current main. Native ELF SHA256:
`f3f94c6b5c1f7bc1da0aef1a114a32ad465e42d2e2864ae1598910e6b013e413`.

Check, compile and native execution succeeded. The independent C++23 oracle
parsed864 actual native constructor/comparison/hull rows and required exact signed
endpoint bits/status/decoration/order/count and completion. Inputs cover both
infinities, finite extrema, negative/positive subnormals, both zero signs, NaN,
singletons, reversed bounds and six decoration values. The native fixture adds
membership and invalid-result assertions. No floating-point arithmetic guarantee
or statistical coverage is inferred from these structural checks.

Controls: changed status is rejected with exit8; truncated output with exit9;
private-field forgery is rejected by the compiler with exit1 and a private-field
diagnostic. Source and selected compiler SHA256 checks pass after execution.
`contract-17107-text.tar.gz` retains final text receipts without ELFs. Earlier
`contract-17096-text.tar.gz` retains the first successful structural replay before
control exit statuses and private-field enforcement were added to the gate.

## Preserved diagnostics

- 17084/source01: check succeeds; lowering rejects nested print arguments whose
  scalar kind is unresolved. Typed i64 print locals fix the fixture interface.
- 17087/source02: native assertion fails; no success claim.
- 17090/source03 and17092/source04: diagnostic checkpoints isolate an inferred
  `let inf=bits_to_f64(...)` being negated as integer bits by retained d74. Native
  lower endpoint bits were -9218868437227405312 rather than -4503599627370496.
- 17096/source05: explicit `let inf:f64` restores correct native binary64 bits and
  passes864-row C++ concordance. 17102/source06 adds exact mutant status receipts;
  17107/source07 adds the private-field negative control. All snapshots are retained.

The compiler-kind diagnostic was sent to the existing shared compiler-file owner;
it is not asserted to reproduce in a fresh current-main build. Explicit endpoint
local types are retained in this module. No shared compiler file was edited.

## Remaining programme obligations

SIO177 remains in progress. Frontend/IR primitive integration, separate Knowledge
e/u, interval Jacobians, literal conversion and effect-checked branching are not
implemented by this increment. SIO178 still owes directed arithmetic/backends,
FP-environment checks, million-case per-operation qualification and Lean proofs.
Existing `data::interval`, `verify::interval` and private `epistemic::interval_ieee`
are distinct legacy representations, not aliases of this contract or certified
implementations of ADR019; some use round-to-nearest endpoint arithmetic. This
increment does not promote their containment claims or change their API behavior.

## Engine and CI routing

The initial PR3101 Full Test Suite failed because the bootstrap seed does not
implement the Madaros bits_to_f64 intrinsic used for exact endpoint construction.
The two contract fixtures now declare requires: madaros, with the reason in their
source. The structural gate is registered explicitly in Stdlib Deep Science,
which uses its identified source-built modular compiler and the C++23 oracle.
This does not claim seed support or replace a native check with a skipped test.
Required governance metadata was also added to the three new documents after
Contracts Core localized its absence. New executable receipts follow separately.

Slurm17135 completed the registered full Deep Science integration on frozen
source08, exit0:0, elapsed17m42s. All497 recorded phase rows succeeded, including
interval-contract and final full input/compiler hash checks. The child contract
ELF retains SHA256f3f94c6b5c1f7bc1da0aef1a114a32ad465e42d2e2864ae1598910e6b013e413.
Compact aggregate/child text receipts are in ci-replay-text.tar.gz. This run is
scoped to main a2a44 plus this interval-contract increment; it does not include
PR3102/3103/3104 or the BDF/numdiff branches. Remote CI still reruns on the new head.

The decision was initially numbered018 on the a2a44 base. Remote main c21b0e1
subsequently assigned018 to release trains. This PR renumbers D34 to ADR019 and
preserves the historical archives as originally produced; their contents are not
rewritten. The branch incorporates c21 main for the new release/seed baseline.

## Replay after remote-main synchronization

Feature merge commit8850fc1f1426 incorporates remote main c21b0e139830ca4949f80cba62887a8922a972d7. Upstream reserved ADR018 for release/versioning, so this feature is now ADR019; earlier compressed historical receipts retain their original numbering. Slurm17199 replayed the structural contract with the new CI-source compiler SHA256 a292cc263f3734b1ad6f17a2e6179b8a7874ef3e76d8f5fdfea6859290baf5d3 and passed all864 bit-exact reference rows, numerical mutation8, truncation9, private-field compile refusal1 and source/compiler integrity checks. contract-c21-17199-text.tar.gz contains the compact text receipts, excluding executable binaries.

Compiler custody: GitHub main run40144792765, Witness job121024346990, artifact12580970584, source c21. Artifact ZIP digest e3eaa0099eb039094c47afdcc3f3e22a90a6bf8a2432d0b5d4e867c486feb211 and ELF digest were checked. All5295 input hashes and tracked paths match the exact Git source archive/catalog. The completed Witness log records source-tracking seed and modular12579-function construction. The previous full Deep Science replay17135 remains scoped to the earlier snapshot/compiler;17199 is the focused current-main replay, not another complete Deep Science run.
