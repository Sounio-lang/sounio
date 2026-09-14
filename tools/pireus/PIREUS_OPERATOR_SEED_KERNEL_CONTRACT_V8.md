# Pireus Operator Seed Kernel Contract v8

Status: `SOUNIO_EXECUTABLE`

Concept-ID: `SOUNIO-PIREUS-OPERATOR-SEED-KERNEL`

Semantic-Lane-ID: `pireus-operator-genome-v3-20260829`

## Authority chronology

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> PARITY_OPEN
-> CLAIM_READY
```

Sounio is `SEMANTIC_AUTHORITY`. Lean 4, Koka, C++, and optional Haskell may
run only after a Sounio semantics freeze identified by hash. External LLMs are
`REVIEW_ONLY`. Python and Rust are forbidden as generators, oracles,
validators, freeze producers, and parity legs.

The first committed source is matcher-free. It contains no expected probe
output, basis transcript, operator digest, IR digest, result transcript, or
frozen matcher.

The rich operator result is private to the Sounio authority module. The module
emits the complete transcript while seed, IR, probes, basis witnesses, and
digests are still local, then returns only a compact scalar summary. No nested
operator aggregate crosses the module ABI and no external printer reconstructs
semantic values.

## Parent admission

v8 binds the exact v7 source, freeze receipt, and frozen Sounio process
evidence by full SHA-256. It also checks that the freeze receipt names the
expected v7 semantics hash. The v7 evaluator is not called recursively inside
the v8 process. The outer gate must first run v7 as a separate Guardian-
authorized process, verify that it matches its frozen transcript, and only
then launch v8. v8 extracts the seed words and all 256 seed cells from that
hash-bound transcript; no seed value is embedded in the v8 source.

Admission requires:

```text
parent.valid
parent.error == 0
parent.failures == 0
source_file_sha256 == frozen_source_sha256
freeze_file_sha256 == frozen_freeze_sha256
process_evidence_sha256 == frozen_evidence_sha256
process_evidence.result_valid == true
process_evidence.frozen_match == true
parent.outcome == OperatorSeed
parent.existing_bridge == false
parent.residual_weight > 0
all 256 seed cells are bits
all 256 packed-word/cell replays match
```

Missing process evidence, a recursive parent-evaluation request, or a
pre-filled parent seed fails closed. No target evidence participates in
admission.

## Operator semantics

Indices are `F2^4` encoded as `0..15`. For seed table `S` and exact integer
vectors `a,b`:

```text
output[d] = sum_(i=0..15) (1 - 2*S(i,i XOR d))*a[i]*b[i XOR d]
```

This is a bilinear map `Z^16 x Z^16 -> Z^16`. No algebra laws are inferred.

## Kernel IR

The canonical term order is:

```text
term_id = 16*d + i
destination = d
left = i
right = i XOR d
source_cell = 16*i + right
coefficient = 1 - 2*S(i,right)
```

The executable checks 256 unique term IDs, 256 unique source cells, 16 groups
of 16 terms, 256 XOR addresses, 256 coefficients, and 256 sign round trips.

## Independent execution paths

The table evaluator scans `(left,right)` in row-major order and accumulates at
`left XOR right`. The IR evaluator reads only compiled term arrays in
destination-major order. Both use exact signed 64-bit integers. Their execution
paths are independent, while the `0 -> +1`, `1 -> -1` sign convention remains
one Sounio semantic definition rather than a second oracle.

Three dense input pairs are fixed in the Garden. The first source contains
their inputs but no outputs. For each probe it checks 16 path equalities and
32 explicit `16*A*B` bounds. This is a hard exact-integer safety envelope, not
a claim that every probe or destination attains the bound.

The basis sweep evaluates all 256 ordered pairs `(e_i,e_j)`. Every result must
have exactly one nonzero lane at `i XOR j`, with the seed-derived coefficient;
the other 15 lanes must be zero. The complete sweep performs 4096 lane checks.

## Generated candidate

A valid result produces one `GeneratedOperatorCandidate` carrying the exact
seed words, operator kind, parent outcome, kernel digest, execution digest,
and future-parent eligibility. The execution digest binds both dense-probe and
complete basis-sweep digests. Eligibility means only that later Sounio work may
name this candidate explicitly. It is not a broad or historical novelty claim.

Kernel, probe, basis, receipt, and negative-contract counters are included in
the content-addressed digest graph. These digests bind live Sounio results; the
first source compares none of them against an expected value.

## First transcript

The first Sounio execution emits:

- three frozen-parent identities, exact file/receipt-match fields, and
  externalized process-evidence admission fields;
- all 256 seed cells;
- the complete 256-term IR;
- all 256 basis witnesses;
- three input pairs and both 16-lane outputs;
- the generated candidate and bounded receipt;
- an ordered 35-case request/admission refusal vector;
- seven digests;
- `claim_ready=false`.

Only after this transcript is committed may the source gain an exact matcher.

## Negative contract

The Sounio request layer refuses non-Sounio authority, missing policy, wrong
stage, absent/unfrozen/unbound parent, missing parent process evidence,
recursive parent evaluation, pre-filled parent seed, injected expected
results, target or material evidence, cost/performance evidence, parity
writes, review promotion, novelty/priority/claim promotion, invalid waiver,
bridge-as-seed, zero seed, and malformed seed. The native Guardian separately
denies forbidden processes before launch.

## Claim boundary

A passing first execution establishes only that Sounio lifted the exact
frozen v7 residual into the declared executable twisted-XOR operator and
destination-major IR and produced exact basis and dense-probe outputs.

It does not establish algebraic, material, performance, historical,
scientific, or priority novelty, and it does not reach `CLAIM_READY`.
