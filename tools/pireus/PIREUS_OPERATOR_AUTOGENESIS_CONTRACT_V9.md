# Pireus Operator Autogenesis v9 Contract

Status: GARDEN CONTRACT
Concept-ID candidate: `SOUNIO-PIREUS-OPERATOR-AUTOGENESIS`
Claim ready: no

## Authority

Sounio is the sole semantic authority. The mandatory order is:

```text
GARDEN -> SOUNIO_EXECUTABLE -> SEMANTICS_FROZEN -> PARITY_OPEN -> CLAIM_READY
```

Lean 4 is formal parity, Koka is effect parity, C++ is material parity, and
Haskell is an optional denotational baseline. External LLMs are review-only.
Python and Rust are forbidden as semantic oracles.

## Frozen Parent Boundary

The first executable must bind all of:

- v8 Sounio source SHA-256
  `f90f7fb48fb2e8f79142c43d69a30623289f88ffca516157423ae071539249ac`;
- v8 semantics SHA-256
  `db63ada1b919dbf869bf3e74163a64acaf7c6d2ec496fb964b6ca0c689c5f508`;
- v8 freeze receipt SHA-256
  `871544da76207ff1ea8c3e6c92af2efaaa80ee0dcf01e1f728d24522045b578a`;
- v8 frozen evidence SHA-256
  `6c58170ce4712b1944892cdd488781df29cdeb483712c9b28bf1621214f8f958`.

The v9 source must not contain a prefilled copy of the v8 seed. It must parse
the seed from the hash-bound frozen Sounio process evidence and replay all 256
cells.

## Semantic Types

The Sounio executable must represent at least:

```text
OperatorGenome
  id
  parent_id
  mutation_id
  arity
  value_carrier
  operator_type
  linearity_profile
  address_space
  address_program
  coefficient_program
  reduction_program
  parenthesization_program
  equivalence_contract
  preserved_laws
  intentionally_unpreserved_laws

MutationWitness
  source_kind
  source_digest
  opcode
  parameters
  reconstruction_checks
  reconstruction_failures

GeneratedOperator
  genome
  basis_evaluation_digest
  finite_function_evaluation_digest
  invariant_profile_digest
  quotient_representative
  separating_witness
  admissible
  rejection_reason
```

Concrete Sounio struct names may differ, but no field above may become implicit
or be reconstructed from target-specific metadata.

## Initial Mutation Opcodes

The matcher-free executable may generate only declared operations:

- `RESIDUAL_TWIST`;
- `AFFINE_ADDRESS`;
- `COEFFICIENT_LIFT`;
- `LEFT_ARITY_LIFT`;
- `RIGHT_ARITY_LIFT`;
- `REDUCTION_TREE`.

Every opcode has a total validator. Unknown opcodes fail closed.

The first executable admits only genomes that are linear in every declared
input slot. Coefficients are integer structure constants derived from frozen
`F2` sign cells and declared constants; they cannot depend on operand values.
`finite_function_evaluation_digest` remains absent. A future nonlinear genome
must declare a finite value carrier and a complete function certificate, or be
rejected.

## Execution Certificates

### Lineage certificate

Requires exact parent source, semantics, freeze receipt, and process evidence
hashes; complete v8 seed reconstruction; no recursive parent execution; and no
prefilled seed.

### Generation certificate

Requires deterministic genome IDs, mutation reconstruction, no duplicate
genomes, complete rejection reasons, and a digest over all generated and
rejected entries.

### Evaluation certificate

Requires two independent Sounio evaluation paths for each admitted operator.
Multilinear bilinear operators require 256 one-support basis products.
Multilinear ternary operators require 4096 basis triples. These checks determine
the integer structure constants of the declared multilinear map; they are not
called a complete truth table over `Z^16`. Dense probes may supplement but
never replace the complete basis certificate. Nonlinear operators are refused
unless a later contract supplies an explicitly finite value carrier and a
complete function certificate.

### Equivalence certificate

Requires an explicit finite transformation set, identity, closure, inverse
checks, complete orbit membership for the bounded frontier, and a separating
witness for every nonmembership result.

### Law-profile certificate

Records only checked laws. Associativity and parenthesization equivalence
default to false or unknown, never true by optimizer convention.

### Authority certificate

Requires Sounio producer role `SEMANTIC_AUTHORITY`; all parity, material,
historical, priority, and claim-ready flags remain false in the first
executable.

## Forbidden Inputs

The request is rejected before generation if it contains:

- expected frontier size;
- expected admitted or rejected genome;
- expected class count or representative;
- expected truth table, digest, invariant, or transcript;
- a nonlinear candidate paired only with a basis certificate;
- target instruction, target cost, or performance evidence;
- a parity-language result or proposed semantic write;
- an LLM review promoted to an authority result;
- a historical novelty, priority, or claim-ready request;
- a founder waiver without scope, purpose, and expiration;
- Python or Rust authority.

## First Executable Acceptance

A v9 first executable is accepted only if:

1. the Garden and this contract are committed first;
2. the source contains no exact result matcher;
3. the native Guardian authorizes execution before process launch;
4. v8 frozen parent evidence is replayed and content-bound;
5. every generated genome is reconstructible from parent plus mutation;
6. evaluation and quotient certificates are complete for the bounded frontier;
7. all negative cases pass;
8. two independent Sounio executions are byte-identical;
9. no Python, Rust, raw ELF, parity result, target result, or LLM verdict is
   used as an oracle;
10. `claim_ready=false`.

## Freeze Acceptance

Only after a committed first transcript may an exact matcher be added. The
frozen transcript must preserve the first transcript as an exact prefix and
append only a causal match suffix. The freeze receipt binds source manifest,
parent semantics, first result, semantic material, toolchain, hardware,
commands, Guardian decisions, and negative decisions.

## Parity Boundary

After freeze:

- Lean may prove the declared finite reconstruction and equivalence theorems;
- Koka may reproduce mutation and authority effects;
- C++ may realize and measure frozen candidates on canonical hardware;
- Haskell may supply an optional denotational baseline.

No parity artifact may select a candidate, rewrite the genome, fill an expected
result, or promote a claim.

## Canonical Targets

Material receipts may be produced only for the frozen candidate set on:

- Intel Xeon;
- Apple Silicon;
- NVIDIA DGX;
- dual AMD Alveo U250.

Missing target evidence is recorded as missing. It is never interpreted as
zero cost, unsupported semantics, or permission to change the operator.
