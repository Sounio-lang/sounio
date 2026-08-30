# Pireus Operator Discovery Engine Contract V10

Status: GARDEN_CONTRACT
Concept-ID: SOUNIO-PIREUS-OPERATOR-DISCOVERY-ENGINE
Parent concept: SOUNIO-PIREUS-OPERATOR-AUTOGENESIS

## Authority

The mandatory order is:

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> PARITY_OPEN
-> CLAIM_READY
```

Sounio is `SEMANTIC_AUTHORITY`. Lean4 is `FORMAL_PARITY`. Koka is
`EFFECT_PARITY`. C++ is `MATERIAL_PARITY`. Haskell is an optional
`DENOTATIONAL_BASELINE`. External LLMs are `REVIEW_ONLY`. Python and Rust are
prohibited as semantic or expected-result oracles.

Parity languages may refute or compare a frozen Sounio result. They may not
create, repair, widen, or retrospectively select the semantics.

## Exact Parent

The first V10 source must bind:

```text
parent_schema=pireus-operator-autogenesis.freeze.v9
parent_semantics_sha256=8fad13c09d2b17ea1adcce0a6b89612964e80ddb4a7a576916c2c1da30286df6
parent_freeze_commit=42e42ded032c03a25c694b8675d7ed6987b9812b
parent_parity_open_commit=cd0f147e36
```

Parent facts are inputs, not re-derived expected results.

## Semantic Fragment

The first executable is restricted to bounded integer multilinear maps over
the V9 carrier and address space:

```text
value_carrier=Z_power_16
address_space=F2_power_4
coefficient_domain=Z
coefficient_programs=OPERAND_INDEPENDENT
first_executable_arity_domain={2}
future_arity_domain_may_include={3}
map_signature=(Z_power_16)^arity_to_Z_power_16
```

Every admitted genome must be total, typed, finite in representation, and
evaluable over its complete pure-tensor basis. For an arity-`r` multilinear
map, the semantic basis domain is exactly the `16^r` ordered tuples
`(e_i1, ..., e_ir)`, with a complete 16-component output vector for every
tuple. The stored semantic object therefore contains exactly `16^(r+1)`
integer structure coefficients. In the first bilinear executable this is
`16^3 = 4096`, including explicit zero coefficients. Multilinearity makes those
values determine the full map over `Z^16`. Arbitrary nonlinear `Z^16` maps are
outside V10.

The first atlas proposition is exactly the arity-2 partition represented by V9
genomes 0, 2, and 3. V9 genomes 4 and 5 have arity 3 and are outside this typed
universe. A V10 certificate therefore means
`Novel<V9_bilinear_A3,C2_diag,full_Z_bilinear_tensor,N2>`; it does not mean
separation from every V9 arity or every bilinear change of basis.

The exact V9 evidence format emits one signed nonzero integer destination coefficient
for each of 256 ordered input pairs. The V10 admission parser is hash-bound to
that parent format and expands its 256 rows into a 4,096-cell tensor with all
other coefficients explicitly zero. This is not a generic dense-atlas parser;
a denser parent requires a new Garden and input hash.

The first bootstrap requires the exact 1,270,431-byte parent transcript, parses
all 768 bilinear rows, and computes its SHA-256 inside Sounio. The
implementation compresses every complete 64-byte block directly and applies
the FIPS 180-4 final padding to the exact remaining bytes. It must set
`parent_evidence_match=true` and `parent_evidence_hash_delegated=false` before
the atlas can be admitted. A size match, a hash string quoted by the freeze, or
successful parsing without the direct digest is insufficient.

As an independent semantic cross-check, V10 also reconstructs all three
4,096-cell tensors from the exact V9 definitions `oag_rotl4`,
`oag_affine_address`, and `oag_genome_coefficient`. Atlas admission requires
exact equality with the 256 parent rows, 256 affine rows, and 256
coefficient-lift rows. The reconstruction cannot replace the transcript hash,
and the transcript hash cannot replace reconstruction.

Every affine address constructor must materialize a complete
operand-independent integer structure tensor. Its acceptance certificate must
show that evaluation has the form of a finite sum of fixed coefficients times
one component from each tracked input slot. This is the executable witness that
the constructor stays inside the admitted multilinear fragment.

## Frozen Inputs

Before the first semantic execution, Sounio must identify by hash:

- the V9 parent semantics;
- the operator atlas;
- the equivalence action domain;
- the typed mutation grammar;
- the discovery intent;
- the deterministic ordering and budget;
- the law-spectrum predicates used for admission or ranking.

No frozen input may be inferred from a later Lean4, Koka, C++, Haskell, target,
or LLM result.

## Required Types

The executable must expose semantic equivalents of:

```text
DiscoveryIntent
OperatorGenome
SemanticBasisMap
AtlasClass
EquivalenceAction
ClassSeparator
LawSpectrum
CounterexampleLineage
OperatorDiscoveryCertificate
DiscoveryOutcome
```

`DiscoveryOutcome` is exactly one of:

```text
TYPED_REJECTION
QUOTIENT_COLLISION
N2_RELATIVE_NOVELTY
SEARCH_INCOMPLETE
```

No first-stage outcome encodes historical novelty or priority.

## Finite Group Action

The first quotient domain `Q` is the frozen group `C2_diag`, not an arbitrary
list of transformations. Its nonidentity member is the unsigned permutation
matrix swapping lanes 0 and 1, used diagonally for the codomain and both
bilinear input slots. All signs are `+1`; the first executable claims neither a
nontrivial sign gauge nor a full independent-input basis-change quotient. Its
action on an arity-`r` map `F : (Z^16)^r -> Z^16` is exactly:

```text
apply(q, F)(x1, ..., xr)
  = Pout(q)^-1 F(P1(q)x1, ..., Pr(q)xr)
```

Sounio must verify the permutation and inverse witnesses, the complete group
multiplication table, identity, closure, inverses, associativity, and the
identity and composition action laws before any `Q`-orbit or quotient language
is admitted. Larger affine `F2^4` address groups and sign/cocycle gauges enter a
future `Q` only through an explicitly frozen representation.

Because every member of the first `C2_diag` is involutive, its forward and
inverse permutations are numerically identical. A future group containing a
non-involution must store and verify the inverse-output permutation separately;
the v10 executable does not claim that generalization.

If the first executable intentionally uses a finite transformation set without
all group and action laws, it returns `SEARCH_INCOMPLETE` with a
`FINITE_ACTION_SET_SEPARATION_ONLY` diagnostic. It cannot emit
`N2_RELATIVE_NOVELTY`, equivalence, orbit, or quotient language.

## Relative Novelty Predicate

For a frozen atlas `A`, action domain `Q`, and candidate `C`, V10 may emit
`N2_RELATIVE_NOVELTY` only when:

```text
forall atlas_class a in A,
forall action q in Q,
exists basis_input x,
eval(C, x) != eval(apply(q, a), x)
```

The signed-permutation group action above and the basis input ordering must be
frozen in Sounio. A separator table must contain at least one deterministic
witness for every `(atlas_class, action)` pair, unless a checked class-level
invariant theorem discharges the entire action block and names that theorem in
the certificate.

The first executable should prefer explicit witnesses over theorem shortcuts.

## Canonical Representative

Every typed executable genome receives a canonical semantic representative
under the declared finite group action. Canonicalization must be deterministic
and independent of search discovery order.

The ordering is frozen before execution and must compare complete semantic
content, not hash values alone. Hashes anchor the selected content; they are not
collision-free mathematical equality proofs.

## Mutation Contract

Mutation operates on typed genome constructors. It may:

- replace an affine address transform with another admitted transform;
- replace a coefficient program with another typed program;
- change a defined binary-tree parenthesization;
- compose bounded multilinear constructors only when a slot-linearity rule
  tracks every original input through the wiring and proves that no linear slot
  is duplicated or reused nonlinearly;
- add or remove an explicitly typed zero or residual term;
- change target-neutral layout annotations.

The first constructor family enumerates in fixed order the 3,600 tensor
coordinates whose two input indices are in `1..15`, and both unit coefficient
deltas, for a grammar cardinality of 7,200. Excluding input index 0 preserves
the frozen `e0` left/right unit equations by construction. Candidate id alone
determines the coordinate and sign without consulting atlas comparison
results. The search may stop at its frozen budget and return
`SEARCH_INCOMPLETE`; it may not require that the enumerated prefix contain an
N2 success.

Mutation may not:

- edit a result table directly;
- consult a later parity receipt;
- use target timing to change semantic admission;
- invoke an external LLM as a candidate or expected-result oracle;
- change the frozen atlas, quotient, intent, or search ordering during a run.

## Counterexample-Guided Search

Every rejected candidate produces a typed refinement record. The next mutation
step may consume only the frozen seed plus prior deterministic refinement
records from the same run.

The execution transcript must make the causal order visible:

```text
candidate_id
parent_candidate_id
mutation_opcode
admission_result
first_counterexample_kind
first_counterexample_address
matched_atlas_class
matched_action
```

Search budget exhaustion produces `SEARCH_INCOMPLETE`, never novelty.

## Law Spectrum

V10 may compute checked predicates from exact semantic tables, including:

- associativity and its first counterexample;
- commutativity and its first counterexample;
- left and right alternativity;
- flexibility;
- selected Moufang identities;
- associator and commutator support counts;
- declared rank or zero-divisor probes.

Every predicate must define its quantification domain and arithmetic carrier.
An identity whose residual is multilinear in distinct variables may be decided
on the complete pure-tensor basis grid for those variables. An identity with a
repeated variable, including raw alternativity or Moufang forms, requires a
checked polarization or coefficient-normal-form certificate over `Z`, not only
basis substitutions. Rank and zero-divisor searches over the infinite carrier
must be labeled bounded unless a separate exact theorem closes their domain.
Sampled probes cannot support an exhaustive law claim.

Law-spectrum distinction is N3 evidence only after an N3 gate. The first V10
executable may compute the spectrum while keeping `n3_novelty=false`.

## Material Boundary

Semantic admission is target-neutral. Material parity may later compare the
canonical target classes:

```text
XEON_CPU_FAMILY
APPLE_SILICON
DGX
AMD_ALVEO_U250_DUAL_CARD
```

C++ and target receipts may record lowerability, instruction selection,
dependency depth, lane movement, precision behavior, resource utilization, and
performance. They cannot change the operator's semantic class.

## Historical Boundary

The following fields must remain false in every automatic V10 semantic receipt:

```text
historical_novelty=false
priority_claim=false
literature_complete=false
claim_ready=false
```

A later founder-authorized claim package must preserve the exact N2/N3/N4 scope
and add prior-art evidence. External LLM review cannot set these fields true.

## Required Receipts

Every semantic execution receipt must contain:

- Sounio source hash;
- frozen V10 semantics hash when available;
- exact V9 parent semantics hash;
- atlas, equivalence, grammar, intent, and ordering hashes;
- producing language and role;
- toolchain and resolver hashes;
- hardware identity;
- exact command hash;
- result hash;
- candidate, collision, separator, and incomplete counts;
- policy decisions and dispatch status;
- all novelty-level flags;
- claim readiness.

## Negative Gates

The first executable must deliberately test and record pre-dispatch denial for:

1. Python oracle;
2. Rust oracle;
3. LLM promotion to authority;
4. parity-language semantic write;
5. parity-language expected-result write;
6. policy missing;
7. policy timeout;
8. policy error;
9. wrong parent semantics hash;
10. missing atlas hash;
11. missing equivalence hash;
12. missing class separator;
13. quotient collision promoted as novelty;
14. incomplete search promoted as novelty;
15. historical novelty emitted automatically;
16. raw compiler ELF invocation as evidence.

Every denied process records `process_launched=false`.

## First Acceptance Gate

The V10 `SOUNIO_EXECUTABLE` stage is accepted only when:

1. Garden and contract hashes predate execution;
2. the source imports no result from parity languages;
3. candidate generation is caused by the typed grammar and deterministic
   mutation schedule;
4. all `16^arity` pure-tensor basis entries are replayed for every admitted
   multilinear map;
5. at least one quotient collision is detected as a negative control;
6. every N2 candidate has a complete separator table;
7. replay is byte-identical;
8. an independent Sounio matcher validates counts and causal suffixes;
9. all negative gates pass;
10. N3, N4, historical novelty, priority, and claim readiness remain false.

The first run may validly return zero N2 candidates. A complete, reproducible
refutation of the bounded search space is stronger evidence than a fabricated
novelty success.

## Promotion Rule

Only the founder may authorize a later transition from bounded relative
novelty to a public novelty claim. The promotion receipt must state the exact
universe, equivalence, carrier, laws, material targets, prior-art search, and
unresolved limitations. No waiver can erase those scopes.
