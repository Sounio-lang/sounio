# Pireus Operator Novelty Frontier Contract V11

Status: GARDEN_CONTRACT

Concept-ID: SOUNIO-PIREUS-OPERATOR-NOVELTY-FRONTIER

## Producer

The first executable producer is Sounio with role SEMANTIC_AUTHORITY. Parity
languages may only consume a frozen V11 receipt.

## Inputs

- frozen V10 source, semantics, freeze, parity-open, and material receipts;
- the exact V10 parent and three-class atlas reconstruction;
- the frozen unsigned C2_diag action;
- the 7200-member single-coefficient-delta grammar;
- a policy state and Guardian receipt.

No expected frontier counts, outcome partitions, separators, quotient map, or
digests are inputs.

## Output

The Sounio result contains:

    lineage
    grammar_certificate
    atlas_difference_profiles[6]
    candidate_census
    separator_census
    candidate_quotient_certificate
    frontier_digest
    negative_receipt

The ordered candidate census is canonical by ascending candidate ID. Atlas
representatives are ordered by ascending (class, action).

## Completeness

enumeration_complete requires:

    generated == 7200
    typed_admitted + typed_rejected == generated
    atlas_collisions + n2_relative_novel == typed_admitted
    candidate_quotient_singletons + 2 * candidate_quotient_pairs ==
        typed_admitted
    quotient_classes == typed_admitted - candidate_quotient_pairs
    all required separators present
    all direct canaries agree with sparse classification
    all digest inputs consumed in canonical order

No prefix, sample, early winner, or budget-exhausted run is complete.

## Sparse Certificate

The sparse classifier is admissible only when each base difference profile was
produced by a complete 4096-cell scan. Hash equality alone is never tensor
equality. A candidate collision is exact only when the profile identity in the
Garden is satisfied coefficient-wise.

Candidate-to-candidate quotient membership under the nonidentity action is
exactly derived from q(P) - P plus one transported sparse mutation:

    delta * e_m -> delta * chi_q(m) * e_q(m)

The action permutation and character are derived fields. The current unsigned
V10 parent must yield chi_q(m)=+1, but the classifier is invalid if it assumes
that value without checking it. If the resulting difference from P is not
exactly one signed delta at a legal grammar coordinate, the action leaves the
grammar frontier.

The quotient certificate must check q(q(T))=T on the ambient tensors, symmetry
for every candidate pair retained in grammar, canonical unordered pair
counting, fixed-point accounting, and a complete partition of all 7200
candidate IDs. Images outside the grammar are singleton classes.

## Effects

The executable may read frozen parent artifacts and emit its own transcript.
It may not:

- write or reinterpret parent semantics;
- receive expected frontier results;
- select a candidate;
- execute parity or material processes;
- lower to a target;
- promote LLM output;
- invoke Python, Rust, or a raw compiler ELF;
- promote any claim.

## Claim State

The strongest permitted positive field is
bounded_internal_frontier_complete=true, scoped to the exact V11 grammar,
atlas, equality, and action. The following remain false:

    n3_novelty
    n4_novelty
    algorithmic_novelty
    material_novelty
    scientific_novelty
    historical_novelty
    priority_claim
    claim_ready

PARITY_OPEN is not CLAIM_READY.
