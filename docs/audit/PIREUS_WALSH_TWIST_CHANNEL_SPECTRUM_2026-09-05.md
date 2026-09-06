<!-- docs:meta
topic_id: repo.docs.audit.pireus-walsh-twist-channel-spectrum-2026-09-05
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.pireus-walsh-twist-channel-spectrum-2026-09-05
-->

# Pireus Walsh twist channel spectrum audit

```text
Semantic-Lane-ID: pireus-walsh-channel-spectrum-20260905
Owner: codex
Concept-IDs: SOUNIO-NONASSOCIATIVE-ORDER, SOUNIO-SECOND-ORDER-COMPILATION
Intent-Preserved: the exact Cayley-Dickson twist is represented rather than
  erased when Pireus generates a transform-domain operator
Transformation: derive the Walsh character coefficients of every displacement
  row S_d(i)=sigma(i,i xor d) and reconstruct all 256 signs
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: an exact WalshCharacterChannels operator is generated for
  dimension 16; its observed coefficient support is fully dense
Claims-Forbidden: sub-quadratic performance; asymptotic density; hardware
  superiority; compiler promotion; equivalence beyond the tested sign tensor
Assumptions: unnormalized Walsh characters on Z2^4; Convention X; exact i64
  coefficient arithmetic
Write-Set: Garden seed, Sounio authority, frozen spectrum, receipt, gate, audit
Read-Set: frozen Cayley-Dickson sign authority and Pireus operator contracts
Positive-Witness: inverse reconstruction of all 256 signs and Parseval energy
  256 for every displacement row
Negative-Witness: Convention-X mutation changes the freeze; constant-character
  mutation is rejected by reconstruction and Parseval
Acceptance-Gate: scripts/ci/pireus_walsh_twist_channel_spectrum_gate.sh
Integration-Target: origin/main
Authoritative-Only-If: Sounio generates and freezes the coefficient matrix;
  parity implementations may measure but may not create expected values
```

## Generated operator

Pireus generated the exact identity

```text
r[d] = (1/16) * sum_k W[d,k]
       * sum_i character(k,i) * a[i] * b[i xor d].
```

The executable reconstructs every original Cayley-Dickson sign and checks the
unnormalized Parseval identity independently for each row. This makes
`WalshCharacterChannels` an executable operator candidate rather than a prose
analogy or a target-specific recipe.

## Discovered structure

The dimension-16 coefficient matrix has 256 nonzero entries: every displacement
uses every character channel. Rows 0 through 8 have one coefficient `-14` and
fifteen coefficients `2`; later rows use coefficients from `{-6,2,10}`. The
weighted coefficient checksum is 21232.

Full support refuses promotion based on spectral sparsity at this dimension.
It does not prove that the family remains dense asymptotically, nor that a
different block, recursive, or matrix-valued transform cannot be sub-quadratic.
Hardware cost and material promotion remain open.
