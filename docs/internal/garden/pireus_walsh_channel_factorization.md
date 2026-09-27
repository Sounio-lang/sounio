<!-- docs:meta
topic_id: repo.docs.internal.garden.pireus-walsh-channel-factorization
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.garden.pireus-walsh-channel-factorization
-->

# Pireus Walsh twist channels

Status: Garden

## Seed

The diagonal sign-gauge route to a single pure XOR convolution is refuted. A
richer exact representation may still exist by expanding each displacement
row of the Cayley-Dickson twist in the Walsh character basis.

For `S_d(i) = sigma(i, i xor d)`, define the unnormalized coefficients

```text
W[d,k] = sum_i S_d(i) * (-1)^popcount(k and i).
```

Then

```text
S_d(i) = (1/n) * sum_k W[d,k] * (-1)^popcount(k and i),
```

so the twisted product can be represented as a weighted family of ordinary XOR
correlations with character-modulated inputs. Pireus should discover the
number and shape of required channels from the Sounio sign law rather than from
a hand-written backend recipe.

## Differentiating result

- Sparse support may yield a useful exact transform candidate.
- Dense support still yields an exact generated candidate, but blocks promotion
  based on sparsity alone.
- The dimension-16 observation cannot establish asymptotic complexity.
- Candidate generation is not candidate selection or hardware parity.

## Falsifier

The representation is invalid if inverse Walsh reconstruction differs from
the original Cayley-Dickson sign on any displacement/index pair, or if Parseval
energy differs from `n*n` for any row.
