<!-- docs:meta
topic_id: repo.docs.internal.garden.pireus-twist-gauge-factorization
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.garden.pireus-twist-gauge-factorization
-->

# Pireus twist gauge factorization

Status: Garden
Founder direction: Pireus must generate operator novelty, not only choose among
hand-written lowerings.

## Seed

The Cayley-Dickson product on basis indices has the form

```text
r[d] = sum_i sigma(i, i xor d) * a[i] * b[i xor d].
```

Pure XOR convolution admits a Walsh-Hadamard transform. A first differentiating
question is whether the sign twist can be removed by three sign gauges:

```text
sigma(i, j) = alpha(i) * beta(j) * gamma(i xor j),
```

where every gauge takes values in `{+1,-1}`. Writing signs as bits turns this
into a linear system over GF(2), with one equation per basis pair and three
variables per basis index.

If the system is consistent, the gauges generate a transform-compatible
operator candidate. If it is inconsistent, the resulting elimination witness
rejects this entire gauge-to-WHT construction, but does not prove that every
sub-quadratic algorithm is impossible.

## Falsifier

The hypothesis “Cayley-Dickson sign twist is gauge-equivalent to pure XOR
convolution” is false at a given bit width when GF(2) elimination derives a row
with zero coefficients and right-hand side one.

## Boundaries

- Sounio must produce the first executable result and the frozen expectation.
- No external language or LLM may decide consistency or expected ranks.
- A negative result rejects only the three-gauge factorization above.
- It does not establish a complexity lower bound.
- Multiplication order and the exact Cayley-Dickson sign convention remain part
  of the semantic object.
