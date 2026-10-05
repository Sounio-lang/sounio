<!-- docs:meta
topic_id: repo.examples.research.rna-cd-confirmatory.readme
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A5
source_of_truth: docs/governance/topic-registry.v1.json#repo.examples.research.rna-cd-confirmatory.readme
-->

# RNA Cayley-Dickson confirmatory lane

This directory begins the confirmatory hierarchy study derived from the
exploratory KIMI OctTree/Rfam experiments. It does not contain a confirmatory
model result.

The registered question is not merely whether OctTree beats Clifford on a
real-versus-corrupted classifier. That existing grid is retained as an
auxiliary mechanistic control. The primary study tests the interaction between
algebra level and structural class: OctTree-CD8 on strictly nested relations,
and SedenTree-CD16 on relations with crossing pairs, each against a
dimension-matched associative control.

Current evidence level: **fixture-scale data contract**.

## Authority split

- `rna_cd_manifest.sio` is the canonical Sounio producer. It validates a frozen
  flat input, refuses zero-pair Tier I records and split-key leakage, and emits
  deterministic group/fold and crossing-graph metadata.
- `validate_manifest.jl` is an independent Julia validator. It recomputes
  sequence and structure hashes, structural invariants, group identity, split
  assignment, and every output byte.
- A missing runtime or any disagreement is a refusal, never a pass.

The accelerator/training worker is deliberately absent from this first gate.
It may consume a validated Sounio manifest later, but it will not own cohort,
split, metric, or receipt semantics.

## Run the fixture contract

```bash
bash scripts/ci/rna_cd_confirmatory_contract.sh
```

The gate requires both the canonical Sounio compiler and Julia. If Julia is
missing it exits `2` with `RNA_CD_CONFIRMATORY_BLOCKED`; it does not downgrade
to a Sounio-only pass.

Direct producer use:

```bash
bin/souc check examples/research/rna_cd_confirmatory/rna_cd_manifest.sio
bin/souc compile examples/research/rna_cd_confirmatory/rna_cd_manifest.sio \
  -o /tmp/rna-cd-manifest
/tmp/rna-cd-manifest \
  tests/research/rna_cd_confirmatory/valid.tsv \
  /tmp/rna-cd-manifest.tsv \
  rna-cd-confirmatory-v1
```

Independent validation:

```bash
julia --startup-file=no \
  examples/research/rna_cd_confirmatory/validate_manifest.jl \
  tests/research/rna_cd_confirmatory/valid.tsv \
  /tmp/rna-cd-manifest.tsv \
  rna-cd-confirmatory-v1
```

## Current boundary

The fixture accepts canonical RNA symbols `A/C/G/U/N` and four extended
dot-bracket classes: `()`, `[]`, `{}`, and `<>`. It proves that multiple pairing
classes survive the manifest path. Its seven accepted records cover nested
pairs, one crossing relation, a four-pair crossing clique, two disconnected
crossing components, and a crossing component coexisting with an enclosing
nested pair. The negative fixtures fix the refusal reasons for no maskable pair,
family-to-group conflict, and duplicate sequence across groups. Sounio validates
the declared SHA-256 syntax and split identity byte-for-byte; Julia independently
recomputes the cryptographic hashes from content.

This is not yet the full hierarchy artifact. The production schema must add the
canonical pair list, exclusion ledger, registered relation masks, provenance,
and fold-by-stratum availability table. The fixture already emits and validates
the `nested`/`crossing` stratum and crossing-complexity tuple.

It does not yet parse raw Stockholm/WUSS. That next producer must also support
letter pairs such as `A/a`, project consensus pairs jointly across per-sequence
gaps, and emit exclusions. The previous derived FASTA cannot be used for
confirmation: 19,692 of 108,072 structures are unbalanced because a gap at one
endpoint could leave the opposite endpoint behind.

The pre-registered scientific protocol and claim boundary are in
`docs/research/rna_cayley_dickson_confirmatory_preregistration_2026-08-09.md`.
