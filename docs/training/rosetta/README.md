<!-- docs:meta
topic_id: repo.docs.training.rosetta.readme
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.training.rosetta.readme
-->

# Sounio Rosetta Stone

Side-by-side translations of common algorithms from Python to idiomatic Sounio.
Use these to teach LLMs how to translate between Python and Sounio.

Each file shows the same algorithm in both languages, highlighting:
- No semicolons
- `var` instead of mutable variables
- `&!` for mutable references
- Effects system (`with IO, Mut, Div, Panic`)
- Fixed-size arrays instead of lists
- Named function references instead of lambdas
- Error codes instead of exceptions
