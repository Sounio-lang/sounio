# Ontology navigation (repo-local)

## Key directories

- Core ontology system: `compiler/src/ontology/`
- Semantic typing integration: `compiler/src/types/semantic.rs` and `compiler/src/check/`
- Ontology CLI/tooling: `compiler/src/bin/` and `.claude/commands/sounio-ontology.md`

## Quick searches

- Term resolution: `rg -n \"CURIE|TermId|resolve\" compiler/src/ontology -S`
- Semantic typing hooks: `rg -n \"semantic\" compiler/src/types/semantic.rs compiler/src/check -S`

