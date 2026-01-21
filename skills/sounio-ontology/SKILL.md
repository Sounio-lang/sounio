---
name: sounio-ontology
description: "Work on Sounio scientific ontology integration (term resolution, embeddings, semantic typing, federated ontologies); use when editing `compiler/src/ontology/` or ontology-related diagnostics/types."
---

# Sounio Ontology

## Overview

Maintain the ontology-backed semantic type layer and its tooling without forcing heavy, data-dependent workflows into the fast dev loop.

## Workflow

### 1) Identify which ontology surface you’re touching

- Semantic typing and term resolution: `compiler/src/ontology/resolver.rs`, `compiler/src/ontology/mod.rs`
- Storage/caching/loading: `compiler/src/ontology/cache.rs`, `compiler/src/ontology/loader/`
- Embeddings and similarity: `compiler/src/ontology/embedding/`
- Federated ontology support: `compiler/src/ontology/federated.rs`

### 2) Keep tests small

- Prefer unit tests using tiny fixtures/mocks.
- Gate large integration tests behind a feature flag and document runtime expectations.

## References

- Ontology map: `references/ontology-navigation.md`
- Claude command reference: `.claude/commands/sounio-ontology.md`

