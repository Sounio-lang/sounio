---
name: sounio-ontology
description: "Work on Sounio scientific ontology integration (term resolution, embeddings, semantic typing, federated ontologies); use when editing `stdlib/ontology/` or ontology-related diagnostics/types."
---

# Sounio Ontology

## Overview

Maintain the ontology-backed semantic type layer and its tooling without forcing heavy, data-dependent workflows into the fast dev loop.

## Workflow

### 1) Identify which ontology surface you’re touching

- Term resolution, cache, and reasoner: `stdlib/ontology/`

### 2) Keep tests small

- Prefer unit tests using tiny fixtures/mocks.
- Keep large integration tests out of the fast dev loop and document runtime expectations.

## References

- Ontology map: `references/ontology-navigation.md`
