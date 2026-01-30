# Sounio v0.29.0 Release Notes

**Release Date:** December 3, 2025

## Overview

This release introduces the **Epistemic Type System** and **4-Layer Ontology Integration**, enabling Sounio to natively support 15 million ontology terms as first-class types. This is a foundational feature for scientific and biomedical computing.

## New Features

### Epistemic Type System (`src/epistemic/`)

The epistemic module implements the `Knowledge[τ, ε, δ, Φ]` type system:

- **`confidence.rs`** - Confidence levels with Bayesian propagation
  - `ConfidenceLevel`: High (>0.95), Medium (0.7-0.95), Low (0.4-0.7), VeryLow (<0.4)
  - `EpistemicStatus`: Axiomatic, Empirical, Derived, Hypothetical, Retracted
  - Confidence propagation rules for logical operations

- **`provenance.rs`** - Functor trace (Φ) with DAG structure
  - `ProvenanceNode`: Literal, Transformation, Aggregation, Assertion
  - `FunctorTrace`: Composition of provenance transformations
  - Full provenance path tracking for derived knowledge

- **`temporal.rs`** - Temporal indexing (τ)
  - `TemporalIndex`: ContextTime, Version, Interval
  - Support for point-in-time and interval-based validity
  - Version tracking for evolving knowledge

- **`operations.rs`** - Epistemic operations
  - `RevisionStrategy`: Prioritized, AGM, Bayesian
  - `EpistemicConstraint`: MinConfidence, RequireProvenance, TemporalScope
  - `RelationalConstraint`: Subsumption, PartOf, DependsOn

### 4-Layer Ontology Integration (`src/ontology/`)

```
┌────────────────────────────────────────────────────────────────┐
│ L4: Federated (~15M terms)                                     │
│     BioPortal, OLS4 - Runtime resolution via HTTP              │
├────────────────────────────────────────────────────────────────┤
│ L3: Domain (~500K terms)                                       │
│     ChEBI, GO, DOID via Semantic-SQL (lazy-loaded SQLite)      │
├────────────────────────────────────────────────────────────────┤
│ L2: Foundation (~8K terms)                                     │
│     PATO, UO, IAO, Schema.org, FHIR - shipped with stdlib      │
├────────────────────────────────────────────────────────────────┤
│ L1: Primitive (~850 terms)                                     │
│     BFO, RO, COB - compiled into the compiler                  │
└────────────────────────────────────────────────────────────────┘
```

- **`mod.rs`** - Core types and CURIE parsing
  - `ParsedTermRef`: Parse CURIE, OBO-style, and full IRI formats
  - `OntologyLayer`: Primitive, Foundation, Domain, Federated
  - `OntologyStats`: Usage tracking and cache hit rates

- **`primitive.rs`** - L1 Primitives compiled into compiler
  - `BfoClass`: 36 BFO 2020 classes (Entity, Continuant, Occurrent, etc.)
  - `RoRelation`: Relation Ontology relations
  - `CobClass`: Core Ontology for Biology classes
  - Subsumption checking with transitive closure

- **`cache.rs`** - Tiered LRU caching
  - Hot/Warm/Cold cache tiers with automatic promotion
  - Negative caching for missing terms
  - TTL-based expiration
  - Detailed cache statistics

- **`sssom.rs`** - SSSOM mapping support
  - `SssomMappingSet`: Collections of ontology mappings
  - `MappingPredicate`: exactMatch, closeMatch, broadMatch, etc.
  - TSV/JSON parsing for SSSOM files
  - Bidirectional mapping lookup

- **`semantic_sql.rs`** - SQLite-backed ontology store
  - Pre-built Semantic-SQL databases from OBO Foundry
  - Efficient term lookup, subsumption, and ancestor queries
  - Lazy loading of domain ontologies

- **`resolver.rs`** - Main resolution interface
  - `OntologyResolver`: Unified resolution across all layers
  - Priority-based resolution (L1 → L4)
  - Caching integration
  - `SubsumptionResult`: Subsumes, SubsumedBy, Equivalent, Disjoint, Unknown

- **`federated.rs`** - L4 Federated resolution
  - `FederatedSource`: OLS4, BioPortal, Custom endpoints
  - Rate limiting and exponential backoff
  - Query builders with filtering options

## Bug Fixes

- Fixed `TokenKind::Float` → `TokenKind::FloatLit` in autodiff macro
- Fixed operator token kinds in autodiff expression generation
- Fixed borrow checker issues in derive macro parsing
- Made `match_sequence()` and `Bindings` fields public for macro expansion

## Test Coverage

- **34 ontology tests** covering:
  - CURIE/IRI parsing
  - Cache promotion and eviction
  - Negative caching
  - SSSOM mapping operations
  - Primitive ontology subsumption
  - Resolver functionality

- **19 epistemic tests** covering:
  - Confidence propagation
  - Provenance composition
  - Temporal indexing
  - Epistemic constraints

## Dependencies Added

- `lru = "0.12"` - LRU cache implementation
- `rusqlite = "0.31"` (optional, feature = "ontology") - SQLite access

## API Changes

### New Public Types

```rust
// Epistemic
pub use epistemic::{
    ConfidenceLevel, EpistemicStatus, ConfidenceValue,
    ProvenanceNode, FunctorTrace,
    TemporalIndex, ContextTime, Version,
    RevisionStrategy, EpistemicConstraint,
};

// Ontology
pub use ontology::{
    OntologyResolver, ResolvedTerm, SubsumptionResult,
    OntologyCache, CacheConfig, CacheStats,
    ParsedTermRef, OntologyLayer,
    SssomMapping, SssomMappingSet, MappingPredicate,
    PrimitiveStore, BfoClass, RoRelation, CobClass,
    FederatedResolver, FederatedSource, FederatedQuery,
};
```

## Migration Guide

This release is additive - no breaking changes to existing APIs.

To use the new ontology system:

```rust
use sounio::ontology::{OntologyResolver, ResolvedTerm};

let resolver = OntologyResolver::new()?;

// Resolve a term
let aspirin = resolver.resolve("CHEBI:15365")?;

// Check subsumption
let result = resolver.is_subclass_of("CHEBI:15365", "CHEBI:23888")?;
```

## What's Next (v0.30.0)

- Integration of epistemic types with the main type checker
- LLVM codegen for ontology-typed values
- FHIR resource type generation
- Interactive ontology browser in LSP

---

*Sounio: Where ontologies become types.*
