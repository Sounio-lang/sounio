---
title: "Ontology Integration"
description: "Sounio's 4-layer scientific ontology with 15M+ terms, O(1) subsumption queries, and SSSOM cross-ontology mappings."
---

## Ontology Integration

Sounio integrates **15 million+ scientific terms** from major biomedical and scientific ontologies directly into the compiler's type system. This enables compile-time verification of scientific concepts, automatic unit inference, and semantic compatibility checking.

### 4-Layer Architecture

**File**: `compiler/src/ontology/mod.rs:1-162`

| Layer | Terms | Access Time | Storage |
|-------|-------|-------------|---------|
| **L1: Primitive** | ~850 | O(1) | Compiled into binary |
| **L2: Foundation** | ~8,000 | O(1) cached | Shipped with stdlib |
| **L3: Domain** | ~500,000 | O(log n) | SQLite database |
| **L4: Federated** | ~15,000,000 | Network + cache | BioPortal, OLS4 APIs |

#### L1: Primitive Ontologies
Hardcoded static stores, always available:
- **BFO** (Basic Formal Ontology): 36 upper-level classes
- **RO** (Relation Ontology): ~600 biological relations
- **COB** (Core Ontology for Biology): ~200 core classes

#### L2: Foundation Ontologies
Loaded at startup from stdlib:
- **PATO**: ~2,500 phenotype and trait terms
- **UO**: ~1,000 units of measurement
- **IAO**: ~300 information artifact terms
- **Schema.org**: ~2,850 web-standard types
- **FHIR**: ~1,150 healthcare resource types

#### L3: Domain Ontologies
Lazy-loaded from SQLite database (feature-gated):
- **ChEBI**: Chemical entities (150K+)
- **GO**: Gene Ontology (50K+)
- **DOID**: Disease Ontology (12K+)
- **HP**: Human Phenotype Ontology (17K+)
- Additional: UBERON, CL, ENVO, NCBITaxon, etc.

#### L4: Federated Resolution
Runtime network queries with 3-tier caching:
- **BioPortal**: 15M+ terms across 1,000+ ontologies
- **OLS4**: European Bioinformatics Institute API

### Ontology Resolver

**File**: `compiler/src/ontology/resolver.rs:217-244`

```rust
pub struct OntologyResolver {
    config: ResolverConfig,
    cache: OntologyCache,              // 3-tier LRU (hot/warm/cold)
    subsumption_cache: SubsumptionCache,
    mappings: Option<SssomMappingSet>, // Cross-ontology mappings
    stats: OntologyStats,              // Hit counters per layer
}
```

**Resolution flow** (lines 252-329):
1. Parse CURIE (e.g., `ChEBI:15365`)
2. Check cache (hot -> warm -> cold)
3. Try L1 Primitive -> L2 Foundation -> L3 Domain -> L4 Federated
4. Cache result (or negative result)
5. Return `ResolvedTerm` with layer metadata

### Resolved Terms

**File**: `compiler/src/ontology/resolver.rs:98-134`

```rust
pub struct ResolvedTerm {
    curie: String,              // Canonical CURIE (e.g., "CHEBI:15365")
    label: Option<String>,      // Human label (e.g., "aspirin")
    definition: Option<String>, // OBO definition
    superclasses: Vec<String>,  // Direct parents
    synonyms: Vec<String>,      // Alternative names
    layer: OntologyLayer,       // Which layer resolved this
    iri: Option<String>,        // Full IRI
}
```

### Binary Format (.dontology)

**File**: `compiler/src/ontology/native/storage.rs:209-420`

Ontologies are stored in a custom binary format for fast loading:

```
+----------------------------+
| Magic: "DONT" (4 bytes)    |
+----------------------------+
| Version: u32 (currently 1) |
+----------------------------+
| Header: id, version, count |
+----------------------------+
| StringTable: interned      |  <-- Deduplicates all strings
+----------------------------+
| ConceptEntry[]: 20 bytes   |  <-- curie_idx, label_idx,
|   per concept              |      parent_idx, flags
+----------------------------+
```

The `StringTable` deduplicates strings using `intern(s) -> u32` for O(1) retrieval.

### O(1) Subsumption Queries

**File**: `compiler/src/ontology/native/hierarchy.rs`

Subsumption checking (is-a queries) uses the **Bender & Farach-Colton (2000)** algorithm for O(1) Lowest Common Ancestor queries:

```rust
pub struct HierarchyIndex {
    euler_tour: Vec<usize>,         // DFS traversal
    depths: Vec<usize>,             // Depth at each position
    first: HashMap<String, usize>,  // First occurrence per CURIE
    sparse_table: SparseTable,      // RMQ for O(1) LCA
}
```

Key operations:
- `is_ancestor(descendant, ancestor) -> bool` --- O(1) via sparse table
- `lca(term1, term2) -> Option<&str>` --- Lowest Common Ancestor
- `get_ancestors(curie) -> Vec<&str>` --- Transitive closure

### Caching Strategy

**File**: `compiler/src/ontology/cache.rs`

3-tier LRU cache:

```rust
pub struct OntologyCache {
    hot: LRU<CachedTermData>,    // ~1K entries, most recent
    warm: LRU<CachedTermData>,   // ~10K entries, frequently used
    cold: L3Store,                // ~100K entries, persistent
}
```

Negative results are also cached to avoid repeated failed lookups.

### SSSOM Cross-Ontology Mappings

**File**: `compiler/src/ontology/sssom.rs`

The **Simple Standard for Sharing Ontological Mappings (SSSOM)** enables cross-ontology term translation:

```rust
pub struct SssomMapping {
    subject_id: String,          // Source (e.g., "CHEBI:15365")
    predicate: MappingPredicate, // exactMatch, closeMatch, etc.
    object_id: String,           // Target (e.g., "FHIR:medication")
    confidence: f64,             // 0.0-1.0
}
```

This allows, for example, a ChEBI chemical entity to be used where a FHIR medication type is expected, with the mapping confidence factored into the epistemic status.

### Epistemic Augmentation

**File**: `compiler/src/ontology/foundation/mod.rs`

Foundation terms are epistemically augmented---each term gets an initial confidence based on:
- **Curation status**: Peer-reviewed, automated, inferred
- **Provenance**: Source, version, last update
- **Cross-ontology agreement**: How many ontologies agree on the term

```rust
pub struct FoundationTerm {
    entry: TermEntry,
    initial_epistemic: EpistemicStatus,  // Computed confidence
    mappings: Vec<TermMapping>,          // SSSOM links
    embedding: Option<Vec<f32>>,         // Semantic embedding
}
```

### Query Interface

**File**: `compiler/src/ontology/mod.rs:380-457`

```rust
pub trait OntologyAccess {
    fn search(&self, query: &str, limit: usize) -> Vec<OntologyConcept>;
    fn ancestors(&self, curie: &str) -> Vec<String>;
    fn descendants(&self, curie: &str) -> Vec<String>;
    fn is_subclass(&self, child: &str, parent: &str) -> bool;
    fn distance(&self, from: &str, to: &str) -> Option<usize>;
}
```

### Type System Integration

Ontology terms integrate with the type system via:
1. **Type constraints**: `Knowledge<mg>` where `mg` resolves to `UO:0000022` (milligram)
2. **Subsumption checking**: `OntologyResolver.is_subclass_of()` enables type compatibility
3. **Semantic alignment**: The type checker (`check/mod.rs`) maintains `alignments: HashMap<(String, String), f64>` for compatibility scoring
4. **Refinement types**: Predicates can reference ontology terms

### Module System

**File**: `compiler/src/module_loader.rs` (2,296 lines)

The module system resolves imports with stdlib path discovery:

**Priority order**:
1. `SOUNIO_STDLIB_PATH` environment variable
2. `SOUNIO_STDLIB` (legacy)
3. Relative to compiler binary: `../stdlib/`
4. User home: `~/.sounio/stdlib/`
5. System paths: `/usr/local/lib/sounio/stdlib`, `/usr/lib/sounio/stdlib`

**Module resolution** for `import math`:
1. `./math.sio` (local)
2. `./math/mod.sio` (directory)
3. `./math/lib.sio` (library)
4. `$STDLIB/math.sio` (stdlib)
5. `$STDLIB/math/mod.sio` (stdlib directory)

Circular imports are detected via a load stack (`module_loader.rs:66-80`).
