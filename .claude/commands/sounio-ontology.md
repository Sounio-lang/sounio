# Query the scientific ontology

Query and manage Sounio's scientific ontology containing 15M+ terms from CHEBI, GO, UBERON, and more.

## Arguments
- `--init` - Initialize/download ontology database
- `--init-core` - Initialize core ontologies only (faster)
- `--search <query>` - Search for concepts by name
- `--info <curie>` - Get detailed info for a CURIE (e.g., CHEBI:27732)
- `--list` - List available ontologies
- `--relations <curie>` - Show relationships for a concept
- `--ancestors <curie>` - Show ancestor concepts
- `--descendants <curie>` - Show descendant concepts

## Examples
- `/sounio-ontology --init` - Initialize full ontology
- `/sounio-ontology --search caffeine` - Search for caffeine
- `/sounio-ontology --info CHEBI:27732` - Info about caffeine
- `/sounio-ontology --relations GO:0006915` - Relations for apoptosis
- `/sounio-ontology --list` - List ontologies

$ARGUMENTS

Execute from the `compiler/` directory:

```bash
cd /home/demetrios/sounio-1/compiler && cargo run -- ontology <subcommand>
```

For initialization:
```bash
cd /home/demetrios/sounio-1/compiler && cargo run --features ontology -- ontology init [--core-only]
```

## Available Ontologies

| Ontology | Description | Terms |
|----------|-------------|-------|
| CHEBI | Chemical Entities of Biological Interest | ~180K |
| GO | Gene Ontology (functions, processes, components) | ~45K |
| UBERON | Uber-anatomy ontology | ~25K |
| CL | Cell Ontology | ~2.5K |
| DOID | Disease Ontology | ~12K |
| HP | Human Phenotype Ontology | ~17K |
| NCIT | NCI Thesaurus | ~180K |
| PR | Protein Ontology | ~350K |
| MONDO | Disease ontology | ~25K |

## CURIE Format

CURIEs (Compact URIs) identify concepts:
- `CHEBI:27732` - Caffeine
- `GO:0006915` - Apoptotic process
- `UBERON:0000948` - Heart
- `CL:0000540` - Neuron

## Usage in Sounio Code

```sio
// Semantic types with ontology backing
type Caffeine = semantic CHEBI:27732
type Apoptosis = semantic GO:0006915

// Type-safe scientific operations
fn measure_caffeine() -> Knowledge<Caffeine, mg/L> {
    // Ontology validates semantic correctness
}

// Query relationships at compile time
static_assert(is_a(Caffeine, CHEBI:23367))  // Caffeine is-a molecular entity
```

## Ontology Relationships

- `is_a` - Subclass relationship
- `part_of` - Parthood relationship
- `has_role` - Role relationship (CHEBI)
- `regulates` - Regulatory relationship (GO)
- `located_in` - Location relationship
