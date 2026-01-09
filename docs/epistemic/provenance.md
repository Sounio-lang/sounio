# Provenance Tracking in Sounio

Provenance is the cornerstone of scientific reproducibility. In Sounio, every `Knowledge<T>` value carries a complete record of its origin and the transformations it has undergone. This is not metadata that can be discarded -- it is an integral part of the value's identity.

> *"Data without origin is data without trust."*
> -- Sounio Manifesto

## What is Provenance Tracking?

Provenance tracking answers the question: *"Where did this value come from, and how was it derived?"*

For regulatory agencies, auditors, and reproducibility requirements, this question is not optional. When a pharmaceutical calculation yields a dose recommendation, or when a climate model predicts temperature anomalies, stakeholders need to trace every step back to primary sources.

Sounio implements provenance as a **Directed Acyclic Graph (DAG)** where:
- **Nodes** represent values at each stage of computation
- **Edges** represent transformations (functions, conversions, aggregations)
- **Roots** are primary data sources (measurements, literature values, user inputs)
- **Leaves** are the final computed results

## The Provenance Type Structure

### Core Components

```sio
// Simplified view of the Provenance structure
struct Provenance {
    // Ordered sequence of transformations (the functor trace)
    trace: FunctorTrace,

    // Original creation point
    origin: Origin,

    // Hash for integrity verification
    integrity_hash: Option<string>,
}

// Origin types classify where data comes from
enum Origin {
    Literal,                              // Hard-coded in source
    External { uri: string },             // From external source
    Computed { function: string },        // Result of computation
    UserInput { context: string },        // User-provided
    Database { query: string, conn: string },
    OntologyAssertion { ontology: string, term: string },
}
```

### The Functor Trace

The functor trace records every transformation applied to a value. Each transformation includes:

```sio
struct Transformation {
    name: string,                 // Human-readable name
    kind: TransformationKind,     // Category of operation
    inputs: Vec<string>,          // Input types
    output: string,               // Output type
    confidence_factor: f64,       // How much does this affect confidence?
    location: Option<Span>,       // Source location
    metadata: TransformationMetadata,
}

enum TransformationKind {
    Function,        // Pure function application (factor: 1.0)
    Conversion,      // Type conversion/coercion (factor: 0.99)
    Translation,     // Ontology translation (factor: 0.95)
    Aggregation,     // Reduction operations (factor: 0.98)
    Filter,          // Selection operations (factor: 1.0)
    Statistical,     // Statistical transformations (factor: 0.95)
    MLInference,     // Machine learning predictions (factor: 0.85)
    ExternalCall,    // External API calls (factor: 0.90)
    Compressed,      // Placeholder for elided steps
}
```

## Source Attribution

### Experimental Data

```sio
// Create a measurement with explicit experimental provenance
let temperature = Knowledge::from_measurement(
    value: 37.2 degC,
    uncertainty: 0.1 degC,
    instrument: "thermometer_TH-5000",
    calibration_cert: "NIST-2025-001234",
    operator: "Dr. Smith",
    timestamp: Timestamp::now()
)

// The provenance automatically records:
// Origin: Measurement { instrument: "thermometer_TH-5000", ... }
// Trace: empty (primary data)
```

### Literature Values

```sio
// Create a value from published literature
let boltzmann_constant = Knowledge::from_literature(
    value: 1.380649e-23 J/K,
    uncertainty: 0.0 J/K,  // Exact by definition (SI 2019)
    source: Source {
        doi: "10.1088/1681-7575/ab0013",
        title: "The CODATA 2018 values of h, e, k, and NA",
        journal: "Metrologia",
        year: 2019,
        confidence: 1.0,
    }
)

// Provenance records:
// Origin: External { uri: "doi:10.1088/1681-7575/ab0013" }
```

### Computed Values

```sio
// When values are computed, provenance chains automatically
let pressure = ideal_gas_pressure(n_moles, temperature, volume)

// pressure.provenance contains:
// Origin: Computed { function: "ideal_gas_pressure" }
// Trace: [
//   Transform("ideal_gas_pressure", inputs: [n_moles, temperature, volume])
// ]
// Parents: [n_moles.provenance, temperature.provenance, volume.provenance]
```

## Provenance Chains Through Transformations

As values flow through computations, their provenance chains grow:

```sio
// Starting values with different origins
let mass = Knowledge::from_measurement(100.0 g, uncertainty: 0.5 g, ...)
let volume = Knowledge::from_measurement(50.0 mL, uncertainty: 0.2 mL, ...)

// Compute density - provenance automatically tracked
let density = mass / volume  // density: Knowledge<g/mL>

// density.provenance.trace contains:
// [Transform("div", confidence_factor: 1.0, inputs: ["Knowledge<g>", "Knowledge<mL>"])]
//
// density.provenance.parents contains references to mass and volume provenance

// Further transformations extend the chain
let normalized = (density - mean_density) / std_density

// normalized.provenance.trace now has additional entries
// Total confidence factor = product of all transformation factors
```

### Automatic Confidence Degradation

Each transformation type has a default confidence factor. The total confidence through a chain is the product of all factors:

```sio
// If we have:
// measurement (factor: 1.0)
// -> calibration (factor: 0.99)
// -> unit conversion (factor: 0.99)
// -> statistical transform (factor: 0.95)
// -> ML inference (factor: 0.85)

// Total confidence factor = 1.0 * 0.99 * 0.99 * 0.95 * 0.85 = 0.79

// Final confidence = original_confidence * total_factor
```

## Merkle-DAG Verification

Sounio uses Merkle DAG structures for cryptographic verification of provenance chains. This provides:

1. **Tamper evidence**: Any modification to historical records changes the hash
2. **Efficient verification**: Only need to check the root hash
3. **Partial proofs**: Can prove specific paths without revealing entire history

### Hash Structure

```sio
// 256-bit hash (BLAKE3 in production)
struct Hash256 {
    h0: u64,
    h1: u64,
    h2: u64,
    h3: u64,
}

// Each node in the Merkle DAG
struct MerkleNode {
    hash: Hash256,                    // Content hash (computed)
    operation: OperationType,         // What produced this
    parents: Vec<Hash256>,            // Parent hashes
    content_hash: Hash256,            // Hash of actual data
    metadata: NodeMetadata,           // Timestamp, author, etc.
}
```

### Creating a Verified Provenance Chain

```sio
// Create a Merkle DAG for provenance tracking
var dag = MerkleDAG::new()

// Add leaf nodes for primary data
let raw_data_hash = dag.add_leaf_array(
    measurements,
    "Raw temperature readings from sensor array"
)

// Add transformation nodes
let normalized_hash = dag.add_transform(
    "normalize",
    vec![raw_data_hash],
    normalized_result,
    "Min-max normalization to [0,1]"
)

let analyzed_hash = dag.add_transform(
    "analyze",
    vec![normalized_hash],
    analysis_result,
    "Statistical analysis with outlier removal"
)

// Verify the entire chain
let is_valid = dag.verify_chain(&analyzed_hash)

// Generate audit trail
let audit = dag.audit_trail(&analyzed_hash)
for entry in audit {
    print("{}: {} - {} ({})",
        entry.hash, entry.operation,
        entry.description,
        if entry.verified { "OK" } else { "FAIL" })
}
```

## Querying Provenance History

Sounio provides rich querying capabilities for provenance:

```sio
// Get the full transformation path as a string
let path = result.provenance.path_string()
// "Literal -> measure -> calibrate -> convert -> analyze"

// Check if a specific transformation was applied
let was_calibrated = result.provenance.includes(TransformationKind::Conversion)

// Get the total confidence factor through the chain
let factor = result.provenance.total_confidence_factor()

// Get the depth of the provenance chain
let depth = result.provenance.depth()

// Access the origin
match result.provenance.origin {
    Origin::Literal => print("Hard-coded value"),
    Origin::External { uri } => print("From: {}", uri),
    Origin::Computed { function } => print("Computed by: {}", function),
    Origin::Database { query, .. } => print("Query: {}", query),
    Origin::OntologyAssertion { ontology, term } =>
        print("Ontology: {}:{}", ontology, term),
}
```

## SLSA Framework Integration

Sounio integrates with [SLSA (Supply-chain Levels for Software Artifacts)](https://slsa.dev/) to provide build provenance attestation. This ensures that not only is your data traceable, but the code that processed it is also verifiable.

### SLSA Build Levels

| Level | Requirements | Sounio Support |
|-------|-------------|----------------|
| L0 | No guarantees | Base level |
| L1 | Build process documented | Full support |
| L2 | Signed provenance, hosted builder | Supported |
| L3 | Hardened builds, full provenance | Planned |
| L4 | Two-party review | Future |

### Build Provenance Attestation

```sio
// SLSA provenance is attached to stdlib builds
struct SLSAProvenance {
    subject_name_hash: i64,
    subject_digest_hash: i64,
    predicate_type_hash: i64,  // https://slsa.dev/provenance/v1
    build_definition: BuildDefinition,
    run_details: RunDetails,
    attestation_hash: i64,
    slsa_level: i32,
}

// Every computation records which stdlib version was used
struct PROVBundleWithSLSA {
    entity_hash: i64,
    activity_hash: i64,
    agent_hash: i64,
    generation_time: i64,
    stdlib_attestation: StdlibAttestation,
    computation_digest_hash: i64,
    bundle_verification_hash: i64,
}
```

### Verifying Supply Chain

```sio
// Check that computation used a properly attested stdlib
let bundle = result.prov_bundle()
let check = check_supply_chain(bundle, required_level: 1)

if !check.is_compliant {
    if !check.has_slsa_attestation {
        error("No SLSA attestation found")
    }
    if !check.slsa_level_sufficient {
        error("SLSA level {} required, got {}",
            required_level, bundle.stdlib_attestation.slsa.slsa_level)
    }
    if !check.dependencies_pinned {
        error("Dependencies not properly pinned")
    }
}
```

## Supply Chain Traceability

For regulated industries (pharmaceuticals, medical devices, aerospace), supply chain traceability is mandatory. Sounio provides:

### Metrological Traceability

Per VIM (International Vocabulary of Metrology), metrological traceability requires an unbroken chain of calibrations back to recognized standards:

```sio
// Build a traceability chain
var chain = TraceabilityChain::new()

// Add calibration links (working standard -> lab standard -> national standard)
let link1 = calibration_link(
    reference: accredited_lab_standard("ACME-CAL-001"),
    certificate: "CERT-2025-0042",
    calibration_date: 1700000000,
    expiry_date: 1800000000,
    uncertainty: 0.02,
    coverage_factor: 2.0,
    dof: 30.0
)
chain = chain.add_link(link1)

let link2 = calibration_link(
    reference: national_standard("NIST-SRM-1234"),
    certificate: "NIST-CERT-5678",
    calibration_date: 1690000000,
    expiry_date: 1790000000,
    uncertainty: 0.01,
    coverage_factor: 2.0,
    dof: 100.0
)
chain = chain.add_link(link2)

// Validate the chain
let validation = validate_traceability_claim(chain, claimed: true)
if !validation.is_valid {
    // This is a HARD ERROR - not a warning
    // error_code: 1=broken, 2=expired, 3=undocumented, 4=no_recognized_reference
    panic("Traceability claim without valid chain: {}", validation.error_code)
}
```

### The traceability_claim_without_chain Lint

Sounio enforces that **traceability claims must be backed by evidence**. This is a compile-time error, not a warning:

```sio
// This will NOT compile:
let result = Knowledge::new(42.0)
    .with_traceability_claim(true)  // ERROR: traceability_claim_without_chain

// This is correct:
let result = Knowledge::new(42.0)
    .with_traceability_chain(validated_chain)  // OK
```

## Audit Trail Generation

For regulatory submission (21 CFR Part 11, EU Annex 11), Sounio generates complete audit trails:

```sio
// Generate audit trail from Merkle DAG
let trail = dag.to_audit_trail()

// Export as JSON for regulatory systems
let json = trail.to_json()

// Export as CSV
let csv = trail.to_csv()

// Each entry contains:
// - hash: Cryptographic identifier
// - operation: What was performed
// - timestamp: When it occurred
// - parents: What inputs were used
// - user: Who performed it (if tracked)
// - system: What system was used
// - signatures: Electronic signatures
```

### W3C PROV-DM Compliance

Sounio's provenance model aligns with [W3C PROV-DM](https://www.w3.org/TR/prov-dm/):

```sio
// PROV-DM core concepts map to Sounio types:
// Entity -> ProvEntity (value with uncertainty)
// Activity -> ProvActivity (measurement, computation, transformation)
// Agent -> (user, system, or organization responsible)

// PROV relations:
// wasGeneratedBy: Entity was produced by Activity
// used: Activity consumed Entity
// wasDerivedFrom: Entity was derived from another Entity

let entity = entity_measured(id, value: 75.5, uncert: 0.5, conf: 0.95, timestamp)
let activity = activity_measurement(id, start_time, end_time)
let record = prov_record_new(entity, activity)

// Add derivation sources
let derived_record = prov_record_with_sources(
    entity,
    activity_arithmetic(id, op_code: 1, timestamp),  // Addition
    source1_id,
    source2_id
)
```

## Best Practices

1. **Always use typed origins**: Prefer `Knowledge::from_measurement()` over `Knowledge::new()` when data has a known source.

2. **Document transformations**: Custom functions should declare their confidence factors:
   ```sio
   #[epistemic(confidence_factor = 0.95)]
   fn my_transformation(x: Knowledge<f64>) -> Knowledge<f64> { ... }
   ```

3. **Validate traceability at boundaries**: When data crosses system boundaries, validate the provenance chain.

4. **Preserve integrity hashes**: When exporting data, include the Merkle root hash for verification.

5. **Use SLSA attestations**: For production systems, ensure your stdlib builds are SLSA-attested.

## References

- W3C PROV-DM: https://www.w3.org/TR/prov-dm/
- W3C PROV-CONSTRAINTS: https://www.w3.org/TR/prov-constraints/
- SLSA Specification: https://slsa.dev/spec/v1.0/
- SLSA Provenance: https://slsa.dev/provenance/v1
- VIM3 (JCGM 200:2012): International Vocabulary of Metrology
- NIST Policy on Traceability: https://www.nist.gov/traceability
- 21 CFR Part 11: Electronic Records; Electronic Signatures
