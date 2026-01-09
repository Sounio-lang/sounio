# Standards Compliance in Sounio

Sounio is designed from the ground up to comply with international standards for measurement uncertainty, laboratory competence, and data management. This is not an afterthought -- compliance is built into the type system and compiler.

## Overview of Supported Standards

| Standard | Domain | Sounio Support |
|----------|--------|----------------|
| GUM (JCGM 100:2008) | Uncertainty expression | Full |
| ISO/IEC 17025 | Laboratory competence | Partial |
| FAIR Principles | Data management | Full |
| 21 CFR Part 11 | Electronic records (FDA) | Designed for |
| VIM (JCGM 200:2012) | Metrology vocabulary | Aligned |
| W3C PROV-DM | Provenance | Full |

## GUM: Guide to the Expression of Uncertainty in Measurement

The GUM (JCGM 100:2008) is the foundational document for expressing measurement uncertainty. Sounio implements GUM throughout its epistemic type system.

### Type A Evaluation (Statistical Analysis)

Type A uncertainty is evaluated by statistical analysis of repeated observations:

```sio
// Type A uncertainty from repeated measurements
fn type_a_uncertainty(std_dev: f64, n: usize) -> GUMUncertainty {
    // Standard uncertainty of the mean = s / sqrt(n)
    let std_u = std_dev / sqrt_f64(n as f64)

    // Degrees of freedom = n - 1
    let dof = if n > 1 { (n - 1) as f64 } else { 1.0 }

    GUMUncertainty {
        std_uncertainty: std_u,
        degrees_of_freedom: dof,
        sensitivity: 1.0,
    }
}

// Example: 10 temperature measurements with std dev = 0.3
let temp_uncert = type_a_uncertainty(std_dev: 0.3, n: 10)
// std_u = 0.3 / sqrt(10) = 0.095
// dof = 9
```

### Type B Evaluation (Other Methods)

Type B uncertainty is evaluated by other means -- specifications, certificates, prior knowledge:

```sio
// Type B from a priori knowledge
let spec_uncert = type_b_uncertainty(std_u: 0.1)
// dof = infinity (represented as 1e9)

// Type B from rectangular (uniform) distribution
// Used for resolution, rounding, etc.
let resolution_uncert = type_b_uniform(half_width: 0.05)
// u = a / sqrt(3) = 0.05 / 1.732 = 0.029

// Type B from triangular distribution
// Used when central values more likely
let triangular_uncert = type_b_triangular(half_width: 0.05)
// u = a / sqrt(6) = 0.05 / 2.449 = 0.020

// Type B from expanded uncertainty with known coverage factor
let certificate_uncert = type_b_from_expanded(
    expanded_u: 0.1,  // U from calibration certificate
    k: 2.0            // coverage factor stated in certificate
)
// u = U / k = 0.1 / 2.0 = 0.05
```

### Combined Standard Uncertainty

Per GUM Section 5, combine uncertainty components using the law of propagation:

```sio
// For y = f(x1, x2, ..., xn), the combined uncertainty is:
// u_c^2 = sum_i (df/dx_i)^2 * u(x_i)^2 + 2 * sum_i<j (df/dx_i)(df/dx_j) * cov(x_i, x_j)

// Sounio implements this automatically for arithmetic operations:

let mass = gum_simple(value: 100.0, std_u: 0.5)
let volume = gum_simple(value: 50.0, std_u: 0.2)

// Addition: sensitivities are both 1
let sum = gum_add(mass, volume)
// u_c = sqrt(0.5^2 + 0.2^2) = 0.54

// Multiplication: sensitivities are the other operand
let product = gum_mul(mass, volume)
// c1 = |volume| = 50.0, c2 = |mass| = 100.0
// u_c = sqrt((50*0.5)^2 + (100*0.2)^2) = sqrt(625 + 400) = 32.0

// Division: sensitivity wrt denominator is -y/x2
let ratio = gum_div(mass, volume)
// c1 = 1/|volume| = 0.02
// c2 = |mass| / volume^2 = 0.04
// u_c = sqrt((0.02*0.5)^2 + (0.04*0.2)^2) = 0.013
```

### Expanded Uncertainty and Coverage Factors

```sio
// Get coverage factor for 95% confidence at given DoF
let k_95 = coverage_factor_95(dof: 10.0)  // 2.228 (t-distribution)
let k_99 = coverage_factor_99(dof: 10.0)  // 3.169

// For infinite DoF (normal distribution):
let k_normal_95 = k_normal_95()  // 1.96

// Complete GUM result includes expanded uncertainty
struct GUMResult {
    value: f64,                    // Best estimate
    std_uncertainty: f64,          // Combined standard uncertainty u_c
    degrees_of_freedom: f64,       // Effective DoF (Welch-Satterthwaite)
    coverage_factor_95: f64,       // k for 95% coverage
    expanded_uncertainty_95: f64,  // U = k * u_c
}

// Get 95% confidence interval
let (lo, hi) = gum_interval_95(result)
// lo = value - U, hi = value + U

// Get relative uncertainty as percentage
let rel_uncert = relative_uncertainty_percent(result)
// 100 * u_c / |value|
```

### Welch-Satterthwaite Approximation

When combining uncertainty components with different degrees of freedom:

```sio
// Effective degrees of freedom formula:
// v_eff = u_c^4 / sum_i( (c_i * u_i)^4 / v_i )

fn welch_satterthwaite_2(u1: GUMUncertainty, u2: GUMUncertainty) -> f64 {
    let c1 = u1.sensitivity
    let c2 = u2.sensitivity
    let s1 = u1.std_uncertainty
    let s2 = u2.std_uncertainty
    let v1 = u1.degrees_of_freedom
    let v2 = u2.degrees_of_freedom

    // Combined variance
    let u_c_sq = c1*c1*s1*s1 + c2*c2*s2*s2
    let u_c_4 = u_c_sq * u_c_sq

    // Denominator terms
    let term1 = c1*c1*c1*c1 * s1*s1*s1*s1 / v1
    let term2 = c2*c2*c2*c2 * s2*s2*s2*s2 / v2

    u_c_4 / (term1 + term2)
}
```

### GUM Compliance Verification

Sounio can verify that calculations follow GUM methodology:

```sio
// Every GUMResult can be checked for internal consistency
fn verify_gum_result(result: GUMResult) -> bool {
    // Coverage factor appropriate for DoF
    let expected_k = coverage_factor_95(result.degrees_of_freedom)
    let k_ok = abs_f64(result.coverage_factor_95 - expected_k) < 0.01

    // Expanded uncertainty = k * u_c
    let u_ok = abs_f64(result.expanded_uncertainty_95 -
                       result.coverage_factor_95 * result.std_uncertainty) < 1e-10

    k_ok && u_ok
}
```

## ISO/IEC 17025: Laboratory Competence

ISO 17025 specifies requirements for testing and calibration laboratories. Sounio supports key requirements through its type system.

### Metrological Traceability (Section 6.5)

Per ISO 17025:2017 Section 6.5, laboratories must establish metrological traceability. Sounio enforces this:

```sio
// Traceability chain must be complete and documented
struct TraceabilityChain {
    link_count: i32,
    links: [TraceabilityLink; 4],
    terminates_at_si: bool,
    terminates_at_national: bool,
    is_unbroken: bool,
    all_links_valid: bool,
    total_uncertainty: f64,
}

// Each link must include:
struct TraceabilityLink {
    reference_type: ReferenceStandard,  // SI, national, accredited lab, working
    certificate_hash: i64,               // Calibration certificate
    calibration_date: i64,
    expiry_date: i64,
    lab_id_hash: i64,                    // Calibrating laboratory
    procedure_hash: i64,                 // Calibration procedure
    contributed_uncertainty: f64,         // Uncertainty from this link
    coverage_factor: f64,
    is_documented: bool,
    is_expired: bool,
    is_valid: bool,
}
```

### The traceability_claim_without_chain Lint

Sounio enforces that traceability claims must be backed by evidence:

```sio
// This is a HARD ERROR, not a warning
fn validate_traceability_claim(chain: TraceabilityChain, claimed: bool) -> TraceabilityValidation {
    if !claimed {
        return TraceabilityValidation { is_valid: true, ... }
    }

    let finalized = chain_finalize(chain)

    if !finalized.is_complete {
        // Error codes:
        // 1 = broken chain
        // 2 = expired calibration
        // 3 = undocumented link
        // 4 = no recognized reference (SI or national)
        return TraceabilityValidation {
            is_valid: false,
            error_code: finalized.error_code,
            error_level: 2,  // HARD ERROR
            ...
        }
    }

    TraceabilityValidation { is_valid: true, ... }
}
```

### Reference Standards Hierarchy

```sio
// Recognized standard levels (higher = more authoritative)
enum StandardType {
    SI,              // Primary (SI definition)
    National,        // NMI (NIST, PTB, NPL, etc.)
    AccreditedLab,   // ISO 17025 accredited calibration lab
    WorkingStandard, // Internal working standard (must be validated)
}

fn si_primary() -> ReferenceStandard {
    ReferenceStandard {
        standard_type: 0,  // SI
        level: 0,          // Primary
        is_recognized: true,
    }
}

fn national_standard(id_hash: i64) -> ReferenceStandard {
    ReferenceStandard {
        standard_type: 1,  // National
        level: 1,          // Secondary
        identifier_hash: id_hash,
        is_recognized: true,
    }
}
```

### Measurement Uncertainty Requirements

ISO 17025 requires laboratories to evaluate measurement uncertainty. Sounio tracks this automatically:

```sio
// Every Knowledge<T> value carries uncertainty
let result = Knowledge::from_measurement(
    value: 25.4 mm,
    uncertainty: 0.1 mm,
    instrument: "micrometer_M-500",
    calibration: valid_chain,  // Required for traceability claims
    operator: "Tech-042",
    timestamp: Timestamp::now(),
    method: "ISO-XXX procedure"
)

// Total uncertainty includes measurement and traceability components
let total_u = sqrt_f64(
    result.uncertainty^2 +
    result.calibration_chain.total_uncertainty^2
)
```

## FAIR Data Principles

The FAIR principles (Findable, Accessible, Interoperable, Reusable) guide scientific data management. Sounio implements these at the language level.

### Findable

Data should be easy to find for both humans and computers.

```sio
// Every Knowledge value can have persistent identifiers
let measurement = Knowledge::new(42.0)
    .with_doi("10.5281/zenodo.1234567")
    .with_orcid_author("0000-0002-1825-0097")
    .with_ror_institution("https://ror.org/02qz8b764")

// Merkle hashes provide content-addressable identifiers
let hash = measurement.provenance_hash()
// Hash uniquely identifies this value AND its provenance
```

### Accessible

Data should be retrievable using open, standardized protocols.

```sio
// Provenance export to standard formats
let prov = measurement.provenance()

// W3C PROV-DM export
let prov_dm = prov.to_prov_dm()

// JSON-LD for linked data
let json_ld = prov.to_json_ld()

// Audit trail for regulatory systems
let audit = prov.to_audit_trail()
let csv = audit.to_csv()  // For legacy systems
let json = audit.to_json() // For modern APIs
```

### Interoperable

Data should integrate with other data and work with applications for analysis.

```sio
// Ontology integration for semantic interoperability
let concentration = Knowledge::new(5.23 mg/L)
    .with_ontology_term("CHEBI:16236")  // glucose in ChEBI
    .with_unit_ontology("UO:0000273")   // milligram per liter in UO

// Units of measure with dimensional analysis
let dose: mg = 500.0
let volume: mL = 10.0
let conc: mg/mL = dose / volume  // Type-safe unit propagation

// Cross-ontology translation with tracked confidence
let fhir_code = translate_chebi_to_fhir("CHEBI:16236")
// Confidence degraded by 0.95 for translation step
```

### Reusable

Data should be richly described for future reuse.

```sio
// Complete provenance enables reproducibility
struct ProvenanceMetadata {
    user: Option<string>,           // Who created this
    system: Option<string>,         // What system
    version: Option<string>,        // Software version
    regulatory_context: Option<string>, // e.g., "21 CFR Part 11"
    custom: HashMap<string, string>,  // Domain-specific metadata
}

// License and usage rights
let dataset = Knowledge::from_dataset(data)
    .with_license("CC-BY-4.0")
    .with_usage_rights("Research use only")
    .with_citation("Smith et al., 2025, J. Science")
```

### FAIR Compliance Verification

```sio
// Check FAIR compliance of a Knowledge value
fn check_fair_compliance(k: Knowledge<T>) -> FAIRReport {
    FAIRReport {
        // Findable
        has_persistent_id: k.doi().is_some() || k.hash().is_some(),
        has_rich_metadata: k.metadata().len() > 5,

        // Accessible
        has_standard_protocol: k.access_protocol() != "proprietary",
        metadata_accessible: k.provenance().is_some(),

        // Interoperable
        uses_fair_vocabulary: k.ontology_terms().len() > 0,
        uses_standard_format: true,  // Sounio is always standard

        // Reusable
        has_clear_license: k.license().is_some(),
        has_provenance: k.provenance().depth() > 0,
        has_domain_metadata: k.domain_metadata().len() > 0,
    }
}
```

## 21 CFR Part 11 Compliance

21 CFR Part 11 establishes FDA requirements for electronic records and electronic signatures. Sounio is designed to support these requirements.

### Audit Trails (Section 11.10(e))

```sio
// Complete audit trail generation
let trail = dag.to_audit_trail()

// Each entry includes:
struct AuditEntry {
    hash: string,            // Cryptographic identifier
    operation: string,       // What was performed
    operation_kind: string,  // Category
    timestamp: u64,          // When it occurred
    parents: Vec<string>,    // Input data hashes
    user: Option<string>,    // Who performed it
    system: Option<string>,  // System identifier
    signatures: Vec<string>, // Electronic signatures
}

// Audit trail is tamper-evident (Merkle hash)
let trail_hash = trail.dag_hash
// Any modification invalidates the hash
```

### Electronic Signatures (Section 11.50)

```sio
// Cryptographic signatures on provenance nodes
struct ProvenanceSignature {
    authority: string,           // Who signed
    signature: Vec<u8>,          // Cryptographic signature
    timestamp: u64,              // When signed
    algorithm: SignatureAlgorithm,  // Ed25519, ECDSA, RSA
}

// Sign a provenance node
let sig = ProvenanceSignature::new(
    authority: "Dr. Jane Smith",
    private_key: user_key,
    node_hash: &node.id
)
node.sign(sig)

// Verify signature
let valid = sig.verify(public_key, &node.id)
```

### Closed System Requirements (Section 11.10)

```sio
// Metadata for regulatory context
let metadata = ProvenanceMetadata::new()
    .with_user("user_id_12345")
    .with_system("LabSystem_v3.2.1")
    .with_regulatory_context("21 CFR Part 11")
    .with_custom("validation_status", "Validated")
    .with_custom("sop_reference", "SOP-LAB-001")

// Attach to computation
let result = compute_result(...)
    .with_metadata(metadata)
```

### Record Retention

```sio
// Export for archival (multiple formats)
let audit = result.provenance().to_audit_trail()

// JSON for long-term storage
let json = audit.to_json()
save_to_archive(json, retention_years: 15)

// CSV for regulatory submission
let csv = audit.to_csv()

// Both formats include dag_hash for integrity verification
// Future verification: recalculate hash and compare
```

## VIM Alignment

Sounio's terminology aligns with VIM (International Vocabulary of Metrology, JCGM 200:2012).

### Key Terms

| VIM Term | Sounio Concept |
|----------|---------------|
| Measurand | The `T` in `Knowledge<T>` |
| Measured quantity value | `Knowledge.value` |
| Measurement uncertainty | `Knowledge.uncertainty` |
| Coverage interval | `gum_interval_95()` result |
| Coverage probability | Confidence level (0.95) |
| Metrological traceability | `TraceabilityChain` |
| Calibration | `CalibrationLink` |

### Uncertainty vs Error

Per VIM 2.26, uncertainty characterizes the dispersion of values that could reasonably be attributed to the measurand. Sounio maintains this distinction:

```sio
// Uncertainty is NOT error (VIM 2.16)
// Error = measured value - true value (unknowable)
// Uncertainty = range of reasonable values

let measurement = Knowledge::new(
    value: 25.0,           // Best estimate of measurand
    uncertainty: 0.5,      // NOT the error!
    confidence: 0.95       // 95% of reasonable values within +/- 1.96*0.5
)

// Uncertainty can be reduced by more measurements
// Error is fixed (but unknown)
```

## Compliance Export Formats

### For Regulatory Submission

```sio
// Generate compliance documentation
let report = ComplianceReport::generate(computation)

// GUM uncertainty budget table
let budget_table = report.gum_budget_table()
// | Component | Value | u(xi) | ci | |ci*u(xi)| | vi |
// |-----------|-------|-------|----|-----------|----|
// | Mass      | 100.0 | 0.5   | 1  | 0.5       | 9  |
// | Volume    | 50.0  | 0.2   | -2 | 0.4       | inf|
// | Combined  | ...   | 0.64  | -  | -         | 25 |

// Traceability documentation
let traceability_doc = report.traceability_documentation()

// Audit trail for Part 11
let audit_trail = report.part_11_audit_trail()
```

### For Data Repositories

```sio
// Export for Zenodo, Figshare, etc.
let fair_package = knowledge.to_fair_package()

// Includes:
// - data.json: The actual values with uncertainty
// - provenance.jsonld: W3C PROV-DM in JSON-LD
// - metadata.xml: DataCite metadata
// - README.md: Human-readable description
// - CITATION.cff: Citation file format
```

## Verification and Validation

### Continuous Compliance Checking

```sio
// Compile-time checks
// - traceability_claim_without_chain: HARD ERROR
// - uncertainty_ignored: WARNING
// - confidence_below_threshold: WARNING (configurable)

// Runtime verification
fn verify_computation_compliance(result: Knowledge<T>) -> ComplianceStatus {
    var issues = Vec::new()

    // Check provenance completeness
    if result.provenance().depth() == 0 {
        issues.push(Issue::Warning("No provenance recorded"))
    }

    // Check traceability if claimed
    if result.claims_traceable() {
        let validation = validate_traceability_claim(
            result.traceability_chain(),
            claimed: true
        )
        if !validation.is_valid {
            issues.push(Issue::Error("Invalid traceability claim"))
        }
    }

    // Check GUM compliance
    if let Some(gum) = result.as_gum_result() {
        if !verify_gum_result(gum) {
            issues.push(Issue::Error("GUM verification failed"))
        }
    }

    ComplianceStatus { issues }
}
```

### Audit Trail Verification

```sio
// Verify integrity of entire audit trail
fn verify_audit_trail(trail: AuditTrail, dag: MerkleProvenanceDAG) -> bool {
    // Recompute DAG hash
    let computed_hash = dag.compute_dag_hash()

    // Compare with stored hash
    if computed_hash != trail.dag_hash {
        return false  // Tampering detected
    }

    // Verify each node
    for (id, node) in dag.nodes {
        if node.verify().is_err() {
            return false  // Node integrity compromised
        }
    }

    // Verify all parents exist
    dag.verify_all().is_ok()
}
```

## Best Practices

1. **Always document uncertainty sources**: Use `type_a_uncertainty` or `type_b_*` constructors with appropriate parameters.

2. **Maintain traceability chains**: For regulated work, ensure calibration links are complete and unexpired.

3. **Use FAIR metadata**: Add DOIs, ORCIDs, and ontology terms to enable data discovery and reuse.

4. **Generate audit trails**: For 21 CFR Part 11 work, always export complete audit trails.

5. **Verify before submission**: Run compliance verification before regulatory submission.

6. **Archive with hashes**: Store Merkle hashes alongside archived data for future integrity verification.

## References

- JCGM 100:2008 - Guide to the Expression of Uncertainty in Measurement (GUM)
- JCGM 200:2012 - International Vocabulary of Metrology (VIM3)
- ISO/IEC 17025:2017 - General requirements for the competence of testing and calibration laboratories
- 21 CFR Part 11 - Electronic Records; Electronic Signatures
- FAIR Principles: https://www.go-fair.org/fair-principles/
- W3C PROV-DM: https://www.w3.org/TR/prov-dm/
- NIST Policy on Traceability: https://www.nist.gov/traceability
