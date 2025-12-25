//! SLSA Build Provenance Attestation
//!
//! SLSA (Supply chain Levels for Software Artifacts) defines provenance as an
//! attestation of HOW an artifact was built.
//!
//! Per SLSA Build L1, required fields include:
//!   - buildDefinition: What was built and how
//!   - runDetails: Who built it and when
//!
//! This module makes Demetrios results traceable through the supply chain:
//!   "This computation used stdlib build X with provenance Y"
//!
//! Every exported PROV bundle includes the SLSA attestation digest.
//!
//! References:
//!   - SLSA Specification: https://slsa.dev/spec/v1.0/
//!   - SLSA Provenance: https://slsa.dev/provenance/v1
//!   - in-toto Attestation: https://in-toto.io/

extern "C" {
    // No external functions needed for this module
}

fn abs_i64(x: i64) -> i64 {
    if x < 0 { return 0 - x }
    return x
}

// ============================================================================
// SLSA BUILD TYPE
// ============================================================================

/// SLSA buildType identifies the template for the build
/// Standard types:
///   0 = unknown
///   1 = github_actions
///   2 = gitlab_ci
///   3 = local_build
///   4 = demetrios_stdlib
struct BuildType {
    type_code: i32,
    type_uri_hash: i64,    // Hash of buildType URI
}

fn build_type_unknown() -> BuildType {
    return BuildType { type_code: 0, type_uri_hash: 0 }
}

fn build_type_github_actions() -> BuildType {
    return BuildType {
        type_code: 1,
        type_uri_hash: 100001,  // https://slsa.dev/github-actions-workflow/v1
    }
}

fn build_type_demetrios_stdlib() -> BuildType {
    return BuildType {
        type_code: 4,
        type_uri_hash: 100004,  // https://demetrios-lang.org/slsa/stdlib/v1
    }
}

// ============================================================================
// SLSA BUILD DEFINITION
// ============================================================================

/// Per SLSA: buildDefinition describes what was built and how
struct BuildDefinition {
    build_type: BuildType,

    // externalParameters: Inputs from external sources (user, CI, etc.)
    ext_param_hash: i64,       // Hash of external parameters
    ext_param_count: i32,

    // internalParameters: Inputs controlled by the builder
    int_param_hash: i64,
    int_param_count: i32,

    // resolvedDependencies: Artifacts resolved during build
    dep_count: i32,
    dep0_hash: i64,            // Hash of dependency 0 (e.g., stdlib source)
    dep1_hash: i64,            // Hash of dependency 1 (e.g., compiler)
    dep2_hash: i64,
    dep3_hash: i64,
}

fn build_definition_new(build_type: BuildType) -> BuildDefinition {
    return BuildDefinition {
        build_type: build_type,
        ext_param_hash: 0,
        ext_param_count: 0,
        int_param_hash: 0,
        int_param_count: 0,
        dep_count: 0,
        dep0_hash: 0,
        dep1_hash: 0,
        dep2_hash: 0,
        dep3_hash: 0,
    }
}

fn build_definition_add_dep(def: BuildDefinition, dep_hash: i64) -> BuildDefinition {
    var result = def
    let idx = def.dep_count

    if idx == 0 { result.dep0_hash = dep_hash }
    else if idx == 1 { result.dep1_hash = dep_hash }
    else if idx == 2 { result.dep2_hash = dep_hash }
    else if idx == 3 { result.dep3_hash = dep_hash }

    if idx < 4 {
        result.dep_count = idx + 1
    }

    return result
}

// ============================================================================
// SLSA RUN DETAILS
// ============================================================================

/// Per SLSA: runDetails describes who built it and when
struct RunDetails {
    // builder: Who performed the build
    builder_id_hash: i64,      // e.g., "https://github.com/demetrios-lang/actions"
    builder_version: i64,      // Builder version

    // metadata
    invocation_id_hash: i64,   // Unique invocation identifier
    started_at: i64,           // Unix timestamp
    finished_at: i64,

    // byproducts: Additional outputs
    log_hash: i64,             // Hash of build log
}

fn run_details_new(builder_hash: i64, invocation_hash: i64, started: i64) -> RunDetails {
    return RunDetails {
        builder_id_hash: builder_hash,
        builder_version: 1,
        invocation_id_hash: invocation_hash,
        started_at: started,
        finished_at: 0,
        log_hash: 0,
    }
}

fn run_details_finish(details: RunDetails, finished: i64, log_hash: i64) -> RunDetails {
    var result = details
    result.finished_at = finished
    result.log_hash = log_hash
    return result
}

// ============================================================================
// SLSA PROVENANCE PREDICATE
// ============================================================================

/// Complete SLSA provenance statement
struct SLSAProvenance {
    // Subject: What was built
    subject_name_hash: i64,
    subject_digest_hash: i64,

    // Predicate type
    predicate_type_hash: i64,  // https://slsa.dev/provenance/v1

    // Build definition + run details
    build_definition: BuildDefinition,
    run_details: RunDetails,

    // Attestation metadata
    attestation_hash: i64,     // Combined hash for quick verification
    slsa_level: i32,           // SLSA Build Level (1-4)
}

fn slsa_provenance_new(
    subject_name: i64,
    subject_digest: i64,
    build_def: BuildDefinition,
    run: RunDetails,
    level: i32
) -> SLSAProvenance {
    // Compute attestation hash from components
    let combined = subject_digest
                 + build_def.build_type.type_uri_hash
                 + run.builder_id_hash
                 + run.invocation_id_hash

    return SLSAProvenance {
        subject_name_hash: subject_name,
        subject_digest_hash: subject_digest,
        predicate_type_hash: 200001,  // https://slsa.dev/provenance/v1
        build_definition: build_def,
        run_details: run,
        attestation_hash: combined,
        slsa_level: level,
    }
}

// ============================================================================
// STDLIB VERSION ATTESTATION
// ============================================================================

/// Attestation for the Demetrios stdlib itself
struct StdlibAttestation {
    version_major: i32,
    version_minor: i32,
    version_patch: i32,
    git_commit_hash: i64,
    build_timestamp: i64,
    slsa: SLSAProvenance,
}

fn stdlib_attestation_new(
    major: i32,
    minor: i32,
    patch: i32,
    commit_hash: i64,
    build_time: i64
) -> StdlibAttestation {
    let build_type = build_type_demetrios_stdlib()
    var build_def = build_definition_new(build_type)
    build_def = build_definition_add_dep(build_def, commit_hash)

    let run = run_details_new(
        300001,     // Demetrios official builder
        build_time, // Use timestamp as invocation ID
        build_time
    )

    let slsa = slsa_provenance_new(
        400001,     // "demetrios-stdlib"
        commit_hash,
        build_def,
        run,
        1           // SLSA Build L1
    )

    return StdlibAttestation {
        version_major: major,
        version_minor: minor,
        version_patch: patch,
        git_commit_hash: commit_hash,
        build_timestamp: build_time,
        slsa: slsa,
    }
}

// ============================================================================
// PROV BUNDLE WITH SLSA
// ============================================================================

/// Extended PROV bundle that includes SLSA attestation
/// "This result was computed with stdlib build X, built by builder Y"
struct PROVBundleWithSLSA {
    // Standard PROV fields
    entity_hash: i64,
    activity_hash: i64,
    agent_hash: i64,
    generation_time: i64,

    // SLSA attestation for the stdlib
    stdlib_attestation: StdlibAttestation,

    // Link to computation
    computation_digest_hash: i64,

    // Combined verification hash
    bundle_verification_hash: i64,
}

fn prov_bundle_with_slsa(
    entity: i64,
    activity: i64,
    agent: i64,
    gen_time: i64,
    stdlib: StdlibAttestation,
    computation_digest: i64
) -> PROVBundleWithSLSA {
    // Verification hash combines all provenance data
    let verification = entity
                     + activity
                     + stdlib.slsa.attestation_hash
                     + computation_digest

    return PROVBundleWithSLSA {
        entity_hash: entity,
        activity_hash: activity,
        agent_hash: agent,
        generation_time: gen_time,
        stdlib_attestation: stdlib,
        computation_digest_hash: computation_digest,
        bundle_verification_hash: verification,
    }
}

/// Verify that bundle is internally consistent
fn verify_bundle(bundle: PROVBundleWithSLSA) -> bool {
    let expected = bundle.entity_hash
                 + bundle.activity_hash
                 + bundle.stdlib_attestation.slsa.attestation_hash
                 + bundle.computation_digest_hash

    return expected == bundle.bundle_verification_hash
}

// ============================================================================
// AUDIT EXPORT
// ============================================================================

/// Exportable audit record for regulatory compliance
struct AuditRecord {
    // Computation identity
    computation_id: i64,
    computation_time: i64,

    // Result provenance
    result_value_hash: i64,
    uncertainty_hash: i64,

    // Stdlib provenance (SLSA)
    stdlib_version_hash: i64,
    stdlib_slsa_hash: i64,

    // Verification
    is_verified: bool,
    audit_timestamp: i64,
}

fn create_audit_record(
    computation_id: i64,
    result_hash: i64,
    uncertainty_hash: i64,
    bundle: PROVBundleWithSLSA,
    audit_time: i64
) -> AuditRecord {
    let is_verified = verify_bundle(bundle)

    // Combine stdlib version into single hash
    let version_hash = (bundle.stdlib_attestation.version_major as i64) * 1000000
                     + (bundle.stdlib_attestation.version_minor as i64) * 1000
                     + (bundle.stdlib_attestation.version_patch as i64)

    return AuditRecord {
        computation_id: computation_id,
        computation_time: bundle.generation_time,
        result_value_hash: result_hash,
        uncertainty_hash: uncertainty_hash,
        stdlib_version_hash: version_hash,
        stdlib_slsa_hash: bundle.stdlib_attestation.slsa.attestation_hash,
        is_verified: is_verified,
        audit_timestamp: audit_time,
    }
}

// ============================================================================
// INVARIANTS FOR EPISTEMIC SUPPLY CHAIN
// ============================================================================

/// Check supply chain invariants
struct SupplyChainCheck {
    has_slsa_attestation: bool,
    slsa_level_sufficient: bool,
    dependencies_pinned: bool,
    builder_verified: bool,
    is_compliant: bool,
}

fn check_supply_chain(bundle: PROVBundleWithSLSA, required_level: i32) -> SupplyChainCheck {
    let has_slsa = bundle.stdlib_attestation.slsa.attestation_hash != 0
    let level_ok = bundle.stdlib_attestation.slsa.slsa_level >= required_level
    let deps_ok = bundle.stdlib_attestation.slsa.build_definition.dep_count > 0
    let builder_ok = bundle.stdlib_attestation.slsa.run_details.builder_id_hash != 0

    let compliant = has_slsa && level_ok && deps_ok && builder_ok

    return SupplyChainCheck {
        has_slsa_attestation: has_slsa,
        slsa_level_sufficient: level_ok,
        dependencies_pinned: deps_ok,
        builder_verified: builder_ok,
        is_compliant: compliant,
    }
}

// ============================================================================
// TESTS
// ============================================================================

fn test_stdlib_attestation() -> bool {
    let stdlib = stdlib_attestation_new(
        0, 82, 0,           // v0.82.0
        12345678,           // Git commit hash
        1700000000          // Build timestamp
    )

    if stdlib.version_major != 0 { return false }
    if stdlib.version_minor != 82 { return false }
    if stdlib.slsa.slsa_level != 1 { return false }
    if stdlib.slsa.attestation_hash == 0 { return false }

    return true
}

fn test_prov_bundle_with_slsa() -> bool {
    let stdlib = stdlib_attestation_new(0, 82, 0, 12345678, 1700000000)

    let bundle = prov_bundle_with_slsa(
        100,        // entity
        200,        // activity
        300,        // agent
        1700001000, // gen time
        stdlib,
        99999       // computation digest
    )

    // Bundle should be verifiable
    if !verify_bundle(bundle) { return false }

    // Check supply chain
    let check = check_supply_chain(bundle, 1)
    if !check.has_slsa_attestation { return false }
    if !check.slsa_level_sufficient { return false }
    if !check.is_compliant { return false }

    return true
}

fn test_audit_record() -> bool {
    let stdlib = stdlib_attestation_new(0, 82, 0, 12345678, 1700000000)
    let bundle = prov_bundle_with_slsa(100, 200, 300, 1700001000, stdlib, 99999)

    let audit = create_audit_record(
        1001,       // computation ID
        54321,      // result hash
        11111,      // uncertainty hash
        bundle,
        1700002000  // audit time
    )

    if !audit.is_verified { return false }
    if audit.computation_id != 1001 { return false }

    // Version hash should encode v0.82.0 as 0*1M + 82*1K + 0 = 82000
    if audit.stdlib_version_hash != 82000 { return false }

    return true
}

fn test_slsa_level_requirement() -> bool {
    let stdlib = stdlib_attestation_new(0, 82, 0, 12345678, 1700000000)
    let bundle = prov_bundle_with_slsa(100, 200, 300, 1700001000, stdlib, 99999)

    // Require L1 -> should pass
    let check1 = check_supply_chain(bundle, 1)
    if !check1.slsa_level_sufficient { return false }

    // Require L2 -> should fail (we're only L1)
    let check2 = check_supply_chain(bundle, 2)
    if check2.slsa_level_sufficient { return false }

    return true
}

// ============================================================================
// MAIN
// ============================================================================

fn main() -> i32 {
    if !test_stdlib_attestation() { return 1 }
    if !test_prov_bundle_with_slsa() { return 2 }
    if !test_audit_record() { return 3 }
    if !test_slsa_level_requirement() { return 4 }

    return 0
}
