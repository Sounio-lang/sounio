//! W3C PROV-DM Export Module
//!
//! Implements core W3C Provenance Data Model concepts for epistemic values.
//!
//! PROV-DM Core:
//!   - Entity: A thing with fixed aspects (data value, measurement)
//!   - Activity: Something that occurs (computation, measurement)
//!   - Agent: Something that bears responsibility
//!
//! References:
//!   - PROV-DM: https://www.w3.org/TR/prov-dm/
//!   - PROV-CONSTRAINTS: https://www.w3.org/TR/prov-constraints/

extern "C" {
    fn sqrt(x: f64) -> f64;
}

fn sqrt_f64(x: f64) -> f64 {
    if x < 0.0 { return 0.0 }
    return sqrt(x)
}

// ============================================================================
// PROV IDENTIFIERS
// ============================================================================

struct ProvId {
    namespace_id: i32,
    local_id: i64,
}

fn prov_id_new(ns: i32, local: i64) -> ProvId {
    return ProvId { namespace_id: ns, local_id: local }
}

fn prov_id_eq(a: ProvId, b: ProvId) -> bool {
    return a.namespace_id == b.namespace_id && a.local_id == b.local_id
}

// ============================================================================
// ENTITY
// ============================================================================

struct ProvEntity {
    id: ProvId,
    entity_type: i32,    // 0=measured, 1=literature, 2=computed, 3=input
    value: f64,
    uncertainty: f64,
    confidence: f64,
    generated_at: i64,
    reference_hash: i64,
}

fn entity_measured(id: ProvId, value: f64, uncert: f64, conf: f64, timestamp: i64) -> ProvEntity {
    return ProvEntity {
        id: id,
        entity_type: 0,
        value: value,
        uncertainty: uncert,
        confidence: conf,
        generated_at: timestamp,
        reference_hash: 0,
    }
}

fn entity_computed(id: ProvId, value: f64, uncert: f64, conf: f64, timestamp: i64) -> ProvEntity {
    return ProvEntity {
        id: id,
        entity_type: 2,
        value: value,
        uncertainty: uncert,
        confidence: conf,
        generated_at: timestamp,
        reference_hash: 0,
    }
}

fn entity_literature(id: ProvId, value: f64, uncert: f64, conf: f64, ref_hash: i64) -> ProvEntity {
    return ProvEntity {
        id: id,
        entity_type: 1,
        value: value,
        uncertainty: uncert,
        confidence: conf,
        generated_at: 0,
        reference_hash: ref_hash,
    }
}

// ============================================================================
// ACTIVITY
// ============================================================================

struct ProvActivity {
    id: ProvId,
    activity_type: i32,  // 0=measurement, 1=arithmetic, 2=transform, 3=fusion, 4=boost, 5=unwrap
    started_at: i64,
    ended_at: i64,
    operation_code: i32,
    reason_hash: i64,
}

fn activity_measurement(id: ProvId, start: i64, end: i64) -> ProvActivity {
    return ProvActivity {
        id: id,
        activity_type: 0,
        started_at: start,
        ended_at: end,
        operation_code: 0,
        reason_hash: 0,
    }
}

fn activity_arithmetic(id: ProvId, op_code: i32, timestamp: i64) -> ProvActivity {
    return ProvActivity {
        id: id,
        activity_type: 1,
        started_at: timestamp,
        ended_at: timestamp,
        operation_code: op_code,
        reason_hash: 0,
    }
}

fn activity_unwrap(id: ProvId, timestamp: i64, reason_hash: i64) -> ProvActivity {
    return ProvActivity {
        id: id,
        activity_type: 5,
        started_at: timestamp,
        ended_at: timestamp,
        operation_code: 0,
        reason_hash: reason_hash,
    }
}

// ============================================================================
// RELATIONS
// ============================================================================

struct WasGeneratedBy {
    entity_id: ProvId,
    activity_id: ProvId,
    timestamp: i64,
}

struct Used {
    activity_id: ProvId,
    entity_id: ProvId,
    timestamp: i64,
}

struct WasDerivedFrom {
    derived_id: ProvId,
    source_id: ProvId,
    activity_id: ProvId,
}

// ============================================================================
// SIMPLE PROVENANCE RECORD (single derivation chain)
// ============================================================================

// Instead of a full document, track a single value's provenance
struct ProvenanceRecord {
    // Current entity
    entity: ProvEntity,

    // How it was generated
    generation_activity: ProvActivity,

    // Source entities (up to 4 for binary ops)
    source_count: i32,
    source1_id: ProvId,
    source2_id: ProvId,
    source3_id: ProvId,
    source4_id: ProvId,

    // Validation state
    is_valid: bool,
    validation_code: i32,  // 0=ok, 1=timing error, 2=missing source
}

fn prov_record_new(e: ProvEntity, a: ProvActivity) -> ProvenanceRecord {
    return ProvenanceRecord {
        entity: e,
        generation_activity: a,
        source_count: 0,
        source1_id: prov_id_new(0, 0),
        source2_id: prov_id_new(0, 0),
        source3_id: prov_id_new(0, 0),
        source4_id: prov_id_new(0, 0),
        is_valid: true,
        validation_code: 0,
    }
}

fn prov_record_with_sources(
    e: ProvEntity,
    a: ProvActivity,
    src1: ProvId,
    src2: ProvId
) -> ProvenanceRecord {
    return ProvenanceRecord {
        entity: e,
        generation_activity: a,
        source_count: 2,
        source1_id: src1,
        source2_id: src2,
        source3_id: prov_id_new(0, 0),
        source4_id: prov_id_new(0, 0),
        is_valid: true,
        validation_code: 0,
    }
}

// ============================================================================
// VALIDATION
// ============================================================================

fn validate_timing(record: ProvenanceRecord) -> ProvenanceRecord {
    var result = record

    // Generation must not be before activity start
    if record.entity.generated_at > 0 && record.generation_activity.started_at > 0 {
        if record.entity.generated_at < record.generation_activity.started_at {
            result.is_valid = false
            result.validation_code = 1
        }
    }

    return result
}

// ============================================================================
// METROLOGICAL TRACEABILITY
// ============================================================================

struct CalibrationLink {
    reference_hash: i64,
    certificate_hash: i64,
    calibration_date: i64,
    contributed_uncertainty: f64,
    coverage_factor: f64,
}

fn calibration_link_new(ref_hash: i64, cert_hash: i64, date: i64, uncert: f64, k: f64) -> CalibrationLink {
    return CalibrationLink {
        reference_hash: ref_hash,
        certificate_hash: cert_hash,
        calibration_date: date,
        contributed_uncertainty: uncert,
        coverage_factor: k,
    }
}

// Traceability chain (up to 4 links for simplicity)
struct TraceabilityChain {
    link_count: i32,
    link1: CalibrationLink,
    link2: CalibrationLink,
    link3: CalibrationLink,
    link4: CalibrationLink,
    is_complete: bool,
    total_uncertainty: f64,
}

fn empty_link() -> CalibrationLink {
    return CalibrationLink {
        reference_hash: 0,
        certificate_hash: 0,
        calibration_date: 0,
        contributed_uncertainty: 0.0,
        coverage_factor: 1.0,
    }
}

fn traceability_chain_new() -> TraceabilityChain {
    return TraceabilityChain {
        link_count: 0,
        link1: empty_link(),
        link2: empty_link(),
        link3: empty_link(),
        link4: empty_link(),
        is_complete: false,
        total_uncertainty: 0.0,
    }
}

fn add_calibration_link(chain: TraceabilityChain, link: CalibrationLink) -> TraceabilityChain {
    var result = chain
    let idx = chain.link_count

    if idx == 0 { result.link1 = link }
    else if idx == 1 { result.link2 = link }
    else if idx == 2 { result.link3 = link }
    else if idx == 3 { result.link4 = link }

    if idx < 4 {
        result.link_count = idx + 1
        let u = link.contributed_uncertainty
        let prev_u2 = chain.total_uncertainty * chain.total_uncertainty
        result.total_uncertainty = sqrt_f64(prev_u2 + u * u)
    }

    return result
}

fn mark_chain_complete(chain: TraceabilityChain) -> TraceabilityChain {
    var result = chain
    result.is_complete = true
    return result
}

fn is_traceable(chain: TraceabilityChain) -> bool {
    return chain.is_complete && chain.link_count > 0
}

// ============================================================================
// PROV EXPORT STATISTICS
// ============================================================================

struct ProvStats {
    entity_type: i32,
    activity_type: i32,
    source_count: i32,
    is_valid: bool,
    has_traceability: bool,
    total_uncertainty: f64,
}

fn get_prov_stats(record: ProvenanceRecord, chain: TraceabilityChain) -> ProvStats {
    return ProvStats {
        entity_type: record.entity.entity_type,
        activity_type: record.generation_activity.activity_type,
        source_count: record.source_count,
        is_valid: record.is_valid,
        has_traceability: is_traceable(chain),
        total_uncertainty: record.entity.uncertainty,
    }
}

// ============================================================================
// TESTS
// ============================================================================

fn test_basic_prov_record() -> bool {
    let e = entity_measured(prov_id_new(1, 100), 75.5, 0.5, 0.95, 1700000000)
    let a = activity_measurement(prov_id_new(1, 200), 1699999990, 1700000000)

    let record = prov_record_new(e, a)

    if !record.is_valid { return false }
    if record.entity.entity_type != 0 { return false }

    return true
}

fn test_invalid_timing() -> bool {
    // Entity generated BEFORE activity started (invalid)
    let e = entity_measured(prov_id_new(1, 100), 75.5, 0.5, 0.95, 1700000000)
    let a = activity_measurement(prov_id_new(1, 200), 1700000010, 1700000020)

    var record = prov_record_new(e, a)
    record = validate_timing(record)

    return !record.is_valid && record.validation_code == 1
}

fn test_traceability() -> bool {
    var chain = traceability_chain_new()

    let link1 = calibration_link_new(12345, 67890, 1700000000, 0.02, 2.0)
    chain = add_calibration_link(chain, link1)

    let link2 = calibration_link_new(11111, 22222, 1690000000, 0.01, 2.0)
    chain = add_calibration_link(chain, link2)

    chain = mark_chain_complete(chain)

    if !is_traceable(chain) { return false }
    if chain.link_count != 2 { return false }

    // sqrt(0.02² + 0.01²) ≈ 0.0224
    let expected = sqrt_f64(0.02 * 0.02 + 0.01 * 0.01)
    let diff = chain.total_uncertainty - expected
    if diff < -0.001 || diff > 0.001 { return false }

    return true
}

fn test_derived_provenance() -> bool {
    let e = entity_computed(prov_id_new(1, 300), 95.5, 2.1, 0.85, 1700000000)
    let a = activity_arithmetic(prov_id_new(1, 400), 1, 1700000000)  // op 1 = add

    let record = prov_record_with_sources(
        e, a,
        prov_id_new(1, 100),  // source 1
        prov_id_new(1, 200)   // source 2
    )

    if record.source_count != 2 { return false }
    if record.entity.entity_type != 2 { return false }  // computed

    return true
}

// ============================================================================
// MAIN
// ============================================================================

fn main() -> i32 {
    if !test_basic_prov_record() { return 1 }
    if !test_invalid_timing() { return 2 }
    if !test_traceability() { return 3 }
    if !test_derived_provenance() { return 4 }

    return 0
}
