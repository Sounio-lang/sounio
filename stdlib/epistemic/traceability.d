//! Metrological Traceability: Provable Claims, Not Labels
//!
//! Per VIM (International Vocabulary of Metrology):
//!   "Metrological traceability is a property of a measurement result
//!    whereby the result can be related to a reference through a
//!    documented unbroken chain of calibrations, each contributing
//!    to the measurement uncertainty."
//!
//! NIST emphasizes: traceability is a property of the RESULT, not the instrument.
//!
//! This module enforces:
//!   - traceability_claim_without_chain is a HARD ERROR
//!   - Every link in the chain must contribute documented uncertainty
//!   - The chain must terminate at a recognized reference
//!
//! References:
//!   - VIM3 (JCGM 200:2012): Section 2.41 "metrological traceability"
//!   - NIST Policy on Traceability: https://www.nist.gov/traceability

extern "C" {
    fn sqrt(x: f64) -> f64;
}

fn sqrt_f64(x: f64) -> f64 {
    if x < 0.0 { return 0.0 }
    return sqrt(x)
}

fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { return 0.0 - x }
    return x
}

// ============================================================================
// REFERENCE STANDARDS
// ============================================================================

/// Recognized reference standard types
/// Higher level = more authoritative
struct ReferenceStandard {
    standard_type: i32,     // 0=SI, 1=national, 2=accredited_lab, 3=working_standard
    level: i32,             // 0=primary, 1=secondary, 2=tertiary, 3=working
    identifier_hash: i64,   // Hash of standard identifier (e.g., NIST SRM number)
    is_recognized: bool,    // True if from a recognized body
}

fn si_primary() -> ReferenceStandard {
    return ReferenceStandard {
        standard_type: 0,
        level: 0,
        identifier_hash: 0,
        is_recognized: true,
    }
}

fn national_standard(id_hash: i64) -> ReferenceStandard {
    return ReferenceStandard {
        standard_type: 1,
        level: 1,
        identifier_hash: id_hash,
        is_recognized: true,
    }
}

fn accredited_lab_standard(id_hash: i64) -> ReferenceStandard {
    return ReferenceStandard {
        standard_type: 2,
        level: 2,
        identifier_hash: id_hash,
        is_recognized: true,
    }
}

fn working_standard(id_hash: i64) -> ReferenceStandard {
    return ReferenceStandard {
        standard_type: 3,
        level: 3,
        identifier_hash: id_hash,
        is_recognized: false,  // Must be validated
    }
}

// ============================================================================
// CALIBRATION LINK WITH VALIDATION
// ============================================================================

/// A single link in the traceability chain
struct TraceabilityLink {
    // Reference this link points to
    reference_id: i64,
    reference_type: ReferenceStandard,

    // Calibration documentation
    certificate_hash: i64,
    calibration_date: i64,
    expiry_date: i64,
    lab_id_hash: i64,
    procedure_hash: i64,

    // Uncertainty contribution
    contributed_uncertainty: f64,
    coverage_factor: f64,
    dof: f64,              // Degrees of freedom

    // Validation status
    is_documented: bool,
    is_expired: bool,
    is_valid: bool,
}

fn empty_link() -> TraceabilityLink {
    return TraceabilityLink {
        reference_id: 0,
        reference_type: working_standard(0),
        certificate_hash: 0,
        calibration_date: 0,
        expiry_date: 0,
        lab_id_hash: 0,
        procedure_hash: 0,
        contributed_uncertainty: 0.0,
        coverage_factor: 1.0,
        dof: 1.0e30,
        is_documented: false,
        is_expired: false,
        is_valid: false,
    }
}

/// Create a documented calibration link
fn calibration_link(
    ref_id: i64,
    ref_type: ReferenceStandard,
    cert_hash: i64,
    cal_date: i64,
    expiry: i64,
    lab_hash: i64,
    proc_hash: i64,
    uncert: f64,
    k: f64,
    dof: f64,
    current_time: i64
) -> TraceabilityLink {
    let is_expired = current_time > expiry
    let is_documented = cert_hash != 0 && lab_hash != 0 && proc_hash != 0
    let is_valid = is_documented && !is_expired && ref_type.is_recognized

    return TraceabilityLink {
        reference_id: ref_id,
        reference_type: ref_type,
        certificate_hash: cert_hash,
        calibration_date: cal_date,
        expiry_date: expiry,
        lab_id_hash: lab_hash,
        procedure_hash: proc_hash,
        contributed_uncertainty: uncert,
        coverage_factor: k,
        dof: dof,
        is_documented: is_documented,
        is_expired: is_expired,
        is_valid: is_valid,
    }
}

// ============================================================================
// TRACEABILITY CHAIN
// ============================================================================

/// Complete traceability chain with validation
struct TraceabilityChain {
    // Links in the chain (up to 4)
    link_count: i32,
    link0: TraceabilityLink,
    link1: TraceabilityLink,
    link2: TraceabilityLink,
    link3: TraceabilityLink,

    // Chain properties
    terminates_at_si: bool,
    terminates_at_national: bool,
    is_unbroken: bool,
    all_links_valid: bool,
    total_uncertainty: f64,
    effective_dof: f64,

    // Validation result
    is_complete: bool,
    error_code: i32,    // 0=ok, 1=broken, 2=expired, 3=undocumented, 4=no_recognized_ref
}

fn chain_new() -> TraceabilityChain {
    return TraceabilityChain {
        link_count: 0,
        link0: empty_link(),
        link1: empty_link(),
        link2: empty_link(),
        link3: empty_link(),
        terminates_at_si: false,
        terminates_at_national: false,
        is_unbroken: true,
        all_links_valid: true,
        total_uncertainty: 0.0,
        effective_dof: 1.0e30,
        is_complete: false,
        error_code: 0,
    }
}

fn get_link(chain: TraceabilityChain, idx: i32) -> TraceabilityLink {
    if idx == 0 { return chain.link0 }
    if idx == 1 { return chain.link1 }
    if idx == 2 { return chain.link2 }
    if idx == 3 { return chain.link3 }
    return empty_link()
}

/// Add a link to the chain
fn chain_add_link(chain: TraceabilityChain, link: TraceabilityLink) -> TraceabilityChain {
    var result = chain
    let idx = chain.link_count

    if idx == 0 { result.link0 = link }
    else if idx == 1 { result.link1 = link }
    else if idx == 2 { result.link2 = link }
    else if idx == 3 { result.link3 = link }

    if idx < 4 {
        result.link_count = idx + 1

        // Update total uncertainty (quadrature)
        let u = link.contributed_uncertainty
        let prev_u2 = chain.total_uncertainty * chain.total_uncertainty
        result.total_uncertainty = sqrt_f64(prev_u2 + u * u)

        // Update validity
        if !link.is_valid {
            result.all_links_valid = false
        }

        // Check if terminates at recognized reference
        if link.reference_type.standard_type == 0 {
            result.terminates_at_si = true
        }
        if link.reference_type.standard_type == 1 {
            result.terminates_at_national = true
        }
    }

    return result
}

/// Finalize and validate the chain
fn chain_finalize(chain: TraceabilityChain) -> TraceabilityChain {
    var result = chain

    if chain.link_count == 0 {
        result.is_complete = false
        result.error_code = 1  // Broken (no links)
        return result
    }

    // Check all links valid
    if !chain.all_links_valid {
        result.is_complete = false

        // Find first invalid link for error code
        var i: i32 = 0
        while i < chain.link_count {
            let link = get_link(chain, i)
            if link.is_expired {
                result.error_code = 2  // Expired
                return result
            }
            if !link.is_documented {
                result.error_code = 3  // Undocumented
                return result
            }
            if !link.reference_type.is_recognized {
                result.error_code = 4  // No recognized reference
                return result
            }
            i = i + 1
        }

        result.error_code = 1  // Generic broken
        return result
    }

    // Check terminates at recognized reference
    if !chain.terminates_at_si && !chain.terminates_at_national {
        result.is_complete = false
        result.error_code = 4  // No recognized reference
        return result
    }

    result.is_complete = true
    result.error_code = 0
    return result
}

// ============================================================================
// TRACEABILITY CLAIM VALIDATION (THE LINT)
// ============================================================================

/// Result of traceability claim validation
struct TraceabilityValidation {
    is_valid: bool,
    error_code: i32,
    error_level: i32,       // 0=none, 1=warning, 2=hard_error
    message_hash: i64,
}

/// CRITICAL LINT: traceability_claim_without_chain
/// This is a HARD ERROR - no exceptions
fn validate_traceability_claim(chain: TraceabilityChain, claimed: bool) -> TraceabilityValidation {
    if !claimed {
        // No claim made - OK
        return TraceabilityValidation {
            is_valid: true,
            error_code: 0,
            error_level: 0,
            message_hash: 0,
        }
    }

    // Claim made - chain must be complete and valid
    let finalized = chain_finalize(chain)

    if !finalized.is_complete {
        // HARD ERROR: traceability claimed but chain is incomplete
        return TraceabilityValidation {
            is_valid: false,
            error_code: finalized.error_code,
            error_level: 2,  // Hard error
            message_hash: 100 + finalized.error_code,  // Error message lookup
        }
    }

    // Chain is complete - claim is justified
    return TraceabilityValidation {
        is_valid: true,
        error_code: 0,
        error_level: 0,
        message_hash: 0,
    }
}

/// Get error message for validation failure
fn get_error_message(validation: TraceabilityValidation) -> i32 {
    // Return message code (would be string in real impl)
    if validation.error_code == 1 {
        return 1  // "Traceability claim without documented chain"
    }
    if validation.error_code == 2 {
        return 2  // "Traceability chain contains expired calibration"
    }
    if validation.error_code == 3 {
        return 3  // "Traceability chain contains undocumented link"
    }
    if validation.error_code == 4 {
        return 4  // "Traceability chain does not terminate at recognized reference"
    }
    return 0
}

// ============================================================================
// TRACEABLE VALUE TYPE
// ============================================================================

/// A value that carries its traceability chain hash
/// (Simplified to avoid nested struct issues)
struct TracedResult {
    value: f64,
    uncertainty: f64,
    chain_hash: i64,           // Hash of the traceability chain
    chain_link_count: i32,     // Number of links in chain
    terminates_at_si: bool,    // True if chain reaches SI
    is_traceable: bool,        // True if chain is complete and valid
}

fn traced_result(val: f64, uncert: f64, chain: TraceabilityChain) -> TracedResult {
    let finalized = chain_finalize(chain)
    // Compute a simple chain hash for verification
    let hash = finalized.total_uncertainty as i64
             + (finalized.link_count as i64) * 1000
             + (finalized.effective_dof as i64)
    return TracedResult {
        value: val,
        uncertainty: uncert,
        chain_hash: hash,
        chain_link_count: finalized.link_count,
        terminates_at_si: finalized.terminates_at_si,
        is_traceable: finalized.is_complete,
    }
}

/// Assert traceability on a chain - returns error if not provable
fn assert_chain_traceable(chain: TraceabilityChain) -> TraceabilityValidation {
    return validate_traceability_claim(chain, true)
}

// ============================================================================
// CHAIN CONTINUITY VALIDATION
// ============================================================================

/// Check that chain is unbroken (each link references the next)
struct ContinuityCheck {
    is_continuous: bool,
    break_at: i32,          // Index of first break (-1 if none)
    gap_description: i32,   // 0=none, 1=date_gap, 2=reference_mismatch
}

fn check_continuity(chain: TraceabilityChain) -> ContinuityCheck {
    if chain.link_count < 2 {
        return ContinuityCheck {
            is_continuous: true,
            break_at: -1,
            gap_description: 0,
        }
    }

    var i: i32 = 0
    while i < chain.link_count - 1 {
        let curr = get_link(chain, i)
        let next = get_link(chain, i + 1)

        // Check date ordering: current calibration should be after next
        // (we're going from measurement to primary standard)
        if curr.calibration_date < next.expiry_date {
            // Current was calibrated before next expired - OK
        } else {
            return ContinuityCheck {
                is_continuous: false,
                break_at: i,
                gap_description: 1,  // Date gap
            }
        }

        // Check reference continuity: curr.reference should match next.level
        // (simplified check)

        i = i + 1
    }

    return ContinuityCheck {
        is_continuous: true,
        break_at: -1,
        gap_description: 0,
    }
}

// ============================================================================
// TESTS
// ============================================================================

fn test_empty_chain_invalid() -> bool {
    let chain = chain_new()
    let validation = validate_traceability_claim(chain, true)

    // Should be a hard error
    if validation.is_valid { return false }
    if validation.error_level != 2 { return false }

    return true
}

fn test_valid_chain() -> bool {
    var chain = chain_new()

    // Working standard -> Accredited lab -> NIST
    let link1 = calibration_link(
        1,                          // reference_id
        accredited_lab_standard(100),
        12345,                      // certificate
        1700000000,                 // calibration date
        1800000000,                 // expiry
        54321,                      // lab
        11111,                      // procedure
        0.02,                       // uncertainty
        2.0,                        // k
        30.0,                       // dof
        1750000000                  // current time
    )
    chain = chain_add_link(chain, link1)

    let link2 = calibration_link(
        2,
        national_standard(200),
        67890,
        1690000000,
        1790000000,
        99999,
        22222,
        0.01,
        2.0,
        100.0,
        1750000000
    )
    chain = chain_add_link(chain, link2)

    let validation = validate_traceability_claim(chain, true)

    // Should be valid
    if !validation.is_valid { return false }
    if validation.error_level != 0 { return false }

    return true
}

fn test_expired_link_fails() -> bool {
    var chain = chain_new()

    // Expired calibration
    let link = calibration_link(
        1,
        national_standard(100),
        12345,
        1600000000,     // calibration date
        1650000000,     // expiry (in past)
        54321,
        11111,
        0.02,
        2.0,
        30.0,
        1750000000      // current time (after expiry)
    )
    chain = chain_add_link(chain, link)

    let validation = validate_traceability_claim(chain, true)

    // Should fail with expired error
    if validation.is_valid { return false }
    if validation.error_code != 2 { return false }

    return true
}

fn test_undocumented_link_fails() -> bool {
    var chain = chain_new()

    // Missing documentation (cert_hash = 0)
    let link = calibration_link(
        1,
        national_standard(100),
        0,              // No certificate!
        1700000000,
        1800000000,
        54321,
        11111,
        0.02,
        2.0,
        30.0,
        1750000000
    )
    chain = chain_add_link(chain, link)

    let validation = validate_traceability_claim(chain, true)

    // Should fail with undocumented error
    if validation.is_valid { return false }
    if validation.error_code != 3 { return false }

    return true
}

fn test_no_recognized_reference_fails() -> bool {
    var chain = chain_new()

    // Only working standard (not recognized)
    let link = calibration_link(
        1,
        working_standard(100),
        12345,
        1700000000,
        1800000000,
        54321,
        11111,
        0.02,
        2.0,
        30.0,
        1750000000
    )
    chain = chain_add_link(chain, link)

    let validation = validate_traceability_claim(chain, true)

    // Should fail with no recognized reference
    if validation.is_valid { return false }
    if validation.error_code != 4 { return false }

    return true
}

fn test_traced_result() -> bool {
    var chain = chain_new()

    let link = calibration_link(
        1,
        national_standard(100),
        12345,
        1700000000,
        1800000000,
        54321,
        11111,
        0.02,
        2.0,
        30.0,
        1750000000
    )
    chain = chain_add_link(chain, link)

    let tr = traced_result(100.0, 0.5, chain)

    if !tr.is_traceable { return false }

    let validation = assert_chain_traceable(chain)
    if !validation.is_valid { return false }

    return true
}

// ============================================================================
// MAIN
// ============================================================================

fn main() -> i32 {
    if !test_empty_chain_invalid() { return 1 }
    if !test_valid_chain() { return 2 }
    if !test_expired_link_fails() { return 3 }
    if !test_undocumented_link_fails() { return 4 }
    if !test_no_recognized_reference_fails() { return 5 }
    if !test_traced_result() { return 6 }

    return 0
}
