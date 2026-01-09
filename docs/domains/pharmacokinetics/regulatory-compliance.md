# Regulatory Compliance in Sounio

Sounio's pharmacokinetics framework is designed with regulatory submission in mind. This document covers FDA 21 CFR Part 11 compliance, EMA PBPK guidelines, audit trail requirements, and how Sounio's epistemic features support regulatory acceptance.

## Regulatory Landscape

### FDA Guidance

Key FDA guidance documents relevant to Sounio PK modeling:

- **Physiologically Based Pharmacokinetic Analyses - Format and Content** (2018)
- **Population Pharmacokinetics** (2022)
- **In Vitro Drug Interaction Studies** (2020)
- **Clinical Drug Interaction Studies** (2020)

### EMA Guidelines

- **Reporting of PBPK Modelling and Simulation** (2018)
- **Guideline on the Investigation of Drug Interactions** (2012)
- **Guideline on Population PK** (2020)

## FDA 21 CFR Part 11 Compliance

21 CFR Part 11 establishes requirements for electronic records and electronic signatures.

### Key Requirements

1. **Validation**: Systems must be validated
2. **Audit trails**: Computer-generated, timestamped
3. **Record retention**: Accurate and complete copies
4. **System access**: Limited to authorized individuals
5. **Authority checks**: Device/signature checks

### Sounio Implementation

```sio
use pbpk::regulatory::*

/// Audit trail entry
struct AuditEntry {
    timestamp: DateTime,
    user: string,
    action: string,
    entity: string,
    old_value: Option<string>,
    new_value: Option<string>,
    reason: string,
    signature: ElectronicSignature,
}

/// Electronic signature (21 CFR Part 11 compliant)
struct ElectronicSignature {
    signer_id: string,
    timestamp: DateTime,
    meaning: string,         // "author", "reviewer", "approver"
    hash: string,            // SHA-256 of signed content
    certificate: string,     // Digital certificate reference
}

/// Audit trail for a simulation session
struct AuditTrail {
    session_id: string,
    start_time: DateTime,
    entries: Vec<AuditEntry>,
    final_signature: Option<ElectronicSignature>,
}
```

### Audit Trail Functions

```sio
use pbpk::regulatory::*

/// Create new audit trail
fn begin_audit_session(user: string, purpose: string) -> AuditTrail {
    let session_id = generate_uuid()
    let entry = AuditEntry {
        timestamp: now(),
        user: user,
        action: "SESSION_START",
        entity: session_id,
        old_value: None,
        new_value: Some(purpose),
        reason: purpose,
        signature: sign_entry(&user, "SESSION_START")
    }

    return AuditTrail {
        session_id: session_id,
        start_time: now(),
        entries: vec![entry],
        final_signature: None
    }
}

/// Log parameter change
fn log_parameter_change(
    trail: &!AuditTrail,
    user: string,
    param_name: string,
    old_value: f64,
    new_value: f64,
    reason: string
) {
    let entry = AuditEntry {
        timestamp: now(),
        user: user,
        action: "PARAMETER_CHANGE",
        entity: param_name,
        old_value: Some(format!("{}", old_value)),
        new_value: Some(format!("{}", new_value)),
        reason: reason,
        signature: sign_entry(&user, "PARAMETER_CHANGE")
    }
    trail.entries.push(entry)
}

/// Log simulation execution
fn log_simulation(
    trail: &!AuditTrail,
    user: string,
    model_name: string,
    config_hash: string,
    result_hash: string
) {
    let entry = AuditEntry {
        timestamp: now(),
        user: user,
        action: "SIMULATION_RUN",
        entity: model_name,
        old_value: Some(config_hash),
        new_value: Some(result_hash),
        reason: "Model simulation executed",
        signature: sign_entry(&user, "SIMULATION_RUN")
    }
    trail.entries.push(entry)
}

/// Finalize and sign audit trail
fn finalize_audit_trail(
    trail: &!AuditTrail,
    user: string,
    meaning: string
) {
    let content_hash = hash_trail_content(trail)
    trail.final_signature = Some(ElectronicSignature {
        signer_id: user,
        timestamp: now(),
        meaning: meaning,
        hash: content_hash,
        certificate: get_user_certificate(&user)
    })
}
```

## Data Integrity

### ALCOA+ Principles

Sounio supports ALCOA+ (Attributable, Legible, Contemporaneous, Original, Accurate + Complete, Consistent, Enduring, Available):

```sio
use pbpk::regulatory::*

/// Data record with ALCOA+ metadata
struct RegulatoryRecord {
    // Core data
    data: Vec<u8>,

    // ALCOA+ metadata
    attributable: Attribution,
    timestamp: DateTime,                // Contemporaneous
    original_source: string,            // Original
    hash: string,                       // Accurate (integrity check)
    complete: bool,
    consistent_with: Vec<string>,       // Related records
    retention_date: DateTime,           // Enduring
    access_level: AccessLevel           // Available
}

struct Attribution {
    creator: string,
    role: string,
    organization: string,
    system: string,
}

/// Verify data integrity
fn verify_integrity(record: &RegulatoryRecord) -> bool {
    let computed_hash = sha256(&record.data)
    return computed_hash == record.hash
}
```

## PBPK Model Validation

### FDA Acceptance Criteria

The FDA PBPK guidance specifies validation requirements:

```sio
use pbpk::regulatory::*

/// FDA-recommended validation metrics
struct ValidationMetrics {
    // Fold errors
    gmfe: f64,          // Geometric Mean Fold Error
    afe: f64,           // Average Fold Error
    aafe: f64,          // Absolute Average Fold Error

    // Percentage within thresholds
    within_1_5fold: f64,
    within_2fold: f64,
    within_3fold: f64,

    // Individual predictions
    fold_errors: Vec<f64>,
    predictions: Vec<f64>,
    observations: Vec<f64>,
}

/// Calculate GMFE
fn calculate_gmfe(predicted: &Vec<f64>, observed: &Vec<f64>) -> f64 {
    let n = predicted.len()
    var log_sum = 0.0

    for i in 0..n {
        let fe = predicted[i] / observed[i]
        log_sum = log_sum + abs(ln(fe))
    }

    return exp(log_sum / (n as f64))
}

/// Calculate percentage within fold threshold
fn percent_within_fold(
    predicted: &Vec<f64>,
    observed: &Vec<f64>,
    fold: f64
) -> f64 {
    let n = predicted.len()
    var within = 0

    for i in 0..n {
        let fe = predicted[i] / observed[i]
        if fe >= 1.0/fold && fe <= fold {
            within = within + 1
        }
    }

    return (within as f64) / (n as f64)
}

/// Full validation metrics
fn calculate_validation_metrics(
    predicted: &Vec<f64>,
    observed: &Vec<f64>
) -> ValidationMetrics {
    var fold_errors: Vec<f64> = vec![]
    for i in 0..predicted.len() {
        fold_errors.push(predicted[i] / observed[i])
    }

    return ValidationMetrics {
        gmfe: calculate_gmfe(predicted, observed),
        afe: mean(&fold_errors),
        aafe: mean_abs(&fold_errors),
        within_1_5fold: percent_within_fold(predicted, observed, 1.5),
        within_2fold: percent_within_fold(predicted, observed, 2.0),
        within_3fold: percent_within_fold(predicted, observed, 3.0),
        fold_errors: fold_errors,
        predictions: predicted.clone(),
        observations: observed.clone()
    }
}
```

### Model Qualification Criteria

```sio
use pbpk::regulatory::*

/// FDA PBPK qualification levels
enum QualificationLevel {
    Fit,                // Describes observed data
    Extrapolate,        // Predicts beyond training data
    Regulatory,         // Supports labeling decisions
}

/// Check if model meets FDA qualification criteria
fn check_fda_qualification(metrics: &ValidationMetrics) -> QualificationResult {
    // FDA typical acceptance: GMFE <= 2 and >= 50% within 2-fold
    let meets_gmfe = metrics.gmfe <= 2.0
    let meets_within_2fold = metrics.within_2fold >= 0.50

    // Stricter criteria for regulatory impact
    let meets_strict_gmfe = metrics.gmfe <= 1.5
    let meets_strict_2fold = metrics.within_2fold >= 0.80

    if meets_strict_gmfe && meets_strict_2fold {
        return QualificationResult {
            level: QualificationLevel::Regulatory,
            passed: true,
            notes: "Model meets strict qualification criteria"
        }
    } else if meets_gmfe && meets_within_2fold {
        return QualificationResult {
            level: QualificationLevel::Extrapolate,
            passed: true,
            notes: "Model meets standard qualification criteria"
        }
    } else {
        return QualificationResult {
            level: QualificationLevel::Fit,
            passed: false,
            notes: "Model does not meet FDA qualification criteria"
        }
    }
}
```

## Provenance Tracking

Sounio's epistemic types automatically track data provenance:

### Source Attribution

```sio
use epistemic::*

/// Parameter with full provenance chain
let cl_hepatic = Knowledge::new(
    value: 30.0 L/h,
    confidence: 0.85,
    source: "In vitro hepatocyte intrinsic clearance, scaled using IVIVE"
).with_provenance(Provenance {
    study_id: "NCT0012345",
    reference: "Smith et al., DMD 2023",
    measurement_method: "Hepatocyte suspension assay",
    scaling_method: "Well-stirred liver model",
    species: "Human",
    n_subjects: 6,
    timestamp: "2023-04-15"
})

/// Track provenance through calculations
let total_cl = cl_hepatic + cl_renal
// total_cl.provenance includes both parent provenances
```

### Provenance Report Generation

```sio
use pbpk::regulatory::*

/// Generate provenance report for regulatory submission
fn generate_provenance_report(params: &PBPKParams) -> ProvenanceReport {
    var entries: Vec<ProvenanceEntry> = vec![]

    // Collect all parameter provenances
    entries.push(ProvenanceEntry {
        parameter: "CL_hepatic",
        value: params.cl_hepatic.value,
        unit: "L/h",
        source: params.cl_hepatic.provenance.source,
        reference: params.cl_hepatic.provenance.reference,
        confidence: params.cl_hepatic.confidence,
        method: params.cl_hepatic.provenance.measurement_method
    })

    // ... all other parameters

    return ProvenanceReport {
        title: "PBPK Model Parameter Provenance",
        drug_name: params.drug_name,
        model_version: params.model_version,
        generation_date: now(),
        entries: entries,
        references: collect_unique_references(&entries)
    }
}
```

## Report Generation

### FDA Submission Format

```sio
use pbpk::regulatory::*

/// Generate FDA PBPK submission report
fn generate_fda_report(
    drug: &DrugProperties,
    params: &PBPKParams,
    result: &SimulationResult,
    observed_data: &ClinicalData,
    config: &ReportConfig
) -> FDAReport {
    // Section 1: Drug Properties
    let drug_section = DrugPropertiesSection {
        name: drug.name,
        molecular_weight: drug.mw,
        logp: drug.logp,
        pka: drug.pka,
        fu_plasma: drug.fu,
        bp_ratio: drug.bp_ratio,
        solubility: drug.solubility,
        permeability: drug.permeability
    }

    // Section 2: PBPK Model Structure
    let model_section = ModelStructureSection {
        compartments: list_compartments(params),
        elimation_pathways: list_pathways(params),
        enzyme_contributions: list_enzymes(drug),
        transporter_effects: list_transporters(drug)
    }

    // Section 3: Parameter Table
    let param_section = generate_parameter_table(params)

    // Section 4: Validation Results
    let metrics = calculate_validation_metrics(&result.predictions, &observed_data)
    let validation_section = ValidationSection {
        metrics: metrics,
        training_data: observed_data.training,
        test_data: observed_data.test,
        qualification: check_fda_qualification(&metrics)
    }

    // Section 5: Sensitivity Analysis
    let sensitivity_section = run_sensitivity_analysis(params, result)

    // Section 6: Application
    let application_section = ApplicationSection {
        intended_use: config.intended_use,
        population: config.target_population,
        scenarios: config.simulation_scenarios,
        recommendations: generate_recommendations(result, config)
    }

    return FDAReport {
        drug: drug_section,
        model: model_section,
        parameters: param_section,
        validation: validation_section,
        sensitivity: sensitivity_section,
        application: application_section,
        appendices: generate_appendices(params, result),
        audit_trail: config.audit_trail
    }
}
```

### Parameter Table Format

```sio
/// Generate parameter table per FDA format
fn generate_parameter_table(params: &PBPKParams) -> ParameterTable {
    var rows: Vec<ParameterRow> = vec![]

    // Physicochemical
    rows.push(ParameterRow {
        category: "Physicochemical",
        parameter: "Molecular weight",
        value: format!("{:.1}", params.mw),
        unit: "g/mol",
        source: params.mw_source,
        reference: params.mw_reference
    })

    // Absorption
    rows.push(ParameterRow {
        category: "Absorption",
        parameter: "Peff,man",
        value: format!("{:.2e}", params.peff),
        unit: "cm/s",
        source: "Caco-2 with IVIVE",
        reference: params.peff_reference
    })

    // Distribution - Kp values
    for (tissue, kp) in params.kp_values {
        rows.push(ParameterRow {
            category: "Distribution",
            parameter: format!("Kp,{}", tissue),
            value: format!("{:.2}", kp.value),
            unit: "unitless",
            source: "Rodgers-Rowland prediction",
            reference: "Rodgers & Rowland, 2006"
        })
    }

    // Elimination
    rows.push(ParameterRow {
        category: "Elimination",
        parameter: "CLint,mic",
        value: format!("{:.1}", params.clint_mic),
        unit: "uL/min/mg protein",
        source: "HLM incubation",
        reference: params.clint_reference
    })

    return ParameterTable {
        title: "PBPK Model Input Parameters",
        rows: rows
    }
}
```

## Epistemic Features for Regulatory Support

Sounio's epistemic computing provides unique regulatory advantages:

### Confidence-Based Model Qualification

```sio
use epistemic::*
use pbpk::regulatory::*

/// Assess prediction reliability for regulatory decision
fn assess_prediction_reliability(
    prediction: Knowledge[f64],
    regulatory_threshold: f64
) -> RegulatoryAssessment {
    // Base decision on both value and confidence
    let margin = abs(prediction.value - regulatory_threshold) / regulatory_threshold

    if prediction.confidence >= 0.90 && margin >= 0.20 {
        return RegulatoryAssessment {
            decision: "Clear determination possible",
            confidence_level: "High",
            recommendation: "Prediction supports regulatory decision",
            additional_studies: false
        }
    } else if prediction.confidence >= 0.75 {
        return RegulatoryAssessment {
            decision: "Determination possible with caveats",
            confidence_level: "Moderate",
            recommendation: "Prediction supports decision with stated uncertainty",
            additional_studies: prediction.confidence < 0.85
        }
    } else {
        return RegulatoryAssessment {
            decision: "Insufficient confidence",
            confidence_level: "Low",
            recommendation: "Additional clinical data recommended",
            additional_studies: true
        }
    }
}
```

### Uncertainty Quantification Report

```sio
/// Generate uncertainty quantification section
fn generate_uq_section(
    result: &EpistemicSimulationResult
) -> UncertaintySection {
    // Propagated uncertainties
    let cmax_ci = result.cmax.confidence_interval(0.90)
    let auc_ci = result.auc.confidence_interval(0.90)

    // Sensitivity contributions
    let sensitivities = analyze_uncertainty_sources(&result)

    return UncertaintySection {
        title: "Uncertainty Quantification",
        prediction_intervals: vec![
            PredictionInterval {
                parameter: "Cmax",
                point_estimate: result.cmax.value,
                lower_90: cmax_ci.0,
                upper_90: cmax_ci.1,
                confidence: result.cmax.confidence
            },
            PredictionInterval {
                parameter: "AUC",
                point_estimate: result.auc.value,
                lower_90: auc_ci.0,
                upper_90: auc_ci.1,
                confidence: result.auc.confidence
            }
        ],
        major_uncertainty_sources: sensitivities.top_contributors(5),
        recommendations: generate_uq_recommendations(&sensitivities)
    }
}
```

## Electronic Submission

### eCTD Module 2.7

```sio
/// Generate eCTD Module 2.7 clinical summary PBPK section
fn generate_ectd_pbpk_section(
    report: &FDAReport
) -> ECTDSection {
    return ECTDSection {
        module: "2.7.2",
        title: "Summary of Clinical Pharmacology Studies - PBPK Analysis",
        content: format_ectd_content(report),
        tables: extract_tables(report),
        figures: generate_figures(report),
        references: report.references(),
        appendix_references: vec!["5.3.3.5-1", "5.3.3.5-2"]  // Module 5 refs
    }
}
```

## Validation Testing

### System Validation Protocol

```sio
use pbpk::regulatory::*

/// Installation Qualification (IQ)
fn run_installation_qualification() -> IQResult {
    var checks: Vec<IQCheck> = vec![]

    // Version verification
    checks.push(IQCheck {
        item: "Sounio version",
        expected: "0.97.0",
        actual: sounio_version(),
        passed: sounio_version() == "0.97.0"
    })

    // Module availability
    checks.push(IQCheck {
        item: "PBPK module",
        expected: "Available",
        actual: if module_available("pbpk") { "Available" } else { "Missing" },
        passed: module_available("pbpk")
    })

    // ODE solver tests
    checks.push(IQCheck {
        item: "BDF solver",
        expected: "Functional",
        actual: test_bdf_solver(),
        passed: test_bdf_solver() == "Functional"
    })

    return IQResult {
        timestamp: now(),
        checks: checks,
        all_passed: checks.iter().all(|c| c.passed)
    }
}

/// Operational Qualification (OQ)
fn run_operational_qualification() -> OQResult {
    var tests: Vec<OQTest> = vec![]

    // Test exponential decay (known solution)
    let exp_result = test_exponential_decay()
    tests.push(OQTest {
        name: "Exponential decay ODE",
        expected_result: 36.788,  // 100 * exp(-1)
        actual_result: exp_result,
        tolerance: 0.01,
        passed: abs(exp_result - 36.788) < 0.01
    })

    // Test midazolam PBPK reference case
    let mid_result = test_midazolam_reference()
    tests.push(OQTest {
        name: "Midazolam PBPK Cmax",
        expected_result: 0.12,  // mg/L
        actual_result: mid_result.cmax,
        tolerance: 0.02,
        passed: abs(mid_result.cmax - 0.12) < 0.02
    })

    return OQResult {
        timestamp: now(),
        tests: tests,
        all_passed: tests.iter().all(|t| t.passed)
    }
}
```

## Summary

Sounio's pharmacokinetics framework provides comprehensive regulatory support through:

1. **21 CFR Part 11 Compliance**: Audit trails, electronic signatures, data integrity
2. **ALCOA+ Principles**: Attributable, legible, contemporaneous, original, accurate records
3. **FDA Validation Metrics**: GMFE, fold error analysis, qualification criteria
4. **Provenance Tracking**: Full source attribution through epistemic types
5. **Report Generation**: FDA/EMA submission-ready formats
6. **Uncertainty Quantification**: Confidence-based regulatory decision support

These features make Sounio uniquely suited for regulatory pharmacometric submissions where both accuracy and traceable uncertainty are essential.
