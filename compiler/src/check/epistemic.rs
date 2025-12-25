//! Epistemic Type Integration
//!
//! This module extends the type checker with epistemic awareness, enabling
//! dependent ontological types with confidence tracking.
//!
//! # Knowledge[τ, ε, δ, Φ] Type System
//!
//! - τ (tau): Temporal index - when the knowledge is valid
//! - ε (epsilon): Epistemic status - confidence and source
//! - δ (delta): Domain constraint - ontological binding
//! - Φ (phi): Provenance functor - derivation trace
//!
//! # Dependent Ontological Types
//!
//! ```text
//! type Aspirin = Knowledge[
//!     ChEBI:15365,
//!     ε ≥ 0.95,
//!     δ ⊆ SmallMolecule,
//!     Φ: ChEBI → verified
//! ]
//! ```

use std::collections::HashMap;

use crate::epistemic::{
    Confidence, EpistemicStatus, Evidence, OntologyBinding, Revisability, Source,
};
use crate::ontology::{FoundationOntologies, OntologyResolver, ParsedTermRef, SubsumptionResult};

/// Temporal index for epistemic validity tracking
///
/// Represents a point in time for temporal epistemic constraints.
/// Supports ISO 8601 timestamps and Unix epoch seconds.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TemporalIndex {
    /// ISO 8601 timestamp string (e.g., "2024-01-15T10:30:00Z")
    pub timestamp: Option<String>,
    /// Unix epoch seconds (for efficient comparison)
    pub epoch_secs: Option<i64>,
}

impl TemporalIndex {
    /// Create a new temporal index from an ISO 8601 string
    pub fn from_iso(iso: &str) -> Self {
        // Parse the ISO string to epoch seconds if possible
        let epoch = parse_iso_to_epoch(iso);
        Self {
            timestamp: Some(iso.to_string()),
            epoch_secs: epoch,
        }
    }

    /// Create a temporal index from Unix epoch seconds
    pub fn from_epoch(secs: i64) -> Self {
        Self {
            timestamp: None,
            epoch_secs: Some(secs),
        }
    }

    /// Create the "now" temporal index
    pub fn now() -> Self {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        Self {
            timestamp: None,
            epoch_secs: Some(secs),
        }
    }

    /// Check if this temporal index is before another
    pub fn is_before(&self, other: &TemporalIndex) -> Option<bool> {
        match (self.epoch_secs, other.epoch_secs) {
            (Some(a), Some(b)) => Some(a < b),
            _ => None,
        }
    }

    /// Check if this temporal index is after another
    pub fn is_after(&self, other: &TemporalIndex) -> Option<bool> {
        match (self.epoch_secs, other.epoch_secs) {
            (Some(a), Some(b)) => Some(a > b),
            _ => None,
        }
    }

    /// Get the age in seconds from now
    pub fn age_secs(&self) -> Option<i64> {
        let now = TemporalIndex::now();
        match (self.epoch_secs, now.epoch_secs) {
            (Some(then), Some(now_secs)) => Some(now_secs - then),
            _ => None,
        }
    }

    /// Get the age in days from now
    pub fn age_days(&self) -> Option<u32> {
        self.age_secs().map(|secs| (secs / 86400) as u32)
    }
}

/// Parse ISO 8601 timestamp to Unix epoch seconds
///
/// Supports formats:
/// - "2024-01-15T10:30:00Z" (full ISO with Z)
/// - "2024-01-15T10:30:00+00:00" (with timezone offset)
/// - "2024-01-15" (date only, assumes 00:00:00 UTC)
fn parse_iso_to_epoch(iso: &str) -> Option<i64> {
    // Simple parser for common ISO 8601 formats
    // In production, would use chrono or time crate

    let trimmed = iso.trim();

    // Try to parse YYYY-MM-DD format (minimum)
    if trimmed.len() < 10 {
        return None;
    }

    let year: i32 = trimmed[0..4].parse().ok()?;
    let month: u32 = trimmed[5..7].parse().ok()?;
    let day: u32 = trimmed[8..10].parse().ok()?;

    // Basic validation
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }

    // Parse time if present
    let (hour, minute, second) = if trimmed.len() >= 19 && &trimmed[10..11] == "T" {
        let h: u32 = trimmed[11..13].parse().ok()?;
        let m: u32 = trimmed[14..16].parse().ok()?;
        let s: u32 = trimmed[17..19].parse().ok()?;
        (h, m, s)
    } else {
        (0, 0, 0)
    };

    // Calculate days since Unix epoch (1970-01-01)
    // This is a simplified calculation, not handling leap seconds
    let days_since_epoch = days_from_date(year, month, day)?;

    // Convert to seconds
    let secs =
        days_since_epoch as i64 * 86400 + hour as i64 * 3600 + minute as i64 * 60 + second as i64;

    Some(secs)
}

/// Calculate days from Unix epoch to a given date
fn days_from_date(year: i32, month: u32, day: u32) -> Option<i64> {
    // Days in each month (non-leap year)
    const DAYS_IN_MONTH: [u32; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

    fn is_leap_year(y: i32) -> bool {
        (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
    }

    fn days_in_year(y: i32) -> i64 {
        if is_leap_year(y) { 366 } else { 365 }
    }

    // Calculate days from 1970 to start of year
    let mut days: i64 = 0;
    if year >= 1970 {
        for y in 1970..year {
            days += days_in_year(y);
        }
    } else {
        for y in year..1970 {
            days -= days_in_year(y);
        }
    }

    // Add days for months in current year
    for m in 1..month {
        days += DAYS_IN_MONTH[(m - 1) as usize] as i64;
        if m == 2 && is_leap_year(year) {
            days += 1;
        }
    }

    // Add days in current month
    days += (day - 1) as i64;

    Some(days)
}

/// Convert Unix epoch seconds to ISO date string (YYYY-MM-DD)
fn epoch_to_iso_date(epoch_secs: i64) -> String {
    const DAYS_IN_MONTH: [u32; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

    fn is_leap_year(y: i32) -> bool {
        (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
    }

    fn days_in_year(y: i32) -> i64 {
        if is_leap_year(y) { 366 } else { 365 }
    }

    let days = epoch_secs / 86400;
    let mut remaining_days = days;
    let mut year = 1970i32;

    // Find the year
    if remaining_days >= 0 {
        while remaining_days >= days_in_year(year) {
            remaining_days -= days_in_year(year);
            year += 1;
        }
    } else {
        while remaining_days < 0 {
            year -= 1;
            remaining_days += days_in_year(year);
        }
    }

    // Find the month
    let mut month = 1u32;
    loop {
        let days_this_month = if month == 2 && is_leap_year(year) {
            29
        } else {
            DAYS_IN_MONTH[(month - 1) as usize] as i64
        };

        if remaining_days < days_this_month {
            break;
        }
        remaining_days -= days_this_month;
        month += 1;
    }

    let day = remaining_days + 1;

    format!("{:04}-{:02}-{:02}", year, month, day)
}

/// A dependent ontological type with epistemic constraints
#[derive(Debug, Clone)]
pub struct OntologicalType {
    /// The ontology term this type represents
    pub binding: OntologyBinding,
    /// Minimum required confidence
    pub min_confidence: Confidence,
    /// Required evidence types
    pub required_evidence: Vec<EvidenceRequirement>,
    /// Temporal validity constraints
    pub temporal_constraint: Option<TemporalConstraint>,
    /// Provenance requirements
    pub provenance_constraint: Option<ProvenanceConstraint>,
}

/// Evidence requirement for a type constraint
#[derive(Debug, Clone)]
pub enum EvidenceRequirement {
    /// Require any evidence
    Any,
    /// Require publication evidence
    Publication,
    /// Require experimental evidence
    Experimental,
    /// Require computational evidence
    Computational,
    /// Require a specific minimum strength
    MinStrength(Confidence),
}

/// Temporal constraint for type validity
#[derive(Debug, Clone)]
pub enum TemporalConstraint {
    /// Valid at a specific point in time
    AtTime(TemporalIndex),
    /// Valid during an interval
    During {
        start: TemporalIndex,
        end: TemporalIndex,
    },
    /// Must be current (no more than N days old)
    Current { max_age_days: u32 },
}

/// Provenance constraint for derivation tracking
#[derive(Debug, Clone)]
pub enum ProvenanceConstraint {
    /// Must originate from a specific source
    FromSource(Source),
    /// Must pass through a verification step
    Verified,
    /// Maximum derivation depth
    MaxDepth(u32),
    /// Must have human review
    HumanReviewed,
}

/// Result of checking an epistemic constraint
#[derive(Debug, Clone)]
pub enum ConstraintResult {
    /// Constraint satisfied
    Satisfied,
    /// Constraint violated with explanation
    Violated(String),
    /// Constraint cannot be checked (missing information)
    Indeterminate(String),
}

/// Epistemic type checker integration
pub struct EpistemicChecker {
    /// Foundation ontologies for quick lookups
    foundations: FoundationOntologies,
    /// Ontology resolver for full resolution
    resolver: OntologyResolver,
    /// Type bindings in scope
    bindings: HashMap<String, OntologicalType>,
}

impl EpistemicChecker {
    /// Create a new epistemic checker
    pub fn new() -> Self {
        let resolver =
            OntologyResolver::default_resolver().expect("Failed to create ontology resolver");
        Self {
            foundations: FoundationOntologies::bootstrap(),
            resolver,
            bindings: HashMap::new(),
        }
    }

    /// Check if a value's epistemic status satisfies a type's requirements
    pub fn check_constraint(
        &mut self,
        value_status: &EpistemicStatus,
        type_constraint: &OntologicalType,
    ) -> ConstraintResult {
        // Check confidence requirement
        if value_status.confidence.value() < type_constraint.min_confidence.value() {
            return ConstraintResult::Violated(format!(
                "Confidence {} is below required minimum {}",
                value_status.confidence.value(),
                type_constraint.min_confidence.value()
            ));
        }

        // Check evidence requirements
        for requirement in &type_constraint.required_evidence {
            if !self.check_evidence_requirement(&value_status.evidence, requirement) {
                return ConstraintResult::Violated(format!(
                    "Evidence requirement not satisfied: {:?}",
                    requirement
                ));
            }
        }

        // Check temporal constraint
        if let Some(ref temporal) = type_constraint.temporal_constraint {
            match self.check_temporal_constraint(value_status, temporal) {
                ConstraintResult::Violated(msg) => return ConstraintResult::Violated(msg),
                ConstraintResult::Indeterminate(msg) => {
                    return ConstraintResult::Indeterminate(msg);
                }
                _ => {}
            }
        }

        // Check provenance constraint
        if let Some(ref provenance) = type_constraint.provenance_constraint {
            match self.check_provenance_constraint(value_status, provenance) {
                ConstraintResult::Violated(msg) => return ConstraintResult::Violated(msg),
                ConstraintResult::Indeterminate(msg) => {
                    return ConstraintResult::Indeterminate(msg);
                }
                _ => {}
            }
        }

        ConstraintResult::Satisfied
    }

    /// Check evidence requirement
    fn check_evidence_requirement(
        &self,
        evidence: &[Evidence],
        requirement: &EvidenceRequirement,
    ) -> bool {
        match requirement {
            EvidenceRequirement::Any => !evidence.is_empty(),
            EvidenceRequirement::Publication => evidence
                .iter()
                .any(|e| matches!(e.kind, crate::epistemic::EvidenceKind::Publication { .. })),
            EvidenceRequirement::Experimental => evidence
                .iter()
                .any(|e| matches!(e.kind, crate::epistemic::EvidenceKind::Experiment { .. })),
            EvidenceRequirement::Computational => evidence
                .iter()
                .any(|e| matches!(e.kind, crate::epistemic::EvidenceKind::Computation { .. })),
            EvidenceRequirement::MinStrength(min) => {
                evidence.iter().any(|e| e.strength.value() >= min.value())
            }
        }
    }

    /// Check temporal constraint
    ///
    /// Verifies that the epistemic status satisfies temporal validity requirements:
    /// - `AtTime`: Knowledge must be valid at a specific point in time
    /// - `During`: Knowledge must be valid during an interval
    /// - `Current`: Knowledge must not be older than a maximum age
    fn check_temporal_constraint(
        &self,
        status: &EpistemicStatus,
        constraint: &TemporalConstraint,
    ) -> ConstraintResult {
        // Extract the most recent timestamp from evidence
        let evidence_timestamp = self.extract_evidence_timestamp(status);

        match constraint {
            TemporalConstraint::AtTime(required_time) => {
                match &evidence_timestamp {
                    Some(evidence_time) => {
                        // Check if the evidence time matches (within same day for simplicity)
                        match (evidence_time.epoch_secs, required_time.epoch_secs) {
                            (Some(ev_secs), Some(req_secs)) => {
                                // Allow 24-hour window around the required time
                                let diff = (ev_secs - req_secs).abs();
                                if diff <= 86400 {
                                    ConstraintResult::Satisfied
                                } else {
                                    ConstraintResult::Violated(format!(
                                        "Evidence timestamp differs from required time by {} days",
                                        diff / 86400
                                    ))
                                }
                            }
                            _ => ConstraintResult::Indeterminate(
                                "Cannot compare timestamps: missing epoch information".into(),
                            ),
                        }
                    }
                    None => ConstraintResult::Indeterminate(
                        "No timestamp available in evidence to check AtTime constraint".into(),
                    ),
                }
            }

            TemporalConstraint::During { start, end } => match &evidence_timestamp {
                Some(evidence_time) => {
                    match (evidence_time.epoch_secs, start.epoch_secs, end.epoch_secs) {
                        (Some(ev), Some(s), Some(e)) => {
                            if ev >= s && ev <= e {
                                ConstraintResult::Satisfied
                            } else if ev < s {
                                ConstraintResult::Violated(format!(
                                    "Evidence predates validity period by {} days",
                                    (s - ev) / 86400
                                ))
                            } else {
                                ConstraintResult::Violated(format!(
                                    "Evidence postdates validity period by {} days",
                                    (ev - e) / 86400
                                ))
                            }
                        }
                        _ => ConstraintResult::Indeterminate(
                            "Cannot check interval: missing epoch information".into(),
                        ),
                    }
                }
                None => ConstraintResult::Indeterminate(
                    "No timestamp available in evidence to check During constraint".into(),
                ),
            },

            TemporalConstraint::Current { max_age_days } => {
                match &evidence_timestamp {
                    Some(evidence_time) => match evidence_time.age_days() {
                        Some(age) => {
                            if age <= *max_age_days {
                                ConstraintResult::Satisfied
                            } else {
                                ConstraintResult::Violated(format!(
                                    "Evidence is {} days old, exceeds maximum age of {} days",
                                    age, max_age_days
                                ))
                            }
                        }
                        None => ConstraintResult::Indeterminate(
                            "Cannot compute evidence age: missing timestamp".into(),
                        ),
                    },
                    None => {
                        // No timestamp means we can't verify freshness
                        // For medical/regulatory contexts, this should be a violation
                        ConstraintResult::Violated(
                            "No timestamp available to verify evidence freshness".into(),
                        )
                    }
                }
            }
        }
    }

    /// Extract the most recent timestamp from evidence
    ///
    /// Looks through the evidence chain to find timestamp information.
    /// Returns the most recent timestamp found, or None if no timestamps available.
    fn extract_evidence_timestamp(&self, status: &EpistemicStatus) -> Option<TemporalIndex> {
        // First, check if any evidence has timestamp info in its reference
        // Evidence references often contain DOIs or dates
        let mut most_recent: Option<TemporalIndex> = None;

        for evidence in &status.evidence {
            // Try to extract timestamp from evidence reference
            if let Some(ts) = self.parse_timestamp_from_reference(&evidence.reference) {
                match (&most_recent, ts.epoch_secs) {
                    (None, _) => most_recent = Some(ts),
                    (Some(current), Some(new_epoch)) => {
                        if let Some(curr_epoch) = current.epoch_secs
                            && new_epoch > curr_epoch
                        {
                            most_recent = Some(ts);
                        }
                    }
                    _ => {}
                }
            }
        }

        // Also check the source for timestamp information
        if most_recent.is_none()
            && let Source::Measurement { timestamp, .. } = &status.source
            && let Some(ts_str) = timestamp
        {
            most_recent = Some(TemporalIndex::from_iso(ts_str));
        }

        most_recent
    }

    /// Try to parse a timestamp from an evidence reference string
    ///
    /// Handles common formats:
    /// - ISO 8601 dates embedded in strings (YYYY-MM-DD)
    /// - Year-only references
    fn parse_timestamp_from_reference(&self, reference: &str) -> Option<TemporalIndex> {
        // Simple pattern matching for YYYY-MM-DD without regex dependency
        // Look for patterns like "2024-01-15" in the reference

        let bytes = reference.as_bytes();
        let len = bytes.len();

        // Scan for YYYY-MM-DD pattern
        for i in 0..len.saturating_sub(9) {
            // Check for YYYY-MM-DD format
            if i + 10 <= len
                && bytes[i].is_ascii_digit()
                && bytes[i + 1].is_ascii_digit()
                && bytes[i + 2].is_ascii_digit()
                && bytes[i + 3].is_ascii_digit()
                && bytes[i + 4] == b'-'
                && bytes[i + 5].is_ascii_digit()
                && bytes[i + 6].is_ascii_digit()
                && bytes[i + 7] == b'-'
                && bytes[i + 8].is_ascii_digit()
                && bytes[i + 9].is_ascii_digit()
            {
                let date_str = &reference[i..i + 10];
                let year: i32 = date_str[0..4].parse().ok()?;
                let month: u32 = date_str[5..7].parse().ok()?;
                let day: u32 = date_str[8..10].parse().ok()?;

                // Validate
                if (1900..=2100).contains(&year)
                    && (1..=12).contains(&month)
                    && (1..=31).contains(&day)
                {
                    return Some(TemporalIndex::from_iso(date_str));
                }
            }
        }

        None
    }

    /// Check provenance constraint
    fn check_provenance_constraint(
        &mut self,
        status: &EpistemicStatus,
        constraint: &ProvenanceConstraint,
    ) -> ConstraintResult {
        match constraint {
            ProvenanceConstraint::FromSource(required_source) => {
                if self.sources_compatible(&status.source, required_source) {
                    ConstraintResult::Satisfied
                } else {
                    ConstraintResult::Violated(format!(
                        "Source {:?} does not match required {:?}",
                        status.source, required_source
                    ))
                }
            }
            ProvenanceConstraint::Verified => {
                // Check if evidence includes verification
                if status
                    .evidence
                    .iter()
                    .any(|e| matches!(e.kind, crate::epistemic::EvidenceKind::Verified { .. }))
                {
                    ConstraintResult::Satisfied
                } else {
                    ConstraintResult::Violated("No verification evidence found".into())
                }
            }
            ProvenanceConstraint::MaxDepth(_) => {
                ConstraintResult::Indeterminate("Depth tracking not yet implemented".into())
            }
            ProvenanceConstraint::HumanReviewed => {
                // Check for human review evidence
                if status.evidence.iter().any(|e| {
                    matches!(
                        e.kind,
                        crate::epistemic::EvidenceKind::HumanAssertion { .. }
                    )
                }) {
                    ConstraintResult::Satisfied
                } else {
                    ConstraintResult::Violated("No human review evidence found".into())
                }
            }
        }
    }

    /// Check if two sources are compatible using ontology subsumption
    ///
    /// A source is compatible if:
    /// - Exact match (same ontology and term)
    /// - Actual term is a subclass of required term (subsumption)
    /// - Terms are equivalent in the ontology
    /// - Transformations preserve compatibility through the chain
    fn sources_compatible(&mut self, actual: &Source, required: &Source) -> bool {
        match (actual, required) {
            // Exact match for axioms
            (Source::Axiom, Source::Axiom) => true,

            // Ontology assertion with subsumption checking
            (
                Source::OntologyAssertion {
                    ontology: o1,
                    term: t1,
                },
                Source::OntologyAssertion {
                    ontology: o2,
                    term: t2,
                },
            ) => {
                // Must be same ontology for subsumption
                if o1 != o2 {
                    return false;
                }

                // Exact term match
                if t1 == t2 {
                    return true;
                }

                // Check subsumption: actual should be more specific (subclass of required)
                // Example: CHEBI:15365 (aspirin) is subclass of CHEBI:35475 (anti-inflammatory drug)
                let actual_curie = format!("{}:{}", o1, t1);
                let required_curie = format!("{}:{}", o2, t2);

                match self.resolver.is_subclass_of(&actual_curie, &required_curie) {
                    Ok(SubsumptionResult::IsSubclass) => true,
                    Ok(SubsumptionResult::Equivalent) => true,
                    _ => false,
                }
            }

            // Measurement compatibility: protocol subsumes
            (
                Source::Measurement {
                    protocol: Some(p1), ..
                },
                Source::Measurement {
                    protocol: Some(p2), ..
                },
            ) => {
                // Check if measurement protocols are compatible
                // p1 should be equal or more specific than p2
                p1 == p2 || p1.starts_with(p2)
            }

            // Derivation compatibility
            (Source::Derivation(d1), Source::Derivation(d2)) => d1 == d2,

            // External source compatibility
            (Source::External { uri: u1, .. }, Source::External { uri: u2, .. }) => {
                // Same URI or u1 is more specific (sub-path)
                u1 == u2 || u1.starts_with(u2)
            }

            // Model prediction compatibility
            (
                Source::ModelPrediction {
                    model: m1,
                    version: v1,
                },
                Source::ModelPrediction {
                    model: m2,
                    version: v2,
                },
            ) => {
                // Same model required
                if m1 != m2 {
                    return false;
                }
                // If required specifies version, actual must match
                match (v1, v2) {
                    (_, None) => true, // No version required
                    (Some(v1), Some(v2)) => v1 == v2,
                    (None, Some(_)) => false, // Required version but none provided
                }
            }

            // Transformation chains: check the original source
            (Source::Transformation { original, .. }, required) => {
                self.sources_compatible(original, required)
            }

            // Allow transformation to satisfy non-transformation if original matches
            (actual, Source::Transformation { original, .. }) => {
                self.sources_compatible(actual, original)
            }

            // Axiom satisfies any requirement (strongest source)
            (Source::Axiom, _) => true,

            // Default: not compatible
            _ => false,
        }
    }

    /// Check ontological subsumption
    pub fn check_subsumption(&mut self, child: &str, parent: &str) -> SubsumptionResult {
        self.resolver
            .is_subclass_of(child, parent)
            .unwrap_or(SubsumptionResult::Unknown)
    }

    /// Bind a variable to an ontological type
    pub fn bind(&mut self, name: String, ty: OntologicalType) {
        self.bindings.insert(name, ty);
    }

    /// Look up a binding
    pub fn lookup(&self, name: &str) -> Option<&OntologicalType> {
        self.bindings.get(name)
    }

    /// Create an ontological type from a CURIE with default constraints
    pub fn type_from_curie(&self, curie: &str) -> Result<OntologicalType, String> {
        let parsed = ParsedTermRef::parse(curie).map_err(|e| e.to_string())?;

        Ok(OntologicalType {
            binding: parsed.to_binding(),
            min_confidence: Confidence::new(0.0), // No minimum by default
            required_evidence: vec![],
            temporal_constraint: None,
            provenance_constraint: None,
        })
    }

    /// Create an ontological type with confidence requirement
    pub fn type_with_confidence(
        &self,
        curie: &str,
        min_confidence: f64,
    ) -> Result<OntologicalType, String> {
        let mut ty = self.type_from_curie(curie)?;
        ty.min_confidence = Confidence::new(min_confidence);
        Ok(ty)
    }
}

impl Default for EpistemicChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute epistemic heterogeneity between two types
pub fn epistemic_heterogeneity(a: &EpistemicStatus, b: &EpistemicStatus) -> f64 {
    // Difference in confidence levels
    let confidence_diff = (a.confidence.value() - b.confidence.value()).abs();

    // Source compatibility (0 if same, 1 if different)
    let source_diff = if std::mem::discriminant(&a.source) == std::mem::discriminant(&b.source) {
        0.0
    } else {
        0.5
    };

    // Combine heterogeneity factors
    (confidence_diff + source_diff) / 2.0
}

/// Combine epistemic statuses using Bayesian methods
pub fn combine_epistemic_bayesian(statuses: &[EpistemicStatus]) -> EpistemicStatus {
    if statuses.is_empty() {
        return EpistemicStatus::default();
    }

    if statuses.len() == 1 {
        return statuses[0].clone();
    }

    // Bayesian combination of confidence values
    // Using log-odds combination
    let combined_confidence = {
        let mut log_odds_sum = 0.0;

        for status in statuses {
            let p = status.confidence.value().clamp(0.001, 0.999);
            let log_odds = (p / (1.0 - p)).ln();
            log_odds_sum += log_odds;
        }

        let avg_log_odds = log_odds_sum / statuses.len() as f64;
        let combined_p = 1.0 / (1.0 + (-avg_log_odds).exp());
        combined_p.clamp(0.0, 1.0)
    };

    // Combine evidence from all sources
    let combined_evidence: Vec<Evidence> =
        statuses.iter().flat_map(|s| s.evidence.clone()).collect();

    // Use most restrictive revisability
    let combined_revisability = Revisability::Revisable {
        conditions: statuses
            .iter()
            .filter_map(|s| {
                if let Revisability::Revisable { conditions } = &s.revisability {
                    Some(conditions.clone())
                } else {
                    None
                }
            })
            .flatten()
            .collect(),
    };

    EpistemicStatus {
        confidence: Confidence::new(combined_confidence),
        revisability: combined_revisability,
        source: Source::Derivation("bayesian_combination".into()),
        evidence: combined_evidence,
    }
}

/// Combine epistemic statuses using weighted Bayesian methods
///
/// Unlike `combine_epistemic_bayesian`, this weighs each status by its
/// evidence strength, giving more weight to higher-quality evidence.
///
/// # Weighting Strategy
///
/// Each status contributes to the final confidence proportionally to:
/// 1. Its own confidence level
/// 2. The average strength of its evidence
/// 3. The number of independent evidence sources
///
/// This implements a form of "strength-adjusted" Bayesian update.
pub fn combine_epistemic_weighted_bayesian(statuses: &[EpistemicStatus]) -> EpistemicStatus {
    if statuses.is_empty() {
        return EpistemicStatus::default();
    }

    if statuses.len() == 1 {
        return statuses[0].clone();
    }

    // Calculate evidence weight for each status
    let weights: Vec<f64> = statuses.iter().map(calculate_evidence_weight).collect();

    let total_weight: f64 = weights.iter().sum();

    // Weighted Bayesian combination using log-odds
    let combined_confidence = if total_weight > 0.0 {
        let mut weighted_log_odds_sum = 0.0;

        for (status, &weight) in statuses.iter().zip(weights.iter()) {
            let p = status.confidence.value().clamp(0.001, 0.999);
            let log_odds = (p / (1.0 - p)).ln();
            // Weight the log-odds contribution
            weighted_log_odds_sum += log_odds * (weight / total_weight);
        }

        // Scale by number of sources (more evidence = more certainty)
        let evidence_multiplier = (1.0 + (statuses.len() as f64).ln()).min(2.0);
        let adjusted_log_odds = weighted_log_odds_sum * evidence_multiplier;

        let combined_p = 1.0 / (1.0 + (-adjusted_log_odds).exp());
        combined_p.clamp(0.0, 1.0)
    } else {
        // Fallback to simple average if no weights
        statuses.iter().map(|s| s.confidence.value()).sum::<f64>() / statuses.len() as f64
    };

    // Deduplicate and merge evidence, keeping highest strength for duplicates
    let combined_evidence = merge_evidence(statuses);

    // Compute combined revisability
    let combined_revisability = combine_revisability(statuses);

    // Track the combination method in provenance
    let source = Source::Derivation(format!(
        "weighted_bayesian_combination(n={}, total_weight={:.3})",
        statuses.len(),
        total_weight
    ));

    EpistemicStatus {
        confidence: Confidence::new(combined_confidence),
        revisability: combined_revisability,
        source,
        evidence: combined_evidence,
    }
}

/// Calculate the evidence weight for an epistemic status
///
/// Weight is computed as:
/// - Base weight from confidence level
/// - Multiplied by average evidence strength
/// - Bonus for multiple independent evidence sources
fn calculate_evidence_weight(status: &EpistemicStatus) -> f64 {
    let base_weight = status.confidence.value();

    if status.evidence.is_empty() {
        // No evidence: rely solely on confidence
        return base_weight * 0.5; // Penalty for no evidence
    }

    // Average evidence strength
    let avg_strength: f64 = status
        .evidence
        .iter()
        .map(|e| e.strength.value())
        .sum::<f64>()
        / status.evidence.len() as f64;

    // Bonus for multiple evidence sources (diminishing returns)
    let diversity_bonus = 1.0 + (status.evidence.len() as f64).ln() * 0.1;

    // Evidence kind quality multiplier
    let quality_multiplier = status
        .evidence
        .iter()
        .map(|e| evidence_kind_quality(&e.kind))
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(1.0);

    base_weight * avg_strength * diversity_bonus * quality_multiplier
}

/// Quality multiplier based on evidence kind
///
/// Higher quality evidence types get higher multipliers
fn evidence_kind_quality(kind: &crate::epistemic::EvidenceKind) -> f64 {
    use crate::epistemic::EvidenceKind;

    match kind {
        // Highest quality: verified/validated evidence
        EvidenceKind::Verified { .. } => 1.5,

        // High quality: experimental evidence
        EvidenceKind::Experiment { .. } => 1.3,

        // Good quality: publications
        EvidenceKind::Publication { .. } => 1.2,

        // Medium quality: human assertion
        EvidenceKind::HumanAssertion { .. } => 1.1,

        // Standard quality: computational
        EvidenceKind::Computation { .. } => 1.0,

        // Expert opinion: slightly below computational
        EvidenceKind::ExpertOpinion { .. } => 0.95,

        // Dataset: standard quality
        EvidenceKind::Dataset { .. } => 0.9,
    }
}

/// Merge evidence from multiple statuses, deduplicating by reference
fn merge_evidence(statuses: &[EpistemicStatus]) -> Vec<Evidence> {
    use std::collections::HashMap;

    let mut evidence_map: HashMap<String, Evidence> = HashMap::new();

    for status in statuses {
        for evidence in &status.evidence {
            evidence_map
                .entry(evidence.reference.clone())
                .and_modify(|existing| {
                    // Keep the higher strength version
                    if evidence.strength.value() > existing.strength.value() {
                        *existing = evidence.clone();
                    }
                })
                .or_insert_with(|| evidence.clone());
        }
    }

    evidence_map.into_values().collect()
}

/// Combine revisability constraints from multiple statuses
fn combine_revisability(statuses: &[EpistemicStatus]) -> Revisability {
    // Check if any status is non-revisable (strongest constraint)
    for status in statuses {
        if matches!(status.revisability, Revisability::NonRevisable) {
            return Revisability::NonRevisable;
        }
    }

    // Collect all revision conditions
    let all_conditions: Vec<String> = statuses
        .iter()
        .filter_map(|s| {
            if let Revisability::Revisable { conditions } = &s.revisability {
                Some(conditions.clone())
            } else {
                None
            }
        })
        .flatten()
        .collect();

    if all_conditions.is_empty() {
        Revisability::Revisable {
            conditions: vec!["new_evidence".to_string()],
        }
    } else {
        // Deduplicate conditions
        let mut unique_conditions: Vec<String> = all_conditions;
        unique_conditions.sort();
        unique_conditions.dedup();

        Revisability::Revisable {
            conditions: unique_conditions,
        }
    }
}

// ============================================================================
// Compile-Time Confidence Refinements
// ============================================================================

/// Compile-time confidence bound
///
/// Represents a statically-known bound on confidence values.
/// Used for compile-time verification of epistemic constraints.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConfidenceBound {
    /// Exact known value (from literal)
    Exact(f64),
    /// Lower bound only (≥ value)
    AtLeast(f64),
    /// Upper bound only (≤ value)
    AtMost(f64),
    /// Range [min, max]
    Range { min: f64, max: f64 },
    /// Unknown at compile time
    Unknown,
}

impl ConfidenceBound {
    /// Create from a literal confidence value
    pub fn from_literal(value: f64) -> Self {
        ConfidenceBound::Exact(value.clamp(0.0, 1.0))
    }

    /// Create a lower bound
    pub fn at_least(min: f64) -> Self {
        ConfidenceBound::AtLeast(min.clamp(0.0, 1.0))
    }

    /// Create an upper bound
    pub fn at_most(max: f64) -> Self {
        ConfidenceBound::AtMost(max.clamp(0.0, 1.0))
    }

    /// Create a range bound
    pub fn range(min: f64, max: f64) -> Self {
        let min = min.clamp(0.0, 1.0);
        let max = max.clamp(0.0, 1.0);
        if min > max {
            ConfidenceBound::Range { min: max, max: min }
        } else {
            ConfidenceBound::Range { min, max }
        }
    }

    /// Check if this bound definitely satisfies a minimum requirement
    pub fn satisfies_minimum(&self, required: f64) -> ConfidenceCheck {
        match self {
            ConfidenceBound::Exact(v) => {
                if *v >= required {
                    ConfidenceCheck::Satisfied
                } else {
                    ConfidenceCheck::Violated {
                        actual: *v,
                        required,
                    }
                }
            }
            ConfidenceBound::AtLeast(min) => {
                if *min >= required {
                    ConfidenceCheck::Satisfied
                } else {
                    ConfidenceCheck::MaybeSatisfied {
                        known_min: *min,
                        required,
                    }
                }
            }
            ConfidenceBound::AtMost(max) => {
                if *max < required {
                    ConfidenceCheck::Violated {
                        actual: *max,
                        required,
                    }
                } else {
                    ConfidenceCheck::MaybeSatisfied {
                        known_min: 0.0,
                        required,
                    }
                }
            }
            ConfidenceBound::Range { min, max } => {
                if *min >= required {
                    ConfidenceCheck::Satisfied
                } else if *max < required {
                    ConfidenceCheck::Violated {
                        actual: *max,
                        required,
                    }
                } else {
                    ConfidenceCheck::MaybeSatisfied {
                        known_min: *min,
                        required,
                    }
                }
            }
            ConfidenceBound::Unknown => ConfidenceCheck::Unknown,
        }
    }

    /// Combine two bounds (e.g., for binary operations)
    ///
    /// Returns the bound after combining two epistemic values.
    /// Uses conservative approximation for unknown cases.
    pub fn combine_binary(&self, other: &ConfidenceBound, op: ConfidenceCombineOp) -> Self {
        match op {
            ConfidenceCombineOp::Min => self.combine_min(other),
            ConfidenceCombineOp::Max => self.combine_max(other),
            ConfidenceCombineOp::Product => self.combine_product(other),
            ConfidenceCombineOp::Average => self.combine_average(other),
        }
    }

    fn combine_min(&self, other: &ConfidenceBound) -> Self {
        match (self, other) {
            (ConfidenceBound::Exact(a), ConfidenceBound::Exact(b)) => {
                ConfidenceBound::Exact(a.min(*b))
            }
            (ConfidenceBound::Exact(a), ConfidenceBound::AtLeast(b))
            | (ConfidenceBound::AtLeast(b), ConfidenceBound::Exact(a)) => {
                if *a <= *b {
                    ConfidenceBound::Exact(*a)
                } else {
                    ConfidenceBound::AtLeast(*b)
                }
            }
            (ConfidenceBound::AtLeast(a), ConfidenceBound::AtLeast(b)) => {
                // min(≥a, ≥b) = ≥min(a,b) is NOT correct
                // We can only say the result is ≥0
                ConfidenceBound::AtLeast(0.0)
            }
            (
                ConfidenceBound::Range { min: min1, .. },
                ConfidenceBound::Range { min: min2, .. },
            ) => {
                // Conservative: result could be as low as 0
                let lower: f64 = 0.0;
                ConfidenceBound::AtLeast(lower.max(min1.min(*min2) - 0.1))
            }
            _ => ConfidenceBound::Unknown,
        }
    }

    fn combine_max(&self, other: &ConfidenceBound) -> Self {
        match (self, other) {
            (ConfidenceBound::Exact(a), ConfidenceBound::Exact(b)) => {
                ConfidenceBound::Exact(a.max(*b))
            }
            (ConfidenceBound::Exact(a), ConfidenceBound::AtLeast(b))
            | (ConfidenceBound::AtLeast(b), ConfidenceBound::Exact(a)) => {
                ConfidenceBound::AtLeast(a.max(*b))
            }
            (ConfidenceBound::AtLeast(a), ConfidenceBound::AtLeast(b)) => {
                ConfidenceBound::AtLeast(a.max(*b))
            }
            _ => ConfidenceBound::Unknown,
        }
    }

    fn combine_product(&self, other: &ConfidenceBound) -> Self {
        match (self, other) {
            (ConfidenceBound::Exact(a), ConfidenceBound::Exact(b)) => {
                ConfidenceBound::Exact(*a * *b)
            }
            (ConfidenceBound::Exact(a), ConfidenceBound::AtLeast(b))
            | (ConfidenceBound::AtLeast(b), ConfidenceBound::Exact(a)) => {
                ConfidenceBound::AtLeast(*a * *b)
            }
            (ConfidenceBound::AtLeast(a), ConfidenceBound::AtLeast(b)) => {
                ConfidenceBound::AtLeast(*a * *b)
            }
            (
                ConfidenceBound::Range {
                    min: min1,
                    max: max1,
                },
                ConfidenceBound::Range {
                    min: min2,
                    max: max2,
                },
            ) => ConfidenceBound::Range {
                min: *min1 * *min2,
                max: *max1 * *max2,
            },
            _ => ConfidenceBound::Unknown,
        }
    }

    fn combine_average(&self, other: &ConfidenceBound) -> Self {
        match (self, other) {
            (ConfidenceBound::Exact(a), ConfidenceBound::Exact(b)) => {
                ConfidenceBound::Exact((*a + *b) / 2.0)
            }
            (ConfidenceBound::AtLeast(a), ConfidenceBound::AtLeast(b)) => {
                ConfidenceBound::AtLeast((*a + *b) / 2.0)
            }
            (
                ConfidenceBound::Range {
                    min: min1,
                    max: max1,
                },
                ConfidenceBound::Range {
                    min: min2,
                    max: max2,
                },
            ) => ConfidenceBound::Range {
                min: (*min1 + *min2) / 2.0,
                max: (*max1 + *max2) / 2.0,
            },
            _ => ConfidenceBound::Unknown,
        }
    }

    /// Narrow the bound with additional information
    pub fn narrow(&self, other: &ConfidenceBound) -> Self {
        match (self, other) {
            (ConfidenceBound::Unknown, other) => *other,
            (this, ConfidenceBound::Unknown) => *this,
            (ConfidenceBound::Exact(a), ConfidenceBound::Exact(b)) if (a - b).abs() < 1e-10 => {
                ConfidenceBound::Exact(*a)
            }
            (ConfidenceBound::AtLeast(a), ConfidenceBound::AtLeast(b)) => {
                ConfidenceBound::AtLeast(a.max(*b))
            }
            (ConfidenceBound::AtMost(a), ConfidenceBound::AtMost(b)) => {
                ConfidenceBound::AtMost(a.min(*b))
            }
            (ConfidenceBound::AtLeast(min), ConfidenceBound::AtMost(max))
            | (ConfidenceBound::AtMost(max), ConfidenceBound::AtLeast(min)) => {
                ConfidenceBound::Range {
                    min: *min,
                    max: *max,
                }
            }
            (
                ConfidenceBound::Range { min: m1, max: x1 },
                ConfidenceBound::Range { min: m2, max: x2 },
            ) => ConfidenceBound::Range {
                min: m1.max(*m2),
                max: x1.min(*x2),
            },
            (ConfidenceBound::Exact(v), ConfidenceBound::AtLeast(min))
            | (ConfidenceBound::AtLeast(min), ConfidenceBound::Exact(v)) => {
                if *v >= *min {
                    ConfidenceBound::Exact(*v)
                } else {
                    // Contradiction - use the more informative bound
                    ConfidenceBound::AtLeast(*min)
                }
            }
            // Range narrowed with AtLeast: raise the minimum
            (ConfidenceBound::Range { min: m, max: x }, ConfidenceBound::AtLeast(new_min))
            | (ConfidenceBound::AtLeast(new_min), ConfidenceBound::Range { min: m, max: x }) => {
                let new_m = m.max(*new_min);
                if new_m >= *x {
                    // Range collapsed to a point or contradiction
                    ConfidenceBound::AtLeast(new_m)
                } else {
                    ConfidenceBound::Range {
                        min: new_m,
                        max: *x,
                    }
                }
            }
            // Range narrowed with AtMost: lower the maximum
            (ConfidenceBound::Range { min: m, max: x }, ConfidenceBound::AtMost(new_max))
            | (ConfidenceBound::AtMost(new_max), ConfidenceBound::Range { min: m, max: x }) => {
                let new_x = x.min(*new_max);
                if *m >= new_x {
                    // Range collapsed
                    ConfidenceBound::AtMost(new_x)
                } else {
                    ConfidenceBound::Range {
                        min: *m,
                        max: new_x,
                    }
                }
            }
            _ => *self,
        }
    }

    /// Get the known minimum, if any
    pub fn known_min(&self) -> Option<f64> {
        match self {
            ConfidenceBound::Exact(v) => Some(*v),
            ConfidenceBound::AtLeast(v) => Some(*v),
            ConfidenceBound::Range { min, .. } => Some(*min),
            _ => None,
        }
    }

    /// Get the known maximum, if any
    pub fn known_max(&self) -> Option<f64> {
        match self {
            ConfidenceBound::Exact(v) => Some(*v),
            ConfidenceBound::AtMost(v) => Some(*v),
            ConfidenceBound::Range { max, .. } => Some(*max),
            _ => None,
        }
    }
}

/// Operation for combining confidence bounds
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfidenceCombineOp {
    /// Take minimum (conservative)
    Min,
    /// Take maximum (optimistic)
    Max,
    /// Multiply (independent events)
    Product,
    /// Average (equal weighting)
    Average,
}

/// Result of compile-time confidence checking
#[derive(Debug, Clone, PartialEq)]
pub enum ConfidenceCheck {
    /// Constraint is definitely satisfied
    Satisfied,
    /// Constraint is definitely violated
    Violated { actual: f64, required: f64 },
    /// Constraint may or may not be satisfied (need runtime check)
    MaybeSatisfied { known_min: f64, required: f64 },
    /// Cannot determine at compile time
    Unknown,
}

impl ConfidenceCheck {
    /// Returns true if definitely satisfied
    pub fn is_satisfied(&self) -> bool {
        matches!(self, ConfidenceCheck::Satisfied)
    }

    /// Returns true if definitely violated
    pub fn is_violated(&self) -> bool {
        matches!(self, ConfidenceCheck::Violated { .. })
    }

    /// Returns true if a runtime check is needed
    pub fn needs_runtime_check(&self) -> bool {
        matches!(
            self,
            ConfidenceCheck::MaybeSatisfied { .. } | ConfidenceCheck::Unknown
        )
    }
}

/// Compile-time epistemic type with confidence bounds
#[derive(Debug, Clone)]
pub struct EpistemicRefinement {
    /// Compile-time confidence bound
    pub confidence_bound: ConfidenceBound,
    /// Required minimum confidence for this context
    pub required_confidence: Option<f64>,
    /// Source constraint
    pub required_source: Option<Source>,
    /// Whether this value has been validated
    pub validated: bool,
}

impl EpistemicRefinement {
    /// Create a new refinement from a confidence literal
    pub fn from_literal(confidence: f64) -> Self {
        Self {
            confidence_bound: ConfidenceBound::from_literal(confidence),
            required_confidence: None,
            required_source: None,
            validated: false,
        }
    }

    /// Create a refinement with a minimum confidence requirement
    pub fn with_minimum(min_confidence: f64) -> Self {
        Self {
            confidence_bound: ConfidenceBound::at_least(min_confidence),
            required_confidence: Some(min_confidence),
            required_source: None,
            validated: false,
        }
    }

    /// Create an unknown refinement
    pub fn unknown() -> Self {
        Self {
            confidence_bound: ConfidenceBound::Unknown,
            required_confidence: None,
            required_source: None,
            validated: false,
        }
    }

    /// Check if this refinement satisfies a required minimum
    pub fn check_minimum(&self, required: f64) -> ConfidenceCheck {
        self.confidence_bound.satisfies_minimum(required)
    }

    /// Combine with another refinement using an operation
    pub fn combine(&self, other: &EpistemicRefinement, op: ConfidenceCombineOp) -> Self {
        Self {
            confidence_bound: self
                .confidence_bound
                .combine_binary(&other.confidence_bound, op),
            required_confidence: match (self.required_confidence, other.required_confidence) {
                (Some(a), Some(b)) => Some(a.max(b)),
                (Some(a), None) | (None, Some(a)) => Some(a),
                (None, None) => None,
            },
            required_source: None, // Sources don't combine simply
            validated: self.validated && other.validated,
        }
    }

    /// Mark as validated (passed a confidence check)
    pub fn validated(mut self) -> Self {
        self.validated = true;
        self
    }

    /// Narrow with additional type information
    pub fn narrow(&self, other: &EpistemicRefinement) -> Self {
        Self {
            confidence_bound: self.confidence_bound.narrow(&other.confidence_bound),
            required_confidence: match (self.required_confidence, other.required_confidence) {
                (Some(a), Some(b)) => Some(a.max(b)),
                (Some(a), None) | (None, Some(a)) => Some(a),
                (None, None) => None,
            },
            required_source: self
                .required_source
                .clone()
                .or(other.required_source.clone()),
            validated: self.validated || other.validated,
        }
    }
}

impl Default for EpistemicRefinement {
    fn default() -> Self {
        Self::unknown()
    }
}

/// Context for compile-time epistemic analysis
pub struct EpistemicRefinementContext {
    /// Variable bindings with their refinements
    bindings: HashMap<String, EpistemicRefinement>,
    /// Path conditions (from if statements, etc.)
    path_conditions: Vec<ConfidencePathCondition>,
}

/// A path condition that affects confidence bounds
#[derive(Debug, Clone)]
pub enum ConfidencePathCondition {
    /// Variable has confidence >= threshold
    MinConfidence { var: String, threshold: f64 },
    /// Variable has confidence <= threshold
    MaxConfidence { var: String, threshold: f64 },
    /// Variable has been validated
    Validated { var: String },
}

impl EpistemicRefinementContext {
    pub fn new() -> Self {
        Self {
            bindings: HashMap::new(),
            path_conditions: Vec::new(),
        }
    }

    /// Bind a variable to a refinement
    pub fn bind(&mut self, name: String, refinement: EpistemicRefinement) {
        self.bindings.insert(name, refinement);
    }

    /// Look up a variable's refinement
    pub fn lookup(&self, name: &str) -> Option<&EpistemicRefinement> {
        self.bindings.get(name)
    }

    /// Add a path condition
    pub fn assume(&mut self, condition: ConfidencePathCondition) {
        // Apply the condition to relevant bindings
        match &condition {
            ConfidencePathCondition::MinConfidence { var, threshold } => {
                if let Some(refinement) = self.bindings.get_mut(var) {
                    let new_bound = refinement
                        .confidence_bound
                        .narrow(&ConfidenceBound::at_least(*threshold));
                    refinement.confidence_bound = new_bound;
                }
            }
            ConfidencePathCondition::MaxConfidence { var, threshold } => {
                if let Some(refinement) = self.bindings.get_mut(var) {
                    let new_bound = refinement
                        .confidence_bound
                        .narrow(&ConfidenceBound::at_most(*threshold));
                    refinement.confidence_bound = new_bound;
                }
            }
            ConfidencePathCondition::Validated { var } => {
                if let Some(refinement) = self.bindings.get_mut(var) {
                    refinement.validated = true;
                }
            }
        }
        self.path_conditions.push(condition);
    }

    /// Create a child context (for nested scopes)
    pub fn child(&self) -> Self {
        Self {
            bindings: self.bindings.clone(),
            path_conditions: self.path_conditions.clone(),
        }
    }

    /// Check if a confidence requirement can be satisfied
    pub fn check_requirement(&self, var: &str, required: f64) -> ConfidenceCheck {
        match self.lookup(var) {
            Some(refinement) => refinement.check_minimum(required),
            None => ConfidenceCheck::Unknown,
        }
    }
}

impl Default for EpistemicRefinementContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::epistemic::EvidenceKind;

    #[test]
    fn test_epistemic_checker_new() {
        let checker = EpistemicChecker::new();
        assert!(checker.bindings.is_empty());
    }

    #[test]
    fn test_type_from_curie() {
        let checker = EpistemicChecker::new();
        let ty = checker.type_from_curie("CHEBI:15365").unwrap();
        assert!(ty.min_confidence.value() == 0.0);
    }

    #[test]
    fn test_type_with_confidence() {
        let checker = EpistemicChecker::new();
        let ty = checker.type_with_confidence("GO:0008150", 0.95).unwrap();
        assert!(ty.min_confidence.value() >= 0.95);
    }

    #[test]
    fn test_constraint_check_confidence() {
        let mut checker = EpistemicChecker::new();

        let value_status = EpistemicStatus {
            confidence: Confidence::new(0.9),
            ..Default::default()
        };

        let high_requirement = OntologicalType {
            binding: ParsedTermRef::parse("TEST:001").unwrap().to_binding(),
            min_confidence: Confidence::new(0.95),
            required_evidence: vec![],
            temporal_constraint: None,
            provenance_constraint: None,
        };

        let result = checker.check_constraint(&value_status, &high_requirement);
        assert!(matches!(result, ConstraintResult::Violated(_)));

        let low_requirement = OntologicalType {
            binding: ParsedTermRef::parse("TEST:001").unwrap().to_binding(),
            min_confidence: Confidence::new(0.8),
            required_evidence: vec![],
            temporal_constraint: None,
            provenance_constraint: None,
        };

        let result = checker.check_constraint(&value_status, &low_requirement);
        assert!(matches!(result, ConstraintResult::Satisfied));
    }

    #[test]
    fn test_epistemic_heterogeneity() {
        let a = EpistemicStatus {
            confidence: Confidence::new(0.9),
            ..Default::default()
        };
        let b = EpistemicStatus {
            confidence: Confidence::new(0.8),
            ..Default::default()
        };

        let het = epistemic_heterogeneity(&a, &b);
        assert!(het >= 0.0 && het <= 1.0);
        assert!(het > 0.0); // Different confidence should produce non-zero heterogeneity
    }

    #[test]
    fn test_bayesian_combination() {
        let statuses = vec![
            EpistemicStatus {
                confidence: Confidence::new(0.9),
                ..Default::default()
            },
            EpistemicStatus {
                confidence: Confidence::new(0.8),
                ..Default::default()
            },
        ];

        let combined = combine_epistemic_bayesian(&statuses);
        // Combined confidence should be between the two
        assert!(combined.confidence.value() > 0.8 && combined.confidence.value() < 0.9);
    }

    #[test]
    fn test_bind_and_lookup() {
        let mut checker = EpistemicChecker::new();
        let ty = checker.type_from_curie("PATO:0000001").unwrap();
        checker.bind("quality".into(), ty);

        let looked_up = checker.lookup("quality");
        assert!(looked_up.is_some());
    }

    // ===== New tests for sources_compatible with subsumption =====

    #[test]
    fn test_sources_compatible_exact_match() {
        let mut checker = EpistemicChecker::new();

        let source1 = Source::OntologyAssertion {
            ontology: "CHEBI".to_string(),
            term: "15365".to_string(),
        };
        let source2 = Source::OntologyAssertion {
            ontology: "CHEBI".to_string(),
            term: "15365".to_string(),
        };

        assert!(checker.sources_compatible(&source1, &source2));
    }

    #[test]
    fn test_sources_compatible_different_ontologies() {
        let mut checker = EpistemicChecker::new();

        let source1 = Source::OntologyAssertion {
            ontology: "CHEBI".to_string(),
            term: "15365".to_string(),
        };
        let source2 = Source::OntologyAssertion {
            ontology: "GO".to_string(),
            term: "15365".to_string(),
        };

        // Different ontologies should not be compatible
        assert!(!checker.sources_compatible(&source1, &source2));
    }

    #[test]
    fn test_sources_compatible_axiom_satisfies_all() {
        let mut checker = EpistemicChecker::new();

        let axiom = Source::Axiom;
        let ontology = Source::OntologyAssertion {
            ontology: "CHEBI".to_string(),
            term: "15365".to_string(),
        };

        // Axiom (strongest source) should satisfy any requirement
        assert!(checker.sources_compatible(&axiom, &ontology));
    }

    #[test]
    fn test_sources_compatible_derivation() {
        let mut checker = EpistemicChecker::new();

        let d1 = Source::Derivation("bayesian_combination".to_string());
        let d2 = Source::Derivation("bayesian_combination".to_string());
        let d3 = Source::Derivation("weighted_bayesian".to_string());

        assert!(checker.sources_compatible(&d1, &d2));
        assert!(!checker.sources_compatible(&d1, &d3));
    }

    #[test]
    fn test_sources_compatible_external_uri() {
        let mut checker = EpistemicChecker::new();

        let e1 = Source::External {
            uri: "https://pubmed.ncbi.nlm.nih.gov/12345678".to_string(),
            accessed: Some("2024-01-01".to_string()),
        };
        let e2 = Source::External {
            uri: "https://pubmed.ncbi.nlm.nih.gov".to_string(),
            accessed: None,
        };
        let e3 = Source::External {
            uri: "https://doi.org/10.1234".to_string(),
            accessed: None,
        };

        // More specific URI satisfies less specific
        assert!(checker.sources_compatible(&e1, &e2));
        // Different domains should not be compatible
        assert!(!checker.sources_compatible(&e1, &e3));
    }

    #[test]
    fn test_sources_compatible_model_prediction() {
        let mut checker = EpistemicChecker::new();

        let m1 = Source::ModelPrediction {
            model: "PBPK_adult".to_string(),
            version: Some("2.1".to_string()),
        };
        let m2 = Source::ModelPrediction {
            model: "PBPK_adult".to_string(),
            version: None,
        };
        let m3 = Source::ModelPrediction {
            model: "PBPK_pediatric".to_string(),
            version: None,
        };

        // Same model, required doesn't specify version
        assert!(checker.sources_compatible(&m1, &m2));
        // Different models
        assert!(!checker.sources_compatible(&m1, &m3));
    }

    #[test]
    fn test_sources_compatible_transformation_chain() {
        let mut checker = EpistemicChecker::new();

        let original = Source::OntologyAssertion {
            ontology: "CHEBI".to_string(),
            term: "15365".to_string(),
        };
        let transformed = Source::Transformation {
            original: Box::new(original.clone()),
            via: "unit_conversion".to_string(),
        };

        // Transformation should satisfy original source requirement
        assert!(checker.sources_compatible(&transformed, &original));
    }

    // ===== Tests for weighted Bayesian combination =====

    #[test]
    fn test_weighted_bayesian_empty() {
        let result = combine_epistemic_weighted_bayesian(&[]);
        // Empty input returns default EpistemicStatus (confidence = 1.0)
        assert_eq!(
            result.confidence.value(),
            EpistemicStatus::default().confidence.value()
        );
    }

    #[test]
    fn test_weighted_bayesian_single() {
        let status = EpistemicStatus {
            confidence: Confidence::new(0.85),
            ..Default::default()
        };

        let result = combine_epistemic_weighted_bayesian(&[status.clone()]);
        assert_eq!(result.confidence.value(), status.confidence.value());
    }

    #[test]
    fn test_weighted_bayesian_higher_weight_for_stronger_evidence() {
        // Status with strong experimental evidence
        let strong = EpistemicStatus {
            confidence: Confidence::new(0.9),
            evidence: vec![Evidence::new(
                EvidenceKind::Experiment {
                    protocol: "RCT".to_string(),
                },
                "doi:10.1234/strong",
                0.95,
            )],
            ..Default::default()
        };

        // Status with weaker computational evidence
        let weak = EpistemicStatus {
            confidence: Confidence::new(0.7),
            evidence: vec![Evidence::new(
                EvidenceKind::Computation {
                    code_ref: "simulation".to_string(),
                },
                "doi:10.1234/weak",
                0.5,
            )],
            ..Default::default()
        };

        let combined = combine_epistemic_weighted_bayesian(&[strong.clone(), weak.clone()]);

        // Combined should be closer to the strong evidence value
        // because it has higher evidence strength and experimental evidence
        assert!(combined.confidence.value() > 0.75);
    }

    #[test]
    fn test_weighted_bayesian_evidence_deduplication() {
        let evidence1 = Evidence::new(
            EvidenceKind::Publication {
                doi: Some("10.1234/paper".to_string()),
            },
            "doi:10.1234/paper",
            0.7,
        );

        let evidence2 = Evidence::new(
            EvidenceKind::Publication {
                doi: Some("10.1234/paper".to_string()),
            },
            "doi:10.1234/paper", // Same reference
            0.9,                 // Higher strength
        );

        let s1 = EpistemicStatus {
            confidence: Confidence::new(0.8),
            evidence: vec![evidence1],
            ..Default::default()
        };

        let s2 = EpistemicStatus {
            confidence: Confidence::new(0.85),
            evidence: vec![evidence2],
            ..Default::default()
        };

        let combined = combine_epistemic_weighted_bayesian(&[s1, s2]);

        // Should deduplicate, keeping higher strength
        assert_eq!(combined.evidence.len(), 1);
        assert!(combined.evidence[0].strength.value() >= 0.9);
    }

    #[test]
    fn test_weighted_bayesian_revisability_non_revisable() {
        let s1 = EpistemicStatus {
            confidence: Confidence::new(0.9),
            revisability: Revisability::NonRevisable,
            ..Default::default()
        };

        let s2 = EpistemicStatus {
            confidence: Confidence::new(0.8),
            revisability: Revisability::Revisable {
                conditions: vec!["new_data".to_string()],
            },
            ..Default::default()
        };

        let combined = combine_epistemic_weighted_bayesian(&[s1, s2]);

        // NonRevisable should propagate (strongest constraint wins)
        assert!(matches!(combined.revisability, Revisability::NonRevisable));
    }

    #[test]
    fn test_evidence_kind_quality_ordering() {
        // Verified should be highest
        assert!(
            evidence_kind_quality(&EvidenceKind::Verified {
                verifier: "peer_review".to_string()
            }) > evidence_kind_quality(&EvidenceKind::Experiment {
                protocol: "RCT".to_string()
            })
        );

        // Experiment > Publication
        assert!(
            evidence_kind_quality(&EvidenceKind::Experiment {
                protocol: "RCT".to_string()
            }) > evidence_kind_quality(&EvidenceKind::Publication {
                doi: Some("10.1234".to_string())
            })
        );

        // Publication > Computation
        assert!(
            evidence_kind_quality(&EvidenceKind::Publication {
                doi: Some("10.1234".to_string())
            }) > evidence_kind_quality(&EvidenceKind::Computation {
                code_ref: "simulation".to_string()
            })
        );
    }

    #[test]
    fn test_calculate_evidence_weight_no_evidence() {
        let status = EpistemicStatus {
            confidence: Confidence::new(0.8),
            evidence: vec![],
            ..Default::default()
        };

        let weight = calculate_evidence_weight(&status);
        // Should be penalized for no evidence
        assert!(weight < 0.8 * 0.6); // Less than half of confidence
    }

    #[test]
    fn test_calculate_evidence_weight_with_evidence() {
        let status = EpistemicStatus {
            confidence: Confidence::new(0.8),
            evidence: vec![
                Evidence::new(
                    EvidenceKind::Experiment {
                        protocol: "RCT".to_string(),
                    },
                    "ref1",
                    0.9,
                ),
                Evidence::new(
                    EvidenceKind::Publication {
                        doi: Some("10.1234".to_string()),
                    },
                    "ref2",
                    0.85,
                ),
            ],
            ..Default::default()
        };

        let weight = calculate_evidence_weight(&status);

        // Should be boosted by multiple high-quality evidence
        assert!(weight > 0.8); // Higher than base confidence
    }

    // ===== Tests for compile-time confidence refinements =====

    #[test]
    fn test_confidence_bound_exact() {
        let bound = ConfidenceBound::from_literal(0.95);
        assert_eq!(bound, ConfidenceBound::Exact(0.95));

        // Satisfies 0.9 requirement
        assert!(bound.satisfies_minimum(0.9).is_satisfied());
        // Does not satisfy 0.99 requirement
        assert!(bound.satisfies_minimum(0.99).is_violated());
    }

    #[test]
    fn test_confidence_bound_at_least() {
        let bound = ConfidenceBound::at_least(0.8);

        // Definitely satisfies 0.7
        assert!(bound.satisfies_minimum(0.7).is_satisfied());
        // May or may not satisfy 0.9
        assert!(bound.satisfies_minimum(0.9).needs_runtime_check());
    }

    #[test]
    fn test_confidence_bound_range() {
        let bound = ConfidenceBound::range(0.7, 0.9);

        // Definitely satisfies 0.6
        assert!(bound.satisfies_minimum(0.6).is_satisfied());
        // Definitely violates 0.95
        assert!(bound.satisfies_minimum(0.95).is_violated());
        // May satisfy 0.8
        assert!(bound.satisfies_minimum(0.8).needs_runtime_check());
    }

    #[test]
    fn test_confidence_bound_combine_product() {
        let a = ConfidenceBound::Exact(0.9);
        let b = ConfidenceBound::Exact(0.8);

        let combined = a.combine_binary(&b, ConfidenceCombineOp::Product);
        match combined {
            ConfidenceBound::Exact(v) => assert!((v - 0.72).abs() < 0.001),
            _ => panic!("Expected exact bound"),
        }
    }

    #[test]
    fn test_confidence_bound_combine_min() {
        let a = ConfidenceBound::Exact(0.9);
        let b = ConfidenceBound::Exact(0.7);

        let combined = a.combine_binary(&b, ConfidenceCombineOp::Min);
        assert_eq!(combined, ConfidenceBound::Exact(0.7));
    }

    #[test]
    fn test_confidence_bound_narrow() {
        let unknown = ConfidenceBound::Unknown;
        let at_least = ConfidenceBound::at_least(0.8);

        // Unknown narrowed with at_least becomes at_least
        let narrowed = unknown.narrow(&at_least);
        assert_eq!(narrowed, ConfidenceBound::AtLeast(0.8));

        // Two at_least bounds combine to the higher minimum
        let at_least_low = ConfidenceBound::at_least(0.6);
        let at_least_high = ConfidenceBound::at_least(0.9);
        let narrowed = at_least_low.narrow(&at_least_high);
        assert_eq!(narrowed, ConfidenceBound::AtLeast(0.9));
    }

    #[test]
    fn test_epistemic_refinement_from_literal() {
        let refinement = EpistemicRefinement::from_literal(0.95);

        assert!(refinement.check_minimum(0.9).is_satisfied());
        assert!(refinement.check_minimum(0.99).is_violated());
    }

    #[test]
    fn test_epistemic_refinement_combine() {
        let r1 = EpistemicRefinement::from_literal(0.9);
        let r2 = EpistemicRefinement::from_literal(0.8);

        // Product combination
        let combined = r1.combine(&r2, ConfidenceCombineOp::Product);
        match combined.confidence_bound {
            ConfidenceBound::Exact(v) => assert!((v - 0.72).abs() < 0.001),
            _ => panic!("Expected exact bound"),
        }
    }

    #[test]
    fn test_epistemic_refinement_context() {
        let mut ctx = EpistemicRefinementContext::new();

        // Bind a variable with unknown confidence
        ctx.bind(
            "drug_confidence".to_string(),
            EpistemicRefinement::unknown(),
        );

        // Initially unknown
        assert!(ctx.check_requirement("drug_confidence", 0.9) == ConfidenceCheck::Unknown);

        // After assuming it's at least 0.95
        ctx.assume(ConfidencePathCondition::MinConfidence {
            var: "drug_confidence".to_string(),
            threshold: 0.95,
        });

        // Now it definitely satisfies 0.9
        assert!(ctx.check_requirement("drug_confidence", 0.9).is_satisfied());
    }

    #[test]
    fn test_epistemic_refinement_context_path_narrowing() {
        let mut ctx = EpistemicRefinementContext::new();

        // Start with a range
        ctx.bind(
            "measurement".to_string(),
            EpistemicRefinement {
                confidence_bound: ConfidenceBound::range(0.5, 0.95),
                required_confidence: None,
                required_source: None,
                validated: false,
            },
        );

        // After an if statement that checks confidence >= 0.8
        ctx.assume(ConfidencePathCondition::MinConfidence {
            var: "measurement".to_string(),
            threshold: 0.8,
        });

        // Inside that branch, we know it's at least 0.8
        let check = ctx.check_requirement("measurement", 0.75);
        assert!(check.is_satisfied());
    }

    #[test]
    fn test_confidence_check_classification() {
        let satisfied = ConfidenceCheck::Satisfied;
        let violated = ConfidenceCheck::Violated {
            actual: 0.5,
            required: 0.9,
        };
        let maybe = ConfidenceCheck::MaybeSatisfied {
            known_min: 0.7,
            required: 0.9,
        };
        let unknown = ConfidenceCheck::Unknown;

        assert!(satisfied.is_satisfied());
        assert!(!satisfied.needs_runtime_check());

        assert!(violated.is_violated());
        assert!(!violated.needs_runtime_check());

        assert!(!maybe.is_satisfied());
        assert!(!maybe.is_violated());
        assert!(maybe.needs_runtime_check());

        assert!(unknown.needs_runtime_check());
    }

    #[test]
    fn test_confidence_bound_clamps_values() {
        // Values outside [0, 1] should be clamped
        let too_high = ConfidenceBound::from_literal(1.5);
        assert_eq!(too_high, ConfidenceBound::Exact(1.0));

        let too_low = ConfidenceBound::from_literal(-0.5);
        assert_eq!(too_low, ConfidenceBound::Exact(0.0));

        let range = ConfidenceBound::range(-0.1, 1.5);
        match range {
            ConfidenceBound::Range { min, max } => {
                assert_eq!(min, 0.0);
                assert_eq!(max, 1.0);
            }
            _ => panic!("Expected range"),
        }
    }

    // ===== Tests for temporal constraint checking =====

    #[test]
    fn test_temporal_index_from_iso() {
        let ti = TemporalIndex::from_iso("2024-01-15");
        assert!(ti.epoch_secs.is_some());

        // 2024-01-15 00:00:00 UTC should be around 1705276800
        let epoch = ti.epoch_secs.unwrap();
        assert!(epoch > 1705000000 && epoch < 1706000000);
    }

    #[test]
    fn test_temporal_index_from_iso_with_time() {
        let ti = TemporalIndex::from_iso("2024-01-15T12:30:45Z");
        assert!(ti.epoch_secs.is_some());

        // Should be ~12.5 hours after midnight
        let base = TemporalIndex::from_iso("2024-01-15");
        let diff = ti.epoch_secs.unwrap() - base.epoch_secs.unwrap();
        assert!(diff >= 12 * 3600 && diff <= 13 * 3600);
    }

    #[test]
    fn test_temporal_index_comparison() {
        let earlier = TemporalIndex::from_iso("2024-01-01");
        let later = TemporalIndex::from_iso("2024-06-15");

        assert_eq!(earlier.is_before(&later), Some(true));
        assert_eq!(later.is_after(&earlier), Some(true));
        assert_eq!(earlier.is_after(&later), Some(false));
    }

    #[test]
    fn test_temporal_index_age() {
        // Create a timestamp from a known past date
        let old = TemporalIndex::from_iso("2020-01-01");
        let age_days = old.age_days();

        // Should be at least 4 years (1460+ days) old
        assert!(age_days.is_some());
        assert!(age_days.unwrap() > 1460);
    }

    #[test]
    fn test_parse_iso_to_epoch_unix_epoch() {
        // Unix epoch should be 0
        let epoch = parse_iso_to_epoch("1970-01-01");
        assert_eq!(epoch, Some(0));
    }

    #[test]
    fn test_parse_iso_to_epoch_known_date() {
        // 2000-01-01 00:00:00 UTC = 946684800
        let epoch = parse_iso_to_epoch("2000-01-01");
        assert_eq!(epoch, Some(946684800));
    }

    #[test]
    fn test_temporal_constraint_current_satisfied() {
        let checker = EpistemicChecker::new();

        // Create status with recent evidence - use a date close to "now"
        // We use a date that's within 30 days by computing from current time
        let now = TemporalIndex::now();
        let recent_epoch = now.epoch_secs.unwrap() - (10 * 86400); // 10 days ago
        let recent_date = epoch_to_iso_date(recent_epoch);

        let status = EpistemicStatus {
            confidence: Confidence::new(0.9),
            evidence: vec![Evidence::new(
                EvidenceKind::Publication {
                    doi: Some("10.1234/paper".to_string()),
                },
                format!("published:{}", recent_date),
                0.9,
            )],
            source: Source::Measurement {
                instrument: None,
                protocol: None,
                timestamp: Some(recent_date.clone()),
            },
            ..Default::default()
        };

        let constraint = TemporalConstraint::Current { max_age_days: 365 };
        let result = checker.check_temporal_constraint(&status, &constraint);

        // Should be satisfied since evidence is recent
        assert!(matches!(result, ConstraintResult::Satisfied));
    }

    #[test]
    fn test_temporal_constraint_current_violated() {
        let checker = EpistemicChecker::new();

        // Create status with old evidence
        let old_date = "2010-01-01";
        let status = EpistemicStatus {
            confidence: Confidence::new(0.9),
            evidence: vec![Evidence::new(
                EvidenceKind::Publication {
                    doi: Some("10.1234/paper".to_string()),
                },
                format!("published:{}", old_date),
                0.9,
            )],
            source: Source::Measurement {
                instrument: None,
                protocol: None,
                timestamp: Some(old_date.to_string()),
            },
            ..Default::default()
        };

        let constraint = TemporalConstraint::Current { max_age_days: 30 };
        let result = checker.check_temporal_constraint(&status, &constraint);

        // Should be violated since evidence is very old
        assert!(matches!(result, ConstraintResult::Violated(_)));
    }

    #[test]
    fn test_temporal_constraint_during_satisfied() {
        let checker = EpistemicChecker::new();

        let status = EpistemicStatus {
            confidence: Confidence::new(0.9),
            evidence: vec![Evidence::new(
                EvidenceKind::Experiment {
                    protocol: "clinical_trial".to_string(),
                },
                "trial:2023-06-15",
                0.95,
            )],
            ..Default::default()
        };

        let constraint = TemporalConstraint::During {
            start: TemporalIndex::from_iso("2023-01-01"),
            end: TemporalIndex::from_iso("2023-12-31"),
        };

        let result = checker.check_temporal_constraint(&status, &constraint);
        assert!(matches!(result, ConstraintResult::Satisfied));
    }

    #[test]
    fn test_temporal_constraint_during_violated_before() {
        let checker = EpistemicChecker::new();

        let status = EpistemicStatus {
            confidence: Confidence::new(0.9),
            evidence: vec![Evidence::new(
                EvidenceKind::Experiment {
                    protocol: "clinical_trial".to_string(),
                },
                "trial:2020-06-15",
                0.95,
            )],
            ..Default::default()
        };

        let constraint = TemporalConstraint::During {
            start: TemporalIndex::from_iso("2023-01-01"),
            end: TemporalIndex::from_iso("2023-12-31"),
        };

        let result = checker.check_temporal_constraint(&status, &constraint);
        assert!(matches!(result, ConstraintResult::Violated(_)));
    }

    #[test]
    fn test_temporal_constraint_at_time() {
        let checker = EpistemicChecker::new();

        let status = EpistemicStatus {
            confidence: Confidence::new(0.9),
            evidence: vec![Evidence::new(
                EvidenceKind::Verified {
                    verifier: "FDA".to_string(),
                },
                "verification:2024-06-15",
                0.99,
            )],
            ..Default::default()
        };

        // Within same day - should satisfy
        let constraint = TemporalConstraint::AtTime(TemporalIndex::from_iso("2024-06-15"));
        let result = checker.check_temporal_constraint(&status, &constraint);
        assert!(matches!(result, ConstraintResult::Satisfied));

        // Different day by more than 24 hours - should violate
        let constraint = TemporalConstraint::AtTime(TemporalIndex::from_iso("2024-01-01"));
        let result = checker.check_temporal_constraint(&status, &constraint);
        assert!(matches!(result, ConstraintResult::Violated(_)));
    }

    #[test]
    fn test_temporal_constraint_no_timestamp() {
        let checker = EpistemicChecker::new();

        // Status with no timestamp information
        let status = EpistemicStatus {
            confidence: Confidence::new(0.9),
            evidence: vec![Evidence::new(
                EvidenceKind::Publication {
                    doi: Some("10.1234/paper".to_string()),
                },
                "some-reference-without-date",
                0.9,
            )],
            ..Default::default()
        };

        let constraint = TemporalConstraint::Current { max_age_days: 30 };
        let result = checker.check_temporal_constraint(&status, &constraint);

        // For Current constraint, missing timestamp should be a violation
        // (medical/regulatory contexts require verifiable freshness)
        assert!(matches!(result, ConstraintResult::Violated(_)));
    }

    #[test]
    fn test_parse_timestamp_from_reference() {
        let checker = EpistemicChecker::new();

        // Reference with embedded date
        let ts = checker.parse_timestamp_from_reference("study:2024-03-15:results");
        assert!(ts.is_some());
        let ts = ts.unwrap();
        assert!(ts.epoch_secs.is_some());

        // Reference without date
        let ts = checker.parse_timestamp_from_reference("doi:10.1234/xyz");
        assert!(ts.is_none());

        // Reference with invalid date
        let ts = checker.parse_timestamp_from_reference("date:2024-13-45");
        assert!(ts.is_none());
    }

    #[test]
    fn test_days_from_date() {
        // Unix epoch
        assert_eq!(days_from_date(1970, 1, 1), Some(0));

        // One day after epoch
        assert_eq!(days_from_date(1970, 1, 2), Some(1));

        // 2000-01-01 = 10957 days after Unix epoch
        assert_eq!(days_from_date(2000, 1, 1), Some(10957));
    }
}
